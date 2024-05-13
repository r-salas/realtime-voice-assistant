#
#
#   Tasks
#
#

import base64
import json
import time
import warnings
import redis
import boto3

from celery import Celery, Task
from llama_cpp import Llama

import settings
from utils import sent_tokenize_stream, is_gpu_available_for_llama


app = Celery(
    "tasks",
    broker=f"redis://:{settings.REDIS_PASSWORD}@{settings.REDIS_HOST}:{settings.REDIS_PORT}/{settings.REDIS_DATABASE}",
    backend=f"redis://:{settings.REDIS_PASSWORD}@{settings.REDIS_HOST}:{settings.REDIS_PORT}/{settings.REDIS_DATABASE}",
)


class ProcessTask(Task):
    """
    Abstraction of Celery's Task class to support loading ML model.
    """
    abstract = True

    def __init__(self):
        super().__init__()

        self.llm = None
        self.polly = None
        self.redis = None

    def __call__(self, *args, **kwargs):
        """
        Load model on first call (i.e. first task processed)
        Avoids the need to load model on each task request
        """
        if not self.llm:
            print("Loading Llama ...")

            n_gpu_layers = -1
            if not is_gpu_available_for_llama():
                n_gpu_layers = 0
                warnings.warn("No GPU available for Llama. Using CPU.")

            self.llm = Llama.from_pretrained(
                repo_id="SanctumAI/Meta-Llama-3-8B-Instruct-GGUF",
                filename="meta-llama-3-8b-instruct.Q4_0.gguf",
                verbose=False,
                n_gpu_layers=n_gpu_layers,
                n_ctx=8192
            )

        if not self.redis:
            print("Loading Redis ...")
            self.redis = redis.Redis(
                host=settings.REDIS_HOST,
                port=settings.REDIS_PORT,
                db=settings.REDIS_DATABASE,
                password=settings.REDIS_PASSWORD,
            )

        if not self.polly:
            print("Loading Polly ...")
            self.polly = boto3.client(
                "polly",
                region_name="eu-central-1",
                aws_access_key_id=settings.AWS_ACCESS_KEY,
                aws_secret_access_key=settings.AWS_SECRET_KEY,
            )

        return self.run(*args, **kwargs)


@app.task(bind=True, base=ProcessTask, name="tasks.process_transcription")
def process_transcription(self, messages: list):
    self.redis.expire(self.request.id, 300)  # Set key to expire in 5 minutes

    messages = [{"role": "system", "content": settings.SYSTEM_PROMPT}] + messages

    stream_completion = self.llm.create_chat_completion_openai_v1(
        messages=messages,
        stream=True
    )

    bot_responses = []
    for sentence in sent_tokenize_stream(stream_completion):
        sentence = sentence.strip()

        bot_responses.append(sentence)

        start_time = time.time()
        tts_response = self.polly.synthesize_speech(
            OutputFormat="pcm",
            Text=sentence,
            VoiceId="Lucia",
            SampleRate="16000"
        )
        audio_bytes = tts_response["AudioStream"].read()
        print(f"TTS: {time.time() - start_time:.4f}s")

        self.redis.lpush(self.request.id, json.dumps({
            "audio": base64.b64encode(audio_bytes).decode(),
            "sentence": sentence
        }))

    self.redis.lpush(self.request.id, b"==END==")  # Signal end of audio
