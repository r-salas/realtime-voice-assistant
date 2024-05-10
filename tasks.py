#
#
#   Tasks
#
#

import json
import time
import redis
import boto3
import inspect

from celery import Celery
from llama_cpp import Llama
from celery.contrib.abortable import AbortableTask

import settings
from utils import create_wav, sent_tokenize_stream

app = Celery(
    "tasks",
    broker=f"redis://:{settings.REDIS_PASSWORD}@{settings.REDIS_HOST}:{settings.REDIS_PORT}/{settings.REDIS_DATABASE}",
    backend=f"redis://:{settings.REDIS_PASSWORD}@{settings.REDIS_HOST}:{settings.REDIS_PORT}/{settings.REDIS_DATABASE}",
)

redis_client = redis.Redis(host=settings.REDIS_HOST, port=settings.REDIS_PORT, db=settings.REDIS_DATABASE,
                           password=settings.REDIS_PASSWORD)

SYSTEM_PROMPT = inspect.cleandoc("""
    You are voice assistant.
    You are concise.
    You will speak in Spanish.  
    You are working for a telecommunications company named Lowi.
    You are a customer service assistant.
    You are polite.
    You are helpful.
    You will introduce yourself as the virtual assistant of Lowi.
    You will ask the user for their name.
    You will always introduce yourself before speaking.
""")


class ProcessAudioTask(AbortableTask):
    """
    Abstraction of Celery's Task class to support loading ML model.
    """
    abstract = True

    def __init__(self):
        super().__init__()

        self.llm = None
        self.polly = None

    def __call__(self, *args, **kwargs):
        """
        Load model on first call (i.e. first task processed)
        Avoids the need to load model on each task request
        """
        if not self.llm:
            print("Loading Llama ...")
            self.llm = Llama.from_pretrained(
                repo_id="SanctumAI/Meta-Llama-3-8B-Instruct-GGUF",
                filename="meta-llama-3-8b-instruct.Q4_0.gguf",
                verbose=False,
                n_gpu_layers=-1
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


@app.task(bind=True, base=ProcessAudioTask)
def process(self, text: str, messages: list):
    redis_client.expire(self.request.id, 300)  # Set key to expire in 5 minutes

    messages = [{"role": "system", "content": SYSTEM_PROMPT},
                *messages,
                {"role": "user", "content": text}]

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

        redis_client.lpush(self.request.id, audio_bytes)

        if self.is_aborted():
            return

    redis_client.lpush(self.request.id, b"end;json:" + json.dumps({
        "user": text,
        "assistant": " ".join(bot_responses)
    }).encode())  # Signal end of audio
