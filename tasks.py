#
#
#   Tasks
#
#

import io
import inspect
import json
import redis
import boto3

from celery import Celery
from llama_cpp import Llama
from faster_whisper import WhisperModel
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
""")


class ProcessAudioTask(AbortableTask):
    """
    Abstraction of Celery's Task class to support loading ML model.
    """
    abstract = True

    def __init__(self):
        super().__init__()

        self.whisper = None
        self.llm = None
        self.polly = None

    def __call__(self, *args, **kwargs):
        """
        Load model on first call (i.e. first task processed)
        Avoids the need to load model on each task request
        """
        if not self.whisper:
            print("Loading Whisper ...")
            self.whisper = WhisperModel("large-v3", device="cuda", compute_type="float16")

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
                region_name="eu-west-3",
                aws_access_key_id=settings.AWS_ACCESS_KEY,
                aws_secret_access_key=settings.AWS_SECRET_KEY,
            )

        return self.run(*args, **kwargs)


@app.task(bind=True, base=ProcessAudioTask)
def process_audio(self, audio_pcm_l16_bytes: bytes, messages: list):
    redis_client.expire(self.request.id, 300)  # Set key to expire in 5 minutes

    pcm_l16_wav = create_wav(audio_pcm_l16_bytes, 16_000)

    segments, info = self.whisper.transcribe(io.BytesIO(pcm_l16_wav), beam_size=5, language="es")
    audio_transcription = " ".join([segment.text for segment in segments])

    if not audio_transcription:
        redis_client.lpush(self.request.id, b"end;json:" + json.dumps({
            "user": None,
            "assistant": None
        }).encode())  # Signal end of audio

    messages = [{"role": "system", "content": SYSTEM_PROMPT}] + messages

    messages.append({"role": "user", "content": audio_transcription})

    stream_completion = self.llm.create_chat_completion_openai_v1(
        messages=messages,
        stream=True
    )

    bot_responses = []
    for sentence in sent_tokenize_stream(stream_completion):
        if self.is_aborted():
            return

        sentence = sentence.strip()

        bot_responses.append(sentence)

        tts_response = self.polly.synthesize_speech(
            OutputFormat="pcm",
            Text=sentence,
            VoiceId="Lucia",
            SampleRate="16000"
        )

        redis_client.lpush(self.request.id, tts_response["AudioStream"].read())

    redis_client.lpush(self.request.id, b"end;json:" + json.dumps({
        "user": audio_transcription,
        "assistant": " ".join(bot_responses)
    }).encode())  # Signal end of audio
