#
#
#   Settings
#
#

import os
import inspect

AZURE_SPEECH_KEY = os.environ["AZURE_SPEECH_KEY"]
AZURE_SPEECH_REGION = os.environ["AZURE_SPEECH_REGION"]

OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]
OPENAI_MODEL = "gpt-3.5-turbo-0125"
SYSTEM_PROMPT = inspect.cleandoc("""
    You are voice assistant.
    You are concise.
    You are designed to help users with their daily tasks.
    You will always speak in the following language: {language}
""")
AWS_ACCESS_KEY = os.environ["AWS_ACCESS_KEY"]
AWS_SECRET_KEY = os.environ["AWS_SECRET_KEY"]

REDIS_PORT = os.getenv("REDIS_PORT", 6379)
REDIS_DATABASE = os.getenv("REDIS_DATABASE", 0)
REDIS_PASSWORD = os.getenv("REDIS_PASSWORD", "")
REDIS_HOST = os.getenv("REDIS_HOST", "localhost")

DEEPGRAM_API_KEY = os.environ["DEEPGRAM_API_KEY"]
