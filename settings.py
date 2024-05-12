#
#
#   Settings
#
#

import os
import inspect

AWS_ACCESS_KEY = os.environ["AWS_ACCESS_KEY"]
AWS_SECRET_KEY = os.environ["AWS_SECRET_KEY"]

REDIS_PORT = os.getenv("REDIS_PORT", 6379)
REDIS_DATABASE = os.getenv("REDIS_DATABASE", 0)
REDIS_PASSWORD = os.getenv("REDIS_PASSWORD", "")
REDIS_HOST = os.getenv("REDIS_HOST", "localhost")

DEEPGRAM_API_KEY = os.environ["DEEPGRAM_API_KEY"]

SYSTEM_PROMPT = inspect.cleandoc("""
    You are voice assistant.
    You will speak in Spanish.  
    You are working for a telecommunications company named Lowi.
    You are a customer service assistant.
    You are polite.
    You are helpful.
    You will introduce yourself as the virtual assistant of Lowi.
    You will ask the user for their name.
    The first time you speak, you will introduce yourself.
    Your answers will be short and concise.
    You will ask for clarification if you do not understand the user.
""")

DEEPGRAM_LANGUAGE = "es-ES"
