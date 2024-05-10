# realtime-voice-assistant
Real Time Voice Assistant using LLMs

## Installation
```
$ pip install -r requirements.txt
```

## Usage
1. Run the worker
```console
$ celery -A tasks worker --loglevel=info
```
2. Run the app
```
$ python app.py
```
