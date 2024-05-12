# realtime-voice-assistant
Real Time Voice Assistant using LLMs

## Requirements
- Python 3
- Redis

## Installation
1. Install llama-cpp-python according to the instructions in the [llama-cpp-python](https://llama-cpp-python.readthedocs.io/en/stable/#installation) documentation.
2. Install requirements
```
$ pip install -r requirements.txt
```
3. The following environment variables are required:
```
AWS_ACCESS_KEY=<Your AWS Access Key>
AWS_SECRET_KEY=<Your AWS Secret Key>
DEEPGRAM_API_KEY=<Your Deepgram API Key>
```
4. (Optional) Customize your settings in `settings.py`. You may want to customize the language, by default it is set to `es-ES`.

## Usage
1. Run the worker
```console
$ celery -A tasks worker --loglevel=info
```
2. Run the app
```
$ python app.py
```

## Mac OS Notes
It's recommended to use the `solo` pool for the worker on Mac OS. This is because the `prefork` pool is not supported on Mac OS. To run the worker with the `solo` pool, use the following command:
```console
$ celery -A tasks worker --loglevel=info --pool=solo
```
