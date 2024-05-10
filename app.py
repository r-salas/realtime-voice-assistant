#
#
#   Main 3
#
#
import io
import json
import redis
import torch
import queue
import datetime
import threading
import numpy as np
from deepgram import DeepgramClientOptions, DeepgramClient, LiveTranscriptionEvents, LiveOptions

from pvrecorder import PvRecorder
from pydub import AudioSegment
from pydub.playback import play as pydub_play

import tasks

import settings


def play_audio(play_queue: queue.Queue):
    while True:
        audio = play_queue.get()

        if audio is None:
            play_queue.task_done()
            break

        audio_segment = AudioSegment.from_file(
            file=io.BytesIO(audio),
            format="raw",
            frame_rate=16_000,
            channels=1,
            sample_width=2
        )
        pydub_play(audio_segment)

        play_queue.task_done()


def main():
    messages = []

    config = DeepgramClientOptions(
        options={"keepalive": "true"}
    )
    deepgram = DeepgramClient(settings.DEEPGRAM_API_KEY, config)

    dg_connection = deepgram.listen.live.v("1")

    def on_message(instance, result):
        nonlocal messages

        if not result.speech_final:
            return

        transcript = result.channel.alternatives[0].transcript

        if not transcript:
            return

        # TODO: stop recording when user stops speaking

        current_task = tasks.process.delay(transcript, messages)

        while True:
            print(f"Retrieving audio from {current_task.id}")
            _, audio_bytes = redis_client.brpop(current_task.id, timeout=0)

            if audio_bytes.startswith(b"end;json:"):
                data = json.loads(audio_bytes[9:])

                if not data["user"]:
                    break  # No transcription available, continue listening

                print(">>> " + data["user"])
                print(data["assistant"])

                messages.extend([
                    {"role": "user", "content": data["user"]},
                    {"role": "assistant", "content": data["assistant"]}

                ])

                break  # Don't keep receiving audio
            else:
                print(f"Playing audio from {current_task.id}")
                play_queue.put(audio_bytes)

    dg_connection.on(LiveTranscriptionEvents.Transcript, on_message)

    deepgram_options = LiveOptions(
        model="nova-2",
        punctuate=True,
        language='es-ES',
        encoding="linear16",
        channels=1,
        sample_rate=16_000,
        endpointing="100"
    )

    deepgram_addons = {
        # Prevent waiting for additional numbers
        "no_delay": "true"
    }

    if dg_connection.start(deepgram_options, addons=deepgram_addons) is False:
        print("Failed to connect to Deepgram")
        return

    frame_ms = 30
    frame_length = int(16_000 * (frame_ms / 1_000))
    recorder = PvRecorder(frame_length=frame_length)

    play_queue = queue.Queue()

    threading.Thread(target=play_audio, args=(play_queue,)).start()

    redis_client = redis.Redis(host=settings.REDIS_HOST, port=settings.REDIS_PORT, db=settings.REDIS_DATABASE,
                               password=settings.REDIS_PASSWORD)

    recorder.start()
    print(f"Listening on {recorder.selected_device} ...")

    while recorder.is_recording:
        frame = recorder.read()
        frame = np.array(frame, dtype=np.int16)

        dg_connection.send(frame.tobytes())

    play_queue.put(None)
    play_queue.join()


if __name__ == "__main__":
    main()
