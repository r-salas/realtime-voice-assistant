#
#
#   Main 3
#
#
import base64
import json
import queue
import time

import redis
import threading
import simpleaudio
import numpy as np

from deepgram import DeepgramClientOptions, DeepgramClient, LiveTranscriptionEvents, LiveOptions
from pvrecorder import PvRecorder

import settings
import tasks


def process(text_queue: queue.Queue, stop_event: threading.Event):
    messages = []

    redis_client = redis.Redis(host=settings.REDIS_HOST, port=settings.REDIS_PORT, db=settings.REDIS_DATABASE,
                               password=settings.REDIS_PASSWORD)

    while True:
        print("Waiting for transcription ...")
        text = text_queue.get()
        print("Transcription received: ", text)

        stop_event.clear()

        if text is None:
            text_queue.task_done()
            return  # Don't do anything else

        messages.append({"role": "user", "content": text})

        current_task = tasks.process.delay(messages)
        print("Starting task : ", current_task.id)

        first_sentence = True
        while True:
            data = redis_client.brpop(current_task.id, timeout=0.05)

            if stop_event.is_set():
                print(f"Aborting task {current_task.id} and stopping ...")

                stop_event.clear()
                break

            if data is None:
                time.sleep(0.05)

                continue

            print("Audio received")

            _, response = data

            if response == b"==END==":  # No more audios to play
                print("No more audios to play")
                break

            response = json.loads(response)
            sentence = response["sentence"]
            audio_bytes = base64.b64decode(response["audio"])

            if first_sentence:
                messages.append({"role": "assistant", "content": sentence})
            else:
                messages[-1]["content"] += f" {sentence}"

            first_sentence = False

            play_obj = simpleaudio.play_buffer(
                audio_bytes,
                num_channels=1,
                bytes_per_sample=2,
                sample_rate=16_000
            )

            while True:
                if stop_event.is_set():
                    print("Stopping ...")

                    break

                if not play_obj.is_playing():
                    break

                time.sleep(0.05)

            if stop_event.is_set():
                print(f"Aborting task {current_task.id} and stopping ...")

                play_obj.stop()

                stop_event.clear()
                break

        text_queue.task_done()


def main():
    config = DeepgramClientOptions(
        options={"keepalive": "true"}
    )
    deepgram = DeepgramClient(settings.DEEPGRAM_API_KEY, config)

    dg_connection = deepgram.listen.live.v("1")

    stop_event = threading.Event()

    text_queue = queue.Queue()

    def on_message(instance, result):
        transcript = result.channel.alternatives[0].transcript

        if not transcript:
            return

        print("[Partial] >>> ", transcript)

        # Stop and clear queue
        stop_event.set()

        with text_queue.mutex:
            text_queue.queue.clear()

        if result.speech_final:
            print("[Final] >>> ", transcript)
            text_queue.put(transcript)

    dg_connection.on(LiveTranscriptionEvents.Transcript, on_message)

    deepgram_options = LiveOptions(
        model="nova-2",
        punctuate=True,
        language=settings.DEEPGRAM_LANGUAGE,
        encoding="linear16",
        channels=1,
        sample_rate=16_000,
        endpointing="100",
        interim_results=True,
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

    threading.Thread(target=process, args=(text_queue, stop_event)).start()

    recorder.start()
    print(f"Listening on {recorder.selected_device} ...")

    try:
        while recorder.is_recording:
            frame = recorder.read()
            frame = np.array(frame, dtype=np.int16)

            dg_connection.send(frame.tobytes())
    except KeyboardInterrupt:
        pass

    recorder.stop()

    stop_event.set()

    with text_queue.mutex:
        text_queue.queue.clear()

    text_queue.put(None)
    text_queue.join()


if __name__ == "__main__":
    main()
