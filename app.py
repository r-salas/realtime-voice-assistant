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
    frame_ms = 60
    frame_length = int(16_000 * (frame_ms / 1_000))
    recorder = PvRecorder(frame_length=frame_length)

    model, _ = torch.hub.load(
        repo_or_dir='snakers4/silero-vad',
        model='silero_vad',
        force_reload=False,
        onnx=True
    )

    play_queue = queue.Queue()

    threading.Thread(target=play_audio, args=(play_queue,)).start()

    messages = []

    last_speech_datetime = None
    audio_buffer_pcm_lin16 = np.array([], dtype=np.int16)

    redis_client = redis.Redis(host=settings.REDIS_HOST, port=settings.REDIS_PORT, db=settings.REDIS_DATABASE,
                               password=settings.REDIS_PASSWORD)

    recorder.start()
    print(f"Listening on {recorder.selected_device} ...")

    while recorder.is_recording:
        frame = recorder.read()
        frame = np.array(frame, dtype=np.int16)

        norm_frame = frame / 2**15  # Normalize the frame
        frame_tensor = torch.tensor(norm_frame).float()
        speech_prob = model(frame_tensor, 16_000).item()
        has_speech = speech_prob > 0.5

        if not has_speech and last_speech_datetime is None:
            continue   # Silence and user has not spoken yet

        audio_buffer_pcm_lin16 = np.concatenate((audio_buffer_pcm_lin16, frame))

        if has_speech:
            last_speech_datetime = datetime.datetime.now()
            print(f"Speech detected! [{speech_prob:.2f}]")
        else:
            elapsed_microseconds = (datetime.datetime.now() - last_speech_datetime).microseconds
            elapsed_miliseconds = elapsed_microseconds / 1_000

            if elapsed_miliseconds > 500:
                recorder.stop()

                audio_buffer_pcm_lin16_bytes = audio_buffer_pcm_lin16.tobytes()
                current_task = tasks.process_audio.delay(audio_buffer_pcm_lin16_bytes, messages)

                while True:
                    print(f"Retrieving audio from {current_task.id}")
                    _, audio_bytes = redis_client.brpop(current_task.id, timeout=0)

                    if audio_bytes.startswith(b"end;json:"):
                        data = json.loads(audio_bytes[9:])

                        if not data["user"]:
                            break   # No transcription available, continue listening

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

                recorder.start()

                last_speech_datetime = None
                audio_buffer_pcm_lin16 = np.array([], dtype=np.int16)

    play_queue.put(None)
    play_queue.join()


if __name__ == "__main__":
    main()
