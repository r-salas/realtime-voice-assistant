#
#
#   Main 3
#
#
import json

import redis
import torch
import queue
import pyaudio
import datetime
import threading
import numpy as np

from pvrecorder import PvRecorder

import tasks_2 as tasks


def play_audio(play_queue: queue.Queue):
    p = pyaudio.PyAudio()

    bit_depth = 16
    sample_rate = 16_000

    stream = p.open(
        format=p.get_format_from_width(bit_depth // 8),
        channels=1,
        rate=sample_rate,
        output=True,
    )

    while True:
        audio = play_queue.get()

        if audio is None:
            play_queue.task_done()
            break

        stream.write(audio)
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

    current_task = None
    last_speech_datetime = None
    audio_buffer_pcm_lin16 = np.array([], dtype=np.int16)

    redis_client = redis.Redis(host='localhost', port=6379, db=0)

    recorder.start()
    print(f"Listening on {recorder.selected_device} ...")

    while recorder.is_recording:
        frame = recorder.read()
        frame = np.array(frame, dtype=np.int16)

        norm_frame = frame / 32768.0  # Normalize the frame
        frame_tensor = torch.tensor(norm_frame).float()
        speech_prob = model(frame_tensor, 16_000).item()
        has_speech = speech_prob > 0.75

        if not has_speech and last_speech_datetime is None:
            continue   # Silence and user has not spoken yet

        audio_buffer_pcm_lin16 = np.concatenate((audio_buffer_pcm_lin16, frame))

        if has_speech:
            if current_task:
                current_task.abort()  # cancel task
                current_task = None

            last_speech_datetime = datetime.datetime.now()
            print(f"Speech detected! [{speech_prob:.2f}]")
        else:
            elapsed_microseconds = (datetime.datetime.now() - last_speech_datetime).microseconds
            elapsed_miliseconds = elapsed_microseconds / 1_000

            if not current_task and elapsed_miliseconds > 100:
                # MARK: start processing the audio
                audio_buffer_pcm_lin16_bytes = audio_buffer_pcm_lin16.tobytes()
                current_task = tasks.process_audio.delay(audio_buffer_pcm_lin16_bytes, messages)
            elif current_task and elapsed_miliseconds > 600:
                last_speech_datetime = None
                audio_buffer_pcm_lin16 = np.array([], dtype=np.int16)

                recorder.stop()

                while True:
                    print(f"Retrieving audio from {current_task.id}")
                    _, audio_bytes = redis_client.brpop(current_task.id, timeout=0)

                    if audio_bytes.startswith(b"json:"):
                        data = json.loads(audio_bytes[5:])
                        messages.extend([
                            {"role": "user", "content": data["user"]},
                            {"role": "assistant", "content": data["assistant"]}

                        ])

                        print("No more audios to play")
                        break

                    play_queue.put(audio_bytes)

                recorder.start()

    play_queue.put(None)
    play_queue.join()


if __name__ == "__main__":
    main()
