#
#
#   Utils
#
#

import io
import time
import wave
import nltk
import numpy as np
from nltk import sent_tokenize


nltk.download("punkt")


def create_wav(pcm_l16: bytes | np.ndarray, sample_rate: 16_000):
    if isinstance(pcm_l16, np.ndarray):
        pcm_l16 = pcm_l16.tobytes()

    wav_io = io.BytesIO()
    with wave.open(wav_io, "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(pcm_l16)
    return wav_io


def sent_tokenize_stream(stream: iter):
    sentence = ""
    start_time = time.time()
    for chunk in stream:
        text = chunk.choices[0].delta.content

        if text:
            sentence += text

        sentence = sentence.replace(":", ".")
        tokenized_sentences = sent_tokenize(sentence, language="spanish")

        if len(tokenized_sentences) > 1:
            print(f"Sentence: {time.time() - start_time:.4f}s")

            tts_sentence = tokenized_sentences.pop(0)
            sentence = " ".join(tokenized_sentences)

            yield tts_sentence

            start_time = time.time()

    if sentence:
        print(f"Sentence: {time.time() - start_time:.4f}s")

        yield sentence  # Send remaining sentence
