#
#
#   Main
#
#

import concurrent
import inspect
import queue
import threading
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor

import azure.cognitiveservices.speech as speechsdk
import httpx
import pyaudio
from anthropic import AnthropicBedrock
from azure.cognitiveservices.speech import ProfanityOption
from nltk.tokenize import sent_tokenize

import settings
from enums import Language


def _get_azure_speech_config(language: Language) -> speechsdk.SpeechConfig:
    azure_speech_config = speechsdk.SpeechConfig(subscription=settings.AZURE_SPEECH_KEY,
                                                 region=settings.AZURE_SPEECH_REGION)
    azure_speech_config.speech_recognition_language = language.azure_language
    azure_speech_config.set_profanity(ProfanityOption.Raw)  # Don't censor words
    azure_speech_config.enable_dictation()
    azure_speech_config.set_service_property(
        name="speechsdk.PropertyId.Speech_SegmentationSilenceTimeoutMs",
        value="200",
        channel=speechsdk.ServicePropertyChannel.UriQueryParameter)
    azure_speech_config.set_speech_synthesis_output_format(speechsdk.SpeechSynthesisOutputFormat.Riff48Khz16BitMonoPcm)
    azure_speech_config.speech_synthesis_voice_name = language.azure_voice

    return azure_speech_config


def play_audio(play_queue: queue.Queue):
    p = pyaudio.PyAudio()

    bit_depth = 16
    sample_rate = 24_000

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


def buffer_audio(play_queue: queue.Queue, buffer_queue: queue.Queue):
    current_rank = 0
    buffer_by_rank = defaultdict(lambda: {"values": bytearray(), "end": False})

    while True:
        result = buffer_queue.get()

        if result is None:
            buffer_queue.task_done()
            break

        rank, audio = result

        if rank is None:
            # Do nothing, just reset rank
            current_rank = 0
            buffer_by_rank.clear()
            buffer_queue.task_done()
            continue

        if audio is None:
            buffer_by_rank[rank]["end"] = True
        else:
            buffer_by_rank[rank]["values"].extend(audio)

        current_rank_data = buffer_by_rank.pop(current_rank, None)

        if current_rank_data is None:
            # It shouldn't happen, but just in case
            buffer_queue.task_done()
            continue

        if current_rank_data["end"]:
            current_rank += 1  # Once we have the end of the audio, we can move to the next rank

        if current_rank_data["values"]:
            play_queue.put(bytes(current_rank_data["values"]))

        buffer_queue.task_done()


def azure_tts(text: str, language: Language, buffer_queue: queue.Queue, rank: int):
    url = f"https://{settings.AZURE_SPEECH_REGION}.tts.speech.microsoft.com/cognitiveservices/v1"

    headers = {
        "Ocp-Apim-Subscription-Key": settings.AZURE_SPEECH_KEY,
        "Content-Type": "application/ssml+xml",
        "X-Microsoft-OutputFormat": "raw-24khz-16bit-mono-pcm",
        "User-Agent": "httpx",
    }

    text = text.replace("\n", " ")

    data = f"""
        <speak version='1.0' xml:lang='en-US'>
            <voice xml:lang='{language.azure_language}' xml:gender='Female' name='{language.azure_voice}'>
                {text}
            </voice>
        </speak>
    """

    with httpx.Client() as client:
        with client.stream("POST", url, headers=headers, data=data) as r:
            for data in r.iter_bytes(chunk_size=4096):
                buffer_queue.put((rank, data))

    buffer_queue.put((rank, None))  # Signal end of audio


def assistant(messages, language: Language):
    anthropic_client = AnthropicBedrock(
        aws_access_key=settings.AWS_ACCESS_KEY,
        aws_secret_key=settings.AWS_SECRET_KEY,
        aws_region=settings.AWS_REGION,
    )

    system_prompt = inspect.cleandoc(f"""
        You are voice assistant. 
        You are concise.
        You are designed to help users with their daily tasks.
        You will always speak in the following language: {language.value.lower()}
    """)

    with anthropic_client.messages.stream(
            model="anthropic.claude-3-haiku-20240307-v1:0",
            system=system_prompt,
            messages=messages,
            max_tokens=1024,
            temperature=0.2
    ) as stream:
        sentence = ""
        for text in stream.text_stream:
            sentence += text

            tokenized_sentences = sent_tokenize(sentence)

            if len(tokenized_sentences) > 1:
                tts_sentence = tokenized_sentences.pop(0)
                sentence = " ".join(tokenized_sentences)

                yield tts_sentence


def main(language: Language = Language.SPANISH):
    messages = []

    play_queue = queue.Queue()
    buffer_queue = queue.Queue()

    threading.Thread(target=play_audio, args=(play_queue,)).start()
    threading.Thread(target=buffer_audio, args=(play_queue, buffer_queue,)).start()

    try:
        while True:
            user_text = input(">>> ")
            user_text = user_text.encode('utf-8', 'replace').decode('utf-8')

            messages.append({
                "role": "user",
                "content": user_text
            })

            bot_text = ""
            with concurrent.futures.ThreadPoolExecutor() as executor:
                for rank, sentence in enumerate(assistant(messages, language)):
                    bot_text += sentence
                    executor.submit(azure_tts, sentence, language, buffer_queue, rank)

            messages.append({
                "role": "assistant",
                "content": bot_text
            })

            buffer_queue.put((None, None))  # Signal end of messages
    except KeyboardInterrupt:
        pass

    play_queue.put(None)
    buffer_queue.put(None)

    play_queue.join()
    buffer_queue.join()


if __name__ == "__main__":
    main()
