import wave
import asyncio
import os
import pyaudio

from typing import List, Optional, TypeVar, Union, IO
from os import PathLike
from deepgram import Deepgram
WaveWrite = TypeVar("WaveWrite", bound=wave.Wave_write)  # can be any subtype of Wave_write 


class SpeechSynthesizer:
    def __init__(self, default_samplerate: int = 44100, frames_per_buffer: int = 1024, duration: int = 5):
        self.default_samplerate = default_samplerate
        self.frames_per_buffer = frames_per_buffer
        self.duration = duration  # duration in seconds
        self.py_audio = pyaudio.PyAudio()
        self.deepgram = Deepgram(os.environ.get("DG_API_KEY"))

    def init_recording(self, file_name: Union[str, IO[bytes]] = "sound.wav", mode: Optional[str] = "wb") -> WaveWrite:
        wave_file = wave.open(file_name, mode)
        wave_file.setnchannels(2)
        wave_file.setsampwidth(2)
        wave_file.setframerate(self.default_samplerate)
        return wave_file


    def record(self, wave_file: WaveWrite, duration: Optional[int] = 3) -> None:
        audio_stream = self.py_audio.open(
            rate=self.default_samplerate,  # frames per second,
            channels=2,  # stereo, set to 1 for mono
            format=8,  # sample format, 8 bytes. see inspect
            input=True,  # input device flag
            frames_per_buffer=self.frames_per_buffer,  # 1024 samples per frame
        )
        frames = []
        for _ in range(int(self.default_samplerate / self.frames_per_buffer * self.duration)):
            data = audio_stream.read(self.frames_per_buffer)
            frames.append(data)
        wave_file.writeframes(b"".join(frames))
        audio_stream.close()


    async def transcribe(self, file_name: Union[Union[str, bytes, PathLike[str], PathLike[bytes]], int]):
        with open(file_name, "rb") as audio:
            source = {"buffer": audio, "mimetype": "audio/wav"}
            response = await self.deepgram.transcription.prerecorded(source)
            return response["results"]["channels"][0]["alternatives"][0]["words"]


    def produce_transcription(self):
        # Start recording
        print("Python is listening..")
        wave_file = self.init_recording()
        self.record(wave_file, duration=self.duration) 

        # Start transcribing
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        # Argument is a coroutine, and is thus scheduled as a asyncio.Task
        words = loop.run_until_complete(self.transcribe("sound.wav"))
        string_words = " ".join(word_dict.get("word") for word_dict in words if "word" in word_dict)
        loop.close()
        return string_words
    