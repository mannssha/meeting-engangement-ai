import sounddevice as sd
import numpy as np
from scipy.io.wavfile import write


DEVICE = 1
SAMPLE_RATE = 44100


def record_audio(duration=10, filename="meeting_audio.wav"):
    print(f"Recording for {duration} seconds...")
    print("Speak into your microphone.")

    audio = sd.rec(
        int(duration * SAMPLE_RATE),
        samplerate=SAMPLE_RATE,
        channels=1,
        dtype="int16",
        device=DEVICE
    )

    sd.wait()

    audio = audio.flatten()

    write(filename, SAMPLE_RATE, audio)

    print("Recording finished.")
    print(f"Saved: {filename}")

    return audio, SAMPLE_RATE


def normalize_audio(audio):
    return audio.astype(np.float32) / 32768.0


if __name__ == "__main__":
    record_audio()