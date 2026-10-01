from record import record_audio
from audio_features import extract_audio_features


if __name__ == "__main__":

    audio, sample_rate = record_audio(
        duration=10,
        filename="meeting_audio.wav"
    )

    print("\n================ AUDIO ANALYSIS ================\n")

    features = extract_audio_features(audio, sample_rate)

    for name, value in features.items():
        print(f"{name}: {value}")

    print("\n=================================================")