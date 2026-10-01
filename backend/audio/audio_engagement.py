import numpy as np


def analyze_audio(audio, sample_rate):
    from audio_features import extract_audio_features

    features = extract_audio_features(audio, sample_rate)

    # Additional meeting-level indicators
    duration = len(audio) / sample_rate if sample_rate else 0

    speech_seconds = duration * features["speech_ratio"]
    pause_seconds = duration * features["pause_ratio"]

    features["duration_seconds"] = round(duration, 2)
    features["speech_seconds"] = round(speech_seconds, 2)
    features["pause_seconds"] = round(pause_seconds, 2)

    # Simple activity classification
    score = features["audio_engagement_score"]

    if score >= 70:
        level = "High"
    elif score >= 40:
        level = "Medium"
    else:
        level = "Low"

    features["engagement_level"] = level

    return features


def print_audio_report(features):
    print("\n========== AUDIO ENGAGEMENT ==========")

    for key, value in features.items():
        print(f"{key}: {value}")

    print("======================================\n")


if __name__ == "__main__":
    from record import record_audio

    audio, sample_rate = record_audio(
        duration=10,
        filename="meeting_audio.wav"
    )

    results = analyze_audio(audio, sample_rate)
    print_audio_report(results)