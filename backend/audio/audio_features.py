import numpy as np
from scipy.signal import find_peaks


def rms_energy(audio):
    """Calculate RMS energy of the audio."""
    audio = audio.astype(np.float32) / 32768.0
    return float(np.sqrt(np.mean(audio ** 2)))


def decibel(audio):
    """Calculate average loudness in decibels."""
    rms = rms_energy(audio)

    if rms <= 1e-10:
        return -96.0

    return float(20 * np.log10(rms))


def frame_audio(audio, sample_rate, frame_duration=0.03):
    """Split audio into short frames."""
    audio = audio.astype(np.float32) / 32768.0

    frame_size = int(sample_rate * frame_duration)

    if frame_size <= 0:
        return np.array([])

    frames = []

    for i in range(0, len(audio) - frame_size + 1, frame_size):
        frames.append(audio[i:i + frame_size])

    return np.array(frames)


def speech_detection(audio, sample_rate):
    """
    Simple energy-based speech detection.

    Returns:
        speech_ratio
        pause_ratio
        speech_frames
        total_frames
    """

    frames = frame_audio(audio, sample_rate)

    if len(frames) == 0:
        return 0.0, 1.0, 0, 0

    energies = np.sqrt(np.mean(frames ** 2, axis=1))

    # Adaptive threshold based on recording level
    threshold = max(np.percentile(energies, 20) * 2.0, 0.005)

    speech = energies > threshold

    speech_frames = int(np.sum(speech))
    total_frames = len(frames)

    speech_ratio = speech_frames / total_frames
    pause_ratio = 1.0 - speech_ratio

    return (
        float(speech_ratio),
        float(pause_ratio),
        speech_frames,
        total_frames
    )


def zero_crossing_rate(audio, sample_rate):
    """Calculate average zero crossing rate."""

    frames = frame_audio(audio, sample_rate)

    if len(frames) == 0:
        return 0.0

    rates = []

    for frame in frames:
        crossings = np.sum(
            np.abs(np.diff(np.sign(frame))) > 0
        )

        rates.append(crossings / len(frame))

    return float(np.mean(rates))


def spectral_centroid(audio, sample_rate):
    """Calculate average spectral centroid."""

    frames = frame_audio(audio, sample_rate)

    if len(frames) == 0:
        return 0.0

    centroids = []

    for frame in frames:

        window = np.hanning(len(frame))
        spectrum = np.abs(np.fft.rfft(frame * window))

        frequencies = np.fft.rfftfreq(
            len(frame),
            1 / sample_rate
        )

        magnitude_sum = np.sum(spectrum)

        if magnitude_sum == 0:
            centroids.append(0.0)
        else:
            centroid = np.sum(
                frequencies * spectrum
            ) / magnitude_sum

            centroids.append(centroid)

    return float(np.mean(centroids))


def estimate_pitch(audio, sample_rate):
    """
    Estimate fundamental frequency using autocorrelation.
    """

    audio = audio.astype(np.float32) / 32768.0

    # Use a middle section to avoid startup noise
    if len(audio) > sample_rate * 3:
        audio = audio[:sample_rate * 3]

    audio = audio - np.mean(audio)

    if np.max(np.abs(audio)) < 0.001:
        return 0.0

    autocorr = np.correlate(
        audio,
        audio,
        mode="full"
    )

    autocorr = autocorr[len(autocorr) // 2:]

    # Human speech fundamental frequency range
    min_freq = 70
    max_freq = 400

    min_lag = int(sample_rate / max_freq)
    max_lag = int(sample_rate / min_freq)

    if max_lag >= len(autocorr):
        max_lag = len(autocorr) - 1

    if min_lag >= max_lag:
        return 0.0

    region = autocorr[min_lag:max_lag]

    if len(region) == 0:
        return 0.0

    lag = np.argmax(region) + min_lag

    if autocorr[lag] <= 0:
        return 0.0

    pitch = sample_rate / lag

    return float(pitch)


def speaking_rate(audio, sample_rate):
    """
    Estimate speaking activity as speech events per second.
    """

    frames = frame_audio(audio, sample_rate)

    if len(frames) == 0:
        return 0.0

    energies = np.sqrt(np.mean(frames ** 2, axis=1))

    threshold = max(
        np.percentile(energies, 20) * 2.0,
        0.005
    )

    speech = energies > threshold

    # Find transitions from silence -> speech
    transitions = np.diff(
        speech.astype(np.int8)
    )

    speech_starts = np.sum(transitions == 1)

    duration = len(audio) / sample_rate

    if duration <= 0:
        return 0.0

    return float(speech_starts / duration)


def audio_engagement_score(
    speech_ratio,
    pause_ratio,
    rms,
    pitch,
    speaking_rate
):
    """
    Calculate a simple 0-100 audio engagement score.

    Higher speech activity, healthy energy,
    and moderate speaking activity increase score.
    """

    speech_component = speech_ratio * 50

    energy_component = min(
        rms / 0.08,
        1.0
    ) * 20

    pause_component = (
        (1.0 - pause_ratio) * 15
    )

    pitch_component = 10 if 70 <= pitch <= 400 else 0

    rate_component = min(
        speaking_rate / 2.0,
        1.0
    ) * 5

    score = (
        speech_component
        + energy_component
        + pause_component
        + pitch_component
        + rate_component
    )

    return float(np.clip(score, 0, 100))


def extract_audio_features(audio, sample_rate):
    """
    Extract all audio engagement features.
    """

    rms = rms_energy(audio)

    db = decibel(audio)

    speech_ratio, pause_ratio, speech_frames, total_frames = (
        speech_detection(audio, sample_rate)
    )

    pitch = estimate_pitch(audio, sample_rate)

    rate = speaking_rate(audio, sample_rate)

    zcr = zero_crossing_rate(audio, sample_rate)

    centroid = spectral_centroid(audio, sample_rate)

    score = audio_engagement_score(
        speech_ratio,
        pause_ratio,
        rms,
        pitch,
        rate
    )

    return {
        "rms_energy": round(rms, 5),
        "decibel": round(db, 2),
        "speech_ratio": round(speech_ratio, 3),
        "pause_ratio": round(pause_ratio, 3),
        "pitch_hz": round(pitch, 2),
        "speaking_rate": round(rate, 3),
        "zero_crossing_rate": round(zcr, 3),
        "spectral_centroid": round(centroid, 2),
        "speech_frames": speech_frames,
        "total_frames": total_frames,
        "audio_engagement_score": round(score, 2)
    }