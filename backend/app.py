import streamlit as st
import sys
from pathlib import Path
import cv2
import numpy as np
import av
import threading

from streamlit_webrtc import webrtc_streamer, WebRtcMode


# ==========================================================
# PATH
# ==========================================================

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


# ==========================================================
# AUDIO
# ==========================================================

from backend.audio.record import record_audio
from backend.audio.audio_features import extract_audio_features


# ==========================================================
# VISION
# ==========================================================

from backend.vision.face_landmarker import detect_face_landmarks
from backend.vision.gaze import calculate_gaze
from backend.vision.head_pose import (
    estimate_head_pose,
    classify_head_pose
)
from backend.vision.eye_closure import EyeClosureDetector
from backend.vision.engagement_score import EngagementScore


# ==========================================================
# LIVE VIDEO PROCESSOR
# ==========================================================

class VideoProcessor:

    def __init__(self):

        self.lock = threading.Lock()

        self.vision_score = 0.0

        self.gaze = 0.0
        self.gaze_direction = "UNKNOWN"

        self.head = "UNKNOWN"

        self.yaw = 0.0
        self.pitch = 0.0
        self.roll = 0.0

        self.eye_event = "UNKNOWN"
        self.blink_count = 0

        self.face_detected = False

        self.eye_detector = EyeClosureDetector()


    def recv(self, frame):

        image = frame.to_ndarray(
            format="bgr24"
        )

        h, w = image.shape[:2]

        try:

            # ==================================================
            # FACE LANDMARKS
            # ==================================================

            result = detect_face_landmarks(image)

            if (
                result is not None
                and result.face_landmarks
            ):

                landmarks = result.face_landmarks[0]


                # ==================================================
                # GAZE
                # ==================================================

                gaze = calculate_gaze(
                    landmarks
                )


                # ==================================================
                # HEAD POSE
                # ==================================================

                pose = estimate_head_pose(
                    landmarks,
                    w,
                    h
                )

                horizontal, vertical = classify_head_pose(
                    pose["yaw"],
                    pose["pitch"]
                )

                head_direction = (
                    f"{horizontal}/{vertical}"
                )


                # ==================================================
                # EYE
                # ==================================================

                eye = self.eye_detector.process(
                    landmarks
                )


                # ==================================================
                # ENGAGEMENT SCORE
                # ==================================================

                scorer = EngagementScore()

                vision = scorer.calculate(
                    gaze_attention=gaze[
                        "attention_score"
                    ],
                    head_direction=head_direction,
                    eye_event=eye["event"],
                    face_detected=True
                )


                # ==================================================
                # SAVE LIVE RESULTS
                # ==================================================

                with self.lock:

                    self.vision_score = vision["score"]

                    self.gaze = (
                        gaze["attention_score"] * 100
                    )

                    self.gaze_direction = (
                        gaze["direction"]
                    )

                    self.head = head_direction

                    self.yaw = pose["yaw"]
                    self.pitch = pose["pitch"]
                    self.roll = pose["roll"]

                    self.eye_event = eye["event"]

                    self.blink_count = (
                        eye["blink_count"]
                    )

                    self.face_detected = True


                # ==================================================
                # OVERLAY ON LIVE VIDEO
                # ==================================================

                cv2.putText(
                    image,
                    f"Engagement: {vision['score']:.1f}/100",
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9,
                    (0, 255, 0),
                    2
                )

                cv2.putText(
                    image,
                    f"Gaze: {gaze['direction']}",
                    (20, 75),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2
                )

                cv2.putText(
                    image,
                    f"Head: {head_direction}",
                    (20, 110),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2
                )

                cv2.putText(
                    image,
                    f"Eyes: {eye['event']}",
                    (20, 145),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2
                )


            else:

                with self.lock:

                    self.vision_score = 0.0
                    self.face_detected = False

                cv2.putText(
                    image,
                    "NO FACE DETECTED",
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9,
                    (0, 0, 255),
                    2
                )


        except Exception as e:

            cv2.putText(
                image,
                "VISION ERROR",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (0, 0, 255),
                2
            )


        return av.VideoFrame.from_ndarray(
            image,
            format="bgr24"
        )


# ==========================================================
# PAGE
# ==========================================================

st.set_page_config(
    page_title="Meeting Engagement AI",
    page_icon="🎯",
    layout="wide"
)


st.title("🎯 Meeting Engagement AI")

st.write(
    "Multimodal meeting engagement analysis using Audio + Vision"
)


# ==========================================================
# SESSION STATE
# ==========================================================

if "vision_score" not in st.session_state:
    st.session_state.vision_score = 0.0

if "audio_score" not in st.session_state:
    st.session_state.audio_score = 0.0


# ==========================================================
# AUDIO + VIDEO
# ==========================================================

col1, col2 = st.columns(2)


# ==========================================================
# AUDIO
# ==========================================================

with col1:

    st.subheader("🎤 Audio Engagement")

    duration = st.slider(
        "Recording duration (seconds)",
        5,
        30,
        10
    )


    if st.button(
        "🎙️ Start Audio Analysis"
    ):

        with st.spinner(
            "Recording..."
        ):

            audio, rate = record_audio(
                duration=duration
            )

            features = extract_audio_features(
                audio,
                rate
            )


        st.session_state.audio_score = (
            features["audio_engagement_score"]
        )


        st.success(
            "Audio analysis complete!"
        )


        st.metric(
            "Audio Engagement",
            f"{features['audio_engagement_score']:.1f}/100"
        )


        st.write(
            "### Audio Features"
        )


        st.write(
            f"**Speech Ratio:** "
            f"{features['speech_ratio'] * 100:.1f}%"
        )


        st.write(
            f"**Pause Ratio:** "
            f"{features['pause_ratio'] * 100:.1f}%"
        )


        st.write(
            f"**Pitch:** "
            f"{features['pitch_hz']:.1f} Hz"
        )


        st.write(
            f"**Speaking Rate:** "
            f"{features['speaking_rate']:.2f}"
        )


        st.write(
            f"**Loudness:** "
            f"{features['decibel']:.1f} dB"
        )


# ==========================================================
# VIDEO
# ==========================================================

with col2:

    st.subheader("👁️ Live Video Engagement")


    ctx = webrtc_streamer(

        key="meeting-video",

        mode=WebRtcMode.SENDRECV,

        video_processor_factory=VideoProcessor,

        media_stream_constraints={
            "video": True,
            "audio": False
        },

        async_processing=True
    )


    # ======================================================
    # GET LIVE VIDEO RESULTS
    # ======================================================

    if ctx.video_processor:

        vp = ctx.video_processor


        with vp.lock:

            vision_score = vp.vision_score

            gaze = vp.gaze

            gaze_direction = (
                vp.gaze_direction
            )

            head = vp.head

            yaw = vp.yaw
            pitch = vp.pitch
            roll = vp.roll

            eye_event = vp.eye_event

            blink_count = (
                vp.blink_count
            )

            face_detected = (
                vp.face_detected
            )


        # ==================================================
        # SAVE LIVE SCORE TO SESSION STATE
        # ==================================================

        st.session_state.vision_score = (
            vision_score
        )


        # ==================================================
        # DISPLAY
        # ==================================================

        if face_detected:

            st.success(
                "Face detected — video is being analyzed continuously!"
            )


            v1, v2, v3, v4 = st.columns(4)


            v1.metric(
                "Vision Engagement",
                f"{vision_score:.1f}/100"
            )


            v2.metric(
                "Gaze",
                f"{gaze:.1f}%"
            )


            v3.metric(
                "Head",
                head
            )


            v4.metric(
                "Eyes",
                eye_event
            )


            st.write(
                "### Vision Features"
            )


            st.write(
                f"**Gaze Direction:** "
                f"{gaze_direction}"
            )


            st.write(
                f"**Yaw:** "
                f"{yaw:.2f}°"
            )


            st.write(
                f"**Pitch:** "
                f"{pitch:.2f}°"
            )


            st.write(
                f"**Roll:** "
                f"{roll:.2f}°"
            )


            st.write(
                f"**Blink Count:** "
                f"{blink_count}"
            )


        else:

            st.warning(
                "No face detected. "
                "Please look at the camera."
            )


    else:

        st.info(
            "Click START above to begin live video analysis."
        )


# ==========================================================
# MULTIMODAL ENGAGEMENT
# ==========================================================

st.divider()

st.subheader(
    "🎯 Multimodal Engagement"
)


# Get latest scores

vision_score = st.session_state.get(
    "vision_score",
    0.0
)

audio_score = st.session_state.get(
    "audio_score",
    0.0
)


# ==========================================================
# FUSION
# ==========================================================

overall = (
    0.60 * vision_score
    +
    0.40 * audio_score
)


# ==========================================================
# DISPLAY SCORES
# ==========================================================

c1, c2, c3 = st.columns(3)


c1.metric(
    "👁️ Vision",
    f"{vision_score:.1f}/100"
)


c2.metric(
    "🎤 Audio",
    f"{audio_score:.1f}/100"
)


c3.metric(
    "🎯 Overall",
    f"{overall:.1f}/100"
)


# ==========================================================
# ENGAGEMENT LEVEL
# ==========================================================

if overall >= 75:

    st.success(
        "HIGH ENGAGEMENT"
    )

elif overall >= 50:

    st.info(
        "MEDIUM ENGAGEMENT"
    )

elif overall >= 25:

    st.warning(
        "LOW ENGAGEMENT"
    )

else:

    st.error(
        "VERY LOW ENGAGEMENT"
    )


# ==========================================================
# FORMULA
# ==========================================================

st.caption(
    "Overall score = 60% Vision + 40% Audio"
)