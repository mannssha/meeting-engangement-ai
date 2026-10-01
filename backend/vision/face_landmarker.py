import cv2
import mediapipe as mp

from pathlib import Path

from mediapipe.tasks import python
from mediapipe.tasks.python import vision


# --------------------------------------------------
# Model path
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "face_landmarker.task"


# --------------------------------------------------
# MediaPipe configuration
# --------------------------------------------------

base_options = python.BaseOptions(
    model_asset_path=str(MODEL_PATH)
)

options = vision.FaceLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.IMAGE,
    num_faces=1,
    min_face_detection_confidence=0.5,
    min_face_presence_confidence=0.5,
    min_tracking_confidence=0.5,
)


# --------------------------------------------------
# Create the model ONCE
# --------------------------------------------------

landmarker = vision.FaceLandmarker.create_from_options(
    options
)


# --------------------------------------------------
# Face landmark detection
# --------------------------------------------------

def detect_face_landmarks(image):
    """
    Detect facial landmarks from an OpenCV BGR image.

    Parameters
    ----------
    image : numpy.ndarray
        OpenCV BGR image.

    Returns
    -------
    FaceLandmarkerResult
        MediaPipe face landmark detection result.
    """

    # Convert BGR → RGB
    rgb_image = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2RGB
    )

    # Convert to MediaPipe Image
    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb_image
    )

    # Run detection
    result = landmarker.detect(mp_image)

    return result


# --------------------------------------------------
# Cleanup
# --------------------------------------------------

def close_landmarker():
    """
    Release the MediaPipe Face Landmarker.
    """

    landmarker.close()