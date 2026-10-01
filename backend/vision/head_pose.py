import cv2
import numpy as np


NOSE_TIP = 1
CHIN = 152

LEFT_EYE_CORNER = 263
RIGHT_EYE_CORNER = 33

LEFT_MOUTH_CORNER = 291
RIGHT_MOUTH_CORNER = 61


def get_image_points(landmarks, width, height):

    indices = [
        NOSE_TIP,
        CHIN,
        LEFT_EYE_CORNER,
        RIGHT_EYE_CORNER,
        LEFT_MOUTH_CORNER,
        RIGHT_MOUTH_CORNER
    ]

    points = []

    for index in indices:

        landmark = landmarks[index]

        x = landmark.x * width
        y = landmark.y * height

        points.append([x, y])

    return np.array(points, dtype=np.float64)


def estimate_head_pose(
    landmarks,
    width,
    height
):

    image_points = get_image_points(
        landmarks,
        width,
        height
    )

    model_points = np.array([
        [0.0, 0.0, 0.0],
        [0.0, -63.6, -12.5],
        [-43.3, 32.7, -26.0],
        [43.3, 32.7, -26.0],
        [-28.9, -28.9, -24.1],
        [28.9, -28.9, -24.1]
    ], dtype=np.float64)

    focal_length = width

    center = (
        width / 2,
        height / 2
    )

    camera_matrix = np.array([
        [focal_length, 0, center[0]],
        [0, focal_length, center[1]],
        [0, 0, 1]
    ], dtype=np.float64)

    distortion_coefficients = np.zeros(
        (4, 1),
        dtype=np.float64
    )

    success, rotation_vector, translation_vector = cv2.solvePnP(
        model_points,
        image_points,
        camera_matrix,
        distortion_coefficients,
        flags=cv2.SOLVEPNP_ITERATIVE
    )

    if not success:

        return {
            "yaw": 0.0,
            "pitch": 0.0,
            "roll": 0.0
        }

    rotation_matrix, _ = cv2.Rodrigues(
        rotation_vector
    )

    angles = cv2.RQDecomp3x3(
        rotation_matrix
    )

    pitch = angles[0][0]
    yaw = angles[0][1]
    roll = angles[0][2]

    return {
        "yaw": float(yaw),
        "pitch": float(pitch),
        "roll": float(roll)
    }


def classify_head_pose(
    yaw,
    pitch,
    baseline_yaw=0.0,
    baseline_pitch=0.0,
    yaw_threshold=15,
    pitch_threshold=15
):

    relative_yaw = yaw - baseline_yaw
    relative_pitch = pitch - baseline_pitch

    if relative_yaw > yaw_threshold:

        horizontal = "RIGHT"

    elif relative_yaw < -yaw_threshold:

        horizontal = "LEFT"

    else:

        horizontal = "CENTER"

    if relative_pitch > pitch_threshold:

        vertical = "DOWN"

    elif relative_pitch < -pitch_threshold:

        vertical = "UP"

    else:

        vertical = "CENTER"

    return horizontal, vertical