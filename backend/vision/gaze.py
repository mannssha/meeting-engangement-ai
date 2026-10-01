import math


# MediaPipe Face Landmarker iris landmark indices
LEFT_IRIS = [474, 475, 476, 477]
RIGHT_IRIS = [469, 470, 471, 472]

# Approximate eye corner landmarks
LEFT_EYE_CORNERS = [362, 263]
RIGHT_EYE_CORNERS = [33, 133]


def landmark_distance(point1, point2):
    """
    Calculate Euclidean distance between two landmarks.
    """

    return math.sqrt(
        (point1.x - point2.x) ** 2
        + (point1.y - point2.y) ** 2
    )


def average_landmark(landmarks, indices):
    """
    Calculate the average position of selected landmarks.
    """

    x = sum(landmarks[i].x for i in indices) / len(indices)
    y = sum(landmarks[i].y for i in indices) / len(indices)

    return x, y


def calculate_eye_ratio(
    landmarks,
    iris_indices,
    corner_indices
):
    """
    Calculate horizontal iris position inside the eye.

    Returns a value approximately between 0 and 1.

    0.0 → left side of eye
    0.5 → center
    1.0 → right side
    """

    iris_x, iris_y = average_landmark(
        landmarks,
        iris_indices
    )

    left_corner = landmarks[corner_indices[0]]
    right_corner = landmarks[corner_indices[1]]

    min_x = min(
        left_corner.x,
        right_corner.x
    )

    max_x = max(
        left_corner.x,
        right_corner.x
    )

    eye_width = max_x - min_x

    if eye_width <= 0:
        return 0.5

    ratio = (iris_x - min_x) / eye_width

    return max(0.0, min(1.0, ratio))


def calculate_gaze(landmarks):
    """
    Estimate horizontal gaze direction.

    Returns:
        gaze_direction
        gaze_score
        left_ratio
        right_ratio
    """

    left_ratio = calculate_eye_ratio(
        landmarks,
        LEFT_IRIS,
        LEFT_EYE_CORNERS
    )

    right_ratio = calculate_eye_ratio(
        landmarks,
        RIGHT_IRIS,
        RIGHT_EYE_CORNERS
    )

    average_ratio = (
        left_ratio + right_ratio
    ) / 2

    # Initial thresholds.
    # We will personalize these later using
    # the user's baseline.

    if average_ratio < 0.40:
        direction = "LEFT"

    elif average_ratio > 0.60:
        direction = "RIGHT"

    else:
        direction = "CENTER"

    # Convert distance from center into
    # an attention score.
    distance_from_center = abs(
        average_ratio - 0.50
    )

    attention_score = 1.0 - (
        distance_from_center / 0.50
    )

    attention_score = max(
        0.0,
        min(1.0, attention_score)
    )

    return {
        "direction": direction,
        "attention_score": attention_score,
        "left_ratio": left_ratio,
        "right_ratio": right_ratio,
        "average_ratio": average_ratio
    }