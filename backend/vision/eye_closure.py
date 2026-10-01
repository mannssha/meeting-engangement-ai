import time
import math


RIGHT_EYE = {
    "left_corner": 33,
    "right_corner": 133,
    "top_1": 160,
    "top_2": 158,
    "bottom_1": 144,
    "bottom_2": 153
}


LEFT_EYE = {
    "left_corner": 362,
    "right_corner": 263,
    "top_1": 385,
    "top_2": 387,
    "bottom_1": 380,
    "bottom_2": 373
}


class EyeClosureDetector:

    def __init__(
        self,
        closed_threshold=0.20,
        prolonged_duration=2.0
    ):
        self.closed_threshold = closed_threshold
        self.prolonged_duration = prolonged_duration

        self.eye_closed_start = None
        self.blink_count = 0

    @staticmethod
    def distance(point1, point2):

        dx = point1.x - point2.x
        dy = point1.y - point2.y
        dz = point1.z - point2.z

        return math.sqrt(
            dx * dx +
            dy * dy +
            dz * dz
        )

    def calculate_ear(self, landmarks, eye):

        left_corner = landmarks[eye["left_corner"]]
        right_corner = landmarks[eye["right_corner"]]

        top_1 = landmarks[eye["top_1"]]
        top_2 = landmarks[eye["top_2"]]

        bottom_1 = landmarks[eye["bottom_1"]]
        bottom_2 = landmarks[eye["bottom_2"]]

        horizontal = self.distance(
            left_corner,
            right_corner
        )

        vertical_1 = self.distance(
            top_1,
            bottom_1
        )

        vertical_2 = self.distance(
            top_2,
            bottom_2
        )

        if horizontal == 0:
            return 0.0

        return (
            vertical_1 + vertical_2
        ) / (2.0 * horizontal)

    def process(self, landmarks):

        left_ear = self.calculate_ear(
            landmarks,
            LEFT_EYE
        )

        right_ear = self.calculate_ear(
            landmarks,
            RIGHT_EYE
        )

        average_ear = (
            left_ear + right_ear
        ) / 2.0

        now = time.time()

        if average_ear < self.closed_threshold:

            if self.eye_closed_start is None:
                self.eye_closed_start = now

            closure_duration = (
                now - self.eye_closed_start
            )

            if closure_duration >= self.prolonged_duration:
                event = "PROLONGED_CLOSURE"
            else:
                event = "CLOSING"

        else:

            if self.eye_closed_start is not None:

                closure_duration = (
                    now - self.eye_closed_start
                )

                if closure_duration < self.prolonged_duration:
                    self.blink_count += 1
                    event = "BLINK"
                else:
                    event = "OPEN"

                self.eye_closed_start = None

            else:

                closure_duration = 0.0
                event = "OPEN"

        return {
            "left_ear": left_ear,
            "right_ear": right_ear,
            "average_ear": average_ear,
            "event": event,
            "closure_duration": closure_duration,
            "blink_count": self.blink_count
        }