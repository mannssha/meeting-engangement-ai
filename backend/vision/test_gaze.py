import cv2
from collections import deque

from face_landmarker import (
    detect_face_landmarks,
    close_landmarker
)

from gaze import calculate_gaze


def main():

    camera = cv2.VideoCapture(0)

    if not camera.isOpened():
        print("ERROR: Could not open webcam.")
        return

    print("Webcam opened successfully.")
    print("Look straight at the camera.")
    print("Then look left and right.")
    print("Press 'q' to quit.")

    # Store the last 10 gaze measurements
    # to reduce frame-to-frame noise.
    gaze_history = deque(maxlen=10)

    try:

        while True:

            success, frame = camera.read()

            if not success:
                print("ERROR: Could not read frame.")
                break

            # -----------------------------------------
            # Detect facial landmarks
            # -----------------------------------------

            result = detect_face_landmarks(frame)

            # -----------------------------------------
            # Process detected face
            # -----------------------------------------

            if result.face_landmarks:

                landmarks = result.face_landmarks[0]

                # Calculate raw gaze information
                gaze = calculate_gaze(landmarks)

                # Store horizontal iris position
                gaze_history.append(
                    gaze["average_ratio"]
                )

                # -----------------------------------------
                # Smooth gaze position
                # -----------------------------------------

                smoothed_ratio = (
                    sum(gaze_history)
                    / len(gaze_history)
                )

                # -----------------------------------------
                # Determine gaze direction
                # -----------------------------------------

                if smoothed_ratio < 0.40:

                    direction = "LEFT"

                elif smoothed_ratio > 0.60:

                    direction = "RIGHT"

                else:

                    direction = "CENTER"

                # -----------------------------------------
                # Calculate attention score
                # -----------------------------------------

                distance_from_center = abs(
                    smoothed_ratio - 0.50
                )

                score = 1.0 - (
                    distance_from_center / 0.50
                )

                score = max(
                    0.0,
                    min(1.0, score)
                )

                # -----------------------------------------
                # Display information
                # -----------------------------------------

                text = (
                    f"Gaze: {direction} | "
                    f"Attention: {score:.2f}"
                )

                cv2.putText(
                    frame,
                    text,
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 0),
                    2
                )

                # Print to terminal
                print(
                    f"Gaze={direction}, "
                    f"Attention={score:.2f}"
                )

            else:

                # No face detected
                cv2.putText(
                    frame,
                    "No face detected",
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 0, 255),
                    2
                )

                print("No face detected.")

                # Clear old gaze values
                gaze_history.clear()

            # -----------------------------------------
            # Display webcam
            # -----------------------------------------

            cv2.imshow(
                "Gaze Detection",
                frame
            )

            # -----------------------------------------
            # Quit
            # -----------------------------------------

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    finally:

        camera.release()

        cv2.destroyAllWindows()

        # Release MediaPipe model
        close_landmarker()


if __name__ == "__main__":
    main()