import cv2

from face_landmarker import (
    detect_face_landmarks,
    close_landmarker
)


def draw_landmarks(frame, result):
    """
    Draw detected facial landmarks.
    """

    if not result.face_landmarks:
        return frame

    height, width, _ = frame.shape

    for face_landmarks in result.face_landmarks:

        for landmark in face_landmarks:

            x = int(landmark.x * width)
            y = int(landmark.y * height)

            x = max(0, min(x, width - 1))
            y = max(0, min(y, height - 1))

            cv2.circle(
                frame,
                (x, y),
                1,
                (0, 255, 0),
                -1
            )

    return frame


def main():

    camera = cv2.VideoCapture(0)

    if not camera.isOpened():
        print("ERROR: Could not open webcam.")
        return

    print("Webcam opened successfully.")
    print("Press 'q' to quit.")

    try:

        while True:

            success, frame = camera.read()

            if not success:
                print("ERROR: Could not read frame.")
                break

            # Detect face landmarks
            result = detect_face_landmarks(frame)

            # Draw landmarks
            frame = draw_landmarks(
                frame,
                result
            )

            # Show webcam
            cv2.imshow(
                "Face Landmarks",
                frame
            )

            # Quit
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    finally:

        camera.release()
        cv2.destroyAllWindows()

        # Release MediaPipe
        close_landmarker()


if __name__ == "__main__":
    main()