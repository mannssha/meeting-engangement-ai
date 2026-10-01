import cv2

from face_landmarker import (
    detect_face_landmarks,
    close_landmarker
)

from eye_closure import EyeClosureDetector


def main():

    camera = cv2.VideoCapture(0)

    if not camera.isOpened():

        print("ERROR: Could not open webcam.")
        return

    detector = EyeClosureDetector(
        closed_threshold=0.20,
        prolonged_duration=2.0
    )

    print("Webcam opened successfully.")
    print()
    print("Test 1: Keep eyes open.")
    print("Test 2: Blink normally.")
    print("Test 3: Close your eyes for 3 seconds.")
    print()
    print("Press 'q' to quit.")
    print()

    try:

        while True:

            success, frame = camera.read()

            if not success:
                print("ERROR: Could not read frame.")
                break

            result = detect_face_landmarks(frame)

            if result.face_landmarks:

                landmarks = result.face_landmarks[0]

                data = detector.process(
                    landmarks
                )

                left_ear = data["left_ear"]
                right_ear = data["right_ear"]
                average_ear = data["average_ear"]
                event = data["event"]
                duration = data["closure_duration"]
                blink_count = data["blink_count"]

                print(
                    f"EAR={average_ear:.3f} | "
                    f"Event={event} | "
                    f"Closed={duration:.2f}s | "
                    f"Blinks={blink_count}"
                )

                cv2.putText(
                    frame,
                    f"EAR: {average_ear:.3f}",
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 0),
                    2
                )

                cv2.putText(
                    frame,
                    f"Event: {event}",
                    (20, 80),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 0),
                    2
                )

                cv2.putText(
                    frame,
                    f"Blinks: {blink_count}",
                    (20, 120),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 0),
                    2
                )

            else:

                cv2.putText(
                    frame,
                    "No face detected",
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 0, 255),
                    2
                )

            cv2.imshow(
                "Blink Detection",
                frame
            )

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    finally:

        camera.release()
        cv2.destroyAllWindows()
        close_landmarker()


if __name__ == "__main__":
    main()