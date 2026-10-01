import cv2
import numpy as np

from face_landmarker import (
    detect_face_landmarks,
    close_landmarker
)

from head_pose import (
    estimate_head_pose,
    classify_head_pose
)


def main():

    camera = cv2.VideoCapture(0)

    if not camera.isOpened():

        print("ERROR: Could not open webcam.")
        return

    print("Webcam opened successfully.")
    print()
    print("CALIBRATION")
    print("Look naturally at the camera.")
    print("Keep your head still.")
    print("Calibration will take a few seconds.")
    print()

    calibration_yaw = []
    calibration_pitch = []
    calibration_roll = []

    calibration_frames = 60

    calibrated = False

    baseline_yaw = 0.0
    baseline_pitch = 0.0
    baseline_roll = 0.0

    try:

        while True:

            success, frame = camera.read()

            if not success:

                print("ERROR: Could not read frame.")
                break

            height, width, _ = frame.shape

            result = detect_face_landmarks(frame)

            if result.face_landmarks:

                landmarks = result.face_landmarks[0]

                pose = estimate_head_pose(
                    landmarks,
                    width,
                    height
                )

                yaw = pose["yaw"]
                pitch = pose["pitch"]
                roll = pose["roll"]

                # --------------------------------
                # Calibration
                # --------------------------------

                if not calibrated:

                    calibration_yaw.append(yaw)
                    calibration_pitch.append(pitch)
                    calibration_roll.append(roll)

                    progress = len(
                        calibration_yaw
                    )

                    cv2.putText(
                        frame,
                        f"Calibrating: "
                        f"{progress}/{calibration_frames}",
                        (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        (0, 255, 255),
                        2
                    )

                    if progress >= calibration_frames:

                        baseline_yaw = np.median(
                            calibration_yaw
                        )

                        baseline_pitch = np.median(
                            calibration_pitch
                        )

                        baseline_roll = np.median(
                            calibration_roll
                        )

                        calibrated = True

                        print()
                        print("Calibration complete!")
                        print(
                            f"Baseline Yaw: "
                            f"{baseline_yaw:.2f}"
                        )
                        print(
                            f"Baseline Pitch: "
                            f"{baseline_pitch:.2f}"
                        )
                        print(
                            f"Baseline Roll: "
                            f"{baseline_roll:.2f}"
                        )
                        print()

                # --------------------------------
                # After calibration
                # --------------------------------

                if calibrated:

                    horizontal, vertical = (
                        classify_head_pose(
                            yaw,
                            pitch,
                            baseline_yaw,
                            baseline_pitch
                        )
                    )

                    relative_yaw = (
                        yaw - baseline_yaw
                    )

                    relative_pitch = (
                        pitch - baseline_pitch
                    )

                    relative_roll = (
                        roll - baseline_roll
                    )

                    cv2.putText(
                        frame,
                        f"Yaw: {relative_yaw:.1f}",
                        (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 255, 0),
                        2
                    )

                    cv2.putText(
                        frame,
                        f"Pitch: {relative_pitch:.1f}",
                        (20, 70),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 255, 0),
                        2
                    )

                    cv2.putText(
                        frame,
                        f"Roll: {relative_roll:.1f}",
                        (20, 100),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 255, 0),
                        2
                    )

                    cv2.putText(
                        frame,
                        f"Head: "
                        f"{horizontal}/{vertical}",
                        (20, 130),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 255, 0),
                        2
                    )

                    print(
                        f"Yaw={relative_yaw:.1f}, "
                        f"Pitch={relative_pitch:.1f}, "
                        f"Roll={relative_roll:.1f}, "
                        f"Head={horizontal}/{vertical}"
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
                "Head Pose Detection",
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