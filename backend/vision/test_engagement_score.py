from engagement_score import EngagementScore


def main():

    engine = EngagementScore()

    print("=" * 50)
    print("ENGAGEMENT SCORE TEST")
    print("=" * 50)

    test_cases = [

        {
            "name": "Fully Engaged",
            "gaze_attention": 0.95,
            "head_direction": "CENTER/CENTER",
            "eye_event": "OPEN",
            "face_detected": True
        },

        {
            "name": "Looking Slightly Away",
            "gaze_attention": 0.70,
            "head_direction": "RIGHT/CENTER",
            "eye_event": "OPEN",
            "face_detected": True
        },

        {
            "name": "Blinking",
            "gaze_attention": 0.90,
            "head_direction": "CENTER/CENTER",
            "eye_event": "BLINK",
            "face_detected": True
        },

        {
            "name": "Prolonged Eye Closure",
            "gaze_attention": 0.50,
            "head_direction": "CENTER/CENTER",
            "eye_event": "PROLONGED_CLOSURE",
            "face_detected": True
        },

        {
            "name": "No Face",
            "gaze_attention": 0.0,
            "head_direction": "UNKNOWN",
            "eye_event": "UNKNOWN",
            "face_detected": False
        }
    ]

    for test in test_cases:

        result = engine.calculate(
            gaze_attention=test["gaze_attention"],
            head_direction=test["head_direction"],
            eye_event=test["eye_event"],
            face_detected=test["face_detected"]
        )

        print()
        print("Test:", test["name"])
        print("Score:", result["score"])
        print("Status:", result["status"])

        if "gaze_score" in result:

            print(
                "Gaze:",
                result["gaze_score"]
            )

            print(
                "Head:",
                result["head_score"]
            )

            print(
                "Eyes:",
                result["eye_score"]
            )

    print()
    print("=" * 50)


if __name__ == "__main__":
    main()