class EngagementScore:

    def __init__(self):
        self.score = 0.0

    def calculate(
        self,
        gaze_attention,
        head_direction,
        eye_event,
        face_detected=True
    ):
        """
        Calculate engagement score from vision signals.

        Parameters
        ----------
        gaze_attention : float
            Gaze attention score from 0 to 1.

        head_direction : str
            Head direction such as:
            CENTER/CENTER
            LEFT/CENTER
            RIGHT/CENTER
            CENTER/UP
            CENTER/DOWN

        eye_event : str
            OPEN
            BLINK
            CLOSING
            PROLONGED_CLOSURE

        face_detected : bool
            Whether a face is currently detected.

        Returns
        -------
        dict
            Score and engagement status.
        """

        # --------------------------------------------------
        # 1. No face
        # --------------------------------------------------

        if not face_detected:

            self.score = 0.0

            return {
                "score": 0.0,
                "status": "NO_FACE"
            }

        # --------------------------------------------------
        # 2. GAZE SCORE
        # --------------------------------------------------

        gaze_score = max(
            0.0,
            min(1.0, gaze_attention)
        )

        # --------------------------------------------------
        # 3. HEAD POSE SCORE
        # --------------------------------------------------

        head_score = self.calculate_head_score(
            head_direction
        )

        # --------------------------------------------------
        # 4. EYE SCORE
        # --------------------------------------------------

        eye_score = self.calculate_eye_score(
            eye_event
        )

        # --------------------------------------------------
        # 5. WEIGHTED ENGAGEMENT SCORE
        # --------------------------------------------------

        final_score = (
            0.50 * gaze_score +
            0.30 * head_score +
            0.20 * eye_score
        )

        # Convert 0-1 → 0-100

        self.score = final_score * 100

        # --------------------------------------------------
        # 6. STATUS
        # --------------------------------------------------

        status = self.get_status(
            self.score
        )

        return {
            "score": round(self.score, 2),
            "status": status,
            "gaze_score": round(gaze_score * 100, 2),
            "head_score": round(head_score * 100, 2),
            "eye_score": round(eye_score * 100, 2)
        }

    def calculate_head_score(self, head_direction):

        if head_direction == "CENTER/CENTER":
            return 1.0

        elif head_direction in [
            "LEFT/CENTER",
            "RIGHT/CENTER",
            "CENTER/UP",
            "CENTER/DOWN"
        ]:
            return 0.5

        else:
            return 0.2

    def calculate_eye_score(self, eye_event):

        if eye_event == "OPEN":
            return 1.0

        elif eye_event == "BLINK":
            return 0.9

        elif eye_event == "CLOSING":
            return 0.6

        elif eye_event == "PROLONGED_CLOSURE":
            return 0.0

        else:
            return 0.5

    def get_status(self, score):

        if score >= 75:
            return "HIGH"

        elif score >= 50:
            return "MEDIUM"

        elif score >= 25:
            return "LOW"

        else:
            return "VERY_LOW"