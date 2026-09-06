# attendance.py
import cv2
import os
import numpy as np
from database import db


class AttendanceSystem:
    def __init__(self):
        self.dataset_path = "face_dataset"
        self.model_path = "face_trainer.yml"

        os.makedirs(self.dataset_path, exist_ok=True)

        # OpenCV 4.x normally ships this XML with the Python package.
        cascade_path = os.path.join(
            cv2.data.haarcascades,
            "haarcascade_frontalface_default.xml"
        )

        if not os.path.exists(cascade_path):
            raise FileNotFoundError(
                "Face detection model not found. "
                f"Expected file: {cascade_path}"
            )

        self.face_detector = cv2.CascadeClassifier(cascade_path)

        if self.face_detector.empty():
            raise RuntimeError(
                f"Could not load face detection model: {cascade_path}"
            )

        if not hasattr(cv2, "face"):
            raise RuntimeError(
                "OpenCV Face module is unavailable. "
                "Install opencv-contrib-python."
            )

        self.recognizer = cv2.face.LBPHFaceRecognizer_create()

    def register_student(self, student_name):
        """Capture face samples and prevent the same face being registered
        under a different student name.

        If a trained face model already exists, the newly captured face is
        compared against existing identities. A strong match causes the
        registration to be rejected instead of silently creating a second
        name for the same face.
        """
        student_name = student_name.strip()

        if not student_name:
            raise ValueError("Student name cannot be empty.")

        # Don't overwrite an existing student's dataset accidentally.
        student_folder = os.path.join(
            self.dataset_path,
            student_name
        )

        if os.path.exists(student_folder) and any(
            name.lower().endswith((".jpg", ".jpeg", ".png"))
            for name in os.listdir(student_folder)
        ):
            raise ValueError(
                f"Face data for '{student_name}' already exists. "
                "Use a different action to update that student's face data."
            )

        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

        if not cap.isOpened():
            cap.release()
            cap = cv2.VideoCapture(0)

        if not cap.isOpened():
            raise RuntimeError(
                "Could not open the camera. "
                "Check camera permissions and make sure no other app is using it."
            )

        # Load an existing trained model, if one exists, so we can check
        # whether this face already belongs to another registered student.
        existing_model = os.path.exists(self.model_path)
        existing_labels = {}

        if existing_model:
            try:
                existing_labels = self._build_label_map()

                if existing_labels:
                    self.recognizer.read(self.model_path)
            except Exception:
                # Registration can still proceed if there is no usable
                # existing model. A fresh model will be trained afterwards.
                existing_model = False
                existing_labels = {}

        samples = []
        duplicate_votes = {}

        try:
            while len(samples) < 20:
                ret, frame = cap.read()

                if not ret or frame is None:
                    raise RuntimeError(
                        "Could not read a frame from the camera."
                    )

                gray = cv2.cvtColor(
                    frame,
                    cv2.COLOR_BGR2GRAY
                )

                faces = self.face_detector.detectMultiScale(
                    gray,
                    scaleFactor=1.3,
                    minNeighbors=5,
                    minSize=(80, 80)
                )

                # Use the largest detected face so background faces do not
                # accidentally become part of this student's dataset.
                if len(faces) > 0:
                    x, y, w, h = max(
                        faces,
                        key=lambda box: box[2] * box[3]
                    )

                    face = gray[y:y + h, x:x + w]

                    # Check the proposed identity against the existing model.
                    if existing_model and existing_labels:
                        label, confidence = self.recognizer.predict(face)

                        # LBPH: LOWER confidence = stronger similarity.
                        # 45 is deliberately strict to reduce accidental
                        # duplicate registration.
                        if confidence < 45 and label in existing_labels:
                            matched_name = existing_labels[label]
                            duplicate_votes[matched_name] = (
                                duplicate_votes.get(matched_name, 0) + 1
                            )

                            # Three strong matches are enough to reject.
                            if duplicate_votes[matched_name] >= 3:
                                raise ValueError(
                                    f"This face already matches registered "
                                    f"student '{matched_name}'. "
                                    f"Registration as '{student_name}' was rejected."
                                )

                    samples.append(face.copy())

                    cv2.rectangle(
                        frame,
                        (x, y),
                        (x + w, y + h),
                        (255, 0, 0),
                        2
                    )

                cv2.putText(
                    frame,
                    f"Samples: {len(samples)}/20",
                    (20, 35),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9,
                    (0, 255, 0),
                    2
                )

                cv2.putText(
                    frame,
                    "Press Q to cancel",
                    (20, 70),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 255, 255),
                    2
                )

                cv2.imshow("Register Face", frame)

                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

        finally:
            cap.release()
            cv2.destroyAllWindows()

        if not samples:
            raise RuntimeError(
                "No face was detected. Please try again with your face clearly visible."
            )

        # Save only after duplicate verification succeeds.
        os.makedirs(student_folder, exist_ok=True)

        for index, face in enumerate(samples, start=1):
            cv2.imwrite(
                os.path.join(student_folder, f"{index}.jpg"),
                face
            )

        # Immediately rebuild the recognizer so the newly registered
        # student is available for attendance.
        self.train_model()

        return len(samples)

    def _build_label_map(self):
        """Create the same label mapping used by train_model()."""
        label_map = {}
        label_id = 0

        if not os.path.exists(self.dataset_path):
            return label_map

        for student_name in sorted(os.listdir(self.dataset_path)):
            student_folder = os.path.join(
                self.dataset_path,
                student_name
            )

            if not os.path.isdir(student_folder):
                continue

            has_valid_image = False

            for image_name in os.listdir(student_folder):
                img_path = os.path.join(
                    student_folder,
                    image_name
                )

                img = cv2.imread(
                    img_path,
                    cv2.IMREAD_GRAYSCALE
                )

                if img is not None:
                    has_valid_image = True
                    break

            if has_valid_image:
                label_map[label_id] = student_name
                label_id += 1

        return label_map

    def train_model(self):
        """Train LBPH recognizer from all saved student face samples."""
        faces = []
        labels = []
        label_map = {}

        label_id = 0

        if not os.path.exists(self.dataset_path):
            raise RuntimeError("No face dataset exists yet.")

        for student_name in sorted(os.listdir(self.dataset_path)):

            student_folder = os.path.join(
                self.dataset_path,
                student_name
            )

            if not os.path.isdir(student_folder):
                continue

            valid_student_images = 0

            for image_name in os.listdir(student_folder):

                img_path = os.path.join(
                    student_folder,
                    image_name
                )

                img = cv2.imread(
                    img_path,
                    cv2.IMREAD_GRAYSCALE
                )

                if img is None:
                    continue

                faces.append(img)
                labels.append(label_id)
                valid_student_images += 1

            if valid_student_images > 0:
                label_map[label_id] = student_name
                label_id += 1

        if not faces or not labels:
            raise RuntimeError(
                "No valid face samples found. Register at least one student first."
            )

        self.recognizer.train(
            faces,
            np.array(labels)
        )

        self.recognizer.save(self.model_path)

        return label_map

    def mark_attendance(self, label_map):
        """Recognize a registered face and save attendance."""
        if not os.path.exists(self.model_path):
            raise RuntimeError(
                "Face model has not been trained yet. "
                "Register a student first."
            )

        if not label_map:
            raise RuntimeError(
                "No registered students are available."
            )

        self.recognizer.read(self.model_path)

        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

        if not cap.isOpened():
            cap.release()
            cap = cv2.VideoCapture(0)

        if not cap.isOpened():
            raise RuntimeError(
                "Could not open the camera. "
                "Check camera permissions and make sure no other app is using it."
            )

        try:
            while True:
                ret, frame = cap.read()

                if not ret or frame is None:
                    raise RuntimeError(
                        "Could not read a frame from the camera."
                    )

                gray = cv2.cvtColor(
                    frame,
                    cv2.COLOR_BGR2GRAY
                )

                faces = self.face_detector.detectMultiScale(
                    gray,
                    scaleFactor=1.3,
                    minNeighbors=5,
                    minSize=(80, 80)
                )

                for (x, y, w, h) in faces:

                    face = gray[y:y + h, x:x + w]

                    label, confidence = self.recognizer.predict(face)

                    # Lower LBPH confidence means a closer match.
                    # Use a stricter threshold to reduce false positives.
                    if confidence < 55 and label in label_map:

                        student_name = label_map[label]

                        db.cursor.execute(
                            """
                            INSERT INTO attendance
                            (student_name, status)
                            VALUES (?, ?)
                            """,
                            (student_name, "Present")
                        )

                        db.conn.commit()

                        return student_name

                    cv2.rectangle(
                        frame,
                        (x, y),
                        (x + w, y + h),
                        (0, 0, 255),
                        2
                    )

                cv2.imshow("Attendance", frame)

                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

        finally:
            cap.release()
            cv2.destroyAllWindows()

        return None
