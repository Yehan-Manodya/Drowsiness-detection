import mediapipe as mp
import cv2
import numpy as np

class FaceAnalyzer:
    def __init__(self):
        print("Loading MediaPipe...")
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.mp_drawing = mp.solutions.drawing_utils

        # Eye landmark indices
        self.LEFT_EYE  = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
        self.RIGHT_EYE = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]

        # Mouth landmark indices
        self.MOUTH = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 308, 324, 318, 402, 317, 14, 87, 178, 88, 95]

        print("MediaPipe loaded successfully! ✅")

    def calculate_ear(self, landmarks, eye_indices, frame_w, frame_h):
        """Calculate Eye Aspect Ratio (EAR) - lower means more closed"""
        eye_points = []
        for idx in eye_indices:
            x = int(landmarks[idx].x * frame_w)
            y = int(landmarks[idx].y * frame_h)
            eye_points.append([x, y])

        eye_points = np.array(eye_points)

        # Vertical distances
        v1 = np.linalg.norm(eye_points[1] - eye_points[5])
        v2 = np.linalg.norm(eye_points[2] - eye_points[4])

        # Horizontal distance
        h = np.linalg.norm(eye_points[0] - eye_points[3])

        # EAR formula
        ear = (v1 + v2) / (2.0 * h) if h != 0 else 0
        return ear

    def calculate_mar(self, landmarks, mouth_indices, frame_w, frame_h):
        """Calculate Mouth Aspect Ratio (MAR) - higher means more open (yawning)"""
        mouth_points = []
        for idx in mouth_indices:
            x = int(landmarks[idx].x * frame_w)
            y = int(landmarks[idx].y * frame_h)
            mouth_points.append([x, y])

        mouth_points = np.array(mouth_points)

        # Vertical distance
        v = np.linalg.norm(mouth_points[2] - mouth_points[6])

        # Horizontal distance
        h = np.linalg.norm(mouth_points[0] - mouth_points[4])

        # MAR formula
        mar = v / h if h != 0 else 0
        return mar

    def analyze(self, frame):
        """
        Analyzes a frame and returns:
        - face_detected: True/False
        - ear: eye aspect ratio
        - mar: mouth aspect ratio
        - is_drowsy: True if eyes closing or yawning
        - annotated_frame: frame with landmarks drawn
        """
        frame_h, frame_w = frame.shape[:2]
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb_frame)

        if not results.multi_face_landmarks:
            return {
                'face_detected': False,
                'ear': 0,
                'mar': 0,
                'is_drowsy': False,
                'annotated_frame': frame
            }

        landmarks = results.multi_face_landmarks[0].landmark

        # Calculate EAR and MAR
        left_ear  = self.calculate_ear(landmarks, self.LEFT_EYE, frame_w, frame_h)
        right_ear = self.calculate_ear(landmarks, self.RIGHT_EYE, frame_w, frame_h)
        ear = (left_ear + right_ear) / 2.0
        mar = self.calculate_mar(landmarks, self.MOUTH, frame_w, frame_h)

        # Thresholds
        EAR_THRESHOLD = 0.25  # below this = eyes closing
        MAR_THRESHOLD = 0.60  # above this = yawning

        is_drowsy = ear < EAR_THRESHOLD or mar > MAR_THRESHOLD

        # Draw landmarks on frame
        annotated_frame = frame.copy()
        self.mp_drawing.draw_landmarks(
            annotated_frame,
            results.multi_face_landmarks[0],
            self.mp_face_mesh.FACEMESH_CONTOURS
        )

        return {
            'face_detected': True,
            'ear': round(ear, 3),
            'mar': round(mar, 3),
            'is_drowsy': is_drowsy,
            'annotated_frame': annotated_frame
        }