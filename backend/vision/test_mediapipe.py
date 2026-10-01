import cv2
import mediapipe as mp

print("OpenCV version:", cv2.__version__)
print("MediaPipe imported successfully!")

mp_face_mesh = mp.solutions.face_mesh

with mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True
) as face_mesh:
    print("MediaPipe Face Mesh initialized successfully!")

print("Environment setup successful!")