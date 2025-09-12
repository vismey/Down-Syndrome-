# video/face_detection.py
import cv2
import mediapipe as mp

mp_face_mesh = mp.solutions.face_mesh

# Initialize Mediapipe Face Mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

def get_face_landmarks(frame):
    """
    Detect facial landmarks on the given frame.
    Returns the first face landmarks or None.
    """
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_frame)
    
    if results.multi_face_landmarks:
        return results.multi_face_landmarks[0]
    return None

def draw_landmarks(frame, landmarks):
    """
    Draw landmarks on the frame
    """
    if landmarks:
        h, w, _ = frame.shape
        for lm in landmarks.landmark:
            x, y = int(lm.x * w), int(lm.y * h)
            cv2.circle(frame, (x, y), 1, (0, 255, 0), -1)
    return frame
