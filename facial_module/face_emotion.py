# video/facial_emotion.py
from deepface import DeepFace

def get_facial_emotion(frame):
    """
    Returns dominant emotion and confidence for the given frame.
    """
    try:
        result = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)
        dominant_emotion = result['dominant_emotion']
        confidence = max(result['emotion'].values())
        return dominant_emotion, confidence
    except Exception as e:
        # If face not detected or error occurs
        return None, 0