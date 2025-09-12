import cv2
import numpy as np
import random
import time
import threading
from collections import deque
EMOTION_LABELS = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral", "contempt"]

# Smoothing / history length (number of frames to average probabilities over)
PROB_HISTORY_LENGTH = 5
NEUTRAL_THRESHOLD = 0.45  # if max averaged prob < threshold -> 'neutral'

# ----------------- Globals -----------------
last_emotion_time = 0
current_emotion = "neutral"
current_confidence = 50.0
huggingface_error_count = 0
MAX_HF_ERRORS = 5

USE_HUGGING_FACE = False
HUGGING_FACE_LOADING = False
emotion_classifier = None
_model_labels = None            # ordered list of labels returned by model (lowercased)
_model_label_to_index = {}    
_model_num_labels = 0
_prob_history = deque(maxlen=PROB_HISTORY_LENGTH)
_loading_thread = None
_state_lock = threading.Lock()

# Haar cascades loaded once
EYE_CASCADE = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
MOUTH_CASCADE = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_smile.xml')

# ----------------- Model loader -----------------
def load_hardlyhumans_model():
    """Load the HardlyHumans HF pipeline in a background thread."""
    global USE_HUGGING_FACE, HUGGING_FACE_LOADING, emotion_classifier
    global _model_labels, _model_label_to_index, _model_num_labels

    try:
        print("🔄 Loading HardlyHumans/Facial-expression-detection in background...")
        # local imports to avoid blocking import time when this module is imported
        from transformers import pipeline
        import torch

        device = 0 if torch.cuda.is_available() else -1
        # create pipeline (may download weights on first run)
        emotion_classifier = pipeline(
            "image-classification",
            model="HardlyHumans/Facial-expression-detection",
            device=device,
        )
        id2label = emotion_classifier.model.config.id2label
        # id2label keys can be str or ints; sort by int(key)
        sorted_items = sorted(id2label.items(), key=lambda kv: int(kv[0]))
        _model_labels = [v.lower() for (_, v) in sorted_items]
        _model_label_to_index = {label: idx for idx, label in enumerate(_model_labels)}
        _model_num_labels = len(_model_labels)

        with _state_lock:
            USE_HUGGING_FACE = True
            HUGGING_FACE_LOADING = False
        print("✅ HardlyHumans model loaded. Labels:", _model_labels)

    except Exception as e:
        with _state_lock:
            USE_HUGGING_FACE = False
            HUGGING_FACE_LOADING = False
        print(f"⚠️ Failed loading HardlyHumans model: {e}")


def start_hardlyhumans_loading():
    global HUGGING_FACE_LOADING, _loading_thread
    with _state_lock:
        if HUGGING_FACE_LOADING or USE_HUGGING_FACE:
            return
        HUGGING_FACE_LOADING = True
    _loading_thread = threading.Thread(target=load_hardlyhumans_model, daemon=True)
    _loading_thread.start()
    print("🚀 Started background HF model loading...")

# ----------------- OpenCV heuristic fallback -----------------
def analyze_facial_features(face_roi):
    """Very lightweight heuristic using Haar cascades as a fallback."""
    try:
        gray = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
        eyes = EYE_CASCADE.detectMultiScale(gray, 1.1, 4)
        mouths = MOUTH_CASCADE.detectMultiScale(gray, 1.1, 4)

        emotion_scores = {
            'happy': 20,
            'sad': 20,
            'angry': 20,
            'surprise': 20,
            'neutral': 25,
            'fear': 20,
            'disgust': 20,
            'contempt': 15,
        }

        if len(mouths) > 0:
            emotion_scores['happy'] += 15
            emotion_scores['neutral'] += 5
        else:
            emotion_scores['sad'] += 12
            emotion_scores['angry'] += 10

        if len(eyes) >= 2:
            emotion_scores['surprise'] += 10
        elif len(eyes) == 1:
            emotion_scores['sad'] += 8
            emotion_scores['angry'] += 8

        for k in emotion_scores:
            emotion_scores[k] += random.randint(0, 5)

        dominant = max(emotion_scores, key=emotion_scores.get)
        confidence = min(emotion_scores[dominant], 85)
        return dominant, confidence

    except Exception as e:
        print("Fallback heuristic error:", e)
        return "neutral", 50.0

# ----------------- Hugging Face detection (with smoothing & neutral threshold) -----------------
def detect_emotion_hf(face_roi):
    """Use the HF pipeline to get a probability vector, store it in a rolling history,
    average across recent frames, and apply a neutral threshold.
    Returns (emotion_label, confidence_percent).
    """
    global _prob_history, _model_labels, _model_label_to_index, _model_num_labels

    try:
        from PIL import Image

        # Convert BGR -> RGB
        face_rgb = cv2.cvtColor(face_roi, cv2.COLOR_BGR2RGB)
        # Resize to a reasonable size (the pipeline will generally handle resizing but
        # keeping a consistent input size can help stability)
        face_rgb = cv2.resize(face_rgb, (224, 224))
        pil_img = Image.fromarray(face_rgb)

        # ask pipeline for full distribution
        top_k = _model_num_labels if _model_num_labels > 0 else None
        results = emotion_classifier(pil_img, top_k=top_k)

        if not results:
            return "neutral", 50.0

        # Build probability vector aligned with _model_labels
        probs = np.zeros(_model_num_labels, dtype=np.float32)
        for r in results:
            label = str(r.get('label', '')).lower()
            score = float(r.get('score', 0.0))
            idx = _model_label_to_index.get(label)
            if idx is None:
                # sometimes labels are like 'LABEL_0' or '0' — try to parse
                try:
                    maybe_idx = int(label.split('_')[-1])
                    if 0 <= maybe_idx < _model_num_labels:
                        probs[maybe_idx] = score
                except Exception:
                    # unknown label — skip
                    pass
            else:
                probs[idx] = score

        # push into history and compute average
        _prob_history.append(probs)
        avg_probs = np.mean(np.stack(list(_prob_history)), axis=0)

        max_idx = int(np.argmax(avg_probs))
        max_prob = float(avg_probs[max_idx])
        emotion = _model_labels[max_idx]

        # neutral threshold
        if max_prob < NEUTRAL_THRESHOLD:
            return 'neutral', max_prob * 100.0
        else:
            return emotion, max_prob * 100.0

    except Exception as e:
        print("HF detection error:", e)
        return "neutral", 50.0

# ----------------- Public API -----------------
def get_facial_emotion(frame, face_coords=None):
    """Return (emotion_label, confidence_percent).
    Hybrid: attempt HF (if loaded), otherwise fallback to heuristic.
    Uses a background loader to load HF model once.
    """
    global last_emotion_time, current_emotion, current_confidence, huggingface_error_count
    global USE_HUGGING_FACE, HUGGING_FACE_LOADING

    try:
        # extract ROI
        if face_coords is not None:
            x, y, w, h = face_coords
            face_roi = frame[y:y+h, x:x+w]
        else:
            face_roi = frame

        # validate ROI
        if face_roi is None or face_roi.size == 0:
            return current_emotion, current_confidence
        if face_roi.shape[0] < 20 or face_roi.shape[1] < 20:
            return current_emotion, current_confidence

        # start background loading if necessary
        with _state_lock:
            should_start = (not USE_HUGGING_FACE) and (not HUGGING_FACE_LOADING)
        if should_start:
            start_hardlyhumans_loading()

        # try HF if loaded and not too many recent errors
        if USE_HUGGING_FACE and huggingface_error_count < MAX_HF_ERRORS:
            try:
                new_emotion, new_confidence = detect_emotion_hf(face_roi)
                huggingface_error_count = 0
            except Exception as e:
                huggingface_error_count += 1
                print(f"HF error {huggingface_error_count}/{MAX_HF_ERRORS}: {e}")
                if huggingface_error_count >= MAX_HF_ERRORS:
                    with _state_lock:
                        USE_HUGGING_FACE = False
                new_emotion, new_confidence = analyze_facial_features(face_roi)
        else:
            new_emotion, new_confidence = analyze_facial_features(face_roi)

        # stability: only update every 1s
        current_time = time.time()
        if current_time - last_emotion_time > 1.0:
            current_emotion = new_emotion
            current_confidence = new_confidence
            last_emotion_time = current_time

        return current_emotion, current_confidence

    except Exception as e:
        print("Error in get_facial_emotion:", e)
        return current_emotion, current_confidence
