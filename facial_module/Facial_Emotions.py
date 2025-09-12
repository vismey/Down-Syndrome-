import cv2
from .face_detection import get_face_landmarks, draw_landmarks
from .face_emotion import get_facial_emotion

print("Starting facial emotion recognition...")
print("📱 Using OpenCV-based detection initially...")
print("🔄 Hugging Face model will load in background...")

# Initialize webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open webcam")
    exit()

print("Webcam initialized successfully!")
print("Press 'q' to quit the application")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Face detection & landmarks
    landmarks = get_face_landmarks(frame)
    frame = draw_landmarks(frame, landmarks)

    # Facial emotion detection
    emotion, confidence = get_facial_emotion(frame)
    if emotion:
        cv2.putText(frame, f"{emotion} ({confidence:.0f}%)", (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # Display
    cv2.imshow("Facial Emotion Recognition", frame)

    # Quit on 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
