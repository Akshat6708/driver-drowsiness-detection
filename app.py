import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
import winsound
import time

# Load trained model
model = tf.keras.models.load_model("mobilenet_model.h5")
print("Model loaded successfully!")

# Mediapipe setup
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# Webcam setup
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise IOError("Cannot open webcam")

# Drowsiness alert settings
closed_start_time = None
ALERT_THRESHOLD = 3  # seconds
alert_on = False
PRED_THRESHOLD = 0.4  # <0.4 = open, >0.4 = closed (adjustable)

# Eye landmark indices
LEFT_EYE_IDX = [33, 133, 160, 159, 158, 157, 173, 153, 144, 145, 163, 7]
RIGHT_EYE_IDX = [362, 263, 387, 386, 385, 384, 398, 382, 381, 380, 374, 373, 390, 249]

def preprocess_eye(img):
    """Resize and normalize an eye image."""
    eye = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    eye = cv2.resize(eye, (224, 224))
    eye = eye.astype("float32") / 255.0
    eye = np.expand_dims(eye, axis=0)
    return eye

def get_eye_crop(frame, face_landmarks, indexes, w, h, scale=1.4):
    """Crop eye region from landmarks with margin scaling."""
    x_coords = [int(face_landmarks.landmark[i].x * w) for i in indexes]
    y_coords = [int(face_landmarks.landmark[i].y * h) for i in indexes]
    x_min, x_max = min(x_coords), max(x_coords)
    y_min, y_max = min(y_coords), max(y_coords)

    cx, cy = (x_min + x_max) // 2, (y_min + y_max) // 2
    w_box, h_box = int((x_max - x_min) * scale), int((y_max - y_min) * scale)
    x1 = max(cx - w_box // 2, 0)
    y1 = max(cy - h_box // 2, 0)
    x2 = min(cx + w_box // 2, w)
    y2 = min(cy + h_box // 2, h)

    return frame[y1:y2, x1:x2], (x1, y1, x2, y2)

# Start detection loop
print("🎥 Starting detection... Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    h, w, _ = frame.shape
    result = face_mesh.process(rgb)

    eyes_closed = False
    avg_pred = 0
    n_eyes = 0

    if result.multi_face_landmarks:
        face_landmarks = result.multi_face_landmarks[0]

        # Draw face mesh landmarks
        for lm in face_landmarks.landmark:
            x, y = int(lm.x * w), int(lm.y * h)
            cv2.circle(frame, (x, y), 1, (0, 255, 0), -1)

        # Get both eyes
        left_eye_img, (lx1, ly1, lx2, ly2) = get_eye_crop(frame, face_landmarks, LEFT_EYE_IDX, w, h)
        right_eye_img, (rx1, ry1, rx2, ry2) = get_eye_crop(frame, face_landmarks, RIGHT_EYE_IDX, w, h)

        for (eye_img, (x1, y1, x2, y2)) in [(left_eye_img, (lx1, ly1, lx2, ly2)), (right_eye_img, (rx1, ry1, rx2, ry2))]:
            if eye_img.size == 0:
                continue

            # Preprocess and predict
            eye_input = preprocess_eye(eye_img)
            pred = model.predict(eye_input, verbose=0)[0][0]

            # Draw rectangle + show probability
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 0), 2)
            label = f"Pred: {pred:.2f}"
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)

            avg_pred += pred
            n_eyes += 1

        # Decision logic
        if n_eyes > 0:
            avg_pred /= n_eyes
            if avg_pred > PRED_THRESHOLD:  # 1 = closed
                eyes_closed = True
                status = "Eyes Closed"
            else:
                eyes_closed = False
                status = "Eyes Open"
        else:
            status = "Eyes not detected"
    else:
        status = "No face detected"

    # Drowsiness alert
    if eyes_closed:
        if closed_start_time is None:
            closed_start_time = time.time()
        elif time.time() - closed_start_time >= 2:
            status = " DROWSY! WAKE UP!"
            if not alert_on:
                winsound.Beep(2000, 1000)
                alert_on = True
    else:
        closed_start_time = None
        alert_on = False

    color = (0, 255, 0) if "Open" in status else (0, 0, 255)
    cv2.putText(frame, status, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 3, cv2.LINE_AA)

    cv2.imshow("Drowsiness Detection", frame)
    if cv2.waitKey(2) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
