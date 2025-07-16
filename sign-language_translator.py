import cv2
import mediapipe as mp
import numpy as np
import time

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5
)

cap = cv2.VideoCapture(0)

# Set webcam resolution 
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

prev_letter = ''
recognized_text = ''
last_time = time.time()

# Finger tip landmarks
TIPS = [4, 8, 12, 16, 20]

def get_finger_status(landmarks):
    fingers = []

    # Thumb: compare x
    if landmarks[TIPS[0]].x < landmarks[TIPS[0] - 2].x:
        fingers.append(1)
    else:
        fingers.append(0)

    # Other fingers: compare y
    for tip in TIPS[1:]:
        if landmarks[tip].y < landmarks[tip - 2].y:
            fingers.append(1)
        else:
            fingers.append(0)

    return fingers

# Simple logic-based gesture classification
def classify_letter(fingers):
    if fingers == [0, 0, 0, 0, 0]:
        return 'A'
    elif fingers == [0, 1, 1, 1, 1]:
        return 'B'
    elif fingers == [0, 1, 1, 0, 0]:
        return 'C'
    elif fingers == [0, 1, 0, 0, 0]:
        return 'D'
    elif fingers == [0, 0, 0, 0, 1]:
        return 'E'
    elif fingers == [1, 1, 0, 0, 0]:
        return 'F'
    elif fingers == [1, 1, 1, 0, 0]:
        return 'G'
    elif fingers == [1, 1, 1, 1, 0]:
        return 'H'
    elif fingers == [0, 1, 0, 0, 0]:
        return 'I'
    elif fingers == [1, 1, 0, 0, 1]:
        return 'J'
    elif fingers == [0, 1, 0, 1, 1]:
        return 'K'
    elif fingers == [1, 0, 0, 0, 0]:
        return 'L'
    elif fingers == [1, 1, 0, 0, 1]:
        return 'M'
    elif fingers == [1, 1, 1, 0, 1]:
        return 'N'
    elif fingers == [0, 1, 1, 1, 1]:
        return 'O'
    elif fingers == [1, 1, 0, 1, 0]:
        return 'P'
    elif fingers == [1, 1, 0, 1, 1]:
        return 'Q'
    elif fingers == [0, 1, 0, 1, 0]:
        return 'R'
    elif fingers == [0, 1, 1, 0, 0]:
        return 'S'
    elif fingers == [0, 1, 1, 1, 0]:
        return 'T'
    elif fingers == [0, 1, 1, 0, 0]:
        return 'U'
    elif fingers == [0, 1, 1, 1, 0]:
        return 'V'
    elif fingers == [0, 1, 1, 1, 1]:
        return 'W'
    elif fingers == [1, 0, 1, 0, 1]:
        return 'X'
    elif fingers == [1, 0, 0, 0, 1]:
        return 'Y'
    elif fingers == [1, 1, 1, 1, 1]:
        return 'Z'
    else:
        return ''



while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame.")
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            fingers = get_finger_status(hand_landmarks.landmark)
            current_letter = classify_letter(fingers)

            if current_letter != '' and current_letter != prev_letter:
                if time.time() - last_time > 1.2:
                    if current_letter == 'Space':
                        recognized_text += ' '
                    else:
                        recognized_text += current_letter
                    prev_letter = current_letter
                    last_time = time.time()

            cv2.putText(frame, f'Letter: {current_letter}', (10, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)

    # Display running text
    cv2.putText(frame, f'Text: {recognized_text}', (10, 450),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 0, 0), 3)

    cv2.imshow("Sign Language Recognizer", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
