import cv2
import mediapipe as mp
import numpy as np
import threading
import queue
import os
import time

# Suppress TensorFlow logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Set camera resolution
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# MediaPipe initialization
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# Drawing canvas and color setup
canvas = np.zeros((480, 640, 3), dtype=np.uint8)
colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255)]
current_color_index = 0
points = []

# Threading setup
frame_queue = queue.Queue()
output_queue = queue.Queue()

def capture_frames():
    while True:
        ret, frame = cap.read()
        if ret:
            frame_queue.put(frame)

def count_fingers(landmarks):
    finger_tips = [4, 8, 12, 16, 20]
    count = 0
    if landmarks[4].x < landmarks[3].x:
        count += 1
    for tip in finger_tips[1:]:
        if landmarks[tip].y < landmarks[tip - 2].y:
            count += 1
    return count

def is_hand_closed(landmarks):
    return landmarks[4].y > landmarks[3].y and all(landmarks[tip].y > landmarks[tip - 2].y for tip in [8, 12, 16, 20])

def process_frames():
    global canvas, current_color_index
    while True:
        frame = frame_queue.get()
        if frame is None:
            continue
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb_frame)

        if results.multi_hand_landmarks:
            for landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame, landmarks, mp_hands.HAND_CONNECTIONS)
                finger_count = count_fingers(landmarks.landmark)

                if is_hand_closed(landmarks.landmark):
                    points.clear()
                elif finger_count == 3:
                    current_color_index = (current_color_index + 1) % len(colors)
                    points.clear()
                    time.sleep(1)  # Delay to avoid rapid color change
                elif finger_count == 5:
                    canvas = np.zeros((480, 640, 3), dtype=np.uint8)
                    points.clear()
                    time.sleep(0.2)
                else:
                    x = int(landmarks.landmark[8].x * frame.shape[1])
                    y = int(landmarks.landmark[8].y * frame.shape[0])
                    points.append((x, y))

        output_queue.put((frame, canvas.copy()))

# Start capture and process threads
capture_thread = threading.Thread(target=capture_frames, daemon=True)
process_thread = threading.Thread(target=process_frames, daemon=True)

capture_thread.start()
process_thread.start()

while True:
    try:
        frame, canvas_display = output_queue.get(timeout=1)
    except queue.Empty:
        continue

    if frame is None or not isinstance(frame, np.ndarray) or frame.size == 0:
        continue

    if len(points) > 1:
        for i in range(1, len(points)):
            cv2.line(canvas_display, points[i-1], points[i], colors[current_color_index], 5)

    cv2.rectangle(frame, (10, 10, 50, 50), colors[current_color_index], -1)
    cv2.putText(frame, f"Color {current_color_index + 1}", (70, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    cv2.imshow("Frame", frame)
    cv2.imshow("Canvas", canvas_display)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()
