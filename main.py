import cv2
import mediapipe as mp
import numpy as np
import json

# Initialize MediaPipe Hand module
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Setup video capture
cap = cv2.VideoCapture(0)
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7, min_tracking_confidence=0.7)

# Variables to store drawing state and data
drawing = False
drawn_points = []

# Function to calculate distance between two points
def calculate_distance(p1, p2):
    return np.linalg.norm(np.array(p1) - np.array(p2))

# Function to clear the canvas
def clear_canvas():
    global drawn_points
    drawn_points = []  # Reset the list of drawn points

# Main loop
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Flip the frame horizontally for a later selfie-view display
    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape

    # Convert the BGR image to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the frame with MediaPipe Hands
    result = hands.process(rgb_frame)

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            # Get coordinates of thumb_tip and index_finger_tip
            thumb_tip = (hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].x * w,
                         hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].y * h)
            index_finger_tip = (hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].x * w,
                                hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y * h)

            # Calculate distance
            distance = calculate_distance(thumb_tip, index_finger_tip)

            # Check if the distance is less than a certain value to start drawing
            if distance < 35:
                drawing = True
            else:
                drawing = False

            if drawing:
                cv2.circle(frame, (int(index_finger_tip[0]), int(index_finger_tip[1])), 5, (0, 255, 0), -1)
                drawn_points.append((int(index_finger_tip[0]), int(index_finger_tip[1])))

            # Draw hand landmarks
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    # Draw the path
    for i in range(1, len(drawn_points)):
        if drawn_points[i - 1] is None or drawn_points[i] is None:
            continue
        cv2.line(frame, drawn_points[i - 1], drawn_points[i], (255, 0, 0), 3)

    # Show the video frame with the drawing
    cv2.imshow('Hand Drawing', frame)

    # Handle key presses
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):  # Quit the program
        break
    elif key == ord('c'):  # Clear the canvas
        clear_canvas()

cap.release()
cv2.destroyAllWindows()

# Save drawn points to a JSON file
with open('drawing_data.json', 'w') as f:
    json.dump(drawn_points, f)
