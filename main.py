import cv2
import mediapipe as mp
import numpy as np
import json
from collections import deque

class HandDrawingApp:
    def __init__(self):
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.hands = self.mp_hands.Hands(
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7
        )

        self.cap = cv2.VideoCapture(0)
        
        self.drawing = False
        self.drawn_points = []
        self.smooth_buffer = deque(maxlen=5)  
        self.brush_color = (255, 0, 0)  
        self.brush_size = 3

    def calculate_distance(self, p1, p2):
        return np.linalg.norm(np.array(p1) - np.array(p2))

    def clear_canvas(self):
        self.drawn_points = []

    def save_canvas(self, frame):
        cv2.imwrite("drawing_output.png", frame)
        with open("drawing_data.json", "w") as f:
            json.dump(self.drawn_points, f)
        print("Drawing saved as image and JSON")

    def run(self):
        while self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)
            h, w, _ = frame.shape
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = self.hands.process(rgb_frame)

            if result.multi_hand_landmarks:
                for hand_landmarks in result.multi_hand_landmarks:
                    thumb_tip = (
                        hand_landmarks.landmark[self.mp_hands.HandLandmark.THUMB_TIP].x * w,
                        hand_landmarks.landmark[self.mp_hands.HandLandmark.THUMB_TIP].y * h
                    )
                    index_tip = (
                        hand_landmarks.landmark[self.mp_hands.HandLandmark.INDEX_FINGER_TIP].x * w,
                        hand_landmarks.landmark[self.mp_hands.HandLandmark.INDEX_FINGER_TIP].y * h
                    )

                    distance = self.calculate_distance(thumb_tip, index_tip)

                    self.drawing = distance < 35

                    if self.drawing:
                        self.smooth_buffer.append(index_tip)
                        avg_x = int(np.mean([p[0] for p in self.smooth_buffer]))
                        avg_y = int(np.mean([p[1] for p in self.smooth_buffer]))
                        point = (avg_x, avg_y)
                        self.drawn_points.append(point)
                        cv2.circle(frame, point, 5, (0, 255, 0), -1)

                    self.mp_drawing.draw_landmarks(
                        frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS
                    )

            for i in range(1, len(self.drawn_points)):
                if self.drawn_points[i - 1] and self.drawn_points[i]:
                    cv2.line(frame, self.drawn_points[i - 1], self.drawn_points[i], self.brush_color, self.brush_size)

            cv2.imshow("Hand Drawing", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"): 
                break
            elif key == ord("c"):  
                self.clear_canvas()
            elif key == ord("s"):  
                self.save_canvas(frame)
            elif key == ord("1"):
                self.brush_color = (255, 0, 0)  
            elif key == ord("2"):
                self.brush_color = (0, 255, 0)  
            elif key == ord("3"):
                self.brush_color = (0, 0, 255) 
            elif key == ord("+"):
                self.brush_size = min(10, self.brush_size + 1)
            elif key == ord("-"):
                self.brush_size = max(1, self.brush_size - 1)

        self.cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    app = HandDrawingApp()
    app.run()
