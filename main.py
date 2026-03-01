import os
import sys
import cv2
import numpy as np
import math
import time
import pyautogui
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from volumenbrightness import VolumeGesture, VolumeController


class HandTracker:
    
    def __init__(self, model_path, cam_width=640, cam_height=480, confidence_threshold=0.5):
        self.cam_width = cam_width
        self.cam_height = cam_height
        self.confidence_threshold = confidence_threshold
        self.hand_confidence = 0.0
        self.landmarker = None
        self.initialize_model(model_path)
    
    def initialize_model(self, model_path):
        if not os.path.exists(model_path):
            print(f"ERROR: Model file not found at {model_path}")
            print("Please download it and place it in the same folder as this script.")
            sys.exit()
        else:
            print(f"Model loaded successfully from: {model_path}")
            print(f"Hand confidence threshold: {self.confidence_threshold * 100:.0f}%")
        
        BaseOptions = mp.tasks.BaseOptions
        HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
        VisionRunningMode = mp.tasks.vision.RunningMode
        
        options = HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=model_path),
            running_mode=VisionRunningMode.VIDEO,
            num_hands=2,
            min_hand_detection_confidence=self.confidence_threshold,
            min_hand_presence_confidence=self.confidence_threshold
        )
        
        self.landmarker = mp.tasks.vision.HandLandmarker.create_from_options(options)
    
    def detect_hand(self, frame):
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
        frame_timestamp_ms = int(time.time() * 1000)
        detection_result = self.landmarker.detect_for_video(mp_image, frame_timestamp_ms)
        
        left_hand = []
        right_hand = []
        self.hand_confidence = 0.0
        
        if detection_result.hand_landmarks and detection_result.handedness:
            for idx, (landmarks, handedness_info) in enumerate(zip(detection_result.hand_landmarks, detection_result.handedness)):
                hand_score = handedness_info[0].score
                hand_label = handedness_info[0].category_name
                
                if hand_score >= self.confidence_threshold:
                    lmList = []
                    for id, landmark in enumerate(landmarks):
                        px, py = int(landmark.x * self.cam_width), int(landmark.y * self.cam_height)
                        lmList.append([id, px, py])
                    
                    if hand_label == "Left":
                        left_hand = lmList
                    elif hand_label == "Right":
                        right_hand = lmList
                    
                    self.hand_confidence = hand_score
        
        return left_hand, right_hand
    
    def visualize_landmarks(self, frame, lmList):
        for lm in lmList:
            cv2.circle(frame, (lm[1], lm[2]), 5, (255, 0, 255), cv2.FILLED)
        return frame
    
    def close(self):
        if self.landmarker:
            self.landmarker.close()


class GestureLogic:
    
    def __init__(self, cam_width=640, cam_height=480, screen_width=None, screen_height=None, 
                 frame_reduction=100, smoothening=9, click_distance_threshold=35):
        self.cam_width = cam_width
        self.cam_height = cam_height
        self.screen_width = screen_width or pyautogui.size()[0]
        self.screen_height = screen_height or pyautogui.size()[1]
        self.frame_reduction = frame_reduction
        self.smoothening = smoothening
        self.click_distance_threshold = click_distance_threshold
        
        self.prev_loc_x = 0
        self.prev_loc_y = 0
        self.curr_loc_x = 0
        self.curr_loc_y = 0
        self.prev_click_state = False
        self.dragging = False
        self.drag_start_x = 0
        self.drag_start_y = 0
    
    def get_fingers_up(self, lmList):
        fingers = []
        
        tipIds = [8, 12, 16, 20]
        pipIds = [6, 10, 14, 18]
        
        for i in range(0, 4):
            if lmList[tipIds[i]][2] < lmList[pipIds[i]][2] - 15:
                fingers.append(1)
            else:
                fingers.append(0)
        
        if lmList[4][1] > lmList[3][1]:
            fingers.append(1)
        else:
            fingers.append(0)
        
        if sum(fingers) == 5:
            fingers.insert(0, 1)
        else:
            fingers.insert(0, 0)
        
        return fingers
    
    def process_gesture(self, lmList, frame):
        if len(lmList) == 0:
            self.prev_click_state = False
            self.dragging = False
            return None, frame
        
        x1, y1 = lmList[8][1:]
        x2, y2 = lmList[12][1:]
        x3, y3 = lmList[16][1:]
        
        fingers = self.get_fingers_up(lmList)
        
        cv2.rectangle(frame, (self.frame_reduction, self.frame_reduction), 
                     (self.cam_width - self.frame_reduction, 
                      self.cam_height - self.frame_reduction), (255, 0, 255), 2)
        
        action = None
        
        if fingers[1] == 1 and fingers[2] == 0 and fingers[3] == 0:
            x_interp = np.interp(x1, (self.frame_reduction, self.cam_width - self.frame_reduction), 
                          (0, self.screen_width))
            y_interp = np.interp(y1, (self.frame_reduction, self.cam_height - self.frame_reduction), 
                          (0, self.screen_height))
            
            x_interp = max(0, min(self.screen_width, x_interp))
            y_interp = max(0, min(self.screen_height, y_interp))
            
            self.curr_loc_x = self.prev_loc_x + (x_interp - self.prev_loc_x) / self.smoothening
            self.curr_loc_y = self.prev_loc_y + (y_interp - self.prev_loc_y) / self.smoothening
            
            action = ("move", self.curr_loc_x, self.curr_loc_y)
            cv2.circle(frame, (x1, y1), 15, (0, 255, 0), cv2.FILLED)
            self.prev_loc_x, self.prev_loc_y = self.curr_loc_x, self.curr_loc_y
            self.prev_click_state = False
            self.dragging = False
        
        elif fingers[1] == 1 and fingers[2] == 1 and fingers[3] == 0:
            length = math.hypot(x2 - x1, y2 - y1)
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            
            cv2.line(frame, (x1, y1), (x2, y2), (255, 0, 255), 3)
            
            is_click_ready = length < self.click_distance_threshold
            
            if is_click_ready and not self.prev_click_state:
                cv2.circle(frame, (cx, cy), 15, (0, 255, 0), cv2.FILLED)
                action = ("click",)
                self.prev_click_state = True
            elif not is_click_ready:
                cv2.circle(frame, (cx, cy), 15, (255, 0, 255), cv2.FILLED)
                self.prev_click_state = False
            else:
                cv2.circle(frame, (cx, cy), 15, (255, 0, 255), cv2.FILLED)
        
        else:
            self.prev_click_state = False
            self.dragging = False
        
        return action, frame


class MouseController:
    
    def __init__(self):
        pyautogui.FAILSAFE = False
    
    def move_mouse(self, x, y):
        pyautogui.moveTo(int(x), int(y))
    
    def click_mouse(self):
        pyautogui.click()


class GestureVirtualMouse:
    
    def __init__(self, model_path, cam_width=640, cam_height=480):
        self.cam_width = cam_width
        self.cam_height = cam_height
        
        self.hand_tracker = HandTracker(model_path, cam_width, cam_height)
        self.gesture_logic = GestureLogic(cam_width, cam_height)
        self.mouse_controller = MouseController()
        self.volume_gesture = VolumeGesture(cam_width, cam_height)
        
        self.cap = cv2.VideoCapture(0)
        self.cap.set(3, cam_width)
        self.cap.set(4, cam_height)
        
        self.prev_time = 0
    
    def run(self):
        try:
            while self.cap.isOpened():
                success, frame = self.cap.read()
                if not success:
                    break

                frame = cv2.flip(frame, 1)

                left_hand, right_hand = self.hand_tracker.detect_hand(frame)

                frame = self.hand_tracker.visualize_landmarks(frame, left_hand)
                frame = self.hand_tracker.visualize_landmarks(frame, right_hand)

                # Swap: left hand for mouse, right hand for volume
                action, frame = self.gesture_logic.process_gesture(left_hand, frame)
                frame, volume_level = self.volume_gesture.process_frame(frame, right_hand)

                if action:
                    if action[0] == "move":
                        self.mouse_controller.move_mouse(action[1], action[2])
                    elif action[0] == "click":
                        self.mouse_controller.click_mouse()

                curr_time = time.time()
                fps = 1 / (curr_time - self.prev_time) if self.prev_time > 0 else 0
                self.prev_time = curr_time
                cv2.putText(frame, f"FPS: {int(fps)}", (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)
                # Short display text
                cv2.putText(frame, "Left: Mouse  Right: Vol", (10, frame.shape[0] - 10), cv2.FONT_HERSHEY_PLAIN, 1.2, (200, 200, 200), 1)

                cv2.imshow("Gesture Virtual Mouse", frame)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        finally:
            self.cleanup()
    
    def cleanup(self):
        self.cap.release()
        cv2.destroyAllWindows()
        self.hand_tracker.close()
        print("Shutting down...")


if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(script_dir, 'hand_landmarker.task')
    print("Starting Gesture Virtual Mouse...")
    app = GestureVirtualMouse(model_path)
    app.run()