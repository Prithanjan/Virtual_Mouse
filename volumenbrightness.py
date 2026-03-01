import os
import cv2
import numpy as np
import math

try:
    from pycaw.pycaw import AudioUtilities, ISimpleAudioVolume
    from comtypes import CLSCTX_ALL
    from ctypes import cast, POINTER
    PYCAW_AVAILABLE = True
except ImportError:
    PYCAW_AVAILABLE = False
    print("Warning: pycaw not installed. Volume control disabled.")


class VolumeController:
    
    def __init__(self):
        self.current_volume = 50
        self.volume = None
        if PYCAW_AVAILABLE:
            try:
                devices = AudioUtilities.GetSpeakers()
                interface = devices.Activate(ISimpleAudioVolume._iid_, CLSCTX_ALL, None)
                self.volume = cast(interface, POINTER(ISimpleAudioVolume))
            except Exception as e:
                print(f"Error initializing volume control: {e}")
                self.volume = None
    
    def set_volume(self, level):
        level = max(0, min(100, int(level)))
        self.current_volume = level
        
        if self.volume and PYCAW_AVAILABLE:
            try:
                normalized_volume = level / 100.0
                self.volume.SetMasterVolume(normalized_volume, None)
            except Exception as e:
                pass
    
    def get_volume(self):
        return self.current_volume
    
    def increase_volume(self, step=5):
        self.set_volume(self.current_volume + step)
    
    def decrease_volume(self, step=5):
        self.set_volume(self.current_volume - step)


class VolumeGesture:
    
    def __init__(self, cam_width=640, cam_height=480):
        self.cam_width = cam_width
        self.cam_height = cam_height
        self.volume_controller = VolumeController()
        self.volume_mode_active = False
    
    def calculate_distance(self, lmList, point1_id, point2_id):
        if len(lmList) == 0:
            return 0
        
        x1, y1 = lmList[point1_id][1], lmList[point1_id][2]
        x2, y2 = lmList[point2_id][1], lmList[point2_id][2]
        distance = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
        return distance
    
    def detect_volume_gesture(self, lmList):
        if len(lmList) == 0:
            self.volume_mode_active = False
            return None
        
        thumb_index_distance = self.calculate_distance(lmList, 4, 8)
        
        if 30 < thumb_index_distance < 150:
            self.volume_mode_active = True
            volume_level = np.interp(thumb_index_distance, (30, 150), (0, 100))
            self.volume_controller.set_volume(volume_level)
            return volume_level
        else:
            self.volume_mode_active = False
            return None
    
    def draw_volume_visualization(self, frame, lmList, volume_level):
        if len(lmList) == 0 or volume_level is None:
            return frame
        
        x1, y1 = lmList[4][1], lmList[4][2]
        x2, y2 = lmList[8][1], lmList[8][2]
        
        cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 255), 3)
        cv2.circle(frame, (x1, y1), 10, (0, 255, 255), cv2.FILLED)
        cv2.circle(frame, (x2, y2), 10, (0, 255, 255), cv2.FILLED)
        
        cv2.putText(frame, f"Volume: {int(volume_level)}%", (20, 50), 
                   cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 255), 2)
        
        bar_length = 300
        bar_start_x = 50
        bar_start_y = 100
        
        cv2.rectangle(frame, (bar_start_x, bar_start_y), 
                     (bar_start_x + bar_length, bar_start_y + 30), (100, 100, 100), 2)
        
        filled_length = int(bar_length * (volume_level / 100))
        cv2.rectangle(frame, (bar_start_x, bar_start_y), 
                     (bar_start_x + filled_length, bar_start_y + 30), (0, 255, 255), -1)
        
        return frame
    
    def process_frame(self, frame, lmList):
        volume_level = self.detect_volume_gesture(lmList)
        
        if self.volume_mode_active:
            frame = self.draw_volume_visualization(frame, lmList, volume_level)
        
        return frame, volume_level