import cv2
import face_recognition
import os
import csv
import numpy as np
from datetime import datetime

# Make sure VS Code interpreter is set to Python 3.11.9
# Install dependencies using python 3.11.9:
#   py -3.11 -m pip install face_recognition (important)
#   py -3.11 -m pip install opencv-python
#   py -3.11 -m pip install numpy 

class FaceRecognizer:

    def __init__(self):
        self.known_encodings = []
        self.known_names = []
        self.marked_today = {}
        self.base_dir = os.path.dirname(os.path.abspath(__file__))
        self.known_faces_dir = os.path.join(self.base_dir, "known_faces")
        self.csv_path = os.path.join(self.base_dir, "attendance.csv")
        self.load_known_faces()
        self.setup_csv()

    def load_known_faces(self):
        for filename in os.listdir(self.known_faces_dir):
            if filename.lower().endswith((".jpg", ".png")):

                img_path = os.path.join(self.known_faces_dir, filename)
                known_img = face_recognition.load_image_file(img_path)
                encodings = face_recognition.face_encodings(known_img)

                if len(encodings) > 0:
                    encoding = encodings[0]
                    name = os.path.splitext(filename)[0]

                    self.known_encodings.append(encoding)
                    self.known_names.append(name)

                    print(f"Loaded: {name}")
                else:
                    print(f"Skipped (no face found): {filename}")

        print("All known faces loaded!")

    def setup_csv(self):
        if not os.path.exists(self.csv_path):
            with open(self.csv_path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(["Name", "Date", "Time"])

    def mark_attendance(self, name):

        if name == "Unknown":
            return

        if name in self.marked_today:
            return

        now = datetime.now()
        date = now.strftime("%Y-%m-%d")
        time = now.strftime("%H:%M")

        with open(self.csv_path, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([name, date, time])

        self.marked_today[name] = now

        print(f"Attendance marked: {name} at {time}")

    def recognize(self, img):

        rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        small_rgb = cv2.resize(rgb_img, (0, 0), fx=0.25, fy=0.25)
        face_locations = face_recognition.face_locations(small_rgb)
        face_encodings = face_recognition.face_encodings(small_rgb, face_locations)

        names = []

        for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):

            top *= 4
            right *= 4
            bottom *= 4
            left *= 4

            name = "Unknown"
            box_color = (0, 0, 255)

            if len(self.known_encodings) > 0:

                face_distances = face_recognition.face_distance(self.known_encodings, face_encoding)
                best_match_index = np.argmin(face_distances)

                if face_distances[best_match_index] < 0.6:
                    name = self.known_names[best_match_index]
                    box_color = (0, 255, 0)

            self.mark_attendance(name)

            names.append(name)

            cv2.rectangle(img, (left, top), (right, bottom), box_color, 2)
            cv2.rectangle(img, (left, bottom), (right, bottom + 60), box_color, cv2.FILLED)
            cv2.putText(img, name, (left + 6, bottom + 25), cv2.FONT_HERSHEY_DUPLEX, 0.8, (0, 0, 0),1)

            status_text = ""

            if name in self.marked_today:
                time_marked = self.marked_today[name]
                elapsed = (datetime.now() - time_marked).total_seconds()

                if elapsed <= 7:
                    status_text = "Attendance Marked!"

            if status_text:
                cv2.putText(img, status_text,
                            (left + 6, bottom + 50),cv2.FONT_HERSHEY_DUPLEX, 0.5, (0, 0, 0),1) 

        return img, names

if __name__ == "__main__":

    recognizer = FaceRecognizer()
    cap = cv2.VideoCapture(0)

    while True:
        success, img = cap.read()

        if not success:
            break

        img = cv2.flip(img, 1)

        img, names = recognizer.recognize(img)

        cv2.imshow("Face recognition attendance system", img)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
