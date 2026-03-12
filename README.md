# 🖱️ Gesture Virtual Mouse & Face Recognition Attendance System

A computer vision project built with Python, OpenCV, and MediaPipe that lets you control your mouse with hand gestures — and automatically takes attendance using face recognition before launching.

---

## 📌 Features

- **Phase 1 — Face Recognition Attendance**
  - Detects and recognizes known faces via webcam
  - Automatically logs Name, Date, and Time to `attendance.csv`
  - Each person is marked only once per session
  - Press `q` to move to the gesture mouse

- **Phase 2 — Gesture Virtual Mouse**
  - Move cursor using your index finger (left hand)
  - Click by bringing index + middle finger together
  - Control system volume using right hand pinch gesture
  - FPS counter displayed on screen

---

## 🗂️ Project Structure

```
Virtual_Mouse/
├── main.py                  # Entry point — runs both phases
├── face_recog.py            # Face recognition attendance module
├── volumenbrightness.py     # Volume control module
├── hand_landmarker.task     # MediaPipe hand model file
├── attendance.csv           # Auto-created on first run
└── known_faces/             # Add face photos here
    ├── Name1.jpg
    ├── Name2.jpg
    └── ...
```

---

## ⚙️ Requirements

- **Python 3.11.9** — required (face_recognition does not support Python 3.12+)
- Windows 10/11
- A working webcam

---

## 📦 Installation

Install all dependencies using Python 3.11:

```bash
py -3.11 -m pip install opencv-python
py -3.11 -m pip install face_recognition
py -3.11 -m pip install mediapipe
py -3.11 -m pip install pyautogui
py -3.11 -m pip install numpy
py -3.11 -m pip install pycaw
py -3.11 -m pip install comtypes
```

> ⚠️ `face_recognition` installs `dlib` automatically. If it fails, make sure you are on **Python 3.11** specifically — dlib does not support newer Python versions yet.

---

## 👤 Adding Faces for Attendance

1. Go to the `known_faces/` folder
2. Add a clear photo of each person named after them:
   ```
   Harshit.jpg
   Prithanjan.jpg
   Saumya.jpg
   ```
3. One face per photo, no blurry images
4. `.jpg` and `.png` both work

The system will automatically load all photos from this folder on startup.

---

## ▶️ Running the Project

Make sure your VS Code / editor interpreter is set to **Python 3.11.9**, then run:

```bash
py -3.11 main.py
```

**Flow:**
1. Attendance window opens → show your face → `Attendance Marked!` appears
2. Press `q` → gesture mouse window opens
3. Use hand gestures to control mouse and volume
4. Press `q` again → program closes

---

## 🤌 Gesture Controls

| Gesture | Action |
|---|---|
| Left hand — index finger up | Move cursor |
| Left hand — index + middle together | Click |
| Right hand — thumb + index pinch | Control volume |

---

## 📋 Attendance Log

Attendance is saved automatically to `attendance.csv`:

```
Name,Date,Time
Harshit,2026-03-12,10:32
Prithanjan,2026-03-12,10:33
```

---

## 🛠️ Built With

- [OpenCV](https://opencv.org/)
- [MediaPipe](https://mediapipe.dev/)
- [face_recognition](https://github.com/ageitgey/face_recognition)
- [PyAutoGUI](https://pyautogui.readthedocs.io/)
- [pycaw](https://github.com/AndreMiras/pycaw)

---

## 👥 Team

- **Prithanjan Acharyya** — Gesture Virtual Mouse
- **Harshit Krishna** — Face Recognition Attendance System  
- **Saumya Kejriwal** — Volume Control
