# Hand Tracking Drawing using OpenCV & MediaPipe

This project demonstrates **real-time hand tracking and virtual drawing** using the **right index finger** detected by a webcam.  
It leverages **OpenCV** for image processing and **MediaPipe** for hand landmark detection.

---

## 📸 Overview

When you run the script, it:
- Captures live video from your webcam.
- Detects your **right hand** using MediaPipe.
- Tracks the **index finger tip (landmark 8)**.
- Draws a continuous line following the movement of your fingertip in real time.

If your right hand moves out of the frame or is not detected, the drawing clears automatically.

---

## 🧠 Tech Stack

- **Python 3.9+**
- **OpenCV** — For image capture and drawing.
- **MediaPipe** — For hand landmark detection.

---

## ⚙️ Installation

1. **Clone this repository**  
   ```bash
   git clone https://github.com/SakshamXI/OPEN_CV.git
   cd OPEN_CV/opencv
   ```

2. **(Optional) Create a virtual environment**  
   ```bash
   python -m venv venv
   # Activate it:
   # On Windows:
   venv\Scripts\activate
   # On Mac/Linux:
   source venv/bin/activate
   ```

3. **Install required Python packages**  
   ```bash
   pip install opencv-python mediapipe
   ```

4. **Run the script**  
   ```bash
   python hand_draw_tracking.py
   ```

---

## ▶️ Usage

- Move your **right index finger** in front of the webcam to draw.  
- Press **ESC** to exit the application.  

---

## 🧩 How It Works

1. Captures frames from the webcam.  
2. Flips them horizontally for a mirror-like experience.  
3. Processes each frame using `mediapipe.solutions.hands`.  
4. Detects landmarks and checks if the detected hand is **Right**.  
5. Draws a small white circle at the index fingertip and connects previous positions to form a line.

---

## 🧑‍💻 Code Structure

```
opencv/
│
├── hand_draw_tracking.py   # Main script
├── requirements.txt        # Dependencies
└── README.md               # Documentation
```

---

## 🏁 Example Output

When you move your right index finger in the air, the script draws a white line tracing your movements in the webcam feed.

---

## 📜 Author

**Saksham Mahajan**  
📂 Folder: `opencv`  
📘 Repo: [OPEN_CV](https://github.com/SakshamXI/OPEN_CV)

---

## 🪪 License

This project is open-source and available under the **MIT License**.
