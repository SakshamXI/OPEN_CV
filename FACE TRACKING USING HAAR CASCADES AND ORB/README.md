FACE TRACKING USING HAAR CASCADES AND ORB
=======================================

A lightweight, real-time face tracking system built using classical computer
vision techniques. This project combines Haar Cascades for face detection and
ORB (Oriented FAST and Rotated BRIEF) for feature-based face matching.

The system compares a reference face against detected faces in a video frame
and returns the best-matching face based on feature distance.


-----------------------------------------------------------------------
FEATURES
-----------------------------------------------------------------------
- Real-time face tracking on CPU
- No deep learning or heavy models
- Rotation and scale-invariant feature matching
- Simple and reusable API
- Privacy-safe demo setup (no images committed)


-----------------------------------------------------------------------
REQUIREMENTS
-----------------------------------------------------------------------
- Python 3.8 or higher
- OpenCV
- NumPy


-----------------------------------------------------------------------
INSTALLATION
-----------------------------------------------------------------------
Install dependencies using pip:

```bash
pip install opencv-python numpy
```


-----------------------------------------------------------------------
REFERENCE IMAGE SETUP
-----------------------------------------------------------------------
For privacy and security reasons, no face images are included in this repository.

To run the demo:
1. Add a clear frontal face image
2. Save it at the following path:

assets/reference.jpg

This image is used as the reference identity for tracking.


-----------------------------------------------------------------------
USAGE
-----------------------------------------------------------------------

Webcam Demo
-----------
Run the demo script to start real-time face tracking using your webcam:

python examples/demo.py

Press ESC to exit the application.

If the reference face is detected in the video stream, a bounding box will be
drawn around the best-matching face.


Programmatic Usage
------------------
The face tracker can be integrated directly into your own application:

import cv2
from face_tracking import face_tracking

reference = cv2.imread("assets/reference.jpg")
frame = cv2.imread("frame.jpg")

bbox = face_tracking(reference, frame, sensitivity=45)

if bbox:
    x, y, w, h = bbox
    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)


-----------------------------------------------------------------------
SENSITIVITY PARAMETER
-----------------------------------------------------------------------
The sensitivity parameter controls how strict the feature matching is.

Lower values  -> Stricter matching (fewer false positives)
Higher values -> More tolerant matching (may allow false matches)

Recommended range:
35 to 60

This value should be tuned based on:
- Lighting conditions
- Camera quality
- Distance from camera
- Face orientation


-----------------------------------------------------------------------
LIMITATIONS
-----------------------------------------------------------------------
- Sensitive to extreme lighting variations
- Performs best with frontal or near-frontal faces
- Haar Cascades may produce false positives
- Not suitable for secure biometric identification
- Accuracy degrades with motion blur and low resolution


-----------------------------------------------------------------------
NOTES
-----------------------------------------------------------------------
- This project performs face tracking and matching, not face recognition
- Uses classical computer vision techniques only
- No deep learning or neural networks are involved
- Designed for CPU-based systems
- Intended for educational and lightweight applications


-----------------------------------------------------------------------
LICENSE
-----------------------------------------------------------------------
MIT License

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files, to deal in the Software
without restriction, including without limitation the rights to use, copy,
modify, merge, publish, distribute, sublicense, and/or sell copies of the
Software, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND.


-----------------------------------------------------------------------
ACKNOWLEDGEMENTS
-----------------------------------------------------------------------
- OpenCV Computer Vision Library
- ORB (Oriented FAST and Rotated BRIEF) by Rublee et al.
- Haar Cascade Face Detector by Viola and Jones
