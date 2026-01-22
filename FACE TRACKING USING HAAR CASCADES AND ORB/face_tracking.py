import cv2
import numpy as np
from typing import Optional, Tuple


def face_tracking(
    reference: np.ndarray,
    frame: np.ndarray,
    sensitivity: int
) -> Optional[Tuple[int, int, int, int]]:

    cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )
    if cascade.empty():
        raise RuntimeError("Failed to load Haar cascade")

    ref_gray = cv2.cvtColor(reference, cv2.COLOR_BGR2GRAY)
    ref_faces = cascade.detectMultiScale(ref_gray, 1.1, 5, minSize=(30, 30))
    if len(ref_faces) == 0:
        return None

    x, y, w, h = ref_faces[0]
    ref_face = cv2.resize(ref_gray[y:y + h, x:x + w], (200, 200))

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = cascade.detectMultiScale(gray, 1.1, 5, minSize=(30, 30))
    if len(faces) == 0:
        return None

    orb = cv2.ORB_create(nfeatures=1000)
    _, ref_des = orb.detectAndCompute(ref_face, None)
    if ref_des is None:
        return None

    matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
    candidates = []

    for fx, fy, fw, fh in faces:
        roi = gray[fy:fy + fh, fx:fx + fw]
        _, des = orb.detectAndCompute(roi, None)
        if des is None:
            continue

        matches = matcher.match(des, ref_des)
        if not matches:
            continue

        matches.sort(key=lambda m: m.distance)
        avg_distance = np.mean([m.distance for m in matches[:20]])

        if avg_distance < sensitivity:
            candidates.append((avg_distance, (fx, fy, fw, fh)))

    if not candidates:
        return None

    candidates.sort(key=lambda x: x[0])
    return candidates[0][1]
