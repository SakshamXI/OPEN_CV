import os
import cv2
from face_tracking import face_tracking


def main():
    root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    reference_path = os.path.join(root_dir, "assets", "reference.jpg")

    if not os.path.exists(reference_path):
        raise RuntimeError(
            "Reference image not found. "
            "Place an image at assets/reference.jpg"
        )

    reference = cv2.imread(reference_path)
    if reference is None:
        raise RuntimeError("Failed to load reference image")

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Cannot open camera")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        bbox = face_tracking(reference, frame, sensitivity=45)

        if bbox:
            x, y, w, h = bbox
            cv2.rectangle(
                frame,
                (x, y),
                (x + w, y + h),
                (0, 255, 0),
                2
            )
            cv2.putText(
                frame,
                "Target Detected",
                (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2
            )

        cv2.imshow("Face Tracking Demo", frame)

        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
