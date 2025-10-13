import cv2,mediapipe as mp

def main():
    camera = cv2.VideoCapture(0)
    cv2.namedWindow("Hand Drawing", cv2.WINDOW_NORMAL)

    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.7)

    lines = []

    while camera.isOpened():
        success, frame_bgr = camera.read()
        if not success:
            break

        frame_bgr = cv2.flip(frame_bgr, 1)
        h, w, _ = frame_bgr.shape

        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        result = hands.process(frame_rgb)

        hand_type = None
        if result.multi_handedness:
            hand_type = result.multi_handedness[0].classification[0].label

        if result.multi_hand_landmarks and hand_type == "Right":
            landmarks = result.multi_hand_landmarks[0]
            x = int(landmarks.landmark[8].x * w)
            y = int(landmarks.landmark[8].y * h)
            lines.append((x, y))
            cv2.circle(frame_bgr, (x, y), 8, (255, 255, 255), -1)
        else:
            lines.clear()

        for i in range(1, len(lines)):
            cv2.line(frame_bgr, lines[i - 1], lines[i], (255, 255, 255), 5)

        cv2.imshow("Hand Drawing", frame_bgr)

        if cv2.waitKey(1) & 0xFF == 27:
            break

    camera.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
