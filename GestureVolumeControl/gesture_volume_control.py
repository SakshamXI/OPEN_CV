import cv2,mediapipe as mp,pyautogui as pg
camera = cv2.VideoCapture(0)
hands = mp.solutions.hands.Hands(min_detection_confidence=0.5)
prev = None
while camera.isOpened():
    rec, frame = camera.read()
    if not rec: break
    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)
    handedness = result.multi_handedness
    hand_type = handedness[0].classification[0].label if handedness else None
    if result.multi_hand_landmarks and hand_type == 'Right':
        lm = result.multi_hand_landmarks[0].landmark
        index = (int(lm[12].x * w), int(lm[12].y * h))
        thumb = (int(lm[4].x * w), int(lm[4].y * h))
        length = int(((index[0] - thumb[0])**2 + (index[1] - thumb[1])**2) ** 0.5)
        '''cv2.circle(frame,index,5,[0]*3,-1)
        cv2.circle(frame, thumb, 5, [0] * 3, -1)
        cv2.line(frame,index,thumb,[0]*3,3)'''

        if prev is not None:
            delta = (length - prev) // 3
            if abs(delta) > 5:
                print(delta)
                if delta > 0:
                    pg.press("volumeup", presses=delta)
                else:
                    pg.press("volumedown", presses=abs(delta))
        prev = length
    else:
        prev = None
    #cv2.imshow("Image",frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break
camera.release()
cv2.destroyAllWindows()
