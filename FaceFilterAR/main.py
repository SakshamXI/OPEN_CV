import cv2
import numpy as np
import mediapipe as mp

import face_filter_engine as ffe


# Available Filters
# --------------------------------------------------
# glasses
#   anchors -> ((170, 247), (340, 246))
#
# moustache
#   anchors -> ((160, 210), (435, 210))
#
# sorting_hat
#   anchors -> ((45, 230), (143, 230))
#
# devil_horns
#   anchors -> ((232, 470), (650, 470))
#
# eye_patch
#   anchors -> ((194, 282), (413, 219))
# --------------------------------------------------


mp_face_mesh = mp.solutions.face_mesh

face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True
)

overlay_image = cv2.imread(
    "assets/glasses.png",
    -1
)

camera = cv2.VideoCapture(0)

while camera.isOpened():

    ret, frame = camera.read()

    if not ret:
        break

    original_frame = frame.copy()

    rgb_frame = frame[:, :, ::-1]

    filtered_frame = ffe.paste_overlay(
        rgb_frame,
        "glasses",
        overlay_image,
        face_mesh,
        ((170, 247), (340, 246))
    )

    comparison_frame = np.hstack(
        (original_frame, filtered_frame)
    )

    comparison_frame = cv2.resize(
        comparison_frame,
        (720, 405)
    )

    cv2.imshow(
        "Face Filter Engine",
        comparison_frame
    )

    if cv2.waitKey(1) & 0xFF == 27:
        break


face_mesh.close()
camera.release()
cv2.destroyAllWindows()