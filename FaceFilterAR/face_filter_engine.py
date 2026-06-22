import cv2
import mediapipe as mp
import numpy as np
from typing import Optional


def mesh(
    frame: np.ndarray,
    model: mp.solutions.face_mesh.FaceMesh
) -> Optional[list]:
    """
    Detect facial landmarks in a frame.

    Args:
        frame: RGB image frame.
        model: MediaPipe FaceMesh model instance.

    Returns:
        A list of facial landmarks if a face is detected,
        otherwise None.
    """
    results = model.process(frame)

    if results.multi_face_landmarks:
        return results.multi_face_landmarks[0].landmark

    return None


def distance(
    p1: tuple[int, int],
    p2: tuple[int, int]
) -> float:
    """
    Compute the Euclidean distance between two points.

    Args:
        p1: First 2D point (x, y).
        p2: Second 2D point (x, y).

    Returns:
        Distance between the two points.
    """
    return ((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2) ** 0.5


def rotation_angle(
    p1: tuple[int, int],
    p2: tuple[int, int]
) -> float:
    """
    Compute the angle between two points in degrees.

    Args:
        p1: First 2D point (x, y).
        p2: Second 2D point (x, y).

    Returns:
        Rotation angle in degrees.
    """
    return np.rad2deg(
        np.arctan2(
            p1[1] - p2[1],
            p1[0] - p2[0]
        )
    )


def rotate_and_resize(
    image: np.ndarray,
    angle: float,
    scale: float
) -> tuple[np.ndarray, np.ndarray]:
    """
    Rotate and scale an image while preserving all pixels.

    Args:
        image: Input RGBA overlay image.
        angle: Rotation angle in degrees.
        scale: Scaling factor.

    Returns:
        A tuple containing:
            - The transformed image.
            - The affine transformation matrix.
    """
    h, w = image.shape[:2]

    center = (w // 2, h // 2)

    transform_matrix = cv2.getRotationMatrix2D(
        center,
        angle,
        scale
    )

    cos_theta = abs(transform_matrix[0, 0])
    sin_theta = abs(transform_matrix[0, 1])

    new_w = int(h * sin_theta + w * cos_theta)
    new_h = int(w * sin_theta + h * cos_theta)

    transform_matrix[0, 2] += (new_w / 2) - center[0]
    transform_matrix[1, 2] += (new_h / 2) - center[1]

    transformed_image = cv2.warpAffine(
        image,
        transform_matrix,
        (new_w, new_h)
    )

    return transformed_image, transform_matrix


def paste_overlay(
    frame: np.ndarray,
    overlay_type: str,
    overlay_image: np.ndarray,
    mesh_model: mp.solutions.face_mesh.FaceMesh,
    overlay_anchors: tuple[
        tuple[int, int],
        tuple[int, int]
    ]
) -> np.ndarray:
    """
    Paste a transformed overlay onto a detected face.

    The overlay is automatically scaled, rotated and
    positioned according to facial landmarks.

    Args:
        frame: Input BGR video frame.
        overlay_type: Overlay category name.
        overlay_image: RGBA overlay image.
        mesh_model: MediaPipe FaceMesh model.
        overlay_anchors: Overlay reference anchor points.

    Returns:
        Frame with the overlay applied.
    """
    global FACE_ANCHORS

    left_idx, right_idx = FACE_ANCHORS[overlay_type]

    overlay_anchor_left, overlay_anchor_right = overlay_anchors

    frame_h, frame_w, _ = frame.shape

    landmarks = mesh(frame, mesh_model)

    if not landmarks:
        return frame[:,:,::-1]

    overlay_reference_width = distance(
        overlay_anchor_left,
        overlay_anchor_right
    )

    left_face_point = (
        int(landmarks[left_idx].x * frame_w),
        int(landmarks[left_idx].y * frame_h)
    )

    right_face_point = (
        int(landmarks[right_idx].x * frame_w),
        int(landmarks[right_idx].y * frame_h)
    )

    scale_factor = (
        distance(left_face_point, right_face_point)
        / overlay_reference_width
    )

    angle = rotation_angle(
        right_face_point,
        left_face_point
    )

    transformed_overlay, transform_matrix = rotate_and_resize(
        overlay_image,
        -angle,
        scale_factor
    )

    overlay_h, overlay_w = transformed_overlay.shape[:2]

    transformed_anchor_left = np.int64(
        transform_matrix @ np.array([
            [overlay_anchor_left[0]],
            [overlay_anchor_left[1]],
            [1]
        ])
    )

    transformed_anchor_left = (
        transformed_anchor_left[0, 0],
        transformed_anchor_left[1, 0]
    )

    paste_top_left = (
        left_face_point[0] - transformed_anchor_left[0],
        left_face_point[1] - transformed_anchor_left[1]
    )

    paste_bottom_right = (
        paste_top_left[0] + overlay_w,
        paste_top_left[1] + overlay_h
    )

    crop_left = 0
    crop_top = 0
    crop_right = 1
    crop_bottom = 1

    if paste_top_left[0] < 0:
        crop_left = -paste_top_left[0]

    if paste_top_left[1] < 0:
        crop_top = -paste_top_left[1]

    if paste_bottom_right[0] > frame_w:
        crop_right = paste_bottom_right[0] - frame_w

    if paste_bottom_right[1] > frame_h:
        crop_bottom = paste_bottom_right[1] - frame_h

    transformed_overlay = transformed_overlay[
        crop_top:-crop_bottom,
        crop_left:-crop_right
    ]

    paste_top_left = (
        paste_top_left[0] + crop_left,
        paste_top_left[1] + crop_top
    )

    overlay_w -= crop_left + crop_right
    overlay_h -= crop_top + crop_bottom
    frame = frame[:,:,::-1]
    roi = frame[
        paste_top_left[1]:paste_top_left[1] + overlay_h,
        paste_top_left[0]:paste_top_left[0] + overlay_w
    ]

    alpha_mask = (
        transformed_overlay[:, :, 3] == 255
    )[:, :, np.newaxis]

    frame[
        paste_top_left[1]:paste_top_left[1] + overlay_h,
        paste_top_left[0]:paste_top_left[0] + overlay_w
    ] = np.where(
        alpha_mask,
        transformed_overlay[:, :, :3],
        roi
    )

    return frame


FACE_ANCHORS: dict[str, tuple[int, int]] = {
    "glasses": (33, 263),
    "moustache": (61, 291),
    "beard": (172, 397),
    "mask": (234, 454),
    "crown": (103, 332),
    "hat": (103, 332),
    "bunny_ears": (103, 332),
    "devil_horns": (103, 332),
    "halo": (103, 332),
    "pirate_hat": (103, 332),
    "mouth": (61, 291),
    "eye_patch": (33, 133),
    "left_eye": (33, 133),
    "right_eye": (362, 263),
    "nose": (98, 327),
    "clown_nose": (98, 327)
}