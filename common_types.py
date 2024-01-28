from dataclasses import dataclass

import cv2


@dataclass
class CameraIntrinsics:
    fx: float
    fy: float
    cx: float
    cy: float
    baseline: float


@dataclass
class Frame:
    left_image: cv2.typing.MatLike
    right_image: cv2.typing.MatLike
