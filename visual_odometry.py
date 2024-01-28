import sys
from typing import Protocol, Sequence

import cv2

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from dataset import KittiSequence
from common_types import CameraIntrinsics, Frame


class KeyPointDetector(Protocol):
    def detect(self, img: cv2.typing.MatLike) -> Sequence[cv2.KeyPoint]:
        ...


class VisualOdometry:
    def __init__(self, camera_intrinsics: CameraIntrinsics,
                 keypoint_detector: KeyPointDetector = cv2.FastFeatureDetector.create(threshold=20),
                 min_features: int = 1000, fb_optical_flow_threshold: float = 5.0, max_y_disparity: float = 1.0):
        self.camera_intrinsics = camera_intrinsics
        self.keypoint_detector = keypoint_detector
        self.min_features = min_features
        self.fb_optical_flow_threshold = fb_optical_flow_threshold
        self.max_y_disparity = max_y_disparity

        self.previous_frame = None
        self.tracked_kp = None
        self.intrinsics_matrix = np.array([[self.camera_intrinsics.fx, 0, self.camera_intrinsics.cx],
                                           [0, self.camera_intrinsics.fy, self.camera_intrinsics.cy],
                                           [0, 0, 1]])

    def _fb_optical_flow(self, img_1: cv2.typing.MatLike, img_2: cv2.typing.MatLike, kp_1: cv2.typing.MatLike):
        # Forward/backward optical flow
        kp_2, status, _ = cv2.calcOpticalFlowPyrLK(img_1, img_2, kp_1,
                                                   None,
                                                   maxLevel=5,
                                                   winSize=(9, 9))
        kp_1b, status_b, _ = cv2.calcOpticalFlowPyrLK(img_2, img_1, kp_2,
                                                      None,
                                                      maxLevel=5,
                                                      winSize=(9, 9))
        # Discard keypoints for which backward optical flow doesn't end back on the original keypoint
        mask = np.sqrt(np.sum((kp_1 - kp_1b) ** 2, axis=1)) < self.fb_optical_flow_threshold
        return kp_2, (status.ravel() == 1) & (status_b.ravel() == 1) & mask

    def step(self, frame: Frame) -> cv2.typing.Matx44f:
        if self.previous_frame is None:
            self.previous_frame = frame
            return np.eye(4)

        if self.tracked_kp is None or len(self.tracked_kp) < self.min_features:
            # If the number of tracked keypoints is below min_featurs then detect new keypoints
            kp_left_1 = cv2.KeyPoint.convert(self.keypoint_detector.detect(self.previous_frame.left_image))
        else:
            # Else reuse tracked keypoints
            kp_left_1 = self.tracked_kp

        # Use optical flow to track features from previous to next frame
        kp_left_2, mask = self._fb_optical_flow(self.previous_frame.left_image, frame.left_image, kp_left_1)

        kp_left_1 = kp_left_1[mask]
        kp_left_2 = kp_left_2[mask]

        # Use optical flow to find matching features in next frame right image
        kp_right_2, mask = self._fb_optical_flow(frame.left_image, frame.right_image, kp_left_2)
        kp_left_1 = kp_left_1[mask]
        kp_left_2 = kp_left_2[mask]
        kp_right_2 = kp_right_2[mask]

        x_disp = kp_left_2[:, 0] - kp_right_2[:, 0]
        y_disp = kp_left_2[:, 1] - kp_right_2[:, 1]
        # Discard matched points with negative x disparity
        mask = x_disp > 0.0
        # Discard matched points if they are not on the same epipolar line
        mask &= np.abs(y_disp) < self.max_y_disparity

        # Compute 3D point cloud
        depth = self.camera_intrinsics.fx * self.camera_intrinsics.baseline / x_disp[mask]
        points_3d = np.float32([
            (kp_left_2[mask, 0] - self.camera_intrinsics.cx) * depth / self.camera_intrinsics.fx,
            (kp_left_2[mask, 1] - self.camera_intrinsics.cy) * depth / self.camera_intrinsics.fy,
            depth
        ]).T

        points_2d = kp_left_1[mask].astype(np.float32)

        # Estimate pose using PnP, points_3d contains 3D point cloud in w.r.t to the left camera at time t,
        # points_2d contains the projected points as seen in the left image at time t-1
        _, r_vec, t_vec, _ = cv2.solvePnPRansac(points_3d,
                                                points_2d,
                                                self.intrinsics_matrix,
                                                None,
                                                iterationsCount=100,
                                                reprojectionError=8.0)
        self.tracked_kp = kp_left_2
        r_mat, _ = cv2.Rodrigues(r_vec)

        self.previous_frame = frame
        return np.block([[r_mat, t_vec], [np.zeros((1, 3)), 1.0]])


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: visual_odometry.py dataset_path sequence_id")
        exit(-1)

    sequence = KittiSequence(sys.argv[1], sys.argv[2])
    vo = VisualOdometry(sequence.camera_intrinsics)

    X = []
    Z = []
    relative_pose = np.concatenate([sequence.initial_pose,
                                    np.concatenate([np.zeros((1, 3)), np.ones((1, 1))], axis=1)], axis=0)

    for i, frame in tqdm(enumerate(sequence), total=len(sequence)):
        frame_pose = vo.step(frame)
        relative_pose @= frame_pose
        X.append(relative_pose[0, 3])
        Z.append(relative_pose[2, 3])

    X_gt = sequence.trajectory[:, 0, 3]
    Z_gt = sequence.trajectory[:, 2, 3]
    plt.plot(X, Z, label="traj")
    plt.plot(X_gt, Z_gt, label="gt")
    plt.legend()
    plt.show()
