import os
from pathlib import Path
from typing import Sequence

import cv2
import numpy as np

from common_types import CameraIntrinsics, Frame


class KittiSequence:
    def __init__(self, dataset_path: str, sequence_id: str):
        dataset_path = Path(dataset_path)
        self.sequence_path = dataset_path.joinpath(f"sequences/{sequence_id}")
        self.poses_path = dataset_path.joinpath("poses").joinpath(f"{sequence_id}.txt")
        self.left_images_path = self.sequence_path.joinpath("image_0")
        self.right_images_path = self.sequence_path.joinpath("image_1")
        self.calib_file_path = self.sequence_path.joinpath("calib.txt")
        with open(self.calib_file_path) as f:
            P0 = [float(x) for x in f.readline()[4:].split(" ")]
            P1 = [float(x) for x in f.readline()[4:].split(" ")]

            self.camera_intrinsics = self._proj_to_camera_intrinsics(P1)
        self.trajectory = []
        with open(self.poses_path) as f:
            initial_pose_vals = [float(x) for x in f.readline().split(" ")]
            self.initial_pose = np.array(initial_pose_vals).reshape(3, 4)
            self.trajectory.append(self.initial_pose)
            while l := f.readline():
                pose_vals = [float(x) for x in l.split(" ")]
                self.trajectory.append(np.array(pose_vals).reshape(3, 4))
        self.trajectory = np.array(self.trajectory)
        self.image_filenames = sorted(os.listdir(self.left_images_path))

    def _proj_to_camera_intrinsics(self, projection_data: Sequence[float]) -> CameraIntrinsics:
        P = np.array(projection_data).reshape(3, 4)
        t = P[:, 3]
        t = np.linalg.inv(P[:, 0:3]) @ t
        return CameraIntrinsics(fx=P[0, 0], fy=P[1, 1], cx=P[0, 2], cy=P[1, 2], baseline=np.linalg.norm(t))

    def __len__(self) -> int:
        return len(self.image_filenames)

    def __getitem__(self, item: int) -> Frame:
        return Frame(cv2.imread(str(self.left_images_path.joinpath(self.image_filenames[item])), 0),
                     cv2.imread(str(self.right_images_path.joinpath(self.image_filenames[item])), 0))
