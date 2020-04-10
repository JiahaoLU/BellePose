import numpy as np
import math
import os
import pandas as pd
from DatasetDownloader import reload_anno_pickle, ANNO_SAV_DIR, IMG_DIR, SMALL_ANNO_DIR
from torch.utils.data import Dataset, DataLoader
from ImgUtil import *
from os.path import join
import os
from typing import List, Tuple

class MPIIDataset(Dataset):

    def __init__(self, input_size, anno_dir=SMALL_ANNO_DIR, img_dir=IMG_DIR):
        """
        Initialise the preprocessor as a Data set in torch
        or initialise it as a data provider (cut smaller data, etc)
        :param anno_dir: file path
        """
        if anno_dir != '' and img_dir != '':
            self.anno_dir = anno_dir
            print('Annotation set initiated from {0}.'.format(self.anno_dir))
            self.img_dir = img_dir
            print('Image set initiated from {0}.'.format(self.img_dir))
            self.annotations = reload_anno_pickle(anno_dir)
            self.img_files = os.listdir(img_dir)
            self.base_zoom = 1.5  # 'How big is the input image region comapred to bbox of joints'
            self.input_size = input_size
        else:
            print('No directory specified. No data loaded')

    def __len__(self):
        """
        return number of data
        """
        return self.annotations.shape[0]

    def __getitem__(self, index) -> Tuple[torch.Tensor, np.ndarray, int]:
        """
        called when iterate on Data Loader
        :param index: index of one sample
        :return:
        """
        anno = self.annotations.iloc[index]
        joints = self.get_joints_array(anno)
        bbox_w, bbox_h = self.calc_joint_bbox_size(joints)
        cx, cy = self.calc_joint_center(joints)
        image = load_img(anno['filename'])

        image, joints = self.crop_reshape(image, joints, bbox_w, bbox_h, cx, cy)
        image = image.astype(np.uint8)
        image = rgb2gray(image)
        image = low_pass_filter(enhance_contrast(image))
        image = enhance_contrast(gradient(image, -1))
        image = transform_to_tensor(image, self.input_size)
        headsize = self.get_headsize(joints[0:2, :])
        joints = joints[2:, :].astype(np.uint8).reshape(-1)
        return image, joints, int(headsize)

    def get_njoints(self):
        """
        :return: an array of size (num_fields)
        """
        return self.get_joints_array(self.annotations.iloc[0]).shape[0]

    def get_joints_array(self, anno: pd.Series) -> np.ndarray:
        labels = [anno['head_x1'], anno['head_y1'],
                  anno['head_x2'], anno['head_y2']]
        for i in range(16):
            labels.append(anno[str(i)][0])
            labels.append(anno[str(i)][1])
        return np.asarray(labels).reshape(-1, 2)

    def get_headsize(self, head_bias: np.ndarray) -> int:
        return min(abs(head_bias[0, 0] - head_bias[1, 0]),
                   abs(head_bias[0, 1] - head_bias[1, 1]))

    def crop_reshape(self, image: np.ndarray, joints: np.ndarray,
                     bbox_w: int, bbox_h: int, cx: float, cy: float):
        bbox_h, bbox_w = bbox_h * self.base_zoom, bbox_w * self.base_zoom
        y_min = max(cy - bbox_h / 2, 0)
        y_max = min(cy + bbox_h / 2, image.shape[0])
        x_min = max(cx - bbox_w / 2, 0)
        x_max = min(cx + bbox_w / 2, image.shape[1])
        image = image[int(y_min):int(y_max), int(x_min):int(x_max)]
        joints -= np.array([x_min, y_min])
        # in cv2, frac tuple follows the order (w, h)
        fx, fy = self.input_size / image.shape[1], self.input_size / image.shape[0]
        cx, cy = image.shape[1] // 2, image.shape[0] // 2
        image, joints = self.zoom(image, joints, cx, cy, fx, fy)
        return image, joints

    def zoom(self, image, joints, center_x, center_y, fx, fy):
        joint_vecs = joints - np.array([center_x, center_y])
        image = cv2.resize(image, None, fx=fx, fy=fy)
        joint_vecs *= np.array([fx, fy])
        center_x, center_y = center_x * fx, center_y * fy
        joints = joint_vecs + np.array([center_x, center_y])
        return image, joints

    def calc_joint_center(self, joints: np.ndarray) -> List[float]:
        x_center = (np.min(joints[2:, 0]) + np.max(joints[2:, 0])) / 2
        y_center = (np.min(joints[2:, 1]) + np.max(joints[2:, 1])) / 2
        return [x_center, y_center]

    def calc_joint_bbox_size(self, joints: np.ndarray) -> Tuple[int, int]:
        return np.max(joints[:, 0]) - np.min(joints[:, 0]), \
               np.max(joints[:, 1]) - np.min(joints[:, 1])


if __name__ == "__main__":
    ds = MPIIDataset(256)
    IMG, ANNO, h = ds[0]
    print(h)
    ANNO_offset = ANNO + 0.1 * min(calc_bbox_size(ANNO)) / 4 / 1.414
    # print(np.linalg.norm(ANNO - ANNO_offset) / min(calc_bbox_size(ANNO)))

    # print(IMG.shape)
    show_img(IMG, [ANNO, ANNO_offset], True)
