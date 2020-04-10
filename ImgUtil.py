# encoding: utf-8
"""
@author: Jiahao LU
@contact: lujiahao8146@gmail.com
@file: ImgUtil.py
@time: 2020/4/7
@desc: Util for editing images
"""
import matplotlib.pyplot as plt
from matplotlib import cm
# from skimage import data as skidata
from skimage import exposure, img_as_float
import numpy as np
import cv2
from DatasetDownloader import IMG_DIR
from torchvision import transforms
import torch
from PIL import Image
from typing import Tuple, List, TypeVar

COLOR_MAP = {0: 'r', 1: 'g', 2: 'b', 3: 'c', 4: 'm', 5: 'y', 6: 'k', 7: 'w'}
T = TypeVar('T')

def rgb2gray(rgb) -> np.ndarray:
    gray = np.dot(rgb[..., :3], [0.299, 0.587, 0.144])
    return np.uint8(gray / np.max(gray) * 255)
    # try:
    #     return cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY)
    # except cv2.error:
    #     print('image loading error')


def is_colorful(img: T) -> bool:
    if type(img) == torch.Tensor:
        return img.shape[0] == 3
    return len(img.shape) == 3


def is_gray(img: T) -> bool:
    if type(img) == torch.Tensor:
        return img.shape[0] == 1
    return len(img.shape) == 2


def load_img(dir: str) -> np.ndarray:
    try:
        return cv2.imread(IMG_DIR + dir).astype(np.uint8)
    except:
        print('Error when loading image')


def show_img(img: T, joints: List[np.ndarray] = [], include_joints: bool = False):
    plt.axis('off')
    if is_colorful(img):
        if type(img) == torch.Tensor:
            img = img.permute(1, 2, 0)

            plt.imshow(img)
        else:
            plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    elif is_gray(img):
        if type(img) == torch.Tensor:
            img = img.squeeze(0)
            img = (img / img.max() * 255).long()
        plt.imshow(img, cmap=plt.get_cmap('gray'))
    else:
        raise ValueError('Image of wrong dimension.')
    if include_joints:
        if not joints:
            raise ValueError('List of joints cannot be empty')
        size_limit = min(img.shape[0], img.shape[1])
        for i, j in enumerate(joints):
            if j.size % 2 != 0:
                plt.show()
                raise ValueError('Joint array of wrong size. Cannot show img')
            j = j.astype(np.uint8).reshape(-1, 2)
            plt.scatter(np.clip(j[:, 0], 0, size_limit), np.clip(j[:, 1], 0, size_limit),
                        c=COLOR_MAP[i % 8], label='joint ' + str(i))
    plt.legend()
    plt.show()


def enhance_contrast(img: np.ndarray, method='clahe') -> np.ndarray:
    if method == 'log':
        return exposure.adjust_log(img, gain=2)
    elif method == 'clahe':
        if is_colorful(img):
            l, a, b = cv2.split(cv2.cvtColor(img, cv2.COLOR_BGR2LAB))
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
            limg = cv2.merge((clahe.apply(l), a, b))
            return cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
        elif is_gray(img):
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            return clahe.apply(img)
        else:
            raise ValueError('Input is not an image.')
    elif method == 'equalize':
        if is_colorful(img):
            return cv2.equalizeHist(img)
        elif is_gray(img):
            return cv2.equalizeHist(img)
        else:
            raise ValueError('Input is not an image.')


def is_low_contrast(img: np.ndarray):
    return exposure.is_low_contrast(img)


def gradient(img: np.ndarray, axis: int = -1, kernel: int = 5) -> np.ndarray:
    if axis not in [0, -1, 1]:
        raise ValueError("need right code for axis.")
    if axis == -1:
        return cv2.Laplacian(img, cv2.CV_8U, ksize=kernel)
    elif axis >= 0:  # 0: x, 1: y
        return cv2.Sobel(img, cv2.CV_8U, 1 - axis, axis, ksize=kernel)


def low_pass_filter(img: np.ndarray, kernel: int = 5) -> np.ndarray:
    return cv2.GaussianBlur(img, (kernel, kernel), cv2.CV_8U)


def calc_bbox_size(joints: np.ndarray) -> Tuple[int, int]:
    if len(joints.shape) == 2:
        return np.max(joints[:, 0]) - np.min(joints[:, 0]), \
               np.max(joints[:, 1]) - np.min(joints[:, 1])
    elif len(joints.shape) == 1:
        if joints.size % 2 != 0:
            raise ValueError('Joints array has wrong size. Cannot get bbox size')
        return np.max(joints[0::2]) - np.min(joints[0::2]), \
               np.max(joints[1::2]) - np.min(joints[1::2])


def transform_to_tensor(mat: np.ndarray, input_size) -> torch.Tensor:
    pilimg = Image.fromarray(np.uint8(mat / mat.max() * 255))
    process = transforms.Compose([
        transforms.Resize(input_size),
        # transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
        # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return process(pilimg)


if __name__ == '__main__':
    img = rgb2gray(load_img('000156511.jpg'))
    img =low_pass_filter(enhance_contrast(img))
    img2 = low_pass_filter(enhance_contrast(img), kernel=3)
    laplacian = enhance_contrast(gradient(img, -1))
    laplacian2 = enhance_contrast(gradient(img2, -1))
    # sobelx = enhance_contrast(gradient(img, 0))
    # sobely = enhance_contrast(gradient(img, 1))

    plt.subplot(2, 2, 1), plt.imshow(img, cmap='gray')
    plt.title('Original'), plt.xticks([]), plt.yticks([])
    plt.subplot(2, 2, 2), plt.imshow(laplacian, cmap='gray')
    plt.title('Laplacian k = 5'), plt.xticks([]), plt.yticks([])
    plt.subplot(2, 2, 3), plt.imshow(laplacian2, cmap='gray')
    plt.title('Laplacian k = 3'), plt.xticks([]), plt.yticks([])
    plt.subplot(2, 2, 4), plt.imshow(transform_to_tensor(laplacian), cmap='gray')
    plt.title('resize'), plt.xticks([]), plt.yticks([])

    # plt.subplot(2, 2, 3), plt.imshow(sobelx, cmap='gray')
    # plt.title('Sobel X'), plt.xticks([]), plt.yticks([])
    # plt.subplot(2, 2, 4), plt.imshow(sobely, cmap='gray')
    # plt.title('Sobel Y'), plt.xticks([]), plt.yticks([])


    plt.show()