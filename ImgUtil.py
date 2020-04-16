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


def enhance_contrast(img: np.ndarray, method='equalize') -> np.ndarray:
    if method == 'log':
        return exposure.adjust_log(img)
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
            yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
            yuv[:, :, 0] = cv2.equalizeHist(yuv[:, :, 0])
            return cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR)
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
        L = cv2.Laplacian(img, cv2.CV_8U, ksize=kernel)
        return cv2.threshold(L, 127, 255, 0)[1]
    elif axis >= 0:  # 0: x, 1: y
        return cv2.Sobel(img, cv2.CV_8U, 1 - axis, axis, ksize=kernel)


def laplace_contour(img, lb=64, do_box=True, btype='bbox'):
    imgray = rgb2gray(img)
    thresh = gradient(imgray, -1)
    # _, thresh = cv2.threshold(imgray, 127,255,0)
    contours, hier = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if do_box:
        cnt_draw = draw_bbox(img, contours, btype)
    else:
        cnt_draw = draw_approx_hull_polygon(img, contours, btype)
    return cnt_draw


def canny_contour(img, lb=64, do_box=True, btype='bbox'):
    imgray = rgb2gray(img)
    thresh = cv2.Canny(imgray, lb, 256)
    # _, thresh = cv2.threshold(imgray, 127,255,0)
    contours, hier = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if do_box:
        cnt_draw = draw_bbox(img, contours, btype)
    else:
        try:
            cnt_draw = draw_approx_hull_polygon(img, contours, btype=btype)
        except:
            cnt_draw = draw_approx_hull_polygon(img, contours)
    # return thresh
    return cnt_draw


def draw_approx_hull_polygon(img, cnts, btype='approx'):
    # img = np.copy(img)
    drawing = np.copy(img)
    if btype == 'contour':
        cv2.drawContours(drawing, cnts, -1, (255, 0, 0), 3)  # blue
    elif btype == 'approx':
        epsilion = drawing.shape[0] / 32
        approxes = [cv2.approxPolyDP(cnt, epsilion, True) for cnt in cnts]
        cv2.polylines(drawing, approxes, True, (0, 255, 0), 2)  # green
    elif btype == 'hull':
        hulls = [cv2.convexHull(cnt) for cnt in cnts]
        cv2.polylines(drawing, hulls, True, (0, 0, 255), 2)  # red
    else:
        raise ValueError('Box type not existed')
    return cv2.cvtColor(drawing, cv2.COLOR_BGR2RGB)


def draw_bbox(image, cnts, btype='bbox', min_frac: float = 0.1):  # conts = contours
    min_size = float(min(image.shape[0], image.shape[1]) * min_frac)
    box = np.copy(image)

    if btype == 'bbox':
        for cnt in cnts:
            x, y, w, h = cv2.boundingRect(cnt)
            if max(w, h) > min_size:
                cv2.rectangle(box, (x, y), (x + w, y + h), (255, 0, 0), 2)  # blue
    elif btype == 'minbox':
        for cnt in cnts:
            min_rect = cv2.minAreaRect(cnt)  # min_area_rectangle
            min_rect = np.int0(cv2.boxPoints(min_rect))
            cv2.drawContours(box, [min_rect], 0, (0, 255, 0), 2)  # green
    elif btype == 'circle':
        for cnt in cnts:
            (x, y), radius = cv2.minEnclosingCircle(cnt)
            center, radius = (int(x), int(y)), int(radius)  # for the minimum enclosing circle
            box = cv2.circle(box, center, radius, (0, 0, 255), 2)  # red
    else:
        raise ValueError('Box type not existed')
    return cv2.cvtColor(box, cv2.COLOR_BGR2RGB)


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
        # transforms.Normalize(mean=[0.5], std=[0.5])
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return process(pilimg)


if __name__ == '__main__':
    img_color = load_img('000337885.jpg')
    pic = rgb2gray(img_color)
    # pic =low_pass_filter(enhance_contrast(pic))
    # img2 = low_pass_filter(enhance_contrast(pic), kernel=3)

    laplacian = enhance_contrast(gradient(pic, -1))
    # sobelx = enhance_contrast(gradient(pic, 0))
    canbox = canny_contour(img_color, 50)
    cancon = canny_contour(img_color, 50, do_box=False)
    plt.subplot(2, 2, 1), plt.imshow(canny_contour(img_color,50,do_box=False, btype='contour'), cmap='gray',  )
    plt.title('Original'), plt.xticks([]), plt.yticks([])
    plt.subplot(2, 2, 2), plt.imshow(canbox, cmap='gray')
    plt.title('Box'), plt.xticks([]), plt.yticks([])
    plt.subplot(2, 2, 3), plt.imshow(laplace_contour(img_color), cmap='gray')
    plt.title('Box by Laplacian'), plt.xticks([]), plt.yticks([])
    plt.subplot(2, 2, 4), plt.imshow(cancon, cmap='gray')
    plt.title('Hull'), plt.xticks([]), plt.yticks([])

    # plt.subplot(1, 2, 1), plt.imshow(laplacian, cmap='gray')
    # plt.title('Sans filtre'), plt.xticks([]), plt.yticks([])
    # plt.subplot(1, 2, 2), plt.imshow(low_pass_filter(laplacian), cmap='gray')
    # plt.title('filtre gaussien'), plt.xticks([]), plt.yticks([])

    # logg = enhance_contrast(pic, method='log')
    # equal = enhance_contrast(pic, method='equalize')
    # cla = enhance_contrast(pic, method='clahe')

    # plt.subplot(2, 2, 1), plt.imshow(pic, cmap='gray')
    # plt.title('Original'), plt.xticks([]), plt.yticks([])
    # plt.subplot(2, 2, 2), plt.imshow(logg, cmap='gray')
    # plt.title('Log'), plt.xticks([]), plt.yticks([])
    # plt.subplot(2, 2, 3), plt.imshow(equal, cmap='gray')
    # plt.title('histogram equalization'), plt.xticks([]), plt.yticks([])
    # plt.subplot(2, 2, 4), plt.imshow(cla, cmap='gray')
    # plt.title('CLAHE'), plt.xticks([]), plt.yticks([])

    plt.show()