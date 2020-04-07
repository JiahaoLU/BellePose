# encoding: utf-8
"""
@author: Jiahao LU
@contact: lujiahao8146@gmail.com
@file: DatasetDownloader.py
@time: 2020/4/5
@desc: Data preprocess
"""

from scipy.io import loadmat
import numpy as np
import pickle
import re
import pandas as pd
from tqdm import tqdm
import os

ANNO_DIR = './mpii_dataset/mpii_annotations/mpii_human_pose_v1_u12_1.mat'
IMG_DIR = './mpii_dataset/mpii_images/'
ANNO_SAV_DIR = './mpii_dataset/mpii_annotations/mpii.annotations.pkl'
SMALL_ANNO_DIR = './mpii_dataset/mpii_annotations/mpii_annotations_s.pkl'


def picklize_annotations():
    mat = loadmat(ANNO_DIR)['RELEASE']  # numpy.ndarray
    df = pd.DataFrame(columns=['filename', 'train', 'is_visible',
                               'head_x1', 'head_y1', 'head_x2', 'head_y2',
                               '0', '1', '2', '3', '4', '5', '6', '7',
                               '8', '9', '10', '11', '12', '13', '14', '15'])

    with open(ANNO_SAV_DIR, 'wb') as pklfile:
        tmp = {}
        for anno, train_flag in tqdm(iterable=zip(mat['annolist'][0, 0][0],
                                                  mat['img_train'][0, 0][0]),
                                     total=len(mat['img_train'][0, 0][0])):
            tmp['filename'] = anno['image']['name'][0, 0][0]
            tmp['train'] = int(train_flag)

            if 'annopoints' in str(anno['annorect'].dtype):
                annopoints = anno['annorect']['annopoints'][0]
                if re.match('.*x1.*y1.*x2.*y2.*', str(anno['annorect'].dtype)) is None:
                    continue
                head_x1s = anno['annorect']['x1'][0]
                head_y1s = anno['annorect']['y1'][0]
                head_x2s = anno['annorect']['x2'][0]
                head_y2s = anno['annorect']['y2'][0]
                for annopoint, head_x1, head_y1, head_x2, head_y2 in zip(
                        annopoints, head_x1s, head_y1s, head_x2s, head_y2s):
                    if len(annopoint) != 0:
                        tmp['head_x1'] = float(head_x1[0, 0])
                        tmp['head_y1'] = float(head_y1[0, 0])
                        tmp['head_x2'] = float(head_x2[0, 0])
                        tmp['head_y2'] = float(head_y2[0, 0])

                        # joint coordinates
                        annopoint = annopoint['point'][0, 0]
                        j_id = [str(j_i[0, 0]) for j_i in annopoint['id'][0]]
                        if len(j_id) != 16:
                            continue
                        x = [x[0, 0] for x in annopoint['x'][0]]
                        y = [y[0, 0] for y in annopoint['y'][0]]
                        for _j_id, (_x, _y) in zip(j_id, zip(x, y)):
                            tmp[str(_j_id)] = [float(_x), float(_y)]

                        # visiblity list
                        if 'is_visible' in str(annopoint.dtype):
                            vis = [v[0] if v else [0]
                                   for v in annopoint['is_visible'][0]]
                            vis = dict([(k, int(v[0])) if len(v) > 0 else v
                                        for k, v in zip(j_id, vis)])
                        else:
                            vis = dict()
                        tmp['is_visible'] = vis
                        df.loc[-1] = tmp
                        df.index = df.index + 1
                        df = df.sort_index()
                        # for k, v in tmp.items():
                        #     print(v == df[k][0])
        print('pickle file stored in %s' % ANNO_SAV_DIR)
        print(df.head(3))
        print('...')
        pickle.dump(df, pklfile)


def reload_anno_pickle() -> pd.DataFrame:
    try:
        with open(ANNO_SAV_DIR, 'rb') as pklf:
            annos = pickle.load(pklf)
            print('pickle file reload from %s' % ANNO_SAV_DIR)
            print(annos.head(5))
            print('...')
            return annos
    except:
        print('pickle file not found')


def cut_smaller_data_for_exam(large: pd.DataFrame, new_path: str, frac : float = 0.1):
    """
    Cut a smaller data set for debugging code of size
    :param frac:
    :param new_path: name and path for new csv file
    :param size: the size of the smaller data set
    :return: none
    """
    print("Producing smaller data")
    smalldf = large.sample(frac=frac)

    if os.path.exists(new_path):
        os.remove(new_path)
    with open(new_path, 'wb') as pklf:
        pickle.dump(smalldf, pklf)
    print("New smaller file created, length = %d" % smalldf.shape[0])


if __name__ == '__main__':
    picklize_annotations()
    # pf = reload_anno_pickle()
    # cut_smaller_data_for_exam(pf, SMALL_ANNO_DIR)
    # print(re.match('.x1.y1.x2.y2.', "[('scale', 'O'), ('objpos', 'O')]" ))
    # print(re.match('.*x1.*y1.*x2.*y2.*', "[('x1', 'O'), ('y1', 'O'), ('x2', 'O'), ('y2', 'O'), ('annopoints', 'O'), ('scale', 'O'), ('objpos', 'O')]") is None)
