import numpy as np
import math
import os
import pandas as pd
from DatasetDownloader import reload_anno_pickle, ANNO_SAV_DIR, IMG_DIR, SMALL_ANNO_DIR
from torch.utils.data import Dataset, DataLoader
import cv2


class MPIIDataset(Dataset):

    def __init__(self, anno_dir=SMALL_ANNO_DIR, img_dir=IMG_DIR):
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
            self.annotations = reload_anno_pickle()

        else:
            print('No directory specified. No data loaded')

    def __len__(self):
        """
        return number of data
        """
        return self.annotations.shape[0]

    def __getitem__(self, index):
        """
        called when iterate on Data Loader
        :param index: index of one sample
        :return:
        """
        return self.one_hot_data[index], self.labels[index]

    def get_anno_dims(self):
        """
        :return: an array of size (num_fields)
        """
        return self.annotations.shape[1]

    def load_img(self, dir: str):
        try:
            return cv2.imread(IMG_DIR + dir)
        except:
            print('Error when loading image')



if __name__ == "__main__":
    ds = MPIIDataset()
    for fn in ds.annotations['filename']:
        img = ds.load_img(fn)
        print(img.shape)
