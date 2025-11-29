import os
import random

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
import json
import ast
import matplotlib.pyplot as plt

DATASET_NAMES = [
    'CLASSIC',
    'SEM'
]


def dataset_info(dataset_name, is_linux=False):
    if is_linux:

        config = {

            'CLASSIC': {
                'img_height': 512,
                'img_width': 512,
                'test_list': None,
                'train_list': None,
                'data_dir': 'data',  # mean_rgb
                'yita': 0.5
            },
            'SEM': {'img_height': 512,
                    'img_width': 512,
                    'test_list': 'test_default.lst',
                    'train_list': 'train_fullsample_larger.lst',
                    'data_dir': '/mnt/ContourDetectionProject/04_TrainingdataPool/dataset-lists/results_full_sample_larger/',
                    'yita': 0.5}
        }
    else:
        config = {
            'CLASSIC': {'img_height': 512,
                        'img_width': 512,
                        'test_list': None,
                        'train_list': None,
                        'data_dir': 'data2',  # mean_rgb
                        'yita': 0.5},

            'SEM': {'img_height': 512,
                    'img_width': 512,
                    'test_list': 'test_sem.lst',
                    'train_list': 'train_list.lst',
                    'data_dir': 'C:/Users/imeri/Florian Imeri Projects/LDC/dataset-lists/NULLSEM1',
                    'yita': 0.5}
        }
    return config[dataset_name]


class TestDataset(Dataset):
    def __init__(self,
                 data_root,
                 test_data,
                 mean_bgr,
                 img_height,
                 img_width,
                 test_list=None,
                 crop_img=False,
                 arg=None
                 ):
        if test_data not in DATASET_NAMES:
            raise ValueError(f"Unsupported dataset: {test_data}")

        self.data_root = data_root
        self.test_data = test_data
        self.test_list = test_list
        self.args=arg
        # self.arg = arg
        # self.mean_bgr = arg.mean_pixel_values[0:3] if len(arg.mean_pixel_values) == 4 \
        #     else arg.mean_pixel_values
        self.mean_bgr = mean_bgr
        self.img_height = img_height
        self.img_width = img_width
        self.crop_img = crop_img
        self.data_index = self._build_index()

    def _build_index(self):
        sample_indices = []
        if self.test_data == "CLASSIC":
            # for single image testing
            images_path = os.listdir(self.data_root)
            labels_path = None
            sample_indices = [images_path, labels_path]
        else:
            # image and label paths are located in a list file

            if not self.test_list:
                raise ValueError(
                    f"Test list not provided for dataset: {self.test_data}")

            list_name = os.path.join(self.data_root, self.test_list)
            if True: #self.test_data.upper()=='SEM':

                with open(list_name) as f:
                    content = f.read()
                    files = ast.literal_eval(content)
                for pair in files:
                    tmp_img = pair[0]
                    tmp_gt = pair[1]
                    sample_indices.append(
                        (os.path.join(self.data_root, tmp_img),
                         os.path.join(self.data_root, tmp_gt),))

        return sample_indices

    def __len__(self):
        return len(self.data_index[0]) if self.test_data.upper()=='CLASSIC' else len(self.data_index)

    def __getitem__(self, idx):
        # get data sample
        # image_path, label_path = self.data_index[idx]
        if self.data_index[1] is None:
            image_path = self.data_index[0][idx]
        else:
            image_path = self.data_index[idx][0]
        label_path = None if self.test_data == "CLASSIC" else self.data_index[idx][1]
        img_name = os.path.basename(image_path)
        file_name = os.path.splitext(img_name)[0]

        # base dir
        if self.test_data.upper() == 'CLASSIC':
            img_dir = self.data_root
            gt_dir = None
        else:
            img_dir = self.data_root
            gt_dir = self.data_root

        # load data
        image = cv2.imread(os.path.join(img_dir, image_path), cv2.IMREAD_COLOR)

        label = cv2.imread(os.path.join(
                gt_dir, label_path), cv2.IMREAD_GRAYSCALE)


        image, label = self.transform(img=image, gt=label)
        im_shape = [image.shape[1], image.shape[2]]

        return image, label, file_name

        #return dict(images=image, labels=label, file_names=file_name, image_shape=im_shape)

    def transform(self, img, gt):
        # gt[gt< 51] = 0 # test without gt discrimination
        i_h, i_w, _  = img.shape

        crop_size = self.img_height if self.img_height == self.img_width else None
        img = np.array(img, dtype=np.float32)
        img -= self.mean_bgr
        #img /= 255.
        # Centered crop instead of resize
        if i_w > crop_size and i_h > crop_size:

            start_h = (i_h - crop_size) // 2
            start_w = (i_w - crop_size) // 2

            img = img[start_h:start_h + crop_size, start_w:start_w + crop_size]
            gt = gt[start_h:start_h + crop_size, start_w:start_w + crop_size]


        else:
            img = cv2.resize(img, dsize=(crop_size, crop_size))
            gt = cv2.resize(gt, dsize=(crop_size, crop_size))

        img = img.transpose((2, 0, 1))
        img = torch.from_numpy(img.copy()).float()

        if self.test_data == "CLASSIC":
            gt = np.zeros((img.shape[:2]))
            gt = torch.from_numpy(np.array([gt])).float()
        else:
            gt = np.array(gt, dtype=np.float32)
            if len(gt.shape) == 3:
                gt = gt[:, :, 0]
            gt /= 255.
            gt = torch.from_numpy(np.array([gt])).float()

        return img, gt

class DevDataset(Dataset):
    def __init__(self,
                 data_root,
                 mean_bgr,
                 img_height,
                 img_width,
                 test_list=None,
                 crop_img=False,
                 arg=None
                 ):
        #if test_data not in DATASET_NAMES:
        #    raise ValueError(f"Unsupported dataset: {test_data}")

        self.data_root = data_root
        self.test_list = test_list
        self.mean_bgr = mean_bgr
        self.img_height = img_height
        self.img_width = img_width
        self.crop_img = crop_img
        self.data_index = self._build_index()

    def _build_index(self):
        sample_indices = []
        images_path = os.listdir(self.data_root)
        sample_indices = [(img, None) for img in images_path]
        return sample_indices

    def __len__(self):
        return len(self.data_index)

    def __getitem__(self, idx):
        image_path, _ = self.data_index[idx]
        img_name = os.path.basename(image_path)
        img_dir = self.data_root
        image = cv2.imread(os.path.join(img_dir, image_path), cv2.IMREAD_COLOR)
        image, _ = self.transform(img=image)
        return dict(images=image, file_names=img_name)

    def transform(self, img):
        i_h, i_w, _ = img.shape
        img = np.array(img, dtype=np.float32)
        #img -= self.mean_bgr
        img /= 255.
        crop_size = self.img_height if self.img_height == self.img_width else None
        if i_w > crop_size and i_h > crop_size:
            start_h = (i_h - crop_size) // 2
            start_w = (i_w - crop_size) // 2
            img = img[start_h:start_h + crop_size, start_w:start_w + crop_size]
        else:
            img = cv2.resize(img, (self.img_width, self.img_height))

        img = img.transpose((2, 0, 1))
        img = torch.from_numpy(img).float()
        return img, None

class TrainDataset(Dataset):
    train_modes = ['train', 'test', ]
    dataset_types = ['rgbr', ]
    data_types = ['aug', ]

    def __init__(self,
                 data_root,
                 img_height,
                 img_width,
                 mean_bgr,
                 train_mode='train',
                 dataset_type='rgbr',
                 #  is_scaling=None,
                 # Whether to crop image or otherwise resize image to match image height and width.
                 crop_img=False,
                 arg=None
                 ):
        self.data_root = data_root
        self.train_mode = train_mode
        self.dataset_type = dataset_type
        self.data_type = 'aug'  # be aware that this might change in the future
        self.img_height = img_height
        self.img_width = img_width
        self.mean_bgr = mean_bgr
        self.crop_img = crop_img
        self.arg = arg

        self.data_index = self._build_index()

        print(f"mean_bgr: {self.mean_bgr}")

    def _build_index(self):
        assert self.train_mode in self.train_modes, self.train_mode
        assert self.dataset_type in self.dataset_types, self.dataset_type
        assert self.data_type in self.data_types, self.data_type

        data_root = os.path.abspath(self.data_root)
        sample_indices = []

        file_path = os.path.join(data_root, self.arg.train_list)
        if self.arg.train_data.lower()=='bsds':

            with open(file_path, 'r') as f:
                files = f.readlines()
            files = [line.strip() for line in files]

            pairs = [line.split() for line in files]
            for pair in pairs:
                tmp_img = pair[0]
                tmp_gt = pair[1]
                sample_indices.append(
                    (os.path.join(data_root,tmp_img),
                     os.path.join(data_root,tmp_gt),))
        else:
            with open(file_path) as f:
                files = json.load(f)
            for pair in files:
                tmp_img = pair[0]
                tmp_gt = pair[1]
                sample_indices.append(
                    (os.path.join(data_root, tmp_img),
                     os.path.join(data_root, tmp_gt),))

        return sample_indices

    def __len__(self):
        return len(self.data_index)

    def __getitem__(self, idx):
        # get data sample
        image_path, label_path = self.data_index[idx]

        # load data
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        label = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)
        #plt.imshow(label, cmap='gray')
        #plt.show()
        image, label = self.transform(img=image, gt=label)
        return image, label
        #return dict(images=image, labels=label)

    def transform(self, img, gt):
        gt = np.array(gt, dtype=np.float32)
        if len(gt.shape) == 3:
            gt = gt[:, :, 0]

        gt /= 255.

        img = np.array(img, dtype=np.float32)
        # mean_img = np.mean(img, axis=(0, 1), dtype= np.float32)
        # img -= mean_img
        img -= self.mean_bgr
        # img /=255.


        #if random.random() < 0.5:
        #    scale_factor = 1.0
        #else:
        #    scale_factor = random.uniform(1.05, 2.5)

        #if scale_factor != 1.0:
        #    new_h = int(img.shape[0] * scale_factor)
        #    new_w = int(img.shape[1] * scale_factor)
        #    img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        #    gt  = cv2.resize(gt,  (new_w, new_h), interpolation=cv2.INTER_NEAREST)

        i_h, i_w, _  = img.shape

        crop_size = self.img_height if self.img_height == self.img_width else None


        if i_w> crop_size and i_h>crop_size:
            i = random.randint(0, i_h - crop_size)
            j = random.randint(0, i_w - crop_size)
            img = img[i:i + crop_size , j:j + crop_size ]
            gt = gt[i:i + crop_size , j:j + crop_size ]


        else:
            img = cv2.resize(img, dsize=(crop_size, crop_size))
            gt = cv2.resize(gt, dsize=(crop_size, crop_size))


        img = img.transpose((2, 0, 1))
        img = torch.from_numpy(img.copy()).float()
        gt = torch.from_numpy(np.array([gt])).float()

        return img, gt
