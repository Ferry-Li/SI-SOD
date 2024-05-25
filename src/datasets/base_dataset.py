import torch
import numpy as np 
import cv2
import os
import torchvision
from torchvision import transforms
from torch.utils.data import Dataset 
from datasets import transforms as myTransforms

NORMALISE_PARAMS = [np.array([0.406, 0.456, 0.485], dtype=np.float32).reshape((1, 1, 3)), # MEAN, BGR
                    np.array([0.225, 0.224, 0.229], dtype=np.float32).reshape((1, 1, 3))] # STD, BGR

def get_img_transform(img_size=384, 
                      is_train=True,
                      hflip=True,
                      norm_mean_std=NORMALISE_PARAMS):
    transform_list = []
    if is_train:
        # transform_list.append(myTransforms.Normalize(*NORMALISE_PARAMS))
        transform_list.append(myTransforms.Normalize(*NORMALISE_PARAMS))
        transform_list.append(myTransforms.Scale(img_size, img_size))
        transform_list.append(myTransforms.RandomFlip())
        transform_list.append(myTransforms.GaussianNoise())
        transform_list.append(myTransforms.ToTensor())

    else:
        transform_list.append(myTransforms.Normalize(*NORMALISE_PARAMS))
        transform_list.append(myTransforms.Scale(img_size, img_size))
        transform_list.append(myTransforms.ToTensor(BGR=False))

    transform = myTransforms.Compose(transform_list)
    return transform

class ImageMaskDataset(Dataset):
    def __init__(self, data_dir, transform=True, is_train=True, image_size=384, is_png=False):
        self.transform = transform
        self.is_train = is_train
        self.image_size = image_size
        self.img_list = list()
        self.msk_list = list()
        self.name_list = list()
        self.img_postfix = '.png' if is_png else '.jpg'

        with open(os.path.join(data_dir, 'list.txt'), 'r') as lines:
            for line in lines:
                line_arr = line.strip()
                self.img_list.append(os.path.join(data_dir, 'image', line_arr + self.img_postfix))
                self.msk_list.append(os.path.join(data_dir, 'mask', line_arr + '.png'))
                self.name_list.append(line_arr)

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        image = cv2.imread(self.img_list[idx])
        mask = cv2.imread(self.msk_list[idx], 0)
        name = self.name_list[idx]

        if self.transform:
            transform = get_img_transform(img_size=self.image_size, is_train=self.is_train)
            image, mask = transform(image, mask)

        return image, mask, name

class ImageMaskWeightDataset(Dataset):
    def __init__(self, data_dir, transform=True, load_weight=False, is_train=True, image_size=384):
        self.transform = transform
        self.is_train = is_train
        self.image_size = image_size
        self.load_weight = load_weight
        self.img_list = list()
        self.msk_list = list()
        self.weight_list = list()
        self.name_list = list()

        with open(os.path.join(data_dir, 'list.txt'), 'r') as lines:
            for line in lines:
                line_arr = line.strip()
                image_path = os.path.join(data_dir, 'image', line_arr + '.jpg')
                if not os.path.exists(image_path):
                    image_path = os.path.join(data_dir, 'image', line_arr + '.png')
                self.img_list.append(image_path)
                self.msk_list.append(os.path.join(data_dir, 'mask', line_arr + '.png'))
                self.weight_list.append(os.path.join(data_dir, 'weight', line_arr + '.npy'))
                self.name_list.append(line_arr)

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        image = cv2.imread(self.img_list[idx])
        mask = cv2.imread(self.msk_list[idx], 0)
        if self.load_weight:
            weight = np.load(self.weight_list[idx])
        # else:
        #     weight = compute_weight(mask)
        name = self.name_list[idx]

        if self.transform:
            transform = get_img_transform(img_size=self.image_size, is_train=self.is_train)
            image, mask_weight = transform(image, [mask, weight])
            mask, weight = mask_weight

        return image, mask, weight, name
