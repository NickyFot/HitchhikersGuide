import os
import json
import re

import torch
from torch.utils import data
from torchvision import transforms

from .utils import pil_loader

CLASSES = ["original", "eyebrows", "eyes", "nose", "mouth", "faceswap"]
CLASS_SYN = [
    ["original", "genuine", "untampered"],
    ["eyebrows", "brow", "forehead"],
    ["eyes", "eyes", "eyes"],
    ["nose", "nose", "nose"],
    ["mouth", "lips", "smile"],
    ["face", "faceswap", "background"]
]


class AugmentedDataset(data.Dataset):
    def __init__(self, img_dir, transform=None, target_transform=None):
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform
        self._mk_dataset()

    @property
    def dataset(self):
        return self.img_files, self.img_labels, self.videos

    @dataset.setter
    def dataset(self, dataset):
        (img_files, img_labels, videos) = dataset
        self.img_files = img_files
        self.img_labels = img_labels
        self.videos = videos

    def _mk_dataset(self):
        # fake_image_2055_00011.png
        imgs = os.listdir(self.img_dir)
        labels = list()
        vid_idx = dict()
        videos = list()
        for img in imgs:
            label = os.path.splitext(img)[0].split('_')[-1]
            vid = os.path.splitext(img)[0].split('_')[-2]
            if vid not in vid_idx.keys():
                vid_idx[vid] = [len(vid_idx)]
            videos.append(vid_idx[vid])
            if label == "00000":
                labels.append([1., 0., 0., 0., 0., 0.])
            else:
                label = [0, *label]
                label = [float(x) for x in label]
                labels.append(label)
        self.img_files = imgs
        self.img_labels = labels
        self.vid_idx = vid_idx
        self.videos = videos

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_files[idx])
        image = pil_loader(img_path)
        label = self.img_labels[idx]
        vid = self.img_files[idx]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return idx, image, label

    def get_img_path(self, index):
        if isinstance(index, int):
            img_path = os.path.join(self.img_dir, self.img_files[index])
            return img_path
        else:
            img = list()
            for idx in index:
                img_path = os.path.join(self.img_dir, self.img_files[idx])
                img.append(img_path)
            return img
