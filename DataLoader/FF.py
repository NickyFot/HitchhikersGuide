import os
import json

import torch
from torch.nn import functional as F
from torch.utils import data
from torchvision import transforms

from .utils import pil_loader

CLASSES = ["original", "Deepfakes", "Face2Face", "FaceSwap", "NeuralTextures"]


class FFDataset(data.Dataset):
    def __init__(self, root, transform=None, split: str = 'train', detailed_lbl: bool = False):
        self.root = root
        self.transform = transform
        if split not in ['train', 'val', 'test']:
            raise NotImplemented("Invalid split option. Split should be either 'train', 'val', or 'test'")
        self.split = split
        self.root = os.path.join(*[self.root, self.split, 'frames'])
        self.lab = detailed_lbl
        self.dataset = list()
        self.vid_idx = dict()
        self._mk_dataset()

    def num_videos(self):
        return len(self.vid_idx.keys())

    def _mk_dataset(self):
        idx_dict = dict()
        for idx, cl in enumerate(CLASSES):
            cl_dir = os.path.join(self.root, cl)
            for root, dirs, files in os.walk(cl_dir):
                for fname in files:
                    fpath = os.path.join(root, fname)
                    sample = dict()
                    sample['image'] = fpath
                    lab = torch.tensor(idx if not self.lab else [int(cl == "original")])
                    if not self.lab:
                        lab = F.one_hot(lab, num_classes=len(CLASSES))
                    sample['label'] = lab
                    vid = fpath
                    sample['vid'] = vid
                    self.dataset.append(sample)
                    if vid not in idx_dict.keys():
                        idx_dict[vid] = [len(idx_dict)]
        self.vid_idx = idx_dict

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        sample = self.dataset[idx]
        image = pil_loader(sample['image'])
        label = sample['label']
        if self.transform is not None:
            image = self.transform(image)
        return idx, image, label

    def get_img_path(self, index):
        if isinstance(index, int):
            sample = self.dataset[index]
            vid = sample['vid']
            img_path = self.vid_idx[vid]
            return img_path
        else:
            img = list()
            for idx in index:
                sample = self.dataset[idx]
                vid = sample['vid']
                img_path = self.vid_idx[vid]
                img.append(img_path)
            return img