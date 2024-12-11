import torch
from torch.utils.data import Dataset
import os
import os.path
import pandas as pd
from ast import literal_eval

from .utils import pil_loader



ATTRIBUTES = ['original', 'Bangs', 'Eyeglasses', 'Beard', 'Smiling', 'Young']
COMPONENTS = ['original', 'nose', 'eye', 'eyebrow', 'lip', 'hair']
CLASSES = list()
CLASS_SYN = list()
COMPONENTS_SYN = [
    ["original", "genuine", "untampered"],
    ["eyes", "eyes", "eyes"],
    ["nose", "nose", "nose",],
    ["eyebrows", "brow", "forehead"],
    ["mouth", "lips", "smile"],
    ["hair", "bangs", "fringe"]
]

ATTRIBUTES_SYN = [
    ["original", "genuine", "untampered"],
    ["hair", "bangs", "fringe"],
    ["eyes", "glasses", "eyewear"],
    ["beard", "facial hair", "stubble"],
    ["mouth", "lips", "smile"],
    ["age", "young", "old"]
]

class SeqDeepFakeDataset(Dataset):

    def __init__(self,
                 cfg=None,
                 root=None,
                 split="train",
                 transform=None,
                 attributes: bool = False
                 ):
        super().__init__()
        self.mode = split
        self.cfg = cfg
        self.transforms = transform
        self.attributes = attributes
        dataset_name = 'facial_attributes' if self.attributes else'facial_components'
        self.dataset = self.make_dataset(os.path.join(root, f"{dataset_name}/annotations/{split}.csv"), root=root)
        if not CLASSES:
            self._set_classes()

    def __getitem__(self, index: int):
        img_path, label = self.dataset[index]

        label_list = literal_eval(label)
        label = torch.zeros(len(label_list)+1)
        if all(l == 0 for l in label_list):
            label[0] = 1
        else:
            for i in label_list:
                label[i] = 1 if i > 0 else 0
        image = pil_loader(img_path)
        if self.transforms:
            image = self.transforms(image)
        return index, image, label

    def __len__(self):
        return len(self.dataset)

    def make_dataset(self, csv_file, root=None):
        dataset = []
        imgs, labels = self.read_data(csv_file)
        for i in range(len(imgs)):
            if root:
                imgs[i] = os.path.join(root, imgs[i])  # from relative path to absolute path
            dataset.append((imgs[i], labels[i]))
        return dataset

    @staticmethod
    def read_data(file):
        info = pd.read_csv(file)
        img_list = info['file_path'].tolist()
        label_list = info['label'].tolist()
        return img_list, label_list


    def _set_classes(self):
        if self.attributes:
            CLASSES.extend(ATTRIBUTES)
            CLASS_SYN.extend(ATTRIBUTES_SYN)
        else:
            CLASSES.extend(COMPONENTS)
            CLASS_SYN.extend(COMPONENTS_SYN)


    def get_img_path(self, index):
        if isinstance(index, int):
            img_path, label = self.dataset[index]
            return img_path
        else:
            img = list()
            for idx in index:
                img_path, label = self.dataset[idx]
                img.append(img_path)
            return img
