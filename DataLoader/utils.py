import torch
from PIL import Image


def pil_loader(path: str) -> Image:
    """Load image RGB file"""
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')


def collate_fn(batch: list) -> (torch.tensor, torch.tensor, torch.tensor):
    vid = torch.tensor([x[0] for x in batch], dtype=torch.int)
    img = [x[1] for x in batch]
    label = [x[2] for x in batch]
    if not isinstance(label[0], torch.Tensor):
        label = torch.tensor(label, dtype=torch.float)
    else:
        label = torch.stack(label)
    return vid, img, label
