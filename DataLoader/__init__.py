from torch.utils import data
from torch.utils.data.distributed import DistributedSampler
from torchvision import transforms

from datasets import load_dataset

from .FF import FFDataset
from .FF import CLASSES as ff_classes
from .StyleGan import StyleGAN
from .CelebDF import CelebDF
from .DFDC import DFDC
from .DFW import DFW
from .PseudoFake import AugmentedDataset
from .PseudoFake import CLASSES as aug_classes
from .PseudoFake import CLASS_SYN as aug_synonyms
from .SeqDeepFake import SeqDeepFakeDataset
from .SeqDeepFake import CLASSES as seq_classes
from .SeqDeepFake import CLASS_SYN as seq_synonyms

from .utils import collate_fn

__all__ = ['get_loaders', 'CLASSES', 'CLASS_SYNONYMS']

CLASSES = list()
CLASS_SYNONYMS = list()


def get_loaders(cnf):
    if cnf.dataset.dataset_name == 'FF++':
        train_dataset, test_dataset = _get_FF_loaders(cnf)
        if cnf.dataset.binary:
            CLASSES.append("original")
        else:
            CLASSES.extend(ff_classes)
    elif cnf.dataset.dataset_name == 'StyleGAN':
        train_dataset, test_dataset = _get_gan_loaders(cnf)
        CLASSES.append("original")
    elif cnf.dataset.dataset_name == 'CelebDF':
        train_dataset, test_dataset = _get_cdf_loaders(cnf)
        CLASSES.append("original")
    elif cnf.dataset.dataset_name == 'DFDC':
        train_dataset, test_dataset = _get_dfdc_loaders(cnf)
        CLASSES.append("original")
    elif cnf.dataset.dataset_name == 'DFW':
        train_dataset, test_dataset = _get_dfw_loaders(cnf)
        CLASSES.append("original")
    elif cnf.dataset.dataset_name == 'augFF':
        train_dataset, test_dataset = _get_custom_loaders(cnf)
        CLASSES.extend(aug_classes)
        CLASS_SYNONYMS.extend(aug_synonyms)
    elif cnf.dataset.dataset_name == 'SeqDeepFake':
        train_dataset, test_dataset = _get_seq_loaders(cnf)
        CLASSES.extend(seq_classes)
        CLASS_SYNONYMS.extend(seq_synonyms)
    elif cnf.dataset.dataset_name == 'image_edit':
        # or load the separate splits if the dataset has train/validation/test splits
        train_dataset = load_dataset("osunlp/MagicBrush", split="train")
        test_dataset = load_dataset("osunlp/MagicBrush", split="dev")

    else:
        raise NotImplementedError

    if cnf.debug and cnf.dataset.dataset_name != 'FFaug':
        train_dataset.dataset = train_dataset.dataset[:30]
        test_dataset.dataset = test_dataset.dataset[:30]

    if cnf.DDP:
        train_sampler = DistributedSampler(train_dataset)
        test_sampler = DistributedSampler(test_dataset)
    else:
        train_sampler = None
        test_sampler = None
    train_loader = data.DataLoader(
        train_dataset,
        batch_size=cnf.training.batch_size,
        collate_fn=collate_fn,
        num_workers=2,
        sampler=train_sampler
    )
    test_loader = data.DataLoader(
        test_dataset,
        batch_size=cnf.training.batch_size,
        collate_fn=collate_fn,
        num_workers=2,
        sampler=test_sampler
    )
    print(len(train_dataset), len(test_dataset))
    return train_loader, test_loader


def _get_custom_loaders(cnf):
    test_transform = None
    test_dataset = AugmentedDataset(
        img_dir=cnf.dataset.root,
        transform=test_transform,
        target_transform=None
    )
    idx = 60000
    img, lab, vid = test_dataset.dataset
    test_dataset.dataset = (img[:idx], lab[:idx], vid[:idx])
    if cnf.debug:
        img, lab, vid = test_dataset.dataset
        test_dataset.dataset = (img[:10], lab[:10], vid[:10])
    return test_dataset, test_dataset


def _get_FF_loaders(cnf):
    train_transform = None
    train_dataset = FFDataset(
        root=cnf.dataset.root,
        split='train',
        transform=train_transform,
        detailed_lbl=cnf.dataset.binary
    )
    test_transform = None
    test_dataset = FFDataset(
        root=cnf.dataset.root,
        split='test',
        transform=test_transform,
        detailed_lbl=cnf.dataset.binary
    )

    return train_dataset, test_dataset


def _get_seq_loaders(cnf):
    train_transform = None
    train_dataset = SeqDeepFakeDataset(
        root=cnf.dataset.root,
        split='train',
        transform=train_transform,
        attributes=cnf.dataset.attributes
    )
    test_transform = None
    test_dataset = SeqDeepFakeDataset(
        root=cnf.dataset.root,
        split='test',
        transform=test_transform,
        attributes=cnf.dataset.attributes
    )
    return train_dataset, test_dataset


def _get_gan_loaders(cnf):
    train_transform = None
    train_dataset = StyleGAN(
        root=cnf.dataset.root,
        version=cnf.dataset.version,
        transform=train_transform
    )
    return train_dataset, train_dataset

def _get_cdf_loaders(cnf):
    train_transform = None
    train_dataset = CelebDF(
        root=cnf.dataset.root,
        transform=train_transform
    )
    return train_dataset, train_dataset

def _get_dfdc_loaders(cnf):
    train_transform = None
    train_dataset = DFDC(
        root=cnf.dataset.root,
        split='test',
        transform=train_transform
    )
    return train_dataset, train_dataset

def _get_dfw_loaders(cnf):
    train_transform = None
    train_dataset = DFW(
        root=cnf.dataset.root,
        split='test',
        transform=train_transform
    )
    return train_dataset, train_dataset
