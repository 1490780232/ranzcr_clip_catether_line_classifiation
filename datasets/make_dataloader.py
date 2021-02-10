#coding:utf-8
from PIL import Image
import torch
import torchvision.transforms as T
from torch.utils.data import DataLoader
from .Ranzer import RANZCR
from .preprocessing import RandomErasing
import pandas as pd
from sklearn.model_selection import train_test_split

from config import Config


def train_collate_fn(batch):
    """
    # collate_fn这个函数的输入就是一个list，list的长度是一个batch size，list中的每个元素都是__getitem__得到的结果
    """
    imgs, pids, _, _, = zip(*batch)
    pids = torch.tensor(pids, dtype=torch.int64)
    return torch.stack(imgs, dim=0), pids


def val_collate_fn(batch):
    imgs, pids, camids, img_paths = zip(*batch)
    return torch.stack(imgs, dim=0), pids, camids, img_paths


def make_dataloader(cfg):
    train_transforms = T.Compose([
        T.Resize(cfg.INPUT_SIZE),
        T.RandomHorizontalFlip(p=0.5),
        T.Pad(10),
        # T.RandomCrop([256, 128]),
        # T.RandomRotation(12, resample=Image.BICUBIC, expand=False, center=None),
        # T.RandomApply([T.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.3, hue=0),
        #                T.RandomAffine(degrees=0, translate=None, scale=[0.8, 1.2], shear=15, \
        #                               resample=Image.BICUBIC, fillcolor=0)], p=0.5),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        RandomErasing(probability=0.5, sh=0.4, mean=(0.4914, 0.4822, 0.4465))
    ])

    val_transforms = T.Compose([
        T.Resize(cfg.INPUT_SIZE),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    num_workers = cfg.DATALOADER_NUM_WORKERS
    train_csv = pd.read_csv(cfg.paths['csv_path'])
    print(train_csv.head())
    train, test, train_y, test_y = train_test_split(train_csv,
                                                        train_csv,
                                                        test_size=0.1,
                                                        random_state=0)
    train_dataset = RANZCR(train, cfg, train_transforms)
    val_dataset = RANZCR(test, cfg, val_transforms)
    train_loader = DataLoader(train_dataset,
                              batch_size=cfg.BATCH_SIZE,
                              num_workers=num_workers,
                              )
    val_loader = DataLoader(val_dataset,
                            batch_size=cfg.TEST_IMS_PER_BATCH,
                            shuffle=False, num_workers=num_workers,
                            )
    return train_loader, val_loader
if __name__ == '__main__':
    cfg = Config()
    train_loader, test_loader = make_dataloader(cfg)
    for data in train_loader:
        print(data[0], data[1].shape, data[2])