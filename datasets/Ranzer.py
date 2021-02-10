import os
import cv2
import numpy as np
import pandas as pd
# import albumentations
import torch
from torch.utils.data import Dataset
from typing import Optional
from tqdm import tqdm
from config.configs import Config
from PIL import Image, ImageFile

from torch.utils.data import Dataset
import os.path as osp

ImageFile.LOAD_TRUNCATED_IMAGES = True
from torch.nn import BCEWithLogitsLoss
def read_image(img_path):
    """Keep reading image until succeed.
    This can avoid IOError incurred by heavy IO process."""
    got_img = False
    if not osp.exists(img_path):
        raise IOError("{} does not exist".format(img_path))
    while not got_img:
        try:
            img = Image.open(img_path).convert('RGB')
            got_img = True
        except IOError:
            print("IOError incurred when reading '{}'. Will redo. Don't worry. Just chill.".format(img_path))
            pass
    return img
class RANZCR(Dataset):
	def __init__(self, df, cf, transforms):
		self.df = df
		self.config = cf
		self.transforms = transforms

		if self.transforms is None:
			print('Transforms is None!')

	def __len__(self):
		return len(self.df)

	def __getitem__(self, idx):
		image_id = self.df[self.config.image_col_name].values[idx]
		image_path = os.path.join(self.config.paths['train_path'], "{}{}".format(image_id, ".jpg"))
		img = read_image(image_path)

		if self.transforms is not None:
			img = self.transforms(img)
		label = self.df[self.config.class_col_name].values[idx]
		label = torch.as_tensor(data=label, dtype=torch.float32, device=None)
		return image_id, img, label
if __name__ == '__main__':
	config = Config()
	train_csv = pd.read_csv(config.paths['csv_path'])
	print(	train_csv.head())
	train_dataset = RANZCR(train_csv, config, None)
	print(train_dataset[1])