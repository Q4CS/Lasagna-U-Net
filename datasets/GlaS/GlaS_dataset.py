import os
import cv2
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from datasets.GlaS.augmentation import train_transform, val_transform
from PIL import Image
from torchvision.transforms import ToPILImage
import matplotlib.pyplot as plt

os.environ['NO_ALBUMENTATIONS_UPDATE'] = '1'


base_dir = r'/home/datasets/GlaS'


def read_imgs_path_list(data_dir=base_dir, split='train'):
    # train: train, val: testA, test: testB
    # benign (0),  malignant (1)
    imgs_dir = os.path.join(base_dir, split, 'image')  #
    csv_pd = pd.read_csv(os.path.join(data_dir, 'Grade.csv'), header=0, index_col='name')
    # csv_pd.loc['testA_1']['grade (GlaS)']
    imgs_path_list = []
    imgs_cls_list = []
    for img_name in os.listdir(imgs_dir):  # train\image
        img_path = os.path.join(imgs_dir, img_name)
        imgs_path_list.append(img_path)
        cls_label_name = csv_pd.loc[img_name.replace('.bmp', '')]['grade (GlaS)']
        if cls_label_name == 'benign':
            cls_label = 0
        elif cls_label_name == 'malignant':
            cls_label = 1
        else:
            raise ValueError(f'class not support: {cls_label_name}')
        imgs_cls_list.append(cls_label)

        # check whether there is a mask
        mask_path = img_path.replace('image', 'mask').replace('.bmp', '_anno.bmp')
        if not os.path.exists(mask_path):
            raise ValueError(f'mask not exist: {mask_path}')

    return imgs_path_list, imgs_cls_list


class GlaS_Dataset(Dataset):
    def __init__(self, base_dir=base_dir, split='train', img_size=(256, 256)):
        super(GlaS_Dataset, self).__init__()
        self.base_dir = base_dir
        self.split = split

        if self.split == 'train':
            self.transform = train_transform(img_size=img_size)
        elif self.split == 'val':
            self.transform = val_transform(img_size=img_size)
        elif self.split == 'test':
            self.transform = val_transform(img_size=img_size)
        else:
            raise ValueError(f"The split ({self.split}) must be between 'train', 'val' or 'test'")

        self.images_path_list, self.images_cls_list = read_imgs_path_list(data_dir=self.base_dir, split=self.split)

        print(f'split: {self.split}, total {len(self.images_path_list)} samples')

    def __len__(self):
        return len(self.images_path_list)

    def __getitem__(self, idx):
        image_path = self.images_path_list[idx]
        cls_label = self.images_cls_list[idx]
        mask_path = image_path.replace('image', 'mask').replace('.bmp', '_anno.bmp')

        # mask
        mask = Image.open(mask_path)
        # mask = mask.convert('L')
        mask = np.array(mask)
        mask[mask > 0] = 1

        # image
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        transform = self.transform(image=image, mask=mask)
        image = transform['image']
        mask = transform['mask']

        return {'image': image.float(),
                'seg_label': mask,
                'cls_label': cls_label,
                'image_name': image_path}

