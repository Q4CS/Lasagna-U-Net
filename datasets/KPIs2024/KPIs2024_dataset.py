import os
import cv2
import numpy as np
from torch.utils.data import Dataset
from datasets.KPIs2024.augmentation import train_transform, val_transform
from PIL import Image
from torchvision.transforms import ToPILImage
import matplotlib.pyplot as plt
os.environ['NO_ALBUMENTATIONS_UPDATE'] = '1'


base_dir = r'/home/datasets/KPIs2024/Task1_patch_level'

'''
normal (0): Normal group: normal mice, sacrificed at the age of 8 weeks.
56Nx (1): 5/6Nx group: mice underwent 5/6 nephrectomy, sacrificed at 12 weeks after nephrectomy (age of 20 weeks).
DN (2): DN group: eNOS-/-/ lepr(db/db) double-knockout mice, sacrificed at the age of 18 weeks.
NEP25 (3): NEP25 group: transgenic mice that express human CD25 selectively in podocytes (NEP25), sacrificed at 3 weeks after immunotoxin-induced glomerular injury (age of 11 weeks).
'''

def read_imgs_path_list(data_dir, split='train'):
    imgs_dir = os.path.join(base_dir, split)
    imgs_path_list = []
    imgs_cls_list = []
    for class_name in os.listdir(imgs_dir):  # train\56Nx
        class_dir = os.path.join(imgs_dir, class_name)

        if class_name == 'normal':
            cls_label = 0
        elif class_name == '56Nx':
            cls_label = 1
        elif class_name == 'DN':
            cls_label = 2
        elif class_name == 'NEP25':
            cls_label = 3
        else:
            raise ValueError(f'class name error: {class_name}')

        for case_name in os.listdir(class_dir):  # train\56Nx\12_116
            img_dir = os.path.join(class_dir, case_name, 'img')  # train\56Nx\12_116\img
            for img_name in os.listdir(img_dir):
                
                img_path = os.path.join(img_dir, img_name)
                imgs_path_list.append(img_path)
                imgs_cls_list.append(cls_label)

                # check whether there is a mask
                mask_path = img_path.replace('img', 'mask')
                if not os.path.exists(mask_path):
                    raise ValueError(f'mask not exist: {mask_path}')

    return imgs_path_list, imgs_cls_list


class KPIs2024_Dataset(Dataset):
    def __init__(self, base_dir=base_dir, split='train', img_size=(256, 256)):
        super(KPIs2024_Dataset, self).__init__()
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
        mask_path = image_path.replace('img', 'mask')

        # mask
        mask = Image.open(mask_path).convert('L')
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

