import os
import cv2
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from PIL import Image
from datasets.BUSI.augmentation import train_transform, val_transform, center_crop
from utils import split_list

os.environ['NO_ALBUMENTATIONS_UPDATE'] = '1'


base_dir = r'/home/datasets/BUSI/mask_merge_data'


def BUSI_dataset_division(base_dir, shuffle=False):

    benign_dir = os.path.join(base_dir, 'benign')
    malignant_dir = os.path.join(base_dir, 'malignant')

    benign_list = list(range(1, 438))
    malignant_list = list(range(1, 211))
    benign_sub_lists = split_list(benign_list, ratio=[0.7, 0.1, 0.2], shuffle=shuffle)
    malignant_sub_lists = split_list(malignant_list, ratio=[0.7, 0.1, 0.2], shuffle=shuffle)

    train_list = []
    val_list = []
    test_list = []

    for idx in range(3):
        for i in benign_sub_lists[idx]:
            img_path = os.path.join(benign_dir, f'benign ({i}).png')
            mask_path = img_path.replace('.png', '_mask.png')
            cls_label = 0
            if idx == 0:
                train_list.append([img_path, mask_path,cls_label])
            elif idx == 1:
                val_list.append([img_path, mask_path,cls_label])
            elif idx == 2:
                test_list.append([img_path, mask_path,cls_label])

        for j in malignant_sub_lists[idx]:
            img_path = os.path.join(malignant_dir, f'malignant ({j}).png')
            mask_path = img_path.replace('.png', '_mask.png')
            cls_label = 1
            if idx == 0:
                train_list.append([img_path, mask_path,cls_label])
            elif idx == 1:
                val_list.append([img_path, mask_path,cls_label])
            elif idx == 2:
                test_list.append([img_path, mask_path,cls_label])

    return {'train_list':train_list,
            'val_list':val_list,
            'test_list':test_list}


class BUSI_DataSets(Dataset):
    def __init__(
            self,
            base_dir=base_dir,
            split='train',
            img_size=(256, 256)
    ):
        super(BUSI_DataSets, self).__init__()

        self.base_dir = base_dir
        self.split = split
        self.img_size = img_size
        self.dataset_division = BUSI_dataset_division(self.base_dir, shuffle=False)

        if split == 'train':
            self.images_mask_info_list = self.dataset_division['train_list']
            self.transform = train_transform(img_size=img_size)
        elif split == 'val':
            self.images_mask_info_list = self.dataset_division['val_list']
            self.transform = val_transform(img_size=img_size)
        elif split == 'test':
            self.images_mask_info_list = self.dataset_division['test_list']
            self.transform = val_transform(img_size=img_size)
        else:
            raise ValueError(f"The split ({split}) must be between 'train', 'val' or 'test'")

        print(f'split: {self.split}, total {len(self.images_mask_info_list)} samples')

    def __len__(self):
        return len(self.images_mask_info_list)

    def __getitem__(self, idx):
        image_path = self.images_mask_info_list[idx][0]
        mask_path = self.images_mask_info_list[idx][1]
        cls_label = self.images_mask_info_list[idx][2]

        # mask
        mask = np.array(Image.open(mask_path).convert('L')) / 255
        mask_shape = mask.shape
        crop_size = mask_shape[0] if mask_shape[0] <= mask_shape[1] else mask_shape[1]  # min edge
        mask = center_crop(mask, crop_size=[crop_size, crop_size])

        # image
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = center_crop(image, crop_size=[crop_size, crop_size])

        transform = self.transform(image=image, mask=mask)
        image = transform['image']
        mask = transform['mask']

        return {'image': image.float(),
                'seg_label': mask,
                'cls_label': cls_label,
                'image_name': os.path.basename(image_path)}

