import os
import cv2
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
from datasets.ISIC_2018_Task_1.augmentation import train_transform, val_transform
os.environ['NO_ALBUMENTATIONS_UPDATE'] = '1'

base_dir = r'/home/datasets/ISIC_2018_Task_1'


def read_imgs_path_list(base_dir, split='train'):
    imgs_path_list = []
    imgs_dir = os.path.join(base_dir, split + '_img')
    # masks_dir = os.path.join(base_dir, split + '_mask')
    for img_name in os.listdir(imgs_dir):  # base_dir\train_img
        img_path = os.path.join(imgs_dir, img_name)

        # check if the mask exists
        mask_path = img_path.replace('_img', '_mask').replace('.jpg', '_segmentation.png')
        if not os.path.exists(mask_path):
            raise ValueError(f'mask not exist: {mask_path}')

        imgs_path_list.append(img_path)

    return imgs_path_list


class ISIC2018Task1_DataSets(Dataset):
    def __init__(
            self,
            base_dir=base_dir,
            split='train',
            img_size=(256, 256),
    ):
        super(ISIC2018Task1_DataSets, self).__init__()

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

        self.images_path_list = read_imgs_path_list(base_dir=self.base_dir, split=self.split)

        print(f'split: {self.split}, total {len(self.images_path_list)} samples')

    def __len__(self):
        return len(self.images_path_list)

    def __getitem__(self, idx):
        image_path = self.images_path_list[idx]
        mask_path = image_path.replace('_img', '_mask').replace('.jpg', '_segmentation.png')

        # mask
        mask = np.array(Image.open(mask_path).convert('L')) / 255

        # image
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        transform = self.transform(image=image, mask=mask)
        image = transform['image']
        mask = transform['mask']

        return {'image': image.float(),
                'label': mask,
                'image_name': image_path}

