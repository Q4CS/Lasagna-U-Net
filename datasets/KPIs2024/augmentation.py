import os
import albumentations as A
from albumentations.pytorch import ToTensorV2
os.environ['NO_ALBUMENTATIONS_UPDATE'] = '1'


def train_transform(p=1, img_size=(256, 256), mean=(1.0, 1.0, 1.0), std=(1.0, 1.0, 1.0)):
    train_compose = A.Compose(
        [
            A.Resize(img_size[0], img_size[1], always_apply=True),
            # A.RandomResizedCrop(width=img_size[0], height=img_size[1], scale=(0.8, 1.0), ratio=(1.0, 1.0), always_apply=True),
            # Spatial-level transforms
            A.OneOf([
                A.HorizontalFlip(p=0.2),
                A.VerticalFlip(p=0.2),
                A.RandomRotate90(p=0.2),
                A.Transpose(p=0.2),
            ], p=0.5),
            # Pixel-level transforms
            A.OneOf([
                A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.2),
                A.Blur(blur_limit=5, p=0.2),
                A.GaussNoise(p=0.2),
                A.ISONoise(p=0.2),
                A.GaussianBlur(blur_limit=(3, 5), sigma_limit=0, p=0.2),
                A.CoarseDropout(num_holes_range=(1, 5), hole_width_range=(5,20), hole_height_range=(5,20), p=0.2),
            ], p=0.5),
            # A.Normalize(mean=mean, std=std),
            ToTensorV2(always_apply=True),  # apply `ToTensorV2` that converts a NumPy array to a PyTorch tensor
        ], p=p
    )
    return train_compose


def val_transform(p=1, img_size=(256, 256), mean=(1.0, 1.0, 1.0), std=(1.0, 1.0, 1.0)):
    val_compose = A.Compose(
        [
            A.Resize(img_size[0], img_size[1], always_apply=True),
            # A.RandomResizedCrop(width=img_size[0], height=img_size[1], scale=(1.0, 1.0), ratio=(1.0, 1.0), always_apply=True),
            # A.Normalize(mean=mean, std=std),
            ToTensorV2(always_apply=True),  # apply `ToTensorV2` that converts a NumPy array to a PyTorch tensor
        ], p=p
    )
    return val_compose
