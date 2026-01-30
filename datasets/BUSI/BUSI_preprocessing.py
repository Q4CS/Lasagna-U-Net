import os
import numpy as np
from PIL import Image
import shutil


def mask_merge(raw_data_dir, save_mask_merge_data_dir):
    """
    merge mask
    :param raw_data_dir:
    :param save_mask_merge_data_dir:
    :return:
    """
    for class_name in os.listdir(raw_data_dir):  # ['benign', 'malignant', 'normal']
        class_dir = os.path.join(raw_data_dir, class_name)

        save_class_dir = os.path.join(save_mask_merge_data_dir, class_name)
        if not os.path.exists(save_class_dir):
            os.makedirs(save_class_dir)

        img_or_mask_name_list = os.listdir(class_dir)
        for img_or_mask_name in img_or_mask_name_list:  # images_dir
            if 'mask' in img_or_mask_name:
                continue

            img_name = img_or_mask_name
            img_path = os.path.join(class_dir, img_name)
            save_img_path = os.path.join(save_class_dir, img_name)
            # count the number of masks
            masks_name_list = []
            for temp_name in img_or_mask_name_list:
                if (img_name.replace('.png', '') + '_') in temp_name:
                    masks_name_list.append(temp_name)

            if len(masks_name_list) == 1:
                shutil.copyfile(img_path, save_img_path)
                shutil.copyfile(img_path.replace('.png', '_mask.png'), save_img_path.replace('.png', '_mask.png'))
            elif len(masks_name_list) > 1:
                shutil.copyfile(img_path, save_img_path)
                # bg:255, roi:0
                merge_mask_np = np.zeros_like(np.array(Image.open(os.path.join(class_dir, masks_name_list[0])).convert('L')))
                for mask_name in masks_name_list:
                    temp_png_mask_np = np.array(Image.open(os.path.join(class_dir, mask_name)).convert('L'))
                    merge_mask_np = merge_mask_np + temp_png_mask_np

                merge_mask_np = np.where(merge_mask_np > 0, 255, 0)
                Image.fromarray(merge_mask_np).convert('L').save(save_img_path.replace('.png', '_mask.png'))
            else:
                raise ValueError(f'img_or_mask_name:{img_or_mask_name}, !!!!!')
