import os

import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as T
import albumentations as A

CMAP = np.array([
    [0, 0, 0],
    [255, 255, 255]
])

class TempDataset(Dataset):
    """
    data_root:
        train:
            images:
                img0001.jpg
                img0002.jpg
                img0003.jpg
                ...
            masks:
                img0001.png
                img0002.png
                img0003.png
                ...
            ...
        val:
            images:
                img0001.jpg
                img0002.jpg
                img0003.jpg
                ...
            masks:
                img0001.png
                img0002.png
                img0003.png
                ...
            ...
        test:
            images:
                img0001.jpg
                img0002.jpg
                img0003.jpg
                ...
    """

    def __init__(self, data_root: str, transform, dataset_type: str, normalization_tensor: bool = True,
                 mean=None, std=None) -> None:
        if std is None:
            std = [0.229, 0.224, 0.225]
        if mean is None:
            mean = [0.485, 0.456, 0.406]
        if dataset_type in ['train', 'val']:
            images_dir = os.path.join(data_root, dataset_type, 'images')
            masks_dir = os.path.join(data_root, dataset_type, 'masks')

            images_name = sorted(os.listdir(images_dir))
            masks_name = sorted(os.listdir(masks_dir))

            assert len(images_name) == len(masks_name), f'{images_dir} 目录和 {masks_dir} 目录下的文件数量不一致'

            self.images_path = [os.path.join(images_dir, name) for name in images_name]
            self.masks_path = [os.path.join(masks_dir, name) for name in masks_name]

            assert Image.open(self.masks_path[0]).mode == 'P', f'{masks_dir} 目录下存储的图片不是以 P 模式存储'

            self.transform = transform
            self.dataset_type = dataset_type
            self.have_nt = normalization_tensor
            self.mean = mean
            self.std = std

        elif dataset_type == 'test':

            assert os.path.exists(os.path.join(data_root, dataset_type)), f'{data_root} 目录下无 test 目录'

            images_dir = os.path.join(data_root, dataset_type, 'images')

            images_name = sorted(os.listdir(images_dir))

            self.images_path = [os.path.join(images_dir, name) for name in images_name]

            self.transform = transform
            self.dataset_type = dataset_type
            self.have_nt = normalization_tensor
            self.mean = mean
            self.std = std

        else:
            raise Exception(f'dataset_type 的值应在 "train", "val", "test" 中, 现传入 {dataset_type}')

    def __len__(self):
        return len(self.images_path)

    def __getitem__(self, item):
        if self.dataset_type in ['train', 'val']:
            image_path = self.images_path[item]
            mask_path = self.masks_path[item]

            image = np.array(Image.open(image_path))
            mask = np.array(Image.open(mask_path))

            assert self.transform is not None, f'在 train 或 val 时不能传入为 None 的数据增强方案'

            comm_transform = self.transform(image=image, mask=mask)

            image_transform, mask_transform = comm_transform['image'], comm_transform['mask']

            if self.have_nt:
                T_image = T.Compose([
                    T.ToTensor(),
                    T.Normalize(mean=self.mean, std=self.std)
                ])
                return T_image(image_transform).type(torch.float32), torch.from_numpy(mask_transform).type(torch.long)
            else:
                return image_transform.type(torch.float32), mask_transform.type(torch.long)

        elif self.dataset_type == 'test':
            image_path = self.images_path[item]
            image = np.array(Image.open(image_path))

            if self.transform is None:
                T_image = T.Compose([
                    T.ToTensor(),
                    T.Normalize(mean=self.mean, std=self.std)
                ])
                return T_image(image).type(torch.float32)
            else:
                comm_transform = self.transform(image=image)
                image_transform = comm_transform['image']
                T_image = T.Compose([
                    T.ToTensor(),
                    T.Normalize(mean=self.mean, std=self.std)
                ])
                return T_image(image_transform).type(torch.float32)
        else:
            raise Exception(f'dataset_type 的值应在 "train", "val", "test" 中, 现传入 {self.dataset_type}')


# 测试功能
if __name__ == '__main__':
    A_transform = A.Compose([
        A.Resize(34, 16),
        A.Flip(),
    ])
    ds_train = TempDataset(r'E:\temp_dataset', A_transform, 'train', True)
    ds_val = TempDataset(r'E:\temp_dataset', A_transform, 'val', True)
    ds_test = TempDataset(r'E:\temp_dataset', None, 'test', True)

    print(f'>>> {len(ds_train), len(ds_val)}')
    print(f'>>> {ds_train[0][0].shape}, {ds_train[0][1].shape}, {ds_val[0][0].shape}, {ds_val[0][1].shape}')
    print(f'>>> {ds_train[0][0]}, {ds_train[0][1]}, {ds_val[0][0]}, {ds_val[0][1]}')
