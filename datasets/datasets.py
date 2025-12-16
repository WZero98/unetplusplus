import os
import torch
import numpy as np
import pandas as pd
import cv2
import torch
from sympy.codegen.fnodes import merge
from torch.utils.data import Dataset
from torchvision.transforms import v2


def mergemasks(path: str, transform=None):
    mask_path = os.path.join(path, 'masks')
    mask_filenames = os.listdir(mask_path)
    # save merged mask at ImageId/merged_mask
    merged_mask_path = os.path.join(path, 'merged_mask')

    if not os.path.exists(merged_mask_path):
        final_mask = None
        for file in mask_filenames:
            filename = os.path.join(mask_path, file)
            m = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
            m = (m > 0).astype(np.uint8)  # 将图像 m 转换为二值 mask
            if final_mask is None:
                final_mask = m
            else:
                final_mask = np.logical_or(final_mask, m).astype(np.uint8)
        os.mkdir(merged_mask_path)
        print(f'{merged_mask_path} has been created!')
        cv2.imwrite(merged_mask_path + '/' + "merged_mask.png", final_mask * 255)
    else:
        final_mask = cv2.imread(merged_mask_path + '/' + "merged_mask.png", cv2.IMREAD_GRAYSCALE)
        final_mask = (final_mask > 0).astype(np.uint8)  # 将图像 m 转换为二值 mask

    tensor_mask = torch.tensor(final_mask).unsqueeze(0)
    if transform:
        tensor_mask = transform(tensor_mask)
    return tensor_mask


def restorefig(fig_tensor: torch.Tensor, outputpath: str) -> None:
    fig_tensor = fig_tensor.numpy()
    img_uint8 = (fig_tensor * 255).astype(np.uint8)
    cv2.imwrite(outputpath, img_uint8)


class CellNucleiDataset(Dataset):

    def __init__(self, annotation_files: pd.DataFrame, path: str, transform=None):
        """
        :param annotation_files: a dataframe of images and their labels.
        :param path: the path where original images locate.
        :param transform: image transform functions
        """
        super().__init__()
        self.annotation_files = annotation_files['ImageId'].drop_duplicates()
        self.path = path
        self.transform = transform

    def __len__(self):
        return len(self.annotation_files)

    def __getitem__(self, idx):
        img_name = self.annotation_files.iloc[idx]
        mask = mergemasks(os.path.join(self.path, img_name), transform=self.transform)
        img_path = os.path.join(self.path, img_name, 'images', img_name + '.png')
        img = torch.tensor(
            cv2.imread(img_path)
        ).permute(2, 0, 1)
        if self.transform:
            img = self.transform(img)
        return img, mask


if __name__ == '__main__':
    # train_df = pd.read_csv('stage1_train_labels.csv')
    # img_transform = v2.Compose([
    #     v2.Resize(size=(512, 512), antialias=True),
    #     # v2.RandomHorizontalFlip(p=0.5),  # 随机水平翻转
    #     v2.ToDtype(torch.float32, scale=True)
    # ])
    # train_dataset = CellNucleiDataset(train_df, 'stage1_train', transform=img_transform)
    # x = train_dataset[0][0].permute(1, 2, 0)
    # print(torch.max(x))
    #
    # restorefig(x, 'F:\\Projects\\CV\\cell-nuclei-segmentation\\output.png')
    mergemasks('stage1_train/07fb37aafa6626608af90c1e18f6a743f29b6b233d2e427dcd1102df6a916cf5', transform=None)
