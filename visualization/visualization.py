import cv2
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from datasets.datasets import CellNucleiDataset
from torch.optim import AdamW
from model_architectures.UNetPlusPlus import UNetPlusPlus
from torchvision.transforms import v2
import datetime

device = "cuda" if torch.cuda.is_available() else "cpu"
model = UNetPlusPlus(2, True).to(device)
trained_weights = torch.load('../model_weights/checkpoints2025-11-26-13-33-43.pth', weights_only=True)
img_transform = v2.Compose([
    v2.Resize(size=(512, 512), antialias=True),
    # v2.RandomHorizontalFlip(p=0.5),  # 随机水平翻转
    v2.ToDtype(torch.float, scale=False)
])

annotation_file = pd.read_csv("../datasets/stage1_train_labels.csv")
train_df, val_df = train_test_split(annotation_file, test_size=0.2, random_state=23, shuffle=True)
train_dataset = CellNucleiDataset(train_df, '../datasets/stage1_train', transform=img_transform)
val_dataset = CellNucleiDataset(val_df, '../datasets/stage1_train', transform=img_transform)

model.load_state_dict(trained_weights)
img = val_dataset[15][0].to(device)
ground_truth = val_dataset[15][1]
model.eval()
with torch.no_grad():
    predicted = (model(img.unsqueeze(0))[0] + model(img.unsqueeze(0))[1] + model(img.unsqueeze(0))[2] + model(img.unsqueeze(0))[3])/4

x = img.permute(1, 2, 0).to('cpu').numpy().astype(np.uint8)
truth = ((ground_truth.permute(1, 2, 0).numpy())*255).astype(np.uint8)
output = (
    (
        (
            predicted
            .to('cpu')
            .squeeze(0)
            .permute(1, 2, 0)
            .numpy()
        ) > 0.5
    )*255
).astype(np.uint8)

# output figs
cv2.imwrite('Image3.png', x)
cv2.imwrite('ground_truth3.png', truth)
cv2.imwrite('predicted3.png', output)
