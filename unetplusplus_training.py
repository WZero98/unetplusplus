import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from datasets.datasets import CellNucleiDataset
from torch.optim import AdamW
from model_architectures.UNetPlusPlus import UNetPlusPlus
from train.trainer import train
from train.loss_functions import BinaryCEDice, MulticlassCEDice
from train.metrics import compute_iou
from torchvision.transforms import v2
import datetime

if __name__ == "__main__":
    date = datetime.datetime.strftime(datetime.datetime.now(), '%Y-%m-%d-%H-%M-%S')
    device = "cuda" if torch.cuda.is_available() else "cpu"
    batch_size = 8
    epochs = 10
    patience = 10
    img_transform = v2.Compose([
        v2.Resize(size=(512, 512), antialias=True),
        # v2.RandomHorizontalFlip(p=0.5),  # 随机水平翻转
        v2.ToDtype(torch.float, scale=False)
    ])
    annotation_file = pd.read_csv("datasets/stage1_train_labels.csv")
    train_df, val_df = train_test_split(annotation_file, test_size=0.2, random_state=23, shuffle=True)
    train_dataset = CellNucleiDataset(train_df, './datasets/stage1_train', transform=img_transform)
    val_dataset = CellNucleiDataset(val_df, './datasets/stage1_train', transform=img_transform)
    model = UNetPlusPlus(2, True).to(device)
    criterion = BinaryCEDice(reduction='mean', combine_weights=[0.5, 0.5])
    optimizer = AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    # 从先前训练的模型权重继续
    trained_weights = torch.load('./model_weights/checkpoints2025-11-26-13-33-43.pth', weights_only=True)

    train(
        model,
        DataLoader(train_dataset, batch_size=batch_size, shuffle=True),
        DataLoader(val_dataset, batch_size=batch_size, shuffle=True),
        criterion,
        optimizer,
        epochs,
        patience,
        save_path=f'./model_weights/checkpoints{date}.pth',
        device=device
    )

