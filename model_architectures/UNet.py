import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Module
from typing import Optional, Tuple, Literal
from collections import OrderedDict
from copy import deepcopy
from .BaseBlocks import ConvNormNonlinearDropout, Upsampling, StackedConvLayers


class UNet(Module):
    """
    U-net architecture
    """
    # class variables
    INPUT_CHANNELS: int = 3
    STACKED_LAYERS_NUM: int = 2
    CONV_LY = nn.Conv2d
    NONELINEAR_LY = nn.ReLU
    NORM_LY = nn.BatchNorm2d

    def __init__(
            self,
            num_classes: int = 2
    ):
        super().__init__()
        if num_classes == 2:
            self.num_classes = num_classes - 1
            self.output_clf = nn.Sigmoid
            self.output_clf_kwargs = None
        else:
            self.num_classes = num_classes
            self.output_clf = nn.Softmax
            self.output_clf_kwargs = {'dim': 1}

        self.stack_blocks_1 = StackedConvLayers(
            in_channels=self.INPUT_CHANNELS,
            out_channels=32,
            num_layers=self.STACKED_LAYERS_NUM,
            conv_ly=self.CONV_LY,
            nonlinear_ly=self.NONELINEAR_LY,
            norm_ly=self.NORM_LY
        )
        self.maxpool_1 = nn.MaxPool2d(2, 2, 0)
        self.stack_blocks_2 = StackedConvLayers(
            in_channels=32,
            out_channels=32*2,
            num_layers=self.STACKED_LAYERS_NUM,
            conv_ly=self.CONV_LY,
            nonlinear_ly=self.NONELINEAR_LY,
            norm_ly=self.NORM_LY
        )
        self.maxpool_2 = nn.MaxPool2d(2, 2, 0)
        self.stack_blocks_3 = StackedConvLayers(
            in_channels=32*2,
            out_channels=32*4,
            num_layers=self.STACKED_LAYERS_NUM,
            conv_ly=self.CONV_LY,
            nonlinear_ly=self.NONELINEAR_LY,
            norm_ly=self.NORM_LY
        )
        self.maxpool_3 = nn.MaxPool2d(2, 2, 0)
        self.stack_blocks_4 = StackedConvLayers(
            in_channels=32*4,
            out_channels=32*8,
            num_layers=self.STACKED_LAYERS_NUM,
            conv_ly=self.CONV_LY,
            nonlinear_ly=self.NONELINEAR_LY,
            norm_ly=self.NORM_LY
        )
        self.maxpool_4 = nn.MaxPool2d(2, 2, 0)
        self.stack_blocks_5 = StackedConvLayers(
            in_channels=32*8,
            out_channels=32*16,
            num_layers=self.STACKED_LAYERS_NUM,
            conv_ly=self.CONV_LY,
            nonlinear_ly=self.NONELINEAR_LY,
            norm_ly=self.NORM_LY,
            dropout=0.5
        )
        self.upsampling_1 = Upsampling(in_channels=32*16, out_channels=32*8, scale_factor=2, mode='bilinear')
        self.stack_blocks_6 = StackedConvLayers(
            in_channels=32*16,
            out_channels=32*8,
            num_layers=self.STACKED_LAYERS_NUM,
            conv_ly=self.CONV_LY,
            nonlinear_ly=self.NONELINEAR_LY,
            norm_ly=self.NORM_LY
        )
        self.upsampling_2 = Upsampling(in_channels=32*8, out_channels=32*4, scale_factor=2, mode='bilinear')
        self.stack_blocks_7 = StackedConvLayers(
            in_channels=32*8,
            out_channels=32*4,
            num_layers=self.STACKED_LAYERS_NUM,
            conv_ly=self.CONV_LY,
            nonlinear_ly=self.NONELINEAR_LY,
            norm_ly=self.NORM_LY
        )
        self.upsampling_3 = Upsampling(in_channels=32*4, out_channels=32*2, scale_factor=2, mode='bilinear')
        self.stack_blocks_8 = StackedConvLayers(
            in_channels=32*4,
            out_channels=32*2,
            num_layers=self.STACKED_LAYERS_NUM,
            conv_ly=self.CONV_LY,
            nonlinear_ly=self.NONELINEAR_LY,
            norm_ly=self.NORM_LY
        )
        self.upsampling_4 = Upsampling(in_channels=32*2, out_channels=32,scale_factor=2, mode='bilinear')
        self.stack_blocks_9 = StackedConvLayers(
            in_channels=32*2,
            out_channels=32,
            num_layers=self.STACKED_LAYERS_NUM,
            conv_ly=self.CONV_LY,
            nonlinear_ly=self.NONELINEAR_LY,
            norm_ly=self.NORM_LY
        )
        self.output_layer = ConvNormNonlinearDropout(
            in_channels=32,
            out_channels=self.num_classes,
            conv_ly=self.CONV_LY,
            conv_kwargs={
                'kernel_size': 1,
                'stride': 1,
                'padding': 0
            },
            nonlinear_ly=self.output_clf,
            nonlinear_kwargs=self.output_clf_kwargs
        )

    def forward(self, x_input):
        # down sampling
        x_0_0 = self.stack_blocks_1(x_input)
        x_1_0 = self.stack_blocks_2(self.maxpool_1(x_0_0))
        x_2_0 = self.stack_blocks_3(self.maxpool_2(x_1_0))
        x_3_0 = self.stack_blocks_4(self.maxpool_3(x_2_0))
        x_4_0 = self.stack_blocks_5(self.maxpool_4(x_3_0))

        # concatenate
        x_3_1 = self.stack_blocks_6(torch.cat((F.adaptive_avg_pool2d(self.upsampling_1(x_4_0), (64, 64)), x_3_0), dim=1))
        x_2_1 = self.stack_blocks_7(torch.cat((F.adaptive_avg_pool2d(self.upsampling_2(x_3_1), (128, 128)), x_2_0), dim=1))
        x_1_1 = self.stack_blocks_8(torch.cat((F.adaptive_avg_pool2d(self.upsampling_3(x_2_1), (256, 256)), x_1_0), dim=1))
        x_0_1 = self.stack_blocks_9(torch.cat((F.adaptive_avg_pool2d(self.upsampling_4(x_1_1), (512,512)), x_0_0), dim=1))

        seg_output = self.output_layer(x_0_1)
        return seg_output


def parameters_sum(models: Module):
    return sum(p.numel() for p in models.parameters() if p.requires_grad)


if __name__ == '__main__':
    from datasets.datasets import CellNucleiDataset, restorefig
    import pandas as pd
    from torchvision.transforms import v2
    # model = StackedConvLayers(3, 64, 2)
    model = UNet(2)
    print(model)
    print(parameters_sum(model))

    # 实际图片测试
    train_df = pd.read_csv('../datasets/stage1_train_labels.csv')
    img_transform = v2.Compose([
        v2.Resize(size=(512, 512), antialias=True),
        # v2.RandomHorizontalFlip(p=0.5),  # 随机水平翻转
        v2.ToDtype(torch.float32, scale=True)
    ])
    train_dataset = CellNucleiDataset(train_df, '../datasets/stage1_train', transform=img_transform)
    x = train_dataset[0][0]
    output = model(x.unsqueeze(0))
    print(output.shape)

