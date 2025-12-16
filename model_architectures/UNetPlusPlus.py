import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Module
from typing import Optional, Tuple, Literal
from collections import OrderedDict
from .BaseBlocks import ConvNormNonlinearDropout, Upsampling, StackedConvLayers


class UNetPlusPlus(Module):
    """
    U-net++ architecture
    """
    # class variables
    INPUT_CHANNELS: int = 3
    STACKED_LAYERS_NUM: int = 2
    CONV_LY = nn.Conv2d
    NONELINEAR_LY = nn.ReLU
    NORM_LY = nn.BatchNorm2d

    def __init__(
            self,
            num_classes: int = 2,
            deep_supervision: bool = True
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

        self.deep_supervision = deep_supervision

        self.x_0_0_blocks = StackedConvLayers(
            in_channels=self.INPUT_CHANNELS,
            out_channels=32,
            num_layers=self.STACKED_LAYERS_NUM,
            conv_ly=self.CONV_LY,
            nonlinear_ly=self.NONELINEAR_LY,
            norm_ly=self.NORM_LY
        )
        self.x_1_0_blocks = StackedConvLayers(
            in_channels=32,
            out_channels=64,
            num_layers=self.STACKED_LAYERS_NUM,
            conv_ly=self.CONV_LY,
            nonlinear_ly=self.NONELINEAR_LY,
            norm_ly=self.NORM_LY
        )
        self.x_2_0_blocks = StackedConvLayers(
            in_channels=64,
            out_channels=128,
            num_layers=self.STACKED_LAYERS_NUM,
            conv_ly=self.CONV_LY,
            nonlinear_ly=self.NONELINEAR_LY,
            norm_ly=self.NORM_LY
        )
        self.x_3_0_blocks = StackedConvLayers(
            in_channels=128,
            out_channels=256,
            num_layers=self.STACKED_LAYERS_NUM,
            conv_ly=self.CONV_LY,
            nonlinear_ly=self.NONELINEAR_LY,
            norm_ly=self.NORM_LY
        )
        self.x_4_0_blocks = StackedConvLayers(
            in_channels=256,
            out_channels=512,
            num_layers=self.STACKED_LAYERS_NUM,
            conv_ly=self.CONV_LY,
            nonlinear_ly=self.NONELINEAR_LY,
            norm_ly=self.NORM_LY
        )
        self.x_0_1_blocks = StackedConvLayers(
            in_channels=32 * 2,
            out_channels=32,
            num_layers=self.STACKED_LAYERS_NUM,
            conv_ly=self.CONV_LY,
            nonlinear_ly=self.NONELINEAR_LY,
            norm_ly=self.NORM_LY
        )
        self.x_0_2_blocks = StackedConvLayers(
            in_channels=32 * 3,
            out_channels=32,
            num_layers=self.STACKED_LAYERS_NUM,
            conv_ly=self.CONV_LY,
            nonlinear_ly=self.NONELINEAR_LY,
            norm_ly=self.NORM_LY
        )
        self.x_0_3_blocks = StackedConvLayers(
            in_channels=32 * 4,
            out_channels=32,
            num_layers=self.STACKED_LAYERS_NUM,
            conv_ly=self.CONV_LY,
            nonlinear_ly=self.NONELINEAR_LY,
            norm_ly=self.NORM_LY
        )
        self.x_0_4_blocks = StackedConvLayers(
            in_channels=32 * 5,
            out_channels=32,
            num_layers=self.STACKED_LAYERS_NUM,
            conv_ly=self.CONV_LY,
            nonlinear_ly=self.NONELINEAR_LY,
            norm_ly=self.NORM_LY
        )

        self.x_1_1_blocks = StackedConvLayers(
            in_channels=64 * 2,
            out_channels=64,
            num_layers=self.STACKED_LAYERS_NUM,
            conv_ly=self.CONV_LY,
            nonlinear_ly=self.NONELINEAR_LY,
            norm_ly=self.NORM_LY
        )
        self.x_1_2_blocks = StackedConvLayers(
            in_channels=64 * 3,
            out_channels=64,
            num_layers=self.STACKED_LAYERS_NUM,
            conv_ly=self.CONV_LY,
            nonlinear_ly=self.NONELINEAR_LY,
            norm_ly=self.NORM_LY
        )
        self.x_1_3_blocks = StackedConvLayers(
            in_channels=64 * 4,
            out_channels=64,
            num_layers=self.STACKED_LAYERS_NUM,
            conv_ly=self.CONV_LY,
            nonlinear_ly=self.NONELINEAR_LY,
            norm_ly=self.NORM_LY
        )
        self.x_2_1_blocks = StackedConvLayers(
            in_channels=128 * 2,
            out_channels=128,
            num_layers=self.STACKED_LAYERS_NUM,
            conv_ly=self.CONV_LY,
            nonlinear_ly=self.NONELINEAR_LY,
            norm_ly=self.NORM_LY
        )
        self.x_2_2_blocks = StackedConvLayers(
            in_channels=128 * 3,
            out_channels=128,
            num_layers=self.STACKED_LAYERS_NUM,
            conv_ly=self.CONV_LY,
            nonlinear_ly=self.NONELINEAR_LY,
            norm_ly=self.NORM_LY
        )
        self.x_3_1_blocks = StackedConvLayers(
            in_channels=256 * 2,
            out_channels=256,
            num_layers=self.STACKED_LAYERS_NUM,
            conv_ly=self.CONV_LY,
            nonlinear_ly=self.NONELINEAR_LY,
            norm_ly=self.NORM_LY
        )

        self.downsampling_blocks = nn.ModuleDict({
            'x_0_0_to_x_1_0': nn.MaxPool2d(2, 2, 0),
            'x_1_0_to_x_2_0': nn.MaxPool2d(2, 2, 0),
            'x_2_0_to_x_3_0': nn.MaxPool2d(2, 2, 0),
            'x_3_0_to_x_4_0': nn.MaxPool2d(2, 2, 0)
        })

        self.upsampling_blocks = nn.ModuleDict({
            'x_1_0_to_x_0_1': Upsampling(64, 32, 3, 1, 1, scale_factor=2, mode='bilinear'),
            'x_2_0_to_x_1_1': Upsampling(128, 64, 3, 1, 1, scale_factor=2, mode='bilinear'),
            'x_3_0_to_x_2_1': Upsampling(256, 128, 3, 1, 1, scale_factor=2, mode='bilinear'),
            'x_4_0_to_x_3_1': Upsampling(512, 256, 3, 1, 1, scale_factor=2, mode='bilinear'),
            'x_1_1_to_x_0_2': Upsampling(64, 32, 3, 1, 1, scale_factor=2, mode='bilinear'),
            'x_2_1_to_x_1_2': Upsampling(128, 64, 3, 1, 1, scale_factor=2, mode='bilinear'),
            'x_3_1_to_x_2_2': Upsampling(256, 128, 3, 1, 1, scale_factor=2, mode='bilinear'),
            'x_1_2_to_x_0_3': Upsampling(64, 32, 3, 1, 1, scale_factor=2, mode='bilinear'),
            'x_2_2_to_x_1_3': Upsampling(128, 64, 3, 1, 1, scale_factor=2, mode='bilinear'),
            'x_1_3_to_x_0_4': Upsampling(64, 32, 3, 1, 1, scale_factor=2, mode='bilinear'),
        })
        self.clf_blocks = nn.ModuleDict({
            'x_0_1_clf':
            ConvNormNonlinearDropout(
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
            ),
            'x_0_2_clf':
            ConvNormNonlinearDropout(
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
            ),
            'x_0_3_clf':
            ConvNormNonlinearDropout(
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
            ),
            'x_0_4_clf':
            ConvNormNonlinearDropout(
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
        })

    def forward(self, x_input):
        x_0_0 = self.x_0_0_blocks(x_input)
        x_1_0 = self.x_1_0_blocks(self.downsampling_blocks['x_0_0_to_x_1_0'](x_0_0))
        x_2_0 = self.x_2_0_blocks(self.downsampling_blocks['x_1_0_to_x_2_0'](x_1_0))
        x_3_0 = self.x_3_0_blocks(self.downsampling_blocks['x_2_0_to_x_3_0'](x_2_0))
        x_4_0 = self.x_4_0_blocks(self.downsampling_blocks['x_3_0_to_x_4_0'](x_3_0))

        x_0_1 = self.x_0_1_blocks(torch.cat((x_0_0, self.upsampling_blocks['x_1_0_to_x_0_1'](x_1_0)), dim=1))

        x_1_1 = self.x_1_1_blocks(torch.cat((x_1_0, self.upsampling_blocks['x_2_0_to_x_1_1'](x_2_0)), dim=1))
        x_0_2 = self.x_0_2_blocks(torch.cat((x_0_0, x_0_1, self.upsampling_blocks['x_1_1_to_x_0_2'](x_1_1)), dim=1))

        x_2_1 = self.x_2_1_blocks(torch.cat((x_2_0, self.upsampling_blocks['x_3_0_to_x_2_1'](x_3_0)), dim=1))
        x_1_2 = self.x_1_2_blocks(torch.cat((x_1_0, x_1_1, self.upsampling_blocks['x_2_1_to_x_1_2'](x_2_1)), dim=1))
        x_0_3 = self.x_0_3_blocks(
            torch.cat((x_0_0, x_0_1, x_0_2, self.upsampling_blocks['x_1_2_to_x_0_3'](x_1_2)), dim=1))

        x_3_1 = self.x_3_1_blocks(torch.cat((x_3_0, self.upsampling_blocks['x_4_0_to_x_3_1'](x_4_0)), dim=1))
        x_2_2 = self.x_2_2_blocks(torch.cat((x_2_0, x_2_1, self.upsampling_blocks['x_3_1_to_x_2_2'](x_3_1)), dim=1))
        x_1_3 = self.x_1_3_blocks(
            torch.cat((x_1_0, x_1_1, x_1_2, self.upsampling_blocks['x_2_2_to_x_1_3'](x_2_2)), dim=1))
        x_0_4 = self.x_0_4_blocks(
            torch.cat((x_0_0, x_0_1, x_0_2, x_0_3, self.upsampling_blocks['x_1_3_to_x_0_4'](x_1_3)), dim=1))

        if self.deep_supervision:
            output_t = (
                self.clf_blocks['x_0_4_clf'](x_0_4),
                self.clf_blocks['x_0_3_clf'](x_0_3),
                self.clf_blocks['x_0_2_clf'](x_0_2),
                self.clf_blocks['x_0_1_clf'](x_0_1)
            )
        else:
            output_t = self.clf_blocks['x_0_4_clf'](x_0_4)
        return output_t


def parameters_sum(models: Module):
    return sum(p.numel() for p in models.parameters() if p.requires_grad)


if __name__ == '__main__':
    from datasets.datasets import CellNucleiDataset, restorefig
    import pandas as pd
    from torchvision.transforms import v2

    # model = StackedConvLayers(3, 64, 2)
    model = UNetPlusPlus(2, True)
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
    print([output[i].shape for i in range(4)])
