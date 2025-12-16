import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from torch.nn import Module
from typing import Optional, Tuple, Literal, List
from collections import OrderedDict
from copy import deepcopy
from torch import Tensor


class BinaryDiceLoss(nn.Module):
    """
    for binary output
    """
    def __init__(self, smooth: float = 1e-5):
        super().__init__()
        self.smooth = smooth

    def forward(self, inputs: Tensor, targets: Tensor) -> Tensor:
        """
        :param inputs: shape [batch_size, 1, height, width] for binary targets
        :param targets: shape [batch_size, 1, height, width]
        :return: a Tensor shape [batch_size, 1, height, width]
        """
        # 将inputs和targets展平成向量
        inputs = torch.flatten(inputs, 1, 3)
        targets = torch.flatten(targets, 1, 3)

        intersection = 2 * torch.sum(inputs * targets, dim=1) + self.smooth
        pixels = torch.sum(inputs, dim=1) + torch.sum(targets, dim=1) + self.smooth
        dice = intersection/pixels
        dice_loss = (1 - dice).mean()
        return dice_loss


class MulticlassDiceLoss(nn.Module):
    """
    for multi-class output
    """
    def __init__(self, smooth: float = 1e-5, class_weight: List[float] | Tuple[float] | None = None):
        """
        :param smooth:
        :param class_weight: 多分类情形下不同分类的权重，默认相同
        """
        super().__init__()
        self.smooth = smooth
        self.class_weight = class_weight

    def forward(self, inputs: Tensor, targets: Tensor) -> Tensor:
        """
        :param inputs: shape [batch_size, num_classes, height, width] for multi-class targets
        :param targets: shape [batch_size, num_classes, height, width]
        :return: a Tensor shape [batch_size, num_classes, height, width]
        """
        num_classes = inputs.shape[1]
        # 将inputs和targets展平成3维向量 (batch_size, num_classes, height*width)
        inputs = torch.flatten(inputs, 2, 3)
        targets = torch.flatten(targets, 2, 3)

        intersection = 2 * torch.sum(inputs * targets, dim=2) + self.smooth  # shape (batch_size, num_classes)
        pixels = torch.sum(inputs, dim=2) + torch.sum(targets, dim=2) + self.smooth  # shape (batch_size, num_classes)
        dice = intersection/pixels  # shape (batch_size, num_classes)

        if self.class_weight:
            weight = torch.tensor(self.class_weight) / torch.sum(torch.tensor(self.class_weight))
            weight = weight.reshape((num_classes, 1))
            dice_mean_multiclass = dice.matmul(weight)
        else:
            dice_mean_multiclass = dice.mean(dim=1)
        dice_loss = (1 - dice_mean_multiclass).mean()
        return dice_loss


class BinaryCEDice(Module):
    """
    custom loss functions, BCE + Dice for binary targets
    """
    def __init__(
            self,
            batch_weight: Optional[Tensor] = None,
            reduction: Literal['mean', 'sum', 'none'] = 'mean',
            smooth: float = 1e-5,
            combine_weights: Optional[Tensor] | List[float] | Tuple[float] = None,
    ):
        """
        :param batch_weight: BCE parameters
        :param reduction: BCE parameters
        :param smooth: Dice parameters
        :param combine_weights: 组合函数间的权重，默认相等 (Weight bce, Weight dice)
        """
        super().__init__()
        self.batch_weight = batch_weight
        self.reduction = reduction
        self.smooth = smooth
        self.combine_weights = combine_weights

    def forward(self, inputs: Tensor, targets: Tensor) -> Tensor:
        ce = nn.BCELoss(self.batch_weight, reduction=self.reduction)
        dice = BinaryDiceLoss(self.smooth)
        if self.combine_weights:
            combine_weights = torch.tensor(self.combine_weights) / torch.sum(torch.tensor(self.combine_weights))
            combine_loss = ce(inputs, targets) * combine_weights[0] + dice(inputs, targets) * combine_weights[1]
        else:
            combine_loss = ce(inputs, targets) + dice(inputs, targets)
        return combine_loss


class MulticlassCEDice(Module):
    def __init__(
            self,
            weight: Optional[Tensor] = None,
            reduction: Literal['mean', 'sum', 'none'] = 'mean',
            smooth: float = 1e-5,
            class_weight: List[float] | Tuple[float] | None = None,
            combine_weights: Optional[Tensor] | List[float] | Tuple[float] = None
    ):
        """
        :param weight: CE parameters
        :param reduction: CE parameters
        :param smooth: Dice parameters
        :param class_weight: Dice parameters 多分类情形下不同分类的权重，默认相同
        :param combine_weights: 组合函数间的权重，默认相等 (Weight bce, Weight dice)
        """
        super().__init__()
        self.weight = weight
        self.reduction = reduction
        self.smooth = smooth
        self.class_weight = class_weight
        self.combine_weights = combine_weights

    def forward(self, inputs: Tensor, targets: Tensor) -> Tensor:
        ce = nn.CrossEntropyLoss(self.weight, reduction=self.reduction)
        dice = MulticlassDiceLoss(self.smooth, self.class_weight)
        if self.combine_weights:
            combine_weights = torch.tensor(self.combine_weights) / torch.sum(torch.tensor(self.combine_weights))
            combine_loss = ce(inputs, targets) * combine_weights[0] + dice(inputs, targets) * combine_weights[1]
        else:
            combine_loss = ce(inputs, targets) + dice(inputs, targets)
        return combine_loss


if __name__ == '__main__':
    test_input = torch.tensor([
        [
            [[0.9, 0.8, 0.01],
            [0.0, 0.95, 0.02]],
            [[0.9, 0.8, 0.01],
             [0.7, 0.95, 0.02]],
            [[0.9, 0.1, 0.01],
             [0.8, 0.95, 0.02]]
        ],
        [
            [[0.9, 0.5, 0.01],
             [0.9, 0.05, 0.02]],
            [[0.9, 0.5, 0.01],
             [0.7, 0.95, 0.02]],
            [[0.9, 0.1, 0.01],
             [0.8, 0.98, 0.02]]
        ]
    ])

    test_target = torch.tensor([
        [
            [[1, 1, 0],
             [0, 1, 0]],
            [[1, 1, 0],
             [1, 1, 0]],
            [[1, 0, 0],
             [1, 1, 0]]
        ],
        [
            [[1, 1, 0],
             [1, 0, 0]],
            [[1, 1, 0],
             [1, 1, 0]],
            [[1, 0, 0],
             [1, 1, 0]]
        ]
    ], dtype=torch.float)

    # loss_f = MulticlassDiceLoss(class_weight=[8, 1, 1])
    loss_f = MulticlassCEDice(combine_weights=[1, 1.5])
    loss_value = loss_f(test_input, test_target)
    print(loss_value)
    print(test_input.shape)
