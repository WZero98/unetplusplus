import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from torch.nn import Module
from typing import Optional, Tuple, Literal, List
from collections import OrderedDict
from copy import deepcopy
from torch import Tensor


def compute_iou(inputs, targets, eps=1e-7, cutoff=0.5):
    """
    :param inputs: shape [batch_size, num_classes, height, width] for multi-class targets
    :param targets: shape [batch_size, num_classes, height, width]
    :param eps:
    :param cutoff: 阈值
    :return:
    """
    # binarization
    inputs = (inputs > cutoff).float()  # binarization

    intersection = (inputs * targets).sum(dim=(2, 3))
    union = inputs.sum(dim=(2, 3)) + targets.sum(dim=(2, 3)) - intersection
    iou = (intersection + eps) / (union + eps)  # shape [batch_size, num_classes]
    return iou.mean().item()
