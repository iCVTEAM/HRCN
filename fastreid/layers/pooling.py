# encoding: utf-8
"""
@author:  l1aoxingyu
@contact: sherlockliao01@gmail.com
"""

import torch
import torch.nn.functional as F
from torch import nn


class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)


class GeneralizedMeanPooling(nn.Module):
    r"""Applies a 2D power-average adaptive pooling over an input signal composed of several input planes.
    The function computed is: :math:`f(X) = pow(sum(pow(X, p)), 1/p)`
        - At p = infinity, one gets Max Pooling
        - At p = 1, one gets Average Pooling
    The output is of size H x W, for any input size.
    The number of output features is equal to the number of input planes.
    Args:
        output_size: the target output size of the image of the form H x W.
                     Can be a tuple (H, W) or a single H for a square image H x H
                     H and W can be either a ``int``, or ``None`` which means the size will
                     be the same as that of the input.
    """

    def __init__(self, norm, output_size=1, eps=1e-6):
        super(GeneralizedMeanPooling, self).__init__()
        assert norm > 0
        self.p = float(norm)
        self.output_size = output_size
        self.eps = eps

    def forward(self, x):
        x = x.clamp(min=self.eps).pow(self.p)
        return torch.nn.functional.adaptive_avg_pool2d(x, self.output_size).pow(1. / self.p)

    def __repr__(self):
        return self.__class__.__name__ + '(' \
               + str(self.p) + ', ' \
               + 'output_size=' + str(self.output_size) + ')'


class GeneralizedMeanPoolingP(GeneralizedMeanPooling):
    """ Same, but norm is trainable
    """

    def __init__(self, norm=3, output_size=1, eps=1e-6):
        super(GeneralizedMeanPoolingP, self).__init__(norm, output_size, eps)
        self.p = nn.Parameter(torch.ones(1) * norm)


class AdaptiveAvgMaxPool2d(nn.Module):
    def __init__(self, output_size=1):
        super(AdaptiveAvgMaxPool2d, self).__init__()
        self.avgpool = FastGlobalAvgPool2d()

    def forward(self, x):
        x_avg = self.avgpool(x)
        x_max = F.adaptive_max_pool2d(x, 1)
        x = x_max + x_avg
        return x


class FastGlobalAvgPool2d(nn.Module):
    def __init__(self, flatten=False):
        super(FastGlobalAvgPool2d, self).__init__()
        self.flatten = flatten

    def forward(self, x):
        if self.flatten:
            in_size = x.size()
            return x.view((in_size[0], in_size[1], -1)).mean(dim=2)
        else:
            return x.view(x.size(0), x.size(1), -1).mean(-1).view(x.size(0), x.size(1), 1, 1)


class CenterPool(nn.Module):
    def __init__(self, part_num=4, concat=False, device='cpu', pool_type='circle', keep_first=True, reweight=True):
        super(CenterPool, self).__init__()
        self.part_num = part_num
        self.concat = concat
        self.device = device
        self.pool_type = pool_type
        self.keep_first = keep_first
        self.reweighting = [0.2 for _ in range(part_num)] + [1.0] if reweight else None
        self.adj_mat = []

        for i in range(self.part_num + 1):
            if i == 0:
                self.adj_mat.append([0] + [1 for _ in range(self.part_num)])
            else:
                self.adj_mat.append([1] + [1 if j == i or j == i - 2 else 0 for j in range(self.part_num)])

        self.adj_mat = torch.FloatTensor(self.adj_mat)

    def features_reweighting(self, masks, total_pixels):
        for i, mask in enumerate(masks):
            effective_pixels = torch.sum(mask, [1, 0])
            self.reweighting[i] = effective_pixels.item() / total_pixels

    def forward(self, x):
        B, C, H, W = x.shape

        if self.pool_type == 'horizontal':
            new_x = x.clone()
            features = [new_x[:, :, :, W * (i + 0) // self.part_num:W * (i + 1) // self.part_num].to(self.device) for
                        i in range(self.part_num)] + [x.clone()]
            return torch.cat(features, dim=1) if self.concat else features
        elif self.pool_type == 'vertical':
            new_x = x.clone()
            features = [new_x[:, :, H * (i + 0) // self.part_num:H * (i + 1) // self.part_num, :].to(self.device) for
                        i in range(self.part_num)] + [x.clone()]
            return torch.cat(features, dim=1) if self.concat else features
        elif self.pool_type == 'grid':
            new_part_num = self.part_num // 2
            new_x = x.clone()
            features = []
            for i in range(new_part_num):
                features.extend([new_x[:, :, H * (i + 0) // new_part_num:H * (i + 1) // new_part_num,
                                            W * (j + 0) // new_part_num:W * (j + 1) // new_part_num].to(self.device)
                                            for j in range(new_part_num)])
            features.append(x.clone())
            return torch.cat(features, dim=1) if self.concat else features

        max_radiu = min(H, W) / 2
        min_radiu = max_radiu / self.part_num
        center_x = (W - 1) / 2
        center_y = (H - 1) / 2

        index_H = torch.arange(0, H, requires_grad=False).expand(W, H).permute(1, 0).contiguous()
        index_W = torch.arange(0, W, requires_grad=False).expand(H, W)
        one_mat = torch.ones(H, W, requires_grad=False)
        zero_mat = torch.zeros(H, W, requires_grad=False)

        if self.pool_type == 'circle':
            circle_masks = [
                torch.where((index_H - center_y).pow(2) + (index_W - center_x).pow(2) <= pow(r * min_radiu, 2),
                            one_mat, zero_mat) for r in range(1, self.part_num + 1)]

            if self.reweighting is not None:
                self.features_reweighting(circle_masks, H * W)

            if self.keep_first:
                features = [x.clone() * circle_masks[i].to(self.device) for i in range(self.part_num)] + [x.clone()]
            else:
                features = [x.clone() * circle_masks[i].to(self.device) for i in range(1, self.part_num)] + [x.clone()]

        elif self.pool_type == 'square':
            square_masks = [
                torch.where((((index_H - center_y).pow(2) <= pow((H * r) // (2 * self.part_num), 2))
                             & ((index_W - center_x).pow(2) <= pow((W * r) // (2 * self.part_num), 2))),
                            one_mat, zero_mat) for r in range(1, self.part_num + 1)]

            if self.reweighting is not None:
                self.features_reweighting(square_masks, H * W)
            if self.keep_first:
                features = [x.clone() * square_masks[i].to(self.device) for i in range(self.part_num)] + [x.clone()]
            else:
                features = [x.clone() * square_masks[i].to(self.device) for i in range(1, self.part_num)] + [x.clone()]
        else:
            ring_masks = [
                torch.where((index_H - center_y).pow(2) + (index_W - center_x).pow(2) <= pow(r * min_radiu, 2),
                            one_mat, zero_mat) for r in range(1, self.part_num + 1)]
            if self.keep_first:
                ring_masks = [ring_masks[0]] + [ring_masks[i + 1] - ring_masks[i] for i in range(self.part_num - 1)]
            else:
                ring_masks = [ring_masks[1]] + [ring_masks[i + 1] - ring_masks[i] for i in range(1, self.part_num - 1)]

            if self.reweighting is not None:
                self.features_reweighting(ring_masks, H * W)

            if self.keep_first:
                features = [x.clone() * ring_masks[i].to(self.device) for i in range(self.part_num)] + [x.clone()]
            else:
                features = [x.clone() * ring_masks[i].to(self.device) for i in range(1, self.part_num)] + [x.clone()]

        return torch.cat(features, dim=1) if self.concat else features
