# encoding: utf-8
"""
@author:  liaoxingyu
@contact: sherlockliao01@gmail.com
"""

import torch
from torch import nn

from fastreid.layers import *
from fastreid.modeling.backbones import build_backbone
from fastreid.modeling.heads import build_reid_heads
from fastreid.modeling.losses import *
from .build import META_ARCH_REGISTRY


@META_ARCH_REGISTRY.register()
class Baseline(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self._cfg = cfg
        assert len(cfg.MODEL.PIXEL_MEAN) == len(cfg.MODEL.PIXEL_STD)
        self.register_buffer("pixel_mean", torch.tensor(cfg.MODEL.PIXEL_MEAN).view(1, -1, 1, 1))
        self.register_buffer("pixel_std", torch.tensor(cfg.MODEL.PIXEL_STD).view(1, -1, 1, 1))

        # backbone
        self.backbone = build_backbone(cfg)

        # head
        pool_type = cfg.MODEL.HEADS.POOL_LAYER
        if pool_type == 'fastavgpool':
            pool_layer = FastGlobalAvgPool2d()
        elif pool_type == 'avgpool':
            pool_layer = nn.AdaptiveAvgPool2d(1)
        elif pool_type == 'maxpool':
            pool_layer = nn.AdaptiveMaxPool2d(1)
        elif pool_type == 'gempool':
            pool_layer = GeneralizedMeanPoolingP()
        elif pool_type == "avgmaxpool":
            pool_layer = AdaptiveAvgMaxPool2d()
        elif pool_type == "identity":
            pool_layer = nn.Identity()
        else:
            raise KeyError(f"{pool_type} is invalid, please choose from "
                           f"'avgpool', 'maxpool', 'gempool', 'avgmaxpool' and 'identity'.")

        in_feat = cfg.MODEL.HEADS.IN_FEAT
        num_classes = cfg.MODEL.HEADS.NUM_CLASSES
        self.circle_pool_layer = CenterPool(cfg.MODEL.HEADS.CENTER_NUM,
                                            False,
                                            cfg.MODEL.DEVICE,
                                            cfg.MODEL.HEADS.CENTER_POOL_TYPE,
                                            cfg.MODEL.HEADS.KEEP_FIRST)
        self.heads = build_reid_heads(cfg, in_feat, num_classes, pool_layer, cfg.MODEL.HEADS.CENTER_NUM, None)

    @property
    def device(self):
        return self.pixel_mean.device

    def forward(self, batched_inputs):
        images = self.preprocess_image(batched_inputs)
        features, features_list = self.backbone(images)

        if self.training:
            assert "targets" in batched_inputs, "Vehicle ID annotation are missing in training!"
            targets = batched_inputs["targets"].long().to(self.device)

            if targets.sum() < 0: targets.zero_()

            part_features_list = self.circle_pool_layer(features)
            all_features_list = features_list + part_features_list
            total_cls_outputs, total_pred_class_logits, total_features = self.heads(all_features_list, targets)

            return [total_cls_outputs, total_pred_class_logits, total_features], targets

        else:
            part_features_list = self.circle_pool_layer(features)
            all_features_list = features_list + part_features_list
            total_features = self.heads(all_features_list)
            global_features = total_features[-1]

            return global_features

    def preprocess_image(self, batched_inputs):
        """
        Normalize and batch the input images.
        """
        if isinstance(batched_inputs, dict):
            images = batched_inputs["images"].to(self.device)
        elif isinstance(batched_inputs, torch.Tensor):
            images = batched_inputs.to(self.device)
        images.sub_(self.pixel_mean).div_(self.pixel_std)
        return images

    def losses(self, outputs, gt_labels):
        r"""
        Compute loss from modeling's outputs, the loss function input arguments
        must be the same as the outputs of the model forwarding.
        """
        cls_outputs, pred_class_logits, pred_features = outputs
        loss_dict = {}
        loss_names = self._cfg.MODEL.LOSSES.NAME

        # Log prediction accuracy
        for i, pred_class_logit in enumerate(pred_class_logits):
            CrossEntropyLoss.log_accuracy(pred_class_logit.detach(), gt_labels, name='{}'.format(i))

        if "CrossEntropyLoss" in loss_names:
            for i, outputs in enumerate(cls_outputs):
                loss_dict['loss_cls_{}'.format(i)] = CrossEntropyLoss(self._cfg)(outputs, gt_labels)

        if "TripletLoss" in loss_names:
            for i, features in enumerate(pred_features):
                loss_dict['loss_triplet_{}'.format(i)] = TripletLoss(self._cfg)(features, gt_labels)

        return loss_dict
