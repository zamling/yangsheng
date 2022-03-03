import sys

import torch
import torch.nn as nn


class LossBuilder(nn.Module):
    '''
    Dice Loss on `mask`,
    Dice Loss on `edge`,
    image-scale BCE loss on `cls`.
    Note: The meaning of inputs can be figured out in `SegDetectorLossBuilder`.
    '''

    def __init__(self, eps=1e-6, mask_scale=0.16, edge_scale=0.8, cls_scale=0.04):
        super(LossBuilder, self).__init__()
        from .dice_loss import BinaryDiceLoss
        self.dice_loss = BinaryDiceLoss()
        from.balance_cross_entropy_loss import ImageCrossEntropyLoss
        self.bce_loss = ImageCrossEntropyLoss()

        self.mask_scale = mask_scale
        self.edge_scale = edge_scale
        self.cls_scale = cls_scale

    def forward(self, pred, batch):
        '''

        :param pred: dict, both shape:(N,1,H,W)
        :param batch: dict,  both shape:(N,1,H,W)
        :return:
        '''
        tgt_mask = batch['mask']
        tgt_edge = batch['edge']
        tgt_label = batch['label']

        pred_mask = pred['mask']
        pred_edge = pred['edge']
        pred_label = pred_mask.flatten(1)
        pred_label,_ = torch.max(pred_label,dim=1)
        tgt_label = tgt_label.type_as(pred_label)
        bce_loss = self.bce_loss(pred_label,tgt_label)
        metrics = dict(cls_loss=bce_loss)

        edge_loss = self.dice_loss(pred_edge, tgt_edge)
        metrics['edge_loss'] = edge_loss

        mask_loss = self.dice_loss(pred_mask,tgt_mask)
        metrics['mask_loss'] = mask_loss
        loss = self.cls_scale * bce_loss + self.edge_scale * edge_loss + self.mask_scale * mask_loss

        return loss, metrics

class LossBuilderV2(nn.Module):
    '''
    Dice Loss on `mask`,
    Dice Loss on `edge`,
    image-scale BCE loss on `cls`.
    Note: The meaning of inputs can be figured out in `SegDetectorLossBuilder`.
    '''

    def __init__(self, eps=1e-6, mask_scale=4, edge_scale=10, cls_scale=5):
        super(LossBuilderV2, self).__init__()
        from .dice_loss import BinaryDiceLoss
        self.dice_loss = BinaryDiceLoss(reduction='none')
        from.balance_cross_entropy_loss import ImageCrossEntropyLoss
        self.bce_loss = ImageCrossEntropyLoss()

        self.mask_scale = mask_scale
        self.edge_scale = edge_scale
        self.cls_scale = cls_scale

    def forward(self, pred, batch):
        '''

        :param pred: dict, both shape:(N,1,H,W)
        :param batch: dict,  both shape:(N,1,H,W)
        :return:
        '''
        tgt_mask = batch['mask']
        tgt_edge = batch['edge']
        tgt_label = batch['label']

        pred_mask = pred['mask']
        pred_edge = pred['edge']
        pred_label = pred_mask.flatten(1)
        pred_label = torch.max(pred_label,dim=1)
        bce_loss = self.bce_loss(pred_label,tgt_label)
        metrics = dict(cls_loss=bce_loss)

        edge_loss = self.dice_loss(pred_edge, tgt_edge)
        edge_loss = torch.mul(edge_loss,tgt_label).mean()
        metrics['edge_loss'] = edge_loss

        mask_loss = self.dice_loss(pred_mask,tgt_mask)

        metrics['mask_loss'] = torch.mul(mask_loss,tgt_label).mean()
        loss = self.cls_scale * bce_loss + self.edge_scale * edge_loss + self.mask_scale * mask_loss

        return loss, metrics


