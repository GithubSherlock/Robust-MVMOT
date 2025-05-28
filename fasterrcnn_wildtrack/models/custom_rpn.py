import torch
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple
from torch import nn, Tensor
from torchvision.models.detection.rpn import RPNHead, RegionProposalNetwork, AnchorGenerator


class CustomRPNHead(RPNHead):
    def forward(self, x: List[Tensor], mask: Optional[Tensor] = None) -> Tuple[List[Tensor], List[Tensor]]:
        # 原始RPN head的前向传播
        logits, bbox_reg = super().forward(x)
        if mask is not None:
            # 对每个特征层的预测应用掩码
            for i in range(len(logits)):
                # 将掩码调整到对应特征图大小
                feat_mask = F.interpolate(mask[None,None], size=logits[i].shape[-2:], mode='nearest')
                # 应用掩码
                logits[i] = logits[i] * feat_mask
                bbox_reg[i] = bbox_reg[i] * feat_mask
        return logits, bbox_reg

class CustomRegionProposalNetwork(RegionProposalNetwork):
    def filter_anchors(self, anchors: List[Tensor], mask: Tensor) -> List[Tensor]:
        filtered_anchors = []
        for anchors_per_image in anchors:
            # 计算锚框的下中心点
            bottom_center_x = (anchors_per_image[:, 0] + anchors_per_image[:, 2]) / 2
            bottom_center_y = anchors_per_image[:, 3]
            # 转换为掩码坐标系
            h, w = mask.shape[-2:]
            x_idx = (bottom_center_x * w).long().clamp(0, w-1)
            y_idx = (bottom_center_y * h).long().clamp(0, h-1)
            # 获取有效锚框
            valid_anchors = mask[0, 0, y_idx, x_idx] > 0
            filtered_anchors.append(anchors_per_image[valid_anchors])
        return filtered_anchors

    def filter_proposals(self, anchors, objectness, pred_bbox_deltas, image_shapes, masks=None):
        """补充proposals过滤方法"""
        proposals = self.box_coder.decode(pred_bbox_deltas, anchors)
        proposals = proposals.view(-1, 4)
        proposals = super().filter_proposals(proposals, objectness, image_shapes)
        
        if masks is not None:
            # 过滤掩码区域外的proposals
            valid_proposals = []
            valid_scores = []
            for props, scores_per_image, mask in zip(proposals, scores, masks):
                bottom_center_x = (props[:, 0] + props[:, 2]) / 2
                bottom_center_y = props[:, 3]
                
                h, w = mask.shape[-2:]
                x_idx = (bottom_center_x * w).long().clamp(0, w-1)
                y_idx = (bottom_center_y * h).long().clamp(0, h-1)
                
                valid_mask = mask[0, 0, y_idx, x_idx] > 0
                
                valid_proposals.append(props[valid_mask])
                valid_scores.append(scores_per_image[valid_mask])
                
            proposals = valid_proposals
            scores = valid_scores
        
        return proposals

    def forward(self, images, features, targets=None, masks=None):
        if masks is not None:
            assert len(masks) == len(images.tensors), "Masks batch size must match images"
        for mask in masks:
            assert mask.dim() == 4, "Mask must be 4D tensor (B,C,H,W)"
        features = list(features.values())
        objectness, pred_bbox_deltas = self.head(features, masks)
        anchors = self.anchor_generator(images, features)
        
        if masks is not None:
            # 过滤无效区域的锚框
            anchors = self.filter_anchors(anchors, masks)
        if self.training:
            assert targets is not None
            labels, matched_gt_boxes = self.assign_targets_to_anchors(anchors, targets) # 计算分类和回归目标
            regression_targets = self.box_coder.encode(matched_gt_boxes, anchors)
            loss_objectness, loss_rpn_box_reg = self.compute_loss( # 应用掩码过滤后计算损失
                objectness, pred_bbox_deltas, labels, regression_targets, masks)
            losses = {
                "loss_objectness": loss_objectness,
                "loss_rpn_box_reg": loss_rpn_box_reg}       
        else:
            losses = {}
        proposals = self.filter_proposals( # 生成提议框
            anchors, objectness, pred_bbox_deltas, 
            images.image_sizes, masks)
        return proposals, losses
        
    def compute_loss(self, objectness, pred_bbox_deltas, labels, regression_targets, masks):
        """计算RPN损失，考虑掩码过滤"""
        sampled_pos_inds, sampled_neg_inds = self.fg_bg_sampler(labels)
        # 合并正负样本索引
        sampled_inds = []
        for pos_inds, neg_inds in zip(sampled_pos_inds, sampled_neg_inds):
            sampled_inds.append(torch.where(pos_inds | neg_inds)[0])
        # 过滤并计算分类损失
        objectness_flattened = []
        labels_flattened = []
        
        for objectness_per_level, labels_per_level, sampled_inds_per_level in zip(
            objectness, labels, sampled_inds):
            objectness_flattened.append(objectness_per_level[sampled_inds_per_level])
            labels_flattened.append(labels_per_level[sampled_inds_per_level])
            
        objectness_cat = torch.cat(objectness_flattened, dim=0)
        labels_cat = torch.cat(labels_flattened, dim=0)
        
        loss_objectness = F.binary_cross_entropy_with_logits(objectness_cat, labels_cat) # 计算分类损失
        sampled_pos_inds_cat = torch.cat([torch.where(pos_inds)[0] for pos_inds in sampled_pos_inds]) # 计算回归损失
        
        if sampled_pos_inds_cat.numel() > 0:
            pred_bbox_deltas_cat = torch.cat([t[pos_inds] for t, pos_inds in zip(pred_bbox_deltas, sampled_pos_inds)], dim=0)
            regression_targets_cat = torch.cat([t[pos_inds] for t, pos_inds in zip(regression_targets, sampled_pos_inds)], dim=0)
            
            loss_rpn_box_reg = F.smooth_l1_loss(
                pred_bbox_deltas_cat,
                regression_targets_cat,
                beta=1.0 / 9.0,
                reduction="sum"
            ) / max(1, sampled_pos_inds_cat.numel())
        else:
            loss_rpn_box_reg = pred_bbox_deltas.sum() * 0 
        return loss_objectness, loss_rpn_box_reg