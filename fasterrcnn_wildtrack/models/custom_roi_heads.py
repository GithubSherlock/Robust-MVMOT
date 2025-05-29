import torch
import torch.nn.functional as F
import logging
from torchvision.models.detection.roi_heads import RoIHeads


def fastrcnn_loss(class_logits, box_regression, labels, regression_targets):
    """Computing Classification and Regression Losses for Fast R-CNN"""
    labels = torch.cat(labels, dim=0)
    regression_targets = torch.cat(regression_targets, dim=0)
    classification_loss = F.cross_entropy(class_logits, labels) # Classification Loss
    sampled_pos_inds_subset = torch.where(labels > 0)[0] # Regression loss - calculated for positive samples (foreground) only
    if sampled_pos_inds_subset.numel() > 0:
        box_regression = box_regression[sampled_pos_inds_subset]
        regression_targets = regression_targets[sampled_pos_inds_subset]
        
        box_loss = F.smooth_l1_loss( # Use Smoothing L1 Loss
            box_regression,
            regression_targets,
            beta=1.0 / 9.0,
            reduction="sum",
        ) / sampled_pos_inds_subset.numel()  # Normalize by the number of positive samples
    else:
        box_loss = box_regression.sum() * 0
    return classification_loss, box_loss

class CustomRoIHeads(RoIHeads):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    def forward(self, features, proposals, image_shapes, targets=None):
        """Modified RoI processing method to get masks from targets"""
        if self.training:
            proposals, matched_idxs, labels, regression_targets = self.select_training_samples(proposals, targets)
            if targets is not None and all("mask" in t for t in targets): # Getting tasks from targets
                masks = [t["mask"] for t in targets]
                
                valid_proposals = []
                valid_matched_idxs = []
                valid_labels = []
                valid_regression_targets = []
                
                for props, mask, match_ids, labs, reg_targets in zip(
                    proposals, masks, matched_idxs, labels, regression_targets):
                    
                    bottom_center_x = (props[:, 0] + props[:, 2]) / 2 # Use fit point
                    bottom_center_y = props[:, 3]
                    
                    h, w = mask.shape[-2:] # Convert to mask coordinates
                    x_idx = (bottom_center_x * w).long().clamp(0, w-1)
                    y_idx = (bottom_center_y * h).long().clamp(0, h-1)
                    valid_mask = mask[0, 0, y_idx, x_idx] > 0 # Retention in white areas (mask value 1)
                    if valid_mask.sum() == 0: # Hedge
                        valid_mask[0] = True
                        logging.warning("Warning: No valid proposals after mask filtering, keeping one proposal")

                    valid_proposals.append(props[valid_mask])
                    valid_matched_idxs.append(match_ids[valid_mask])
                    valid_labels.append(labs[valid_mask])
                    valid_regression_targets.append(reg_targets[valid_mask])
                
                proposals = valid_proposals
                matched_idxs = valid_matched_idxs
                labels = valid_labels
                regression_targets = valid_regression_targets

        box_features = self.box_roi_pool(features, proposals, image_shapes) # Extract RoI features
        box_features = self.box_head(box_features)
        class_logits, box_regression = self.box_predictor(box_features)

        if self.training:
            loss_classifier, loss_box_reg = fastrcnn_loss(
                class_logits, box_regression, labels, regression_targets)
            losses = {
                "loss_classifier": loss_classifier,
                "loss_box_reg": loss_box_reg}
            return proposals, losses
        else:
            boxes, scores, labels = self.postprocess_detections(
                class_logits, box_regression, proposals, image_shapes)
            
            if targets is not None and all("mask" in t for t in targets): # Mask filtering in the inference step
                masks = [t["mask"] for t in targets]
                filtered_boxes = []
                filtered_scores = []
                filtered_labels = []
                
                for boxes_per_image, scores_per_image, labels_per_image, mask in zip(
                    boxes, scores, labels, masks):
                    if len(boxes_per_image) > 0:
                        bottom_center_x = (boxes_per_image[:, 0] + boxes_per_image[:, 2]) / 2
                        bottom_center_y = boxes_per_image[:, 3]
                        
                        h, w = mask.shape[-2:]
                        x_idx = (bottom_center_x * w).long().clamp(0, w-1)
                        y_idx = (bottom_center_y * h).long().clamp(0, h-1)
                        
                        valid_mask = mask[0, 0, y_idx, x_idx] > 0
                        
                        filtered_boxes.append(boxes_per_image[valid_mask])
                        filtered_scores.append(scores_per_image[valid_mask])
                        filtered_labels.append(labels_per_image[valid_mask])
                    else:
                        filtered_boxes.append(torch.empty((0, 4), device=boxes_per_image.device))
                        filtered_scores.append(torch.empty(0, device=scores_per_image.device))
                        filtered_labels.append(torch.empty(0, dtype=torch.int64, device=labels_per_image.device))
                
                boxes, scores, labels = filtered_boxes, filtered_scores, filtered_labels

            result = []
            for boxes_per_image, scores_per_image, labels_per_image in zip(boxes, scores, labels):
                result.append({
                    "boxes": boxes_per_image,
                    "scores": scores_per_image,
                    "labels": labels_per_image,})
            return result
