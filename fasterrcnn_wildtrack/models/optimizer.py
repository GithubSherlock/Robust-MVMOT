import torch
import logging

def yosinski_optimizer(learning_rate, model):
    """
        Learning Rate Strategy for Faster R-CNN with ResNet50-FPN Backbone

        This implementation adopts a scientifically-designed differential learning rate strategy based on several key research findings:

        1. Backbone High-level Features (Layer3, Layer4) - 0.1x base_lr
        - Based on Yosinski et al. (NIPS 2014) finding that higher layers contain more task-specific features
        - Small learning rate preserves useful pre-trained features while allowing minor adaptations
        - Supported by He et al. (CVPR 2016) transfer learning practices

        2. Feature Pyramid Network (FPN) - 0.5x base_lr
        - Following Lin et al. (CVPR 2017) FPN architecture principles
        - Medium learning rate balances feature extraction and adaptation
        - Allows proper adjustment of multi-scale feature fusion capabilities

        3. Detection Heads (RPN & ROI) - 1.0x base_lr
        - Based on Ren et al. (NIPS 2015) Faster R-CNN training strategy
        - Full learning rate for task-specific layers with random initialization
        - Enables rapid adaptation to target detection task

        4. Implementation Details:
        - Base learning rate: 1e-3 (standard for fine-tuning)
        - SGD optimizer with momentum=0.9 and weight_decay=5e-4
        - Automatic parameter grouping based on layer names
        - Integration with layer freezing strategy (requires_grad check)

        References:
        - How transferable are features in deep neural networks? (Yosinski et al., 2014)
        - Feature Pyramid Networks for Object Detection (Lin et al., 2017)
        - Faster R-CNN: Towards Real-Time Object Detection (Ren et al., 2015)
        - Deep Residual Learning for Image Recognition (He et al., 2016)
    """
    params = []
    params.append({'params': [p for n, p in model.named_parameters() 
                            if ('backbone.body.layer3' in n or 'backbone.body.layer4' in n) 
                            and p.requires_grad],'lr': learning_rate * 0.1})
    params.append({'params': [p for n, p in model.named_parameters() 
                            if 'backbone.fpn' in n and p.requires_grad],'lr': learning_rate * 0.5})
    params.append({'params': [p for n, p in model.named_parameters() 
                            if ('rpn' in n or 'roi_heads' in n) and p.requires_grad],'lr': learning_rate})
    params.append({'params': [p for n, p in model.named_parameters() 
                            if not any(x in n for x in ['backbone', 'rpn', 'roi_heads']) 
                            and p.requires_grad],'lr': learning_rate})
    optimizer = torch.optim.SGD(params, momentum=0.9, weight_decay=5e-4)
    logging.info("Parameter groups configuration:")
    for idx, group in enumerate(optimizer.param_groups):
        param_count = sum(p.numel() for p in group['params'])
        logging.info(f"Group {idx}: {param_count} parameters, lr={group['lr']}")
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)
    return optimizer, lr_scheduler

def _get_valid_indices_(boxes, mask):
    if mask.dim() > 2:
        mask = mask.squeeze(0)
    # 计算边界框中心点
    fit_points = torch.stack([(boxes[:, 0] + boxes[:, 2])/2, boxes[:, 3]], dim=1)
    # 获取中心点处的mask值并过滤
    valid_indices = []
    # print(f"Mask shape: {mask.shape}")
    # print(f"Mask value range: {mask.min()}-{mask.max()}")
    for fit_point in fit_points:
        x, y = fit_point.long()
        if (0 <= x < mask.shape[1] and 0 <= y < mask.shape[0] and mask[y, x] == 1):
            # mask_value = mask[y, x].item()
            # print(f"Mask value at ({x},{y}): {mask_value}")
            valid_indices.append(True)
        else:
            valid_indices.append(False)
    # 只保留有效的目标
    valid_indices = torch.tensor(valid_indices, dtype=torch.bool, device=boxes.device)
    return valid_indices

def filter_by_mask(targets, masks):
    filtered_targets = []
    for batch_idx, (target, mask) in enumerate(zip(targets, masks)):
        boxes = target['boxes']
        # print(f"Before filtering: {len(boxes)} targets")
        # print(f"Mask valid area: {(mask == 1).float().mean().item():.2%}")
        valid_indices = _get_valid_indices_(boxes, mask)
        filtered_target = {
            'boxes': boxes[valid_indices],
            'labels': target['labels'][valid_indices],
            'image_id': target['image_id'] if 'image_id' in target else batch_idx,
            'area': target['area'][valid_indices] if 'area' in target else None,
            'iscrowd': target['iscrowd'][valid_indices] if 'iscrowd' in target else None}
        # print(f"After filtering: {len(filtered_target['boxes'])} targets")
        filtered_targets.append(filtered_target)
    return filtered_targets

if __name__ == "__main__":
    pass
