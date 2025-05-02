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

if __name__ == "__main__":
    pass