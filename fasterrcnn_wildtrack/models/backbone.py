import torchvision
from torchvision.models.detection import FasterRCNN_ResNet50_FPN_Weights
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone

from models.custom_faster_rcnn import CustomFasterRCNN


def _base_model_(pretrained=True):
    """Build the base model"""
    weight = FasterRCNN_ResNet50_FPN_Weights.DEFAULT if pretrained else None
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=weight)

    return model

def resnet_backbone(weights_name="resnet50", pretrained=True):
    """
    创建带FPN的backbone
    Args:
        weights: resnet18, resnet34, resnet50, resnet101
        pretrained: 是否使用预训练权重
    """
    WEIGHTS_MAP = {
        "resnet18": torchvision.models.ResNet18_Weights.IMAGENET1K_V1,
        "resnet34": torchvision.models.ResNet34_Weights.IMAGENET1K_V1,
        "resnet50": torchvision.models.ResNet50_Weights.IMAGENET1K_V1,
        "resnet101": torchvision.models.ResNet101_Weights.IMAGENET1K_V1}
    backbone_weights = WEIGHTS_MAP[weights_name] if pretrained else None

    return resnet_fpn_backbone(
        backbone_name=weights_name,
        weights=backbone_weights,
        trainable_layers=3)  # Value of trainable layers: 1,2,3,4,5

def build_fasterrcnn_model(weights_name="resnet18", num_classes=2, pretrained=True):
    WEIGHTS_MAP = {"fasterrcnn_resnet50_fpn": FasterRCNN_ResNet50_FPN_Weights.COCO_V1,
        "fasterrcnn_resnet50_fpn_v2": FasterRCNN_ResNet50_FPN_Weights.COCO_V1}
    if weights_name in WEIGHTS_MAP: # Choose backbone with FPN
        pretrained_model = torchvision.models.detection.fasterrcnn_resnet50_fpn(
            weights=WEIGHTS_MAP[weights_name] if pretrained else None)
        backbone = pretrained_model.backbone # Based on COCO dataset
    elif weights_name in ["resnet18", "resnet34", "resnet50", "resnet101"]:
        backbone = resnet_backbone(weights_name=weights_name, pretrained=pretrained) # Based on ImageNet1K dataset
    else:
        raise KeyError(f"Unsupported weights type: {weights_name}")

    model = CustomFasterRCNN(backbone=backbone, num_classes=num_classes)
    freeze_layers = { # Freeze strategy based on the weights type
        "resnet18": ["conv1"],
        "resnet34": ["conv1"],
        "resnet50": ["conv1"],
        "fasterrcnn_resnet50_fpn": ["conv1"],
        "fasterrcnn_resnet50_fpn_v2": ["conv1"],
        "resnet101": ["conv1"]}
    for name, param in model.named_parameters(): # Freeze specific layers based on the weights type
        param.requires_grad = True
        for layer in freeze_layers[weights_name]:
            if f"backbone.body.{layer}" in name:
                param.requires_grad = False
    return model

if __name__ == "__main__":
    # base_model = _base_model_() # Create different models for comparative experiments
    model = build_fasterrcnn_model()
