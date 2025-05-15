import torchvision
from torchvision.models.detection import FasterRCNN_ResNet50_FPN_Weights
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

def _base_model_(pretrained=True):
    """Build the base model"""
    weight = FasterRCNN_ResNet50_FPN_Weights.DEFAULT if pretrained else None
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=weight)
    return model

def build_fasterrcnn_freeze(num_classes=2):
    """
    Build a Faster R-CNN model with a scientific layered freezing strategy
    
    Freezing strategy:
    - layer1: Completely frozen (basic features)
    - layer2: First two blocks frozen, last block trainable
    - layer3, 4: Completely trainable (task-related features)
    - Detection head: Completely trainable
    """
    model = _base_model_(pretrained=True)
    for param in model.parameters(): # First set all parameters trainable
        param.requires_grad = True
    for name, param in model.named_parameters():
        if "backbone.body.layer1" in name: # Freeze the layer1
            param.requires_grad = False
        if "backbone.body.layer2.0" in name or "backbone.body.layer2.1" in name: # Freeze the first two blocks of layer2
            param.requires_grad = False
    in_features = model.roi_heads.box_predictor.cls_score.in_features # Change the detection head
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model

def build_fasterrcnn_finetuning(num_classes=2):
    """Build a fully fine-tuned Faster R-CNN model (all layers trainable)"""
    model = _base_model_(pretrained=True)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model

def build_fasterrcnn_from_scratch(num_classes=2):
    """Build a Faster R-CNN model trained from scratch (no pre-training)"""
    model = _base_model_(pretrained=False)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model

def print_model_params_state(model):
    """Print the training state of each layer's parameters (for debugging)"""
    for name, param in model.named_parameters():
        print(f"{name}: {param.requires_grad}")

if __name__ == "__main__":
    # Create different models for comparative experiments
    base_model = _base_model_()
    freezed_model = build_fasterrcnn_freeze()
    finetune_model = build_fasterrcnn_finetuning()
    scratch_model = build_fasterrcnn_from_scratch()
    print_model_params_state(freezed_model) # Check parameter freezing status
