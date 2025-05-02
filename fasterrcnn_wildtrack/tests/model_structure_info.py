import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from collections import defaultdict
from torchvision.models.detection import FasterRCNN_ResNet50_FPN_Weights
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from models.backbone import _base_model_, build_fasterrcnn_freeze, build_fasterrcnn_finetuning, build_fasterrcnn_from_scratch

def print_layer_prefixes(model):
    prefix_dict = defaultdict(list)
    for name, _ in model.named_parameters():
        prefix = name.split('.')[0]
        prefix_dict[prefix].append(name)
    
    print("Main components in the model:")
    print("=" * 50)
    for prefix, param_names in prefix_dict.items():
        print(f"\n{prefix}:")
        print("-" * 30)
        for name in param_names[:3]:
            print(f"  - {name}")
        if len(param_names) > 3:
            print(f"  ... etc. total{len(param_names)}parameters\n")

def inspect_model_structure(model):
    # Print the model structure
    print("="*50)
    print("Overview of the model structure:")
    print("="*50)
    
    backbone_params = []
    rpn_params = []
    roi_params = []
    
    for name, param in model.named_parameters():
        if name.startswith('backbone'):
            backbone_params.append((name, param.requires_grad, param.shape))
        elif name.startswith('rpn'):
            rpn_params.append((name, param.requires_grad, param.shape))
        elif name.startswith('roi_heads'):
            roi_params.append((name, param.requires_grad, param.shape))
    
    print("\n1. Backbone (feature extractor) parameters:")
    print("-"*50)
    for name, requires_grad, shape in backbone_params:
        print(f"{name}:")
        print(f"  Trainable: {requires_grad}")
        print(f"  Shape: {shape}")
    
    print("\n2. RPN (Regional Proposal Network) Parameters:")
    print("-"*50)
    for name, requires_grad, shape in rpn_params:
        print(f"{name}:")
        print(f"  Trainable: {requires_grad}")
        print(f"  Shape: {shape}")
    
    print("\n3. ROI Heads (Target Detection Heads) Parameters:")
    print("-"*50)
    for name, requires_grad, shape in roi_params:
        print(f"{name}:")
        print(f"  Trainable: {requires_grad}")
        print(f"  Shape: {shape}")
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print("\n"+"="*50)
    print(f"Total number of parameters: {total_params:,}")
    print(f"Trainable parametrs: {trainable_params:,}")
    print(f"Freezed parameters: {total_params - trainable_params:,}")
    print("="*50)

def save_txt_file(model, save=True):
    from contextlib import redirect_stdout
    with open('structure_info_freezed_model.txt', 'w', encoding='utf-8') as f:
        with redirect_stdout(f):
            print_layer_prefixes(model)
            inspect_model_structure(model)

if __name__ == "__main__":
    base_model = _base_model_()
    freezed_model = build_fasterrcnn_freeze()
    finetune_model = build_fasterrcnn_finetuning()
    scratch_model = build_fasterrcnn_from_scratch()
    
    save_txt_file(model=freezed_model, save=True)
    print_layer_prefixes(freezed_model)
    inspect_model_structure(freezed_model)
