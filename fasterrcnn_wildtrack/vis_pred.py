import os
import torch
from pathlib import Path

from models.predicor import DetectionPredictor
from datasets.wildtrack2_dataloader import create_wildtrack_dataloaders

def find_model_weights():
    """
    Scan all model weights files in the model_weights folder
    Returns: A list containing the paths to all model files.
    """
    model_pathes = []
    weights_dir = Path('model_weights')
    
    if not weights_dir.exists():
        print("Warning: model_weights folder does not exist")
        return model_pathes
    for ext in ['.pth', '.ckpt']: # Search for .pth and .ckpt files
        model_pathes.extend([str(p) for p in weights_dir.rglob(f'*{ext}')])
    model_pathes.sort()
    if not model_pathes:
        print("Warning: No model weights found.")
    else:
        print(f"Find the following model files:")
        for path in model_pathes:
            print(f"- {path}")
    return model_pathes

def main(BATCH_SIZE: int, NUM_WORKERS: int, use_pretrained: bool, num_runs: int = 3):
    data_root = "/home/s-jiang/Documents/datasets/Wildtrack2"
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    vis_params = {
        'box_color': (0, 255, 0),
        'thickness': 3,
        'alpha': 0.6,
        'show_labels': True,
        'text_color': (255, 255, 255),
        'text_thickness': 2,
        'text_size': 1}
    
    _, _, test_loader = create_wildtrack_dataloaders(
        data_root,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS)
    
    if use_pretrained:
        model_pathes = ['pretrained']
    else:
        model_pathes = find_model_weights()
        if not model_pathes:
            print("No model files found, program exits.")
            return
    
    for model_path in model_pathes:
        try:
            predictor = DetectionPredictor(
                data_root=data_root,
                model_path=model_path,
                subset_ratio=0.8,  # Use of 80% of test data 0-1
                noise_factor=0.05,  # Add 5% Gaussian noise 0.01-0.1
                device=device,
                vis_params=vis_params,
                use_pretrained=use_pretrained)
            
            if predictor.actual_use_pretrained:
                print("\nProcessing model: PyTorch_fasterrcnn_resnet50_fpn")
            else:
                print(f"\nProcessing model: {model_path}")
            
            save_dir = "vis_results"
            os.makedirs(save_dir, exist_ok=True)
            for run in range(num_runs):
                print(f"\nRun {run + 1}/{num_runs}")
                metrics = predictor.predict_and_evaluate(test_loader, save_dir)
                print("Prediction completed! The results saved at:", save_dir)
                print("\nEvaluation Results:")
                print(f"mAP: {metrics['map']:.4f}")
                print(f"mAP@50: {metrics['map_50']:.4f}")
                print(f"mAP@75: {metrics['map_75']:.4f}")
        except Exception as e:
            print(f"Error processing model {model_path}: {str(e)}")
            continue

if __name__ == "__main__":
    main(BATCH_SIZE=1, NUM_WORKERS=8, use_pretrained=True)
    # main(BATCH_SIZE=1, NUM_WORKERS=8, use_pretrained=False)
