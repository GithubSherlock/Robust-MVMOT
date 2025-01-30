import torch
import torch.nn as nn
from models import MVDet, Segnet, SplitSegnet,Segnet_e

print(torch.backends.cudnn.deterministic)

best_model = '/media/rasho/Data 1/Arbeit/saved_models/EarlyBird_models/retrain/train_wild_135_mvdet_res18/checkpoints/last.ckpt'
# new_model = '/media/rasho/Data 1/Arbeit/saved_models/EarlyBird_models/train_wild_135_mvdet_res18/checkpoints/last.ckpt'
old_baseline = '/media/rasho/Data 1/Arbeit/saved_models/EarlyBird_models/new_old_models/baseline_models/train_wild_135_mvdet_2_res18/checkpoints/last.ckpt'
# Load checkpoints
checkpoint1 = torch.load(best_model)
checkpoint2 = torch.load(old_baseline)

print(checkpoint1.keys())


# Assuming model structure is the same
model1_weights = checkpoint1['state_dict']
model2_weights = checkpoint2['state_dict']

# Compare the keys (layer names) and shapes of the weights
for key in model1_weights.keys():
    if key not in model2_weights:
        print(f"Layer '{key}' is in Model 1 but not in Model 2")
    elif model1_weights[key].shape != model2_weights[key].shape:
        print(f"Layer '{key}' has different shapes in the two models:")
        print(f"Model 1: {model1_weights[key].shape}, Model 2: {model2_weights[key].shape}")

for key in model2_weights.keys():
    if key not in model1_weights:
        print(f"Layer '{key}' is in Model 2 but not in Model 1")

# # check cam_compressor
# print(model1_weights['cam_compressor'].shape)
# print(model2_weights['cam_compressor'].shape)


# # Compare layer by layer
# for key in model1_weights:
#     if key in model2_weights:
#         difference = torch.abs(model1_weights[key] - model2_weights[key])
#         print(f'Layer: {key}, Max Difference: {torch.max(difference)}')
#     else:
#         print(f'Layer {key} not found in model2')