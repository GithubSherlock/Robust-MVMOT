import matplotlib.pyplot as plt
import os
import torch

def print_infos(x):
    print(f'{len(x) = :<10}')
def vis_BEV_featuers(BEV_featuers):
    print(f'{BEV_featuers.shape = }')  # torch.Size([1, 256, 120, 360])

    imaged = torch.mean(BEV_featuers, dim=1)  # .permute(1, 2, 0).cpu()
    print(f'{imaged.shape = }')

    imaged = torch.squeeze(imaged, dim=1).permute(1, 2, 0).cpu()


    pass

if __name__ == '__main__':
    x = [0,2,4,1,5]
    super_long_temp = [0,2,4,1,5]

    print(f'{len(x) = :>30}')
    print(f'{len(super_long_temp) = :>30}')
