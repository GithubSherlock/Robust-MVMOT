import random
import os
from pathlib import Path

x = [1,2,3,4,5,6,7]

for _ in range(10):
    cameras = random.sample(x, 2)
    cameras.sort()
    print(cameras)

# th_2 = '/media/rasho/Data 1/Arbeit/saved_models/EarlyBird_models/train_wild_02_splitSegnet_camDropout_res18_Z4'
path = Path('/home/deep/PythonCode/EarlyBird/World_track-main/configs/m_mvdet.yml')
if not path.exists():
    print(f"{path} does not exist")
else:
    print(f"{path} exists")