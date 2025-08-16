# test_import.py - 放在 datasets 目录下
import sys
import os

# 添加路径
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.insert(0, project_root)
sys.path.insert(0, current_dir)

print(f"项目根目录: {project_root}")
print(f"当前目录: {current_dir}")

# 测试导入
try:
    print("尝试导入 WildtrackDataset...")
    from wildtrack2_datasets import WildtrackDataset
    print("✓ WildtrackDataset 导入成功")
    
    print("尝试导入 MOTDataset...")
    from mot_datasets import MOTDataset
    print("✓ MOTDataset 导入成功")
    
    print("尝试导入 collate_fn...")
    utils_path = os.path.join(project_root, 'utils')
    sys.path.insert(0, utils_path)
    from utils import collate_fn
    print("✓ collate_fn 导入成功")
    
    print("所有导入测试通过!")
    
except ImportError as e:
    print(f"导入失败: {e}")
    import traceback
    traceback.print_exc()
