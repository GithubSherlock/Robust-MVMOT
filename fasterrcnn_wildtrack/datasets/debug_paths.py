# 创建 debug_paths.py 文件
import os
import sys

print("=== 路径诊断 ===")
print(f"当前工作目录: {os.getcwd()}")
print(f"当前文件路径: {os.path.abspath(__file__)}")

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)

print(f"当前文件目录: {current_dir}")
print(f"项目根目录: {project_root}")

print(f"\n当前目录内容:")
for item in os.listdir(current_dir):
    print(f"  {item}")

print(f"\n项目根目录内容:")
for item in os.listdir(project_root):
    print(f"  {item}")

# 检查关键文件是否存在
key_files = [
    os.path.join(current_dir, 'wildtrack2_datasets.py'),
    os.path.join(current_dir, 'mot_datasets.py'),
    os.path.join(project_root, 'utils', 'utils.py'),
]

print(f"\n关键文件检查:")
for file_path in key_files:
    exists = os.path.exists(file_path)
    print(f"  {file_path}: {'存在' if exists else '不存在'}")

print(f"\nPython 路径:")
for path in sys.path:
    print(f"  {path}")
