import os
import torch
import numpy as np
import configparser
from torchvision.io import read_image
from torchvision.io import ImageReadMode
from torchvision import tv_tensors
from torchvision.transforms.v2 import functional as F
import torchvision.transforms.v2 as T


class MOTDataset(torch.utils.data.Dataset):
    """
    MOT (Multiple Object Tracking) 数据集加载器
    支持 MOT15, MOT16, MOT17, MOT20 等数据集
    """
    def __init__(self, root, datasets=['MOT16'], split='train', transforms=None,
             sample_ratio=None, max_samples=None, random_seed=42, enable_sampling=True):
        """
        增强的MOT数据集初始化方法
        
        Args:
            root (str): 数据集根路径
            datasets (list or str): 数据集名称列表
            split (str): 数据集分割 ('train', 'val', 'test')
            transforms: 数据变换
            sample_ratio (float): 采样比例 (0.0-1.0)
            max_samples (int): 最大样本数限制
            random_seed (int): 随机种子
            enable_sampling (bool): 是否启用采样功能
        """
        self.root = root
        self.transforms = transforms
        self.split = split
        self.random_seed = random_seed
        self.enable_sampling = enable_sampling

        # 🎯 智能采样参数设置
        if not enable_sampling:
            self.sample_ratio = 1.0
            self.max_samples = None
        else:
            if sample_ratio is None:
                # 根据数据集分割自动设置采样比例
                sampling_config = {
                    'train': 0.8,      # 训练集80%
                    'val': 0.3,        # 验证集30%
                    'test': 0.2        # 测试集20%
                }
                self.sample_ratio = sampling_config.get(split, 1.0)
            else:
                self.sample_ratio = max(0.01, min(1.0, sample_ratio))  # 限制在合理范围

        self.max_samples = max_samples
        # 处理数据集参数
        if isinstance(datasets, str):
            self.datasets = [datasets]
        else:
            self.datasets = datasets
        
        # 存储所有序列信息
        self.sequences = []
        self.sequence_info = {}
        self.images = []  # 存储所有图像路径信息
        
        # 加载所有指定的数据集
        self._load_datasets()
        
        # 🔥 应用数据采样
        original_size = len(self.images)
        if self.sample_ratio < 1.0 or self.max_samples is not None:
            self.images = self._sample_data()
        
        print(f"📊 MOT {split.upper()} Dataset Summary:")
        print(f"   - Loaded {len(self.sequences)} sequences")
        print(f"   - Original images: {original_size}")
        print(f"   - Sampled images: {len(self.images)}")
        if original_size > 0:
            actual_ratio = len(self.images) / original_size
            print(f"   - Actual sampling ratio: {actual_ratio:.2%}")
        print(f"   - Sample ratio setting: {self.sample_ratio:.2%}")
        if self.max_samples:
            print(f"   - Max samples limit: {self.max_samples}")
    
    def _sample_data(self):
        """
        优化的数据采样方法
        支持按比例采样和最大样本数限制
        """
        # 设置随机种子确保可重现性
        np.random.seed(self.random_seed)
        torch.manual_seed(self.random_seed)
        
        total_samples = len(self.images)
        
        # 计算目标样本数
        if self.max_samples is not None:
            target_samples = min(self.max_samples, int(total_samples * self.sample_ratio))
        else:
            target_samples = int(total_samples * self.sample_ratio)
        
        # 确保至少有一个样本
        target_samples = max(1, target_samples)
        
        # 随机采样索引
        if target_samples >= total_samples:
            sampled_indices = list(range(total_samples))
        else:
            sampled_indices = np.random.choice(
                total_samples, 
                size=target_samples, 
                replace=False
            ).tolist()
        
        # 返回采样后的图像列表
        sampled_images = [self.images[i] for i in sorted(sampled_indices)]
        
        print(f"🎯 采样结果: {len(sampled_images)}/{total_samples} "
            f"(比例: {len(sampled_images)/total_samples:.2%})")
        
        return sampled_images
    
    def get_dataset_stats(self):
        """
        📈 获取数据集统计信息
        """
        stats = {
            'total_sequences': len(self.sequences),
            'total_images': len(self.images),
            'datasets': self.datasets,
            'split': self.split,
            'sample_ratio': self.sample_ratio,
            'max_samples': self.max_samples,
            'sampling_enabled': self.enable_sampling
        }
        
        # 按数据集统计
        dataset_counts = {}
        for img_info in self.images:
            dataset_name = img_info['dataset']
            dataset_counts[dataset_name] = dataset_counts.get(dataset_name, 0) + 1
        
        stats['dataset_distribution'] = dataset_counts
        return stats
    
    def print_dataset_info(self):
        """
        🖨️ 打印详细的数据集信息
        """
        stats = self.get_dataset_stats()
        
        print("\n" + "="*60)
        print(f"📋 MOT Dataset Information - {self.split.upper()}")
        print("="*60)
        print(f"🎯 Sampling Configuration:")
        print(f"   - Sample ratio: {stats['sample_ratio']:.2%}")
        if stats['max_samples']:
            print(f"   - Max samples: {stats['max_samples']}")
        print(f"   - Random seed: {self.random_seed}")
        
        print(f"\n📊 Dataset Statistics:")
        print(f"   - Total sequences: {stats['total_sequences']}")
        print(f"   - Total images: {stats['total_images']}")
        
        print(f"\n📁 Dataset Distribution:")
        for dataset, count in stats['dataset_distribution'].items():
            percentage = (count / stats['total_images']) * 100
            print(f"   - {dataset}: {count} images ({percentage:.1f}%)")
        
        print(f"\n🎬 Top 5 Sequences by Image Count:")
        sorted_sequences = sorted(stats['sequence_distribution'].items(), 
                                key=lambda x: x[1], reverse=True)
        for seq, count in sorted_sequences[:5]:
            percentage = (count / stats['total_images']) * 100
            print(f"   - {seq}: {count} images ({percentage:.1f}%)")
        
        print("="*60 + "\n")
    
    def _load_datasets(self):
        """加载所有指定的数据集"""
        for dataset_name in self.datasets:
            dataset_path = os.path.join(self.root, dataset_name, self.split)
            if not os.path.exists(dataset_path):
                print(f"Warning: Dataset path {dataset_path} does not exist, skipping...")
                continue
            
            # 获取该数据集下的所有序列
            sequences = [d for d in os.listdir(dataset_path) 
                        if os.path.isdir(os.path.join(dataset_path, d))]
            sequences.sort()
            
            for seq_name in sequences:
                seq_path = os.path.join(dataset_path, seq_name)
                self._load_sequence(dataset_name, seq_name, seq_path)
    
    def _load_sequence(self, dataset_name, seq_name, seq_path):
        """加载单个序列的信息"""
        # 检查必要的文件夹是否存在
        img_dir = os.path.join(seq_path, 'img1')
        gt_dir = os.path.join(seq_path, 'gt')
        det_dir = os.path.join(seq_path, 'det')
        seqinfo_path = os.path.join(seq_path, 'seqinfo.ini')
        
        if not all(os.path.exists(p) for p in [img_dir, seqinfo_path]):
            print(f"Warning: Missing required files in {seq_path}, skipping...")
            return
        
        # 读取序列信息
        seq_info = self._read_seqinfo(seqinfo_path)
        seq_info['dataset'] = dataset_name
        seq_info['sequence'] = seq_name
        seq_info['path'] = seq_path
        
        # 存储序列信息
        full_seq_name = f"{dataset_name}-{seq_name}"
        self.sequences.append(full_seq_name)
        self.sequence_info[full_seq_name] = seq_info
        
        # 加载图像文件列表
        img_files = [f for f in os.listdir(img_dir) 
                    if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        img_files.sort()
        
        # 加载检测数据（如果存在）
        det_data = {}
        if os.path.exists(os.path.join(det_dir, 'det.txt')):
            det_data = self._load_detections(os.path.join(det_dir, 'det.txt'))
        
        # 加载真值数据（如果存在）
        gt_data = {}
        if os.path.exists(os.path.join(gt_dir, 'gt.txt')):
            gt_data = self._load_ground_truth(os.path.join(gt_dir, 'gt.txt'))
        
        # 为每个图像创建条目
        for img_file in img_files:
            frame_id = int(os.path.splitext(img_file)[0])
            
            img_info = {
                'dataset': dataset_name,
                'sequence': seq_name,
                'full_sequence': full_seq_name,
                'frame_id': frame_id,
                'img_path': os.path.join(img_dir, img_file),
                'img_name': img_file,
                'detections': det_data.get(frame_id, []),
                'ground_truth': gt_data.get(frame_id, []),
                'seq_info': seq_info
            }
            
            self.images.append(img_info)
    
    def _read_seqinfo(self, seqinfo_path):
        """读取序列信息文件"""
        config = configparser.ConfigParser()
        config.read(seqinfo_path)
        
        seq_info = {}
        if 'Sequence' in config:
            for key, value in config['Sequence'].items():
                # 尝试转换为数字类型
                try:
                    if '.' in value:
                        seq_info[key] = float(value)
                    else:
                        seq_info[key] = int(value)
                except ValueError:
                    seq_info[key] = value
        
        return seq_info
    
    def _load_detections(self, det_file):
        """加载检测文件"""
        detections = {}
        try:
            with open(det_file, 'r') as f:
                for line in f:
                    parts = line.strip().split(',')
                    if len(parts) >= 7:
                        frame_id = int(parts[0])
                        # det.txt格式: frame, track_id(-1), left, top, width, height, conf, x, y, z
                        detection = {
                            'track_id': int(parts[1]),  # 通常为-1
                            'bbox': [float(parts[2]), float(parts[3]), 
                                   float(parts[4]), float(parts[5])],  # left, top, width, height
                            'confidence': float(parts[6]) if len(parts) > 6 else 1.0
                        }
                        
                        if frame_id not in detections:
                            detections[frame_id] = []
                        detections[frame_id].append(detection)
        except Exception as e:
            print(f"Error loading detections from {det_file}: {e}")
        
        return detections
    
    def _load_ground_truth(self, gt_file):
        """加载真值文件"""
        ground_truth = {}
        try:
            with open(gt_file, 'r') as f:
                for line in f:
                    parts = line.strip().split(',')
                    if len(parts) >= 8:
                        frame_id = int(parts[0])
                        # gt.txt格式: frame, id, left, top, width, height, conf, class, visibility
                        gt_obj = {
                            'track_id': int(parts[1]),
                            'bbox': [float(parts[2]), float(parts[3]), 
                                   float(parts[4]), float(parts[5])],  # left, top, width, height
                            'confidence': float(parts[6]) if len(parts) > 6 else 1.0,
                            'class': int(parts[7]) if len(parts) > 7 else 1,
                            'visibility': float(parts[8]) if len(parts) > 8 else 1.0
                        }
                        
                        if frame_id not in ground_truth:
                            ground_truth[frame_id] = []
                        ground_truth[frame_id].append(gt_obj)
        except Exception as e:
            print(f"Error loading ground truth from {gt_file}: {e}")
        
        return ground_truth
    
    def __getitem__(self, idx):
        """获取单个数据样本"""
        if idx >= len(self.images):
            raise IndexError(f"Index {idx} out of range for dataset with {len(self.images)} images")
    
        img_info = self.images[idx]
        # 读取图像
        img = read_image(img_info['img_path'])
        img = tv_tensors.Image(img)
        
        # 构建边界框和标签
        boxes = []
        labels = []
        track_ids = []
        
        # 对于目标检测任务，使用检测数据(det)作为真值
        if self.split == 'train' and img_info['detections']:
            # 训练时使用检测数据作为真值（用于目标检测任务）
            for det_obj in img_info['detections']:
                # 转换从 left,top,width,height 到 xmin,ymin,xmax,ymax
                left, top, width, height = det_obj['bbox']
                boxes.append([left, top, left + width, top + height])
                labels.append(1)  # 假设所有对象都是行人类别
                track_ids.append(det_obj.get('track_id', -1))
        elif self.split == 'test' and img_info['detections']:
            # 测试时也使用检测数据
            for det_obj in img_info['detections']:
                left, top, width, height = det_obj['bbox']
                boxes.append([left, top, left + width, top + height])
                labels.append(1)  # 假设所有对象都是行人类别
                track_ids.append(det_obj.get('track_id', -1))
        
        # 处理空检测的情况
        if len(boxes) == 0:
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.int64)
            track_ids = torch.zeros((0,), dtype=torch.int64)
            area = torch.zeros((0,), dtype=torch.float32)
        else:
            boxes = torch.as_tensor(boxes, dtype=torch.float32)
            labels = torch.as_tensor(labels, dtype=torch.int64)
            track_ids = torch.as_tensor(track_ids, dtype=torch.int64)
            area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        
        num_objs = len(boxes)
        iscrowd = torch.zeros((num_objs,), dtype=torch.int64)
        
        # 构建目标字典
        target = {
            "boxes": tv_tensors.BoundingBoxes(boxes, format="XYXY", canvas_size=F.get_size(img)),
            "labels": labels,
            "image_id": idx,
            "area": area,
            "iscrowd": iscrowd,
            "track_ids": track_ids,
            "frame_id": img_info['frame_id'],
            "sequence": img_info['full_sequence']
        }
        
        # 应用变换
        if self.transforms is not None:
            img, target = self.transforms(img, target)
        
        # 类型转换
        if img.dtype == torch.uint8:
            img = img.float() / 255.0
        
        return img, target, img_info['img_name'], None  # 返回None作为mask以保持接口一致
    
    def __len__(self):
        return len(self.images)
    
    def get_sequence_info(self, sequence_name=None):
        """获取序列信息"""
        if sequence_name is None:
            return self.sequence_info
        else:
            return self.sequence_info.get(sequence_name, {})
    
    def get_sequences(self):
        """获取所有序列名称"""
        return self.sequences
    
    def print_dataset_info(self):
        """打印数据集信息"""
        print("=" * 50)
        print("MOT Dataset Information")
        print("=" * 50)
        print(f"Datasets: {self.datasets}")
        print(f"Split: {self.split}")
        print(f"Total sequences: {len(self.sequences)}")
        print(f"Total images: {len(self.images)}")
        print()
        
        # 按数据集分组统计
        dataset_stats = {}
        for img_info in self.images:
            dataset = img_info['dataset']
            if dataset not in dataset_stats:
                dataset_stats[dataset] = {'sequences': set(), 'images': 0}
            dataset_stats[dataset]['sequences'].add(img_info['sequence'])
            dataset_stats[dataset]['images'] += 1
        
        for dataset, stats in dataset_stats.items():
            print(f"{dataset}:")
            print(f"  - Sequences: {len(stats['sequences'])}")
            print(f"  - Images: {stats['images']}")
            print(f"  - Sequence names: {sorted(list(stats['sequences']))}")
            print()
    
    def print_sequence_info(self, sequence_name=None):
        """打印序列详细信息"""
        if sequence_name is None:
            sequences_to_print = self.sequences
        else:
            sequences_to_print = [sequence_name] if sequence_name in self.sequences else []
        
        for seq_name in sequences_to_print:
            info = self.sequence_info[seq_name]
            print(f"Sequence: {seq_name}")
            print("-" * 30)
            for key, value in info.items():
                print(f"  {key}: {value}")
            print()


def _print_sample_info(dataset, idx=0):
    """打印数据样本信息"""
    if len(dataset) == 0:
        print("Dataset is empty!")
        return
    
    img, target, img_name, mask = dataset[idx]
    
    print("=" * 30)
    print("Sample Information")
    print("=" * 30)
    print(f"Image name: {img_name}")
    print(f"Image shape: {img.shape}")
    print(f"Image dtype: {img.dtype}")
    print(f"Image value range: [{img.min():.3f}, {img.max():.3f}]")
    print()
    
    print("Target Information:")
    print(f"  - Number of objects: {len(target['boxes'])}")
    print(f"  - Bounding boxes: {target['boxes'].tolist()}")
    print(f"  - Labels: {target['labels'].tolist()}")
    print(f"  - Track IDs: {target['track_ids'].tolist()}")
    print(f"  - Frame ID: {target['frame_id']}")
    print(f"  - Sequence: {target['sequence']}")
    print(f"  - Areas: {target['area'].tolist()}")
    print()


if __name__ == "__main__":
    ROOT_DIR = "/home/s-jiang/Documents/datasets/"
    DATASETS = ['MOT16', 'MOT17', 'MOT20']
    def single_dataset_test(ROOT_DIR=ROOT_DIR):
        try:
            dataset_single = MOTDataset(ROOT_DIR, datasets='MOT16', split='train')
            dataset_single.print_dataset_info()
            dataset_single.print_sequence_info()
            if len(dataset_single) > 0:
                _print_sample_info(dataset_single, 0)
        except Exception as e:
            print(f"Error loading MOT16: {e}")
        print("\n" + "="*60 + "\n")
    
    def multi_dataset_test(ROOT_DIR=ROOT_DIR):
        print("Testing multiple datasets ...")
        try:
            dataset_multi = MOTDataset(ROOT_DIR, datasets=DATASETS, split='train')
            dataset_multi.print_dataset_info()
            if len(dataset_multi) > 0:
                _print_sample_info(dataset_multi, 0)
        except Exception as e:
            print(f"Error loading multiple datasets: {e}")
        print("\n" + "="*60 + "\n")
    
    def test_split_test(ROOT_DIR=ROOT_DIR):
        print("Testing test split ...")
        try:
            dataset_test = MOTDataset(ROOT_DIR, datasets=DATASETS, split='test')
            dataset_test.print_dataset_info()
            if len(dataset_test) > 0:
                _print_sample_info(dataset_test, 0)
        except Exception as e:
            print(f"Error loading multiple test-sets: {e}")
    
    # single_dataset_test(ROOT_DIR=ROOT_DIR)
    multi_dataset_test(ROOT_DIR=ROOT_DIR)
    test_split_test(ROOT_DIR=ROOT_DIR)
