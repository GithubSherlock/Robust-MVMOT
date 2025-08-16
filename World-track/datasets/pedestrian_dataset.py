import math
import os
import json
import random
from operator import itemgetter

import torch
import numpy as np
from torchvision.datasets import VisionDataset
import torchvision.transforms.functional as F
from PIL import Image
import matplotlib.pyplot as plt
from datasets.multiviewx_dataset import MultiviewX
from datasets.wildtrack_dataset import Wildtrack
from utils import geom, basic, vox


class PedestrianDataset(VisionDataset):
    """
    多视角行人检测数据集类
    
    该类基于Wildtrack数据集，为多视角行人检测和跟踪任务提供数据加载功能。
    主要功能包括：
    - 多视角图像数据加载
    - 3D世界坐标和2D图像坐标的标注处理
    - 数据增强（训练时）
    - BEV（鸟瞰图）和图像空间的真值生成
    """
    def __init__(
            self,
            base, # Wildtrack数据集实例
            # split='train',  # 新增参数：'train', 'val', 'test'
            # train_ratio=0.8,  # 训练集比例
            # val_ratio=0.1,   # 验证集比例
            # test_ratio=0.1,  # 测试集比例
            is_train=True,
            is_testing=False,
            is_predict=False,
            resolution=(160, 4, 250), # 体素分辨率 (Y, Z, X)
            bounds=(-500, 500, -320, 320, 0, 2), # 3D空间边界 (x_min, x_max, y_min, y_max, z_min, z_max)
            final_dim: tuple = (720, 1280), # 最终图像尺寸 (H, W)
            resize_lim: list = (0.8, 1.2), # 随机缩放范围
            ):
        """
        初始化行人检测数据集
        
        Args:
            base: Wildtrack数据集实例，提供基础数据访问功能
            is_train: 训练模式标志
            is_testing: 测试模式标志  
            is_predict: 预测模式标志
            resolution: 3D体素网格分辨率 (Y轴, Z轴, X轴)
            bounds: 3D世界坐标边界范围
            final_dim: 数据增强后的最终图像尺寸
            resize_lim: 训练时随机缩放的范围
        """
        super().__init__(base.root)
        self.base = base
        self.root, self.num_cam, self.num_frame = base.root, base.num_cam, base.num_frame
        self.used_cameras = base.used_cameras
        # img_shape and worldgrid_shape is the original shape matching the annotations in dataset
        # MultiviewX: [1080, 1920], [640, 1000] Wildtrack: [1080, 1920], [480, 1440]
        self.img_shape = base.img_shape # 原始图像尺寸 [H, W]
        self.worldgrid_shape = base.worldgrid_shape # 世界网格尺寸 [N_row, N_col]
        self.is_train = is_train
        self.is_testing = is_testing
        self.is_predict = is_predict
        self.bounds = bounds # 3D空间边界
        self.resolution = resolution # 体素分辨率
        self.data_aug_conf = {'final_dim': final_dim, 'resize_lim': resize_lim} # 数据增强配置：最终图像尺寸；缩放范围
        self.kernel_size = 1.5 # 高斯核大小（用于热力图生成）
        self.max_objects = 60 # 最大目标数量
        self.img_downsample = 4 # 图像下采样倍数
        self.load_masks = True # 是否加载相机掩码
        self.Y, self.Z, self.X = self.resolution # 解包体素分辨率
        self.scene_centroid = torch.tensor((0., 0., 0.)).reshape([1, 3]) # 场景中心点（用于3D变换）

        self.vox_util = vox.VoxelUtil(
            self.Y, self.Z, self.X,
            scene_centroid=self.scene_centroid,
            bounds=self.bounds,
            assert_cube=False)


        if self.is_train:
            frame_range = range(0, int(self.num_frame * 0.9))
        else:
            frame_range = range(int(self.num_frame * 0.9), self.num_frame)
        if self.is_testing:
            frame_range = range(int(self.num_frame * 0.9), self.num_frame)
            frame_range = range(0, int(self.num_frame * 0.9))

        self.img_fpaths = self.base.get_image_fpaths(frame_range) # 获取图像文件路径 {cam: {frame: img_path}}
        self.world_gt = {} # 世界坐标真值 {frame: (points, person_ids)}
        self.imgs_gt = {} # 图像坐标真值 {frame: {cam: (bboxes, person_ids)}}
        self.pid_dict = {} # 行人ID映射字典
        self.masks = {} # 相机掩码 {cam: mask_tensor}
        self.download(frame_range) # 下载并处理标注数据
        if self.load_masks:
            self.load_camera_masks() # 加载相机掩码

        self.gt_fpath = os.path.join(self.root, 'gt.txt') # 设置真值文件路径并准备真值数据
        self.prepare_gt()

        self.calibration = {}
        self.setup()

    def setup(self):
        """
        设置相机标定参数
        
        将基础数据集中的内参和外参矩阵转换为PyTorch张量，
        并调整格式以适应后续的几何变换计算。
        """
        # 获取所有相机的内参矩阵并转换为张量
        # 形状: (S, 3, 3) -> (S, 4, 4) 其中S为相机数量
        intrinsic = torch.tensor(np.stack(self.base.intrinsic_matrices, axis=0), dtype=torch.float32)  # S,3,3
        intrinsic = geom.merge_intrinsics(*geom.split_intrinsics(intrinsic)).squeeze()  # S,4,4
        self.calibration['intrinsic'] = intrinsic
        self.calibration['extrinsic'] = torch.eye(4)[None].repeat(intrinsic.shape[0], 1, 1)
        self.calibration['extrinsic'][:, :3] = torch.tensor(
            np.stack(self.base.extrinsic_matrices, axis=0), dtype=torch.float32)

    def prepare_gt(self):
        """
        准备并保存真值数据到文件
        
        从JSON标注文件中提取世界坐标真值，转换为统一格式并保存为文本文件。
        输出格式：每行包含 [frame_id, grid_x, grid_y]
        """
        og_gt = []
        for fname in sorted(os.listdir(os.path.join(self.root, 'annotations_positions'))):
            frame = int(fname.split('.')[0])
            with open(os.path.join(self.root, 'annotations_positions', fname)) as json_file:
                all_pedestrians = json.load(json_file)
            for single_pedestrian in all_pedestrians:
                def is_in_cam(cam):
                    return not (single_pedestrian['views'][cam]['xmin'] == -1 and
                                single_pedestrian['views'][cam]['xmax'] == -1 and
                                single_pedestrian['views'][cam]['ymin'] == -1 and
                                single_pedestrian['views'][cam]['ymax'] == -1)

                in_cam_range = sum(is_in_cam(cam) for cam in self.used_cameras)  # range(self.num_cam))
                if not in_cam_range:
                    continue
                grid_x, grid_y = self.base.get_worldgrid_from_pos(single_pedestrian['positionID'])
                og_gt.append(np.array([frame, grid_x, grid_y]))
        og_gt = np.stack(og_gt, axis=0)
        os.makedirs(os.path.dirname(self.gt_fpath), exist_ok=True)
        np.savetxt(self.gt_fpath, og_gt, '%d')

    def download(self, frame_range):
        """
        下载并处理指定帧范围内的标注数据
        
        从JSON标注文件中提取世界坐标和图像坐标的标注信息，
        构建用于训练的数据结构。
        
        Args:
            frame_range: 需要处理的帧范围
        """
        num_frame, num_world_bbox, num_imgs_bbox = 0, 0, 0
        for fname in sorted(os.listdir(os.path.join(self.root, 'annotations_positions'))):
            frame = int(fname.split('.')[0])
            if frame in frame_range:  # is the frame in the annotated datas
                num_frame += 1
                with open(os.path.join(self.root, 'annotations_positions', fname)) as json_file:
                    all_pedestrians = json.load(json_file)
                world_pts, world_pids = [], []
                # img_bboxs, img_pids = [[] for _ in range(self.num_cam)], [[] for _ in range(self.num_cam)]
                # img_bboxs, img_pids = [[] for _ in self.used_cameras], [[] for _ in self.used_cameras]
                img_bboxs, img_pids = {cam:[] for cam in self.used_cameras}, {cam:[] for cam in self.used_cameras}

                for pedestrian in all_pedestrians:
                    grid_x, grid_y = self.base.get_worldgrid_from_pos(pedestrian['positionID']).squeeze()
                    if pedestrian['personID'] not in self.pid_dict:
                        self.pid_dict[pedestrian['personID']] = len(self.pid_dict)
                    num_world_bbox += 1
                    world_pts.append((grid_x, grid_y))
                    world_pids.append(pedestrian['personID'])
                    for cam in self.used_cameras:  # range(self.num_cam):
                        if itemgetter('xmin', 'ymin', 'xmax', 'ymax')(pedestrian['views'][cam]) != (-1, -1, -1, -1):
                            img_bboxs[cam].append(itemgetter('xmin', 'ymin', 'xmax', 'ymax')
                                                  (pedestrian['views'][cam]))
                            img_pids[cam].append(pedestrian['personID'])
                            num_imgs_bbox += 1
                self.world_gt[frame] = (np.array(world_pts), np.array(world_pids))
                self.imgs_gt[frame] = {}
                for cam in self.used_cameras:  # range(self.num_cam):
                    # x1y1x2y2
                    self.imgs_gt[frame][cam] = (np.array(img_bboxs[cam]), np.array(img_pids[cam]))

    def get_bev_gt(self, mem_pts, pids):
        """
        生成鸟瞰图（BEV）视角的真值数据
        
        将3D内存坐标转换为BEV热力图、偏移图和ID图，用于训练BEV检测头。
        
        Args:
            mem_pts: 内存坐标系中的3D点 (N, 3)
            pids: 对应的行人ID列表 (N,)
            
        Returns:
            tuple: (center, valid_mask, person_ids, offset)
                - center: 中心点热力图 (1, Y, X)
                - valid_mask: 有效位置掩码 (1, Y, X) 
                - person_ids: 行人ID图 (1, Y, X)
                - offset: 亚像素偏移图 (2, Y, X)
        """
        center = torch.zeros((1, self.Y, self.X), dtype=torch.float32)
        valid_mask = torch.zeros((1, self.Y, self.X), dtype=torch.bool)
        offset = torch.zeros((2, self.Y, self.X), dtype=torch.float32)
        person_ids = torch.zeros((1, self.Y, self.X), dtype=torch.long)
        # size = torch.zeros((1, self.Y, self.X), dtype=torch.float32)

        for pts, pid in zip(mem_pts, pids):
            ct = pts[:2]
            ct_int = ct.int()

            if ct_int[1] >= self.Y or ct_int[0] >= self.X:
                continue

            basic.draw_umich_gaussian(center[0], ct_int, self.kernel_size)
            valid_mask[:, ct_int[1], ct_int[0]] = 1
            offset[:, ct_int[1], ct_int[0]] = ct - ct_int
            person_ids[:, ct_int[1], ct_int[0]] = pid

        return center, valid_mask, person_ids, offset

    def get_img_gt(self, img_pts, img_pids, sx, sy, crop):
        """
        生成图像空间的真值数据
        
        将原始图像坐标的边界框转换为下采样后的热力图、偏移图等，
        用于训练图像检测头。
        
        Args:
            img_pts: 图像边界框坐标 (N, 4) [xmin, ymin, xmax, ymax]
            img_pids: 对应的行人ID (N,)
            sx, sy: x和y方向的缩放因子
            crop: 裁剪参数 (crop_x, crop_y, crop_x+W, crop_y+H)
            
        Returns:
            tuple: (center, offset, size, skeleton, person_ids, valid_mask)
                - center: 中心点和脚部点热力图 (2, H, W)
                - offset: 亚像素偏移图 (2, H, W)
                - size: 边界框尺寸图 (2, H, W)
                - skeleton: 骨架图 (2, H, W) - 当前未使用
                - person_ids: 行人ID图 (1, H, W)
                - valid_mask: 有效掩码 (1, H, W)
        """
        H = int(self.data_aug_conf['final_dim'][0] / self.img_downsample)
        W = int(self.data_aug_conf['final_dim'][1] / self.img_downsample)
        center = torch.zeros((2, H, W), dtype=torch.float32)
        offset = torch.zeros((2, H, W), dtype=torch.float32)
        size = torch.zeros((2, H, W), dtype=torch.float32)
        skeleton = torch.zeros((2, H, W), dtype=torch.float32)

        valid_mask = torch.zeros((1, H, W), dtype=torch.bool)
        person_ids = torch.zeros((1, H, W), dtype=torch.long)

        xmin = (img_pts[:, 0] * sx - crop[0]) / self.img_downsample
        ymin = (img_pts[:, 1] * sy - crop[1]) / self.img_downsample
        xmax = (img_pts[:, 2] * sx - crop[0]) / self.img_downsample
        ymax = (img_pts[:, 3] * sy - crop[1]) / self.img_downsample

        center_pts = np.stack(((xmin + xmax) / 2, (ymin + ymax) / 2), axis=1)
        center_pts = torch.tensor(center_pts, dtype=torch.float32)
        size_pts = np.stack(((-xmin + xmax), (-ymin + ymax)), axis=1)
        size_pts = torch.tensor(size_pts, dtype=torch.float32)
        foot_pts = np.stack(((xmin + xmax) / 2, ymin), axis=1)
        foot_pts = torch.tensor(foot_pts, dtype=torch.float32)
        # head_pts = np.stack(((xmin + xmax) / 2, ymax), axis=1)
        # head_pts = torch.tensor(head_pts, dtype=torch.float32)

        for pt_idx, (pid, wh) in enumerate(zip(img_pids, size_pts)):
            for idx, pt in enumerate((center_pts[pt_idx], foot_pts[pt_idx])):
                if pt[0] < 0 or pt[0] >= W or pt[1] < 0 or pt[1] >= H:
                    continue
                basic.draw_umich_gaussian(center[idx], pt.int(), self.kernel_size)

            ct_int = center_pts[pt_idx].int()
            if ct_int[0] < 0 or ct_int[0] >= W or ct_int[1] < 0 or ct_int[1] >= H:
                continue
            valid_mask[:, ct_int[1], ct_int[0]] = 1
            offset[:, ct_int[1], ct_int[0]] = center_pts[pt_idx] - ct_int
            size[:, ct_int[1], ct_int[0]] = wh
            person_ids[:, ct_int[1], ct_int[0]] = pid

        return center, offset, size, skeleton, person_ids, valid_mask

    def sample_augmentation(self):
        """
        采样数据增强参数
        
        根据训练/验证模式生成不同的图像变换参数。
        训练时使用随机缩放和裁剪，验证时使用固定变换。
        
        Returns:
            tuple: (resize_dims, crop)
                - resize_dims: 缩放后的尺寸 (W, H)
                - crop: 裁剪参数 (x, y, x+W, y+H)
        """
        fH, fW = self.data_aug_conf['final_dim']
        if self.is_train:
            resize = np.random.uniform(*self.data_aug_conf['resize_lim'])
            resize_dims = (int(fW * resize), int(fH * resize))
            newW, newH = resize_dims

            # center it
            crop_h = int((newH - fH) / 2)
            crop_w = int((newW - fW) / 2)

            crop_offset = int(self.data_aug_conf['resize_lim'][0] * self.data_aug_conf['final_dim'][0])
            crop_w = crop_w + int(np.random.uniform(-crop_offset, crop_offset))
            crop_h = crop_h + int(np.random.uniform(-crop_offset, crop_offset))

            crop = (crop_w, crop_h, crop_w + fW, crop_h + fH)
        else:  # validation/test
            # do a perfect resize
            resize_dims = (fW, fH)
            crop_h = 0
            crop_w = 0
            crop = (crop_w, crop_h, crop_w + fW, crop_h + fH)
        return resize_dims, crop

    def get_image_data(self, index, cameras):
        """
        获取指定索引和相机的图像数据及相关标注
        
        加载图像，应用数据增强，更新相机内参，生成图像空间的真值数据。
        
        Args:
            index: 数据索引
            cameras: 相机ID列表
            
        Returns:
            tuple: 包含图像、内参、外参和各种真值的元组
        """
        imgs, intrins, extrins = [], [], []
        centers, offsets, sizes, skeletons, pids, valids = [], [], [], [], [], []
        frame = list(self.world_gt.keys())[index]
        for cam in cameras:
            img = Image.open(self.img_fpaths[cam][frame]).convert('RGB')
            W, H = img.size

            resize_dims, crop = self.sample_augmentation()
            sx = resize_dims[0] / float(W)
            sy = resize_dims[1] / float(H)

            extrin = self.calibration['extrinsic'][cam]
            intrin = self.calibration['intrinsic'][cam]
            intrin = geom.scale_intrinsics(intrin.unsqueeze(0), sx, sy).squeeze(0)

            fx, fy, x0, y0 = geom.split_intrinsics(intrin.unsqueeze(0))

            new_x0 = x0 - crop[0]
            new_y0 = y0 - crop[1]

            pix_T_cam = geom.merge_intrinsics(fx, fy, new_x0, new_y0)
            intrin = pix_T_cam.squeeze(0)  # 4,4
            img = basic.img_transform(img, resize_dims, crop)

            imgs.append(F.to_tensor(img))
            intrins.append(intrin)
            extrins.append(extrin)

            img_pts, img_pids = self.imgs_gt[frame][cam]
            center_img, offset_img, size_img, skeleton_img, pid_img, valid_img = self.get_img_gt(img_pts, img_pids,
                                                                                                 sx, sy, crop)

            centers.append(center_img)
            offsets.append(offset_img)
            sizes.append(size_img)
            skeletons.append(skeleton_img)
            pids.append(pid_img)
            valids.append(valid_img)

        return torch.stack(imgs), torch.stack(intrins), torch.stack(extrins), torch.stack(centers), torch.stack(
            offsets), torch.stack(sizes), torch.stack(skeletons), torch.stack(pids), torch.stack(valids)
    
    def load_camera_masks(self):
        """
        加载相机掩码信息
        
        从base数据集中获取每个相机的掩码，用于标识有效视野区域。
        掩码格式通常为二值图像，1表示有效区域，0表示无效区域。
        
        基于PyTorch数据集最佳实践，掩码应该与图像数据一起提供 [[0]](#__0)
        """
        print("Loading camera masks...")
        self.masks = {}
        mask_folder = os.path.join(self.root, 'Image_subsets', 'mask') # 掩码文件夹路径
        if not os.path.exists(mask_folder):
            print(f"Warning: Mask folder not found at {mask_folder}")
            for cam in self.used_cameras: # 为所有相机创建默认掩码
                self.masks[cam] = torch.ones(self.img_shape, dtype=torch.float32)
            return
        print(f"Looking for masks in: {mask_folder}")
        print(f"Used cameras: {self.used_cameras}")
        for cam in self.used_cameras:
            mask_file = os.path.join(mask_folder, f'C{cam+1}.png') # 掩码文件路径：mask/C{cam+1}.png
            if os.path.exists(mask_file):
                try:
                    mask_img = Image.open(mask_file) # 加载掩码图像
                    if mask_img.mode != 'L': # 转换为灰度图（如果不是的话）
                        mask_img = mask_img.convert('L')
                    mask_array = np.array(mask_img) # 转换为numpy数组并归一化到0-1范围
                    mask_tensor = torch.from_numpy(mask_array / 255.0).float()
                    self.masks[cam] = mask_tensor
                    # print(f"Loaded mask for camera {cam}: shape {mask_tensor.shape}")
                except Exception as e:
                    print(f"Error loading mask for camera {cam} from {mask_file}: {e}")
                    self.masks[cam] = torch.ones(self.img_shape, dtype=torch.float32) # 创建默认掩码
            else:
                print(f"Warning: Mask file not found for camera {cam} at {mask_file}")
                self.masks[cam] = torch.ones(self.img_shape, dtype=torch.float32) # 创建默认掩码
        print(f"Loaded masks for {len(self.masks)} cameras")
    
    def get_camera_mask(self, cam_idx, resize_dims=None, crop=None):
        """
        获取指定相机的掩码，并应用相同的变换
        
        Args:
            cam_idx: 相机索引
            resize_dims: 缩放尺寸 (W, H)
            crop: 裁剪参数 (x, y, x+W, y+H)
            
        Returns:
            torch.Tensor: 变换后的掩码张量
            
        基于自定义数据集加载的最佳实践 [[5]](#__5)
        """
        if cam_idx not in self.masks:
            # 如果没有该相机的掩码，返回全1掩码
            if resize_dims is not None:
                return torch.ones((resize_dims[1], resize_dims[0]), dtype=torch.float32)
            else:
                return torch.ones(self.img_shape, dtype=torch.float32)
        
        mask = self.masks[cam_idx].clone()
        
        # 如果需要应用变换
        if resize_dims is not None or crop is not None:
            # 转换为PIL图像进行变换
            mask_pil = F.to_pil_image(mask.unsqueeze(0))
            
            # 应用与图像相同的变换
            if resize_dims is not None:
                mask_pil = mask_pil.resize(resize_dims, Image.NEAREST)
            
            if crop is not None:
                mask_pil = mask_pil.crop(crop)
            
            # 转换回张量
            mask = F.to_tensor(mask_pil).squeeze(0)
        
        return mask

    def __len__(self):
        """
        返回数据集长度
        
        Returns:
            int: 数据集中的帧数
        """
        return len(self.world_gt.keys())

    def __getitem__(self, index):
        """
        获取数据集中的一个样本
        
        Args:
            index: 样本索引
            
        Returns:
            根据模式返回不同内容：
            - 预测模式：只返回item字典
            - 训练/验证模式：返回(item, target)元组
        """
        frame = list(self.world_gt.keys())[index]
        # cameras = list(range(self.num_cam))  # TODO: cam dropout?
        cameras = list(self.used_cameras)  # TODO: cam dropout?

        # images
        imgs, intrins, extrins, centers_img, offsets_img, sizes_img, skeletons_img, pids_img, valids_img = \
            self.get_image_data(index, cameras)

        worldcoord_from_worldgrid = torch.eye(4)
        worldcoord_from_worldgrid2d = torch.tensor(self.base.worldcoord_from_worldgrid_mat, dtype=torch.float32)
        worldcoord_from_worldgrid[:2, :2] = worldcoord_from_worldgrid2d[:2, :2]
        worldcoord_from_worldgrid[:2, 3] = worldcoord_from_worldgrid2d[:2, 2]
        worldgrid_T_worldcoord = torch.inverse(worldcoord_from_worldgrid)

        worldgrid_pts, world_pids = self.world_gt[frame]
        worldgrid_pts = torch.tensor(worldgrid_pts, dtype=torch.float32)
        worldgrid_pts = torch.cat((worldgrid_pts, torch.zeros_like(worldgrid_pts[:, 0:1])), dim=1).unsqueeze(0)

        if self.is_train:
            # yaw = random.uniform(-math.pi / 16, math.pi / 16)  # +- 180°
            # Rz = torch.tensor([[np.cos(yaw), -np.sin(yaw), 0],
            #                    [np.sin(yaw), np.cos(yaw), 0],
            #                    [0, 0, 1]])
            Rz = torch.eye(3)
            scene_center = torch.tensor([0., 0., 0.], dtype=torch.float32)
            off = self.base.worldcoord_from_worldgrid_mat[0, 0]
            scene_center[:2].uniform_(-off, off)
            augment = geom.merge_rt(Rz.unsqueeze(0), -scene_center.unsqueeze(0)).squeeze()
            worldgrid_T_worldcoord = torch.matmul(worldgrid_T_worldcoord, augment)
            worldgrid_pts = geom.apply_4x4(augment.unsqueeze(0), worldgrid_pts)

        mem_pts = self.vox_util.Ref2Mem(worldgrid_pts, self.Y, self.Z, self.X)
        center_bev, valid_bev, pid_bev, offset_bev = self.get_bev_gt(mem_pts[0], world_pids)

        grid_gt = torch.zeros((self.max_objects, 3), dtype=torch.long)
        grid_gt[:worldgrid_pts.shape[1], :2] = worldgrid_pts[0, :, :2]
        grid_gt[:worldgrid_pts.shape[1], 2] = torch.tensor(world_pids, dtype=torch.long)

        img_gt = self.imgs_gt[frame]
        gt_boxes = torch.zeros((len(img_gt), self.max_objects, 4))  #
        gt_ids = torch.zeros(len(img_gt), self.max_objects, dtype=torch.long)
        for cam_idx, (box, ids) in self.imgs_gt[frame].items():

            # hack sulution
            dict_keys = list(self.imgs_gt[frame].keys())
            gt_boxes[dict_keys.index(cam_idx), :len(ids)] = torch.tensor(box)
            gt_ids[dict_keys.index(cam_idx), :len(ids)] = torch.tensor(ids)

        item = {
            'img': imgs,  # S,3,H,W
            'depth': 0,
            'intrinsic': intrins,  # S,4,4
            'extrinsic': extrins,  # S,4,4
            'ref_T_global': worldgrid_T_worldcoord,  # 4,4
            'num_cameras': self.num_cam,
            'used_cameras': self.used_cameras,
            'town_path': self.root,
            'frame': frame,
            'grid_gt': grid_gt,
            'img_gt': (gt_boxes, gt_ids),
        }

        target = {
            # bev
            'valid_bev': valid_bev,  # 1,Y,X
            'center_bev': center_bev,  # 1,Y,X
            'offset_bev': offset_bev,  # 2,Y,X
            'pid_bev': pid_bev,  # 1,Y,X
            'size_bev': 0,
            'rotbin_bev': 0,
            'rotres_bev': 0,
            'depth': 0,
            # img
            'center_img': centers_img,  # S,1,H/8,W/8
            'offset_img': offsets_img,  # S,2,H/8,W/8
            'size_img': sizes_img,  # S,2,H/8,W/8
            'valid_img': valids_img,  # S,1,H/8,W/8
            'pid_img': pids_img  # S,1,H/8,W/8
        }
        # 在方法最后添加判断
        if self.is_predict:
            return item  # 预测时只返回 item 字典
        return item, target

if __name__ == '__main__':
    from torch.utils.data import DataLoader
    # data = MultiviewX('/home/s-jiang/Documents/datasets/MultiviewX') # /usr/home/tee/Developer/datasets/synthehicle
    data = Wildtrack('/home/s-jiang/Documents/datasets/Wildtrack') # /usr/home/tee/Developer/datasets/synthehicle
    print(len(data))
    loader = DataLoader(data, batch_size=16, shuffle=True)
    _, items = next(enumerate(loader))
