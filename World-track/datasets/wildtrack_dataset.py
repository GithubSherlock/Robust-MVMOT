import os
import numpy as np
import cv2
import xml.etree.ElementTree as ET
import re
from torchvision.datasets import VisionDataset
# 定义所有摄像机的内参标定文件名
intrinsic_camera_matrix_filenames = ['intr_CVLab1.xml', 'intr_CVLab2.xml', 'intr_CVLab3.xml', 'intr_CVLab4.xml',
                                     'intr_IDIAP1.xml', 'intr_IDIAP2.xml', 'intr_IDIAP3.xml']
# 定义所有摄像机的外参标定文件名
extrinsic_camera_matrix_filenames = ['extr_CVLab1.xml', 'extr_CVLab2.xml', 'extr_CVLab3.xml', 'extr_CVLab4.xml',
                                     'extr_IDIAP1.xml', 'extr_IDIAP2.xml', 'extr_IDIAP3.xml']

# USED_CAMERAS = [0, 1, 2]#, 3, 4, 5, 6]
# USED_CAMERAS = [3, 1, 2]
# TEST_CAMERAS = [4, 5, 6]

class Wildtrack(VisionDataset):
    """
    Wildtrack多视角行人检测数据集类
    
    该数据集包含7个摄像机视角的行人检测数据，支持多视角行人跟踪和检测任务。
    数据集特点：
    - 图像尺寸: 1080x1920 (H×W)
    - 世界网格: 480x1440
    - 坐标系统: ij-indexing (与常见的xy-indexing不同)
    - 单位: 厘米(cm)
    """
    def __init__(self, root, train_cameras, test_cameras, is_test=False):
        """
        初始化Wildtrack数据集
        
        Args:
            root (str): 数据集根目录路径
            train_cameras (list): 训练时使用的摄像机ID列表
            test_cameras (list): 测试时使用的摄像机ID列表  
            is_test (bool): 是否为测试模式，默认False
        """
        super().__init__(root)
        # image of shape C,H,W (C,N_row,N_col); xy indexging; x,y (w,h) (n_col,n_row)
        # WILDTRACK has ij-indexing: H*W=480*1440, thus x (i) is \in [0,480), y (j) is \in [0,1440)
        # WILDTRACK has in-consistent unit: centi-meter (cm) for calibration & pos annotation
        self.__name__ = 'Wildtrack' # 设置数据集名称
        # 定义图像尺寸和世界网格尺寸 [H, W] 和 [N_row, N_col]
        # 图像: 1080行×1920列, 世界网格: 480行×1440列
        self.img_shape, self.worldgrid_shape = [1080, 1920], [480, 1440]  # H,W; N_row,N_col
        
        self.train_cameras = train_cameras # 保存训练和测试摄像机配置
        self.test_cameras = test_cameras

        if is_test: # 根据模式选择使用的摄像机
            self.used_cameras = self.test_cameras
        else:
            self.used_cameras = self.train_cameras

        self.num_cam = len(self.used_cameras) # 设置摄像机数量和总帧数
        self.num_frame = 2000
        # 世界坐标系到世界网格的变换矩阵。用于将世界网格坐标转换为实际世界坐标(厘米)
        # world x,y actually means i,j in Wildtrack, which correspond to h,w
        self.worldcoord_from_worldgrid_mat = np.array([[0, 2.5, -300], [2.5, 0, -900], [0, 0, 1]])
        # 获取所有7个摄像机的内参和外参矩阵。使用zip和列表推导式同时获取内参和外参
        self.intrinsic_matrices, self.extrinsic_matrices = zip(
            *[self.get_intrinsic_extrinsic_matrix(cam) for cam in range(7)])  # 7 is all cameras
        self.masks = self._load_camera_masks() # 加载摄像机掩码

    def _load_camera_masks(self):
        """
        加载每个摄像机视角对应的掩码图像
        
        掩码用于标识每个摄像机视角中的有效区域，通常用于：
        - 去除视角中的无关区域
        - 标识摄像机的可视范围
        - 提高检测精度
        
        Returns:
            dict: 字典，键为相机ID，值为对应的掩码图像(二值化的uint8矩阵)
                 掩码中1表示有效区域，0表示无效区域
        """
        masks = {}
        mask_dir = os.path.join(self.root, 'Image_subsets', 'mask')
        for cam in self.used_cameras:
            mask_path = os.path.join(mask_dir, f'C{cam+1}.png')
            if not os.path.exists(mask_path):
                raise FileNotFoundError(f'Mask file not found: {mask_path}')
            
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE) # 以灰度模式读取掩码图像并确保尺寸正确
            # 如果尺寸不匹配，进行resize操作。注意：cv2.resize的参数顺序是(width, height)
            if mask.shape != (self.img_shape[0], self.img_shape[1]):
                mask = cv2.resize(mask, (self.img_shape[1], self.img_shape[0]))
            mask = (mask // 255).astype(np.uint8) # 将白色(255)转换为1，黑色(0)保持为0
            masks[cam] = mask # 存储处理后的掩码
            
        return masks
    
    def parse_camera_number(self, folder_name):
        """
        解析相机文件夹名称，返回相机编号
        支持格式：C1, C2, Camera1, cam1, 1, etc.
        """
        import re
        
        # 跳过明显的非相机文件夹
        skip_folders = ['mask', 'masks', 'annotation', 'annotations', 'gt', 'groundtruth']
        if folder_name.lower() in skip_folders:
            return None
            
        # 提取数字
        numbers = re.findall(r'\d+', folder_name)
        if not numbers:
            return None
            
        try:
            cam_num = int(numbers[0])
            # 转换为0基索引
            return cam_num - 1 if cam_num > 0 else cam_num
        except ValueError:
            return None
        
    def get_image_fpaths(self, frame_range):
        """
        获取指定帧范围内所有摄像机的图像文件路径
        
        Args:
            frame_range (range or list): 需要获取的帧编号范围
            
        Returns:
            dict: 嵌套字典结构 {camera_id: {frame_id: file_path}}
                外层键为摄像机ID，内层键为帧ID，值为图像文件完整路径
        """
        img_fpaths = {cam: {} for cam in self.used_cameras}
        img_subsets_path = os.path.join(self.root, 'Image_subsets')
        
        for camera_folder in sorted(os.listdir(img_subsets_path)):
            camera_path = os.path.join(img_subsets_path, camera_folder)
            if not os.path.isdir(camera_path):
                continue
                
            # 使用解析函数获取相机编号
            cam = self.parse_camera_number(camera_folder)
            if cam is None:
                print(f"Info: Skipping non-camera folder '{camera_folder}'")
                continue
                
            # 验证相机编号是否在使用的相机列表中
            if cam not in self.used_cameras:
                continue
                
            # 处理该相机的图像文件
            for fname in sorted(os.listdir(camera_path)):
                if not fname.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                    continue
                    
                try:
                    frame = int(fname.split('.')[0])
                    if frame in frame_range:  # 只保存在指定帧范围内的图像路径
                        img_fpaths[cam][frame] = os.path.join(camera_path, fname)
                except (ValueError, IndexError):
                    print(f"Warning: Cannot parse frame number from filename '{fname}' in camera {cam}")
                    continue
        
        return img_fpaths

    # def get_image_fpaths(self, frame_range):
    #     """
    #     获取指定帧范围内所有摄像机的图像文件路径
        
    #     Args:
    #         frame_range (range or list): 需要获取的帧编号范围
            
    #     Returns:
    #         dict: 嵌套字典结构 {camera_id: {frame_id: file_path}}
    #              外层键为摄像机ID，内层键为帧ID，值为图像文件完整路径
    #     """
    #     img_fpaths = {cam: {} for cam in self.used_cameras}  # range(self.num_cam)}
    #     for camera_folder in sorted(os.listdir(os.path.join(self.root, 'Image_subsets'))):
    #         cam = int(camera_folder[-1]) - 1
    #         # if cam >= self.num_cam:
    #         if cam not in self.used_cameras:
    #             continue
    #         for fname in sorted(os.listdir(os.path.join(self.root, 'Image_subsets', camera_folder))):
    #             frame = int(fname.split('.')[0])
    #             if frame in frame_range: # 只保存在指定帧范围内的图像路径
    #                 img_fpaths[cam][frame] = os.path.join(self.root, 'Image_subsets', camera_folder, fname)
    #     return img_fpaths

    def get_worldgrid_from_pos(self, pos):
        """
        将一维位置索引转换为二维世界网格坐标
        
        Wildtrack数据集使用一维索引来表示世界网格中的位置，
        该函数将其转换为二维网格坐标(grid_x, grid_y)
        
        Args:
            pos (int): 一维位置索引
            
        Returns:
            np.ndarray: 二维网格坐标 [grid_x, grid_y]
                       grid_x: 行索引 (0 to 479)
                       grid_y: 列索引 (0 to 1439)
        """
        # 世界网格形状为480×1440，使用行优先的一维索引
        # pos = grid_x * 480 + grid_y
        # 因此: grid_y = pos % 480, grid_x = pos // 480
        grid_y = pos % 480
        grid_x = pos // 480
        return np.array([grid_x, grid_y], dtype=int)

    def get_intrinsic_extrinsic_matrix(self, camera_i):
        intrinsic_camera_path = os.path.join(self.root, 'calibrations', 'intrinsic_zero')
        intrinsic_params_file = cv2.FileStorage(os.path.join(intrinsic_camera_path,
                                                             intrinsic_camera_matrix_filenames[camera_i]),
                                                flags=cv2.FILE_STORAGE_READ)
        intrinsic_matrix = intrinsic_params_file.getNode('camera_matrix').mat()
        intrinsic_params_file.release()

        extrinsic_params_file_root = ET.parse(os.path.join(self.root, 'calibrations', 'extrinsic',
                                                           extrinsic_camera_matrix_filenames[camera_i])).getroot()

        rvec = extrinsic_params_file_root.findall('rvec')[0].text.lstrip().rstrip().split(' ')
        rvec = np.array(list(map(lambda x: float(x), rvec)), dtype=np.float32)

        tvec = extrinsic_params_file_root.findall('tvec')[0].text.lstrip().rstrip().split(' ')
        tvec = np.array(list(map(lambda x: float(x), tvec)), dtype=np.float32)

        rotation_matrix, _ = cv2.Rodrigues(rvec)
        translation_matrix = np.array(tvec, dtype=np.float32).reshape(3, 1)
        extrinsic_matrix = np.hstack((rotation_matrix, translation_matrix))

        return intrinsic_matrix, extrinsic_matrix
