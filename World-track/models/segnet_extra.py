import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import utils.geom
import utils.vox
import utils.basic
from icecream import ic

from models.encoder import Encoder_res101, Encoder_res50, Encoder_res18_featDirection, Encoder_res18, Encoder_eff, \
    Encoder_swin_t, Encoder_fasterrcnn_resnet50_fpn, Encoder_fasterrcnn_resnet50_fpn_v2, Encoder_resnet18_fpn_v2
from models.decoders import DecoderBEV, DecoderImage


class Segnet_e(nn.Module):
    def __init__(self, Y, Z, X,
                 do_rgbcompress=True,
                 rand_flip=False,
                 latent_dim=512,  # 512, # 256,
                 feat2d_dim=128,
                 num_classes=2,
                 num_cameras=None,
                 num_ids=None,
                 z_sign=1,
                 encoder_type='res18',

                 waighting_mode='None',
                 feat_direction_mode='None',
                 aggregate_mode='mean',  # mean, max_pool, 2d_conv, 3d_conv
                 apply_mask_filter=True,

                 device=torch.device('cuda')):

        super(Segnet_e, self).__init__()
        assert (encoder_type in ['res101', 'res50', 'effb0', 'effb4', 'res18', 'res18_featDirection', 'vgg11',
                                 'swin_t', 'fasterrcnn_resnet50_fpn', 'fasterrcnn_resnet50_fpn_v2', 'Encoder_resnet18_fpn_v2'])

        self.Y, self.Z, self.X = Y, Z, X
        self.do_rgbcompress = do_rgbcompress
        self.rand_flip = rand_flip
        self.latent_dim = latent_dim
        self.encoder_type = encoder_type
        self.z_sign = z_sign
        self.num_cameras = num_cameras
        self.apply_mask_filter = apply_mask_filter

        self.waighting_mode = waighting_mode
        self.feat_direction = feat_direction_mode
        valid_modes = ['3d_conv', '2d_conv', 'mean', 'max_pool']
        if aggregate_mode not in valid_modes:
            raise ValueError(f"aggregate_mode must be one of {valid_modes}")
        self.aggregate_mode = aggregate_mode
        self.mean = torch.as_tensor([0.485, 0.456, 0.406], device=device).reshape(1, 3, 1, 1)
        self.std = torch.as_tensor([0.229, 0.224, 0.225], device=device).reshape(1, 3, 1, 1)

        # Encoder
        self.encoder_type = encoder_type
        if encoder_type == 'res101':
            self.encoder = Encoder_res101(feat2d_dim)
        elif encoder_type == 'res50':
            self.encoder = Encoder_res50(feat2d_dim)
        elif encoder_type == 'effb0':
            self.encoder = Encoder_eff(feat2d_dim, version='b0')
        elif encoder_type == 'res18':
            self.encoder = Encoder_res18(feat2d_dim)
        elif encoder_type == 'res18_featDirection':
            self.encoder = Encoder_res18_featDirection(feat2d_dim)
        elif encoder_type == 'Encoder_resnet18_fpn_v2':
            self.encoder = Encoder_resnet18_fpn_v2(feat2d_dim)
        elif encoder_type == 'swin_t':
            self.encoder = Encoder_swin_t(feat2d_dim)
        elif encoder_type == 'effb4':
            self.encoder = Encoder_eff(feat2d_dim, version='b4')
        elif encoder_type == 'fasterrcnn_resnet50_fpn':
            self.encoder = Encoder_fasterrcnn_resnet50_fpn(feat2d_dim)
        elif encoder_type == 'fasterrcnn_resnet50_fpn_v2':
            self.encoder = Encoder_fasterrcnn_resnet50_fpn_v2(feat2d_dim)
        else:
            raise ValueError(f"Unexpected encoder_type: {encoder_type}")

        add_dimension = 0
        if self.waighting_mode == 'concat':
            add_dimension += 2
        if self.feat_direction == 'concat':
            add_dimension += 3
        self.feat2d_dim = feat2d_dim + add_dimension

        # BEV compressor
        if self.aggregate_mode == '3d_conv':
            self.cam_compressor = nn.Sequential(
                nn.Conv3d(self.feat2d_dim * self.num_cameras, self.feat2d_dim, kernel_size=3, padding=1,
                          stride=1),
                nn.InstanceNorm3d(self.feat2d_dim), nn.ReLU(),
                nn.Conv3d(self.feat2d_dim, self.feat2d_dim, kernel_size=1),
            )
        elif self.aggregate_mode == '2d_conv':
            self.cam_slice_compressor = nn.ModuleList([nn.Sequential(
                nn.Conv2d(self.feat2d_dim * self.num_cameras, self.feat2d_dim, kernel_size=3, padding=1),
                nn.InstanceNorm3d(self.feat2d_dim), nn.ReLU(),
                nn.Conv2d(self.feat2d_dim, self.feat2d_dim, kernel_size=1),
            ) for _ in range(self.Z)
            ])
        elif self.aggregate_mode == 'mean':  # do not need a layer for avergae pooling
            pass
        elif self.aggregate_mode == 'max_pool':  # do not need a layer for max pooling
            pass

        self.bev_compressor = nn.Sequential(
            nn.Conv2d(self.feat2d_dim * self.Z, latent_dim, kernel_size=3, padding=1),
            nn.InstanceNorm2d(latent_dim), nn.ReLU(),
            nn.Conv2d(latent_dim, latent_dim, kernel_size=1),
        )

        # BEV decoder
        self.decoder = DecoderBEV(
            in_channels=latent_dim,
            n_ids=num_ids,
        )
        # image decoder
        self.decoder_image = DecoderImage(
            n_classes=num_classes,
            n_ids=num_ids,
        )
        # output right here!

        # Weights
        self.center_weight = nn.Parameter(torch.tensor(0.0), requires_grad=True)
        self.offset_weight = nn.Parameter(torch.tensor(0.0), requires_grad=True)
        self.tracking_weight = nn.Parameter(torch.tensor(0.0), requires_grad=True)
        self.size_weight = nn.Parameter(torch.tensor(0.0), requires_grad=True)
        self.rot_weight = nn.Parameter(torch.tensor(0.0), requires_grad=True)

    def feature_weighting(self, out_image_dict, feat_cams_, waighting_mode='multiply'):
        '''
        using the output of the detection head weight image features
        used detection head outputs BB center, pedestrian head position
        '''

        assert waighting_mode in ['multiply', 'concat', 'None']

        if waighting_mode == 'multiply':
            output_to_show = out_image_dict['img_center']

            # Compute mean and std for each image center across all image views
            mean_0 = torch.mean(output_to_show[:, 0, :, :], dim=(1, 2), keepdim=True)
            std_0 = torch.std(output_to_show[:, 0, :, :], dim=(1, 2), keepdim=True)
            mean_1 = torch.mean(output_to_show[:, 1, :, :], dim=(1, 2), keepdim=True)
            std_1 = torch.std(output_to_show[:, 1, :, :], dim=(1, 2), keepdim=True)

            # Normalize each image center
            normalized_center_0 = (output_to_show[:, 0, :, :] - mean_0) / std_0
            normalized_center_1 = (output_to_show[:, 1, :, :] - mean_1) / std_1

            # Apply sigmoid function
            sigmoid_center_0 = utils.basic._sigmoid(normalized_center_0)
            sigmoid_center_1 = utils.basic._sigmoid(normalized_center_1)

            # Sum the results
            weighting = sigmoid_center_0 + sigmoid_center_1

            new_featurs = feat_cams_ * weighting.unsqueeze(1)
        elif waighting_mode == 'concat':
            output_to_show = out_image_dict['img_center']
            new_featurs = torch.cat((feat_cams_, output_to_show), 1)

        elif waighting_mode == 'None':
            new_featurs = feat_cams_
        return new_featurs
    
    def _find_local_maxima(self, heatmap, threshold=0.1, kernel_size=3):
        """
        在热力图中找到局部最大值点
        
        Args:
            heatmap: 输入热力图 (H, W)
            threshold: 最小阈值
            kernel_size: 局部最大值检测的核大小
            
        Returns:
            torch.Tensor: 局部最大值的布尔掩码
        """
        # 应用阈值
        above_threshold = heatmap > threshold
        
        # 使用最大池化找到局部最大值
        pad = kernel_size // 2
        heatmap_padded = F.pad(heatmap.unsqueeze(0).unsqueeze(0), 
                            (pad, pad, pad, pad), mode='constant', value=0)
        local_max = F.max_pool2d(heatmap_padded, kernel_size, stride=1, padding=0)
        local_max = local_max.squeeze(0).squeeze(0)
        
        # 找到与原始值相等的位置（即局部最大值）
        local_maxima = (heatmap == local_max) & above_threshold
        
        return local_maxima
    
    def _filter_predictions_by_mask_with_bbox(self, predictions, masks, img_sizes, threshold=0.5):
        """
        使用掩码过滤预测结果 - 基于重构的边界框信息
        
        Args:
            predictions: 图像空间的预测结果字典
            masks: 相机掩码张量 (B*S, H, W) 或 (B*S, 1, H, W)
            img_sizes: 原始图像尺寸列表 [(H, W), ...]
            threshold: 掩码阈值，默认0.5
            
        Returns:
            dict: 过滤后的预测结果
        """
        if not self.apply_mask_filter or masks is None:
            return predictions
            
        try:
            # 获取预测输出
            img_center = predictions.get('img_center')    # (B*S, n_classes, H_feat, W_feat)
            img_offset = predictions.get('img_offset')    # (B*S, 2, H_feat, W_feat)
            img_size = predictions.get('img_size')        # (B*S, 2, H_feat, W_feat)
            img_id_feat = predictions.get('img_id_feat')  # (B*S, feat_dim, H_feat, W_feat)
            
            if img_center is None or img_size is None:
                print("Warning: Missing required predictions for bbox reconstruction")
                return predictions
                
            B_S, n_classes, H_feat, W_feat = img_center.shape
            device = img_center.device
            
            # 处理掩码维度
            if masks.dim() == 4 and masks.shape[1] == 1:
                masks = masks.squeeze(1)  # (B*S, H, W)
            
            # 计算特征图到原图的缩放比例
            scale_factors = []
            for cam_idx in range(B_S):
                if isinstance(img_sizes, list) and len(img_sizes) > cam_idx:
                    orig_h, orig_w = img_sizes[cam_idx]
                else:
                    orig_h, orig_w = masks[cam_idx].shape
                
                scale_y = orig_h / H_feat
                scale_x = orig_w / W_feat
                scale_factors.append((scale_y, scale_x))
            
            # 为每个相机视角应用掩码过滤
            filtered_center = torch.zeros_like(img_center)
            filtered_offset = torch.zeros_like(img_offset) if img_offset is not None else None
            filtered_size = torch.zeros_like(img_size)
            filtered_id_feat = torch.zeros_like(img_id_feat) if img_id_feat is not None else None
            
            for cam_idx in range(B_S):
                mask = masks[cam_idx]  # (H_orig, W_orig)
                scale_y, scale_x = scale_factors[cam_idx]
                
                # 将掩码调整到特征图尺寸用于快速预筛选
                mask_feat = F.interpolate(
                    mask.float().unsqueeze(0).unsqueeze(0), 
                    size=(H_feat, W_feat), 
                    mode='nearest'
                ).squeeze(0).squeeze(0) > threshold
                
                for class_idx in range(n_classes):
                    center_map = img_center[cam_idx, class_idx]  # (H_feat, W_feat)
                    size_map = img_size[cam_idx]  # (2, H_feat, W_feat) - [width, height]
                    
                    # 找到潜在的目标中心点
                    local_maxima = self._find_local_maxima(center_map, threshold=0.1)
                    peak_coords = torch.nonzero(local_maxima, as_tuple=False)  # (N, 2) - [y, x]
                    
                    if len(peak_coords) == 0:
                        continue
                    
                    # 对每个检测到的目标进行边界框重构和掩码检查
                    valid_peaks = torch.zeros_like(local_maxima, dtype=torch.bool)
                    
                    for peak in peak_coords:
                        y_feat, x_feat = peak[0].item(), peak[1].item()
                        
                        # 获取中心点偏移（如果有的话）
                        if img_offset is not None:
                            offset_y = img_offset[cam_idx, 0, y_feat, x_feat].item()
                            offset_x = img_offset[cam_idx, 1, y_feat, x_feat].item()
                        else:
                            offset_y, offset_x = 0.0, 0.0
                        
                        # 计算精确的中心点位置（特征图坐标）
                        center_y_feat = y_feat + offset_y
                        center_x_feat = x_feat + offset_x
                        
                        # 获取目标尺寸（特征图坐标）
                        width_feat = img_size[cam_idx, 0, y_feat, x_feat].item()
                        height_feat = img_size[cam_idx, 1, y_feat, x_feat].item()
                        
                        # 转换到原图坐标系
                        center_x_orig = center_x_feat * scale_x
                        center_y_orig = center_y_feat * scale_y
                        width_orig = width_feat * scale_x
                        height_orig = height_feat * scale_y
                        
                        # 重构边界框坐标
                        xmin = max(0, int(center_x_orig - width_orig / 2))
                        ymin = max(0, int(center_y_orig - height_orig / 2))
                        xmax = min(mask.shape[1] - 1, int(center_x_orig + width_orig / 2))
                        ymax = min(mask.shape[0] - 1, int(center_y_orig + height_orig / 2))
                        
                        # 检查边界框的有效性
                        if xmin >= xmax or ymin >= ymax:
                            continue
                        
                        # 方法1: 检查边界框下中心点
                        bottom_center_x = int(center_x_orig)
                        bottom_center_y = min(ymax, mask.shape[0] - 1)
                        
                        # 方法2: 检查边界框内的掩码覆盖率
                        bbox_mask_region = mask[ymin:ymax+1, xmin:xmax+1]
                        if bbox_mask_region.numel() > 0:
                            mask_coverage = bbox_mask_region.float().mean().item()
                        else:
                            mask_coverage = 0.0
                        
                        # 方法3: 检查边界框关键点
                        key_points = [
                            (bottom_center_x, bottom_center_y),  # 下中心点
                            (int(center_x_orig), int(center_y_orig)),  # 中心点
                            (xmin, ymax),  # 左下角
                            (xmax, ymax),  # 右下角
                        ]
                        
                        valid_key_points = 0
                        for kx, ky in key_points:
                            if (0 <= kx < mask.shape[1] and 0 <= ky < mask.shape[0] and 
                                mask[ky, kx] > threshold):
                                valid_key_points += 1
                        
                        # 综合判断：下中心点有效 OR 掩码覆盖率足够 OR 关键点大部分有效
                        is_valid = (
                            (0 <= bottom_center_x < mask.shape[1] and 
                            0 <= bottom_center_y < mask.shape[0] and 
                            mask[bottom_center_y, bottom_center_x] > threshold)
                            # or (mask_coverage > 0.3)  # 30%以上的区域有效
                            # or (valid_key_points >= 2)   # 至少2个关键点有效
                        )
                        
                        if is_valid:
                            valid_peaks[y_feat, x_feat] = True
                    
                    # 应用过滤
                    filtered_center[cam_idx, class_idx] = center_map * valid_peaks.float()
                    
                    # 对其他输出应用相同的掩码
                    if filtered_offset is not None:
                        valid_mask_2d = valid_peaks.unsqueeze(0).float()  # (1, H_feat, W_feat)
                        filtered_offset[cam_idx] = img_offset[cam_idx] * valid_mask_2d
                    
                    valid_mask_2d = valid_peaks.unsqueeze(0).float()
                    filtered_size[cam_idx] = img_size[cam_idx] * valid_mask_2d
                    
                    if filtered_id_feat is not None:
                        filtered_id_feat[cam_idx] = img_id_feat[cam_idx] * valid_mask_2d
            
            # 构建过滤后的预测结果
            filtered_predictions = predictions.copy()
            filtered_predictions['img_center'] = filtered_center
            if filtered_offset is not None:
                filtered_predictions['img_offset'] = filtered_offset
            filtered_predictions['img_size'] = filtered_size
            if filtered_id_feat is not None:
                filtered_predictions['img_id_feat'] = filtered_id_feat
                
            return filtered_predictions
            
        except Exception as e:
            print(f"Error in _filter_predictions_by_mask_with_bbox: {str(e)}")
            import traceback
            traceback.print_exc()
            return predictions

    def _extract_detections_from_predictions(self, predictions, conf_threshold=0.3):
        """
        从网络预测中提取检测结果（用于调试和可视化）
        
        Args:
            predictions: 网络预测结果
            conf_threshold: 置信度阈值
            
        Returns:
            list: 每个相机的检测结果 [{'boxes': tensor, 'scores': tensor, 'labels': tensor}, ...]
        """
        img_center = predictions.get('img_center')  # (B*S, n_classes, H_feat, W_feat)
        img_offset = predictions.get('img_offset')  # (B*S, 2, H_feat, W_feat)
        img_size = predictions.get('img_size')      # (B*S, 2, H_feat, W_feat)
        
        if img_center is None or img_size is None:
            return []
        
        B_S, n_classes, H_feat, W_feat = img_center.shape
        detections = []
        
        for cam_idx in range(B_S):
            cam_detections = {'boxes': [], 'scores': [], 'labels': []}
            
            for class_idx in range(n_classes):
                center_map = img_center[cam_idx, class_idx]
                
                # 找到高置信度的检测点
                high_conf_mask = center_map > conf_threshold
                if not high_conf_mask.any():
                    continue
                    
                # 找到局部最大值
                local_maxima = self._find_local_maxima(center_map, threshold=conf_threshold)
                peak_coords = torch.nonzero(local_maxima, as_tuple=False)
                
                for peak in peak_coords:
                    y_feat, x_feat = peak[0].item(), peak[1].item()
                    confidence = center_map[y_feat, x_feat].item()
                    
                    # 获取偏移和尺寸
                    if img_offset is not None:
                        offset_y = img_offset[cam_idx, 0, y_feat, x_feat].item()
                        offset_x = img_offset[cam_idx, 1, y_feat, x_feat].item()
                    else:
                        offset_y, offset_x = 0.0, 0.0
                    
                    center_y = y_feat + offset_y
                    center_x = x_feat + offset_x
                    width = img_size[cam_idx, 0, y_feat, x_feat].item()
                    height = img_size[cam_idx, 1, y_feat, x_feat].item()
                    
                    # 构建边界框
                    xmin = center_x - width / 2
                    ymin = center_y - height / 2
                    xmax = center_x + width / 2
                    ymax = center_y + height / 2
                    
                    cam_detections['boxes'].append([xmin, ymin, xmax, ymax])
                    cam_detections['scores'].append(confidence)
                    cam_detections['labels'].append(class_idx)
            
            # 转换为张量
            if cam_detections['boxes']:
                cam_detections['boxes'] = torch.tensor(cam_detections['boxes'], 
                                                    device=img_center.device)
                cam_detections['scores'] = torch.tensor(cam_detections['scores'], 
                                                    device=img_center.device)
                cam_detections['labels'] = torch.tensor(cam_detections['labels'], 
                                                    device=img_center.device, dtype=torch.long)
            else:
                cam_detections['boxes'] = torch.empty((0, 4), device=img_center.device)
                cam_detections['scores'] = torch.empty(0, device=img_center.device)
                cam_detections['labels'] = torch.empty(0, device=img_center.device, dtype=torch.long)
            
            detections.append(cam_detections)
        
        return detections

    def forward(self, rgb_cams, pix_T_cams, cams_T_global, vox_util, ref_T_global, masks=None, prev_bev=None):
        """
        B = batch size, S = number of cameras, C = 3, H = img height, W = img width
        rgb_cams: (B,S,C,H,W)
        pix_T_cams: (B,S,4,4)
        cams_T_global: (B,S,4,4)
        vox_util: vox util object
        ref_T_global: (B,4,4)
        masks: (B,S,H,W) 相机掩码，新增参数
        prev_bev: 前一帧BEV特征（可选）
        rad_occ_mem0:
            - None when use_radar = False, use_lidar = False
            - (B, 1, Z, Y, X) when use_radar = True, use_metaradar = False
            - (B, 16, Z, Y, X) when use_radar = True, use_metaradar = True
            - (B, 1, Z, Y, X) when use_lidar = True
        """
        B, S, C, H, W = rgb_cams.shape
        assert (C == 3)
        # reshape tensors
        __p = lambda x: utils.basic.pack_seqdim(x, B)
        __u = lambda x: utils.basic.unpack_seqdim(x, B)
        rgb_cams_ = __p(rgb_cams)  # B*S,3,H,W
        pix_T_cams_ = __p(pix_T_cams)  # B*S,4,4
        cams_T_global_ = __p(cams_T_global)  # B*S,4,4

        ################################################ 处理掩码 - 新增代码块
        masks_ = None
        img_sizes = [(H, W)] * (B * S)  # 默认图像尺寸
        if masks is not None:
            masks_ = __p(masks)  # B*S,H,W 或 B*S,1,H,W
            # 如果掩码尺寸与输入图像不同，更新img_sizes
            if masks.dim() == 4:  # (B,S,H,W)
                img_sizes = [(masks.shape[2], masks.shape[3])] * (B * S)
            elif masks.dim() == 5:  # (B,S,1,H,W)
                img_sizes = [(masks.shape[3], masks.shape[4])] * (B * S)

        global_T_cams_ = torch.inverse(cams_T_global_)  # B*S,4,4
        ref_T_cams = torch.matmul(ref_T_global.repeat(S, 1, 1), global_T_cams_)  # B*S,4,4
        cams_T_ref_ = torch.inverse(ref_T_cams)  # B*S,4,4
        ################################################

        # rgb encoder
        device = rgb_cams_.device
        rgb_cams_ = (rgb_cams_ - self.mean.to(device)) / self.std.to(device)
        if self.rand_flip:
            B0, _, _, _ = rgb_cams_.shape
            self.rgb_flip_index = np.random.choice([0, 1], B0).astype(bool)
            rgb_cams_[self.rgb_flip_index] = torch.flip(rgb_cams_[self.rgb_flip_index], [-1])

        if self.encoder_type == 'res18_featDirection':
            # featDirection = torch.zeros_like(rgb_cams_,device=device)
            featDirection = vox_util.camera_viewing_direction(rgb_cams_, pix_T_cams_, cams_T_ref_)  # B*S,1
            encoder_input = torch.cat((rgb_cams_, featDirection), dim=1)
            feat_cams_ = self.encoder(encoder_input)  # B*S,128,H/8,W/8

        else:
            feat_cams_ = self.encoder(rgb_cams_)  # B*S,128,H/8,W/8 torch.Size([4, 128, 180, 320])

        if self.rand_flip:
            feat_cams_[self.rgb_flip_index] = torch.flip(feat_cams_[self.rgb_flip_index], [-1])
        _, C, Hf, Wf = feat_cams_.shape
        sy = Hf / float(H)
        sx = Wf / float(W)

        out_image_dict = self.decoder_image(feat_cams_) # image decoder
        ################################################ 应用掩码过滤 - 新增代码块
        if self.apply_mask_filter and masks_ is not None:
            out_image_dict = self._filter_predictions_by_mask_with_bbox(out_image_dict, masks_, img_sizes)
        ################################################

        Y, Z, X = self.Y, self.Z, self.X
        featpix_T_cams_ = utils.geom.scale_intrinsics(pix_T_cams_, sx, sy)
        # unproject image feature to 3d grid

        feat_cams_ = self.feature_weighting(out_image_dict, feat_cams_, waighting_mode=self.waighting_mode)

        feat_mems_ = vox_util.unproject_image_to_mem(
            feat_cams_,  # B*S,128,H/8,W/8
            utils.basic.matmul2(featpix_T_cams_, cams_T_ref_),  # featpix_T_ref  B*S,4,4
            cams_T_ref_, Y, Z, X, xyz_refA=None, z_sign=self.z_sign, mode='bilinear')

        # concatenate feat_direction to the image features
        if self.feat_direction == 'concat':
            feat_direction_ = vox_util.pixel_viewing_direction(feat_cams_,  # B*S,128,H/8,W/8
                                                               featpix_T_cams_,  # featpix_T_ref  B*S,4,4
                                                               cams_T_ref_,
                                                               utils.basic.matmul2(featpix_T_cams_, cams_T_ref_),
                                                               Y, Z, X)
            feat_mems_ = torch.cat((feat_mems_, feat_direction_), 1)

        feat_mems = __u(feat_mems_)  # B, S, C, Z, Y, X

        # aggregate features
        if self.aggregate_mode == '3d_conv':
            feat_mem = self.cam_compressor(feat_mems.flatten(1, 2))
        elif self.aggregate_mode == '2d_conv':
            feat_mems = feat_mems.flatten(1, 2)
            batch_size, channels, height, depth, width = feat_mems.shape
            all_slices_out = []
            # Apply conv and pooling to each slice separately
            for i in range(depth):
                slice_i = feat_mems[:, :, :, i, :]  # Extract the i-th depth slice
                slice_i = slice_i.view(batch_size, channels, height, width)  # Reshape for 2D conv
                out_slice_i = self.cam_slice_compressor[i](slice_i)  # Apply the i-th camera compressor
                all_slices_out.append(out_slice_i)

            # Stack all 2D processed slices along the depth dimension
            feat_mem = torch.stack(all_slices_out, dim=3)

        elif self.aggregate_mode == 'max_pool': # max pooling
            feat_mem, _ = torch.max(feat_mems, dim=1)  # maximum values, indices
        elif self.aggregate_mode == 'mean': # average pooling
            feat_mem = torch.mean(feat_mems, dim=1)
        else:
            raise ValueError(f"Unexpected aggregate_mode: {self.aggregate_mode}")

        if self.rand_flip:
            self.bev_flip1_index = np.random.choice([0, 1], B).astype(bool)
            self.bev_flip2_index = np.random.choice([0, 1], B).astype(bool)
            feat_mem[self.bev_flip1_index] = torch.flip(feat_mem[self.bev_flip1_index], [-1])
            feat_mem[self.bev_flip2_index] = torch.flip(feat_mem[self.bev_flip2_index], [-3])

        # ic(feat_mem.permute(0, 1, 3, 2, 4).shape, self.feat2d_dim)
        feat_bev_ = feat_mem.permute(0, 1, 3, 2, 4).reshape(B, self.feat2d_dim * Z, Y, X)
        feat_bev = self.bev_compressor(feat_bev_)

        # bev decoder
        # torch.Size([1, 256, 120, 360])
        out_BEV_dict = self.decoder(feat_bev)
        out_dict = {**out_BEV_dict, **out_image_dict} # whole output of bev and image decoder


        return out_dict
