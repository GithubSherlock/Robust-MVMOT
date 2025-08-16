import os
import torch
import torch.nn as nn
import torchvision
import torch.nn.functional as F
from torchvision.models.detection import FasterRCNN_ResNet50_FPN_Weights, fasterrcnn_resnet50_fpn
from efficientnet_pytorch import EfficientNet
from timm.utils.model import freeze_batch_norm_2d


def set_bn_momentum(model, momentum=0.1):
    for m in model.modules():
        if isinstance(m, (nn.InstanceNorm1d, nn.InstanceNorm2d, nn.InstanceNorm3d)):
            m.momentum = momentum


def freeze_bn(model):
    for n, module in model.named_children():
        if len(list(module.children())) > 0:
            freeze_bn(module)

        if isinstance(module, torch.nn.BatchNorm2d):
            setattr(model, n, freeze_batch_norm_2d(module))


class UpsamplingConcat(nn.Module):
    def __init__(self, in_channels, out_channels, scale_factor=2):
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=scale_factor, mode='bilinear', align_corners=False)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)

    def forward(self, x_to_upsample, x):
        x_to_upsample = self.upsample(x_to_upsample)
        x_to_upsample = torch.cat([x, x_to_upsample], dim=1)
        return self.conv(x_to_upsample)


class MultiViewFeatureFusion(nn.Module): # FeatureWeightingModule
    """多视角特征增强模块"""
    def __init__(self, channels):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Conv2d(channels, channels // 4, 1),
            nn.BatchNorm2d(channels // 4),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // 4, channels, 1),
            nn.Sigmoid()
        )
    def forward(self, x):
        # 简单的注意力机制增强特征
        att = self.attention(x)
        return x * att


class Encoder_fasterrcnn_resnet50_fpn(nn.Module):
    # pretrained_path='/home/s-jiang/Documents/Robust-MVMOT/fasterrcnn_wildtrack/model_fasterrcnn_resnet50_fpn_0/best_map_0.6137.pth'
    def __init__(self, C, pretrained_path=None): # 
        super().__init__()
        self.C = C
        self.pretrained_path = pretrained_path
        if pretrained_path and os.path.exists(pretrained_path):
            print(f"Loading local Faster R-CNN model from: {pretrained_path}")
            checkpoint = torch.load(pretrained_path, map_location='cpu')
            from torchvision.models.detection import fasterrcnn_resnet50_fpn
            fasterrcnn = fasterrcnn_resnet50_fpn(num_classes=2)  # 或者根据您的数据集调整类别数
            if 'model_state_dict' in checkpoint: # 加载权重
                fasterrcnn.load_state_dict(checkpoint['model_state_dict'])
            elif 'state_dict' in checkpoint:
                fasterrcnn.load_state_dict(checkpoint['state_dict'])
            else:
                fasterrcnn.load_state_dict(checkpoint)
        else:
            print("Loading default COCO pretrained Faster R-CNN model")
            from torchvision.models.detection import FasterRCNN_ResNet50_FPN_Weights
            fasterrcnn = torchvision.models.detection.fasterrcnn_resnet50_fpn(
                weights=FasterRCNN_ResNet50_FPN_Weights.COCO_V1)
        resnet = fasterrcnn.backbone.body # 直接使用ResNet50作为backbone（去掉FPN部分）
        freeze_bn(resnet)
        self.layer0 = nn.Sequential(*list(resnet.children())[:4])  # conv1, bn1, relu, maxpool
        self.layer1 = resnet.layer1  # 256通道
        self.layer2 = resnet.layer2  # 512通道
        self.layer3 = resnet.layer3  # 1024通道
        # 冻结早期层
        # self._freeze_layers()
        # 简单的上采样结构（参考其他Encoder）
        self.upsampling_layer1 = UpsamplingConcat(1024 + 512, 512)
        self.upsampling_layer2 = UpsamplingConcat(512 + 256, 512)
        self.depth_layer = nn.Conv2d(512, self.C, kernel_size=1, bias=False)
        
    def _freeze_layers(self):
        """冻结指定的层"""
        freeze_layer_names = ["conv1", "layer1", "layer2.0", "layer2.1"]
        for name, param in self.named_parameters():
            should_freeze = any(freeze_name in name for freeze_name in freeze_layer_names)
            if should_freeze:
                param.requires_grad = False
    
    def forward(self, x):
        x0 = self.layer0(x)  # 1/4分辨率
        x1 = self.layer1(x0) # 1/4分辨率, 256通道
        x2 = self.layer2(x1) # 1/8分辨率, 512通道
        x3 = self.layer3(x2) # 1/16分辨率, 1024通道
        x = self.upsampling_layer1(x3, x2) # 简单的上采样融合（与ResNet50 Encoder保持一致）
        x = self.upsampling_layer2(x, x1)
        x = self.depth_layer(x)

        return x


class Encoder_fasterrcnn_resnet50_fpn_v2(nn.Module):
    """改进版FasterRCNN-ResNet50-FPN编码器 - MV-MOD & MOT优化"""
    # pretrained_path='/home/s-jiang/Documents/Robust-MVMOT/fasterrcnn_wildtrack/model_fasterrcnn_resnet50_fpn_0/best_map_0.6137.pth'
    def __init__(self, C, pretrained_path=None, use_fpn=True, freeze_strategy='mv_mot_hybrid'):
        super().__init__()
        self.C = C
        self.use_fpn = use_fpn
        self.freeze_strategy = freeze_strategy
        self.training_stage = 'initial'  # 训练阶段控制
        self._load_pretrained_model(pretrained_path) # 加载预训练模型
        if use_fpn:
            print("🔄 Setting up FPN architecture for MV-MOT...")
            self._setup_fpn_architecture()
        else:
            print("🔄 Setting up ResNet architecture for compatibility...")
            self._setup_resnet_architecture()
        self._apply_mv_mot_freeze_strategy()
        self.freeze_stats = {}
        self._log_freeze_statistics()
    
    def _load_pretrained_model(self, pretrained_path):
        """加载预训练模型"""
        if pretrained_path and os.path.exists(pretrained_path):
            print(f"🔄 Loading local model: {pretrained_path}")
            checkpoint = torch.load(pretrained_path, map_location='cpu')
            self.fasterrcnn = fasterrcnn_resnet50_fpn(num_classes=2)
            if 'model_state_dict' in checkpoint:
                self.fasterrcnn.load_state_dict(checkpoint['model_state_dict'])
            else:
                self.fasterrcnn.load_state_dict(checkpoint)
        else:
            print("🌐 Loading COCO pretrained model")
            self.fasterrcnn = fasterrcnn_resnet50_fpn(
                weights=FasterRCNN_ResNet50_FPN_Weights.COCO_V1)
    
    def _setup_fpn_architecture(self):
        """设置FPN架构（推荐用于MV-MOT）"""
        self.backbone = self.fasterrcnn.backbone  
        self.fpn_adapters = nn.ModuleDict({ # ✅ FPN特征适配器 - 专门为多视角设计
            'P2': nn.Sequential(
                nn.Conv2d(256, 128, 1),
                nn.BatchNorm2d(128),  # 保持可训练
                nn.ReLU(inplace=True),
                nn.Dropout2d(0.1)  # 增加正则化
            ),
            'P3': nn.Sequential(
                nn.Conv2d(256, 256, 1),
                nn.BatchNorm2d(256),  # 保持可训练
                nn.ReLU(inplace=True),
                nn.Dropout2d(0.1)
            ),
            'P4': nn.Sequential(
                nn.Conv2d(256, 512, 1),
                nn.BatchNorm2d(512),  # 保持可训练
                nn.ReLU(inplace=True),
                nn.Dropout2d(0.1)
            ),
        })
        self.mv_fusion = MultiViewFeatureFusion(256) # 多视角特征融合模块
        self.upsampling_layer1 = UpsamplingConcat(512 + 256, 512) # 改进的上采样结构 - 增强时序一致性
        self.upsampling_layer2 = UpsamplingConcat(512 + 128, 512)
        self.depth_layer = nn.Sequential( # 深度预测层 - 添加时序稳定性
            nn.Conv2d(512, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),  # 保持可训练
            nn.ReLU(inplace=True),
            nn.Conv2d(256, self.C, kernel_size=1, bias=False)
        )

    def _setup_resnet_architecture(self):
        """设置ResNet架构（兼容性方案）"""
        resnet = self.fasterrcnn.backbone.body
        self.layer0 = nn.Sequential(*list(resnet.children())[:4])
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.upsampling_layer1 = UpsamplingConcat(1024 + 512, 512) # 增强上采样模块
        self.upsampling_layer2 = UpsamplingConcat(512 + 256, 512)
        self.depth_layer = nn.Sequential( # 深度预测层
            nn.Conv2d(512, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),  # 保持可训练
            nn.ReLU(inplace=True),
            nn.Conv2d(256, self.C, kernel_size=1, bias=False)
        )

    def _apply_mv_mot_freeze_strategy(self):
        """应用MV-MOD & MOT优化的冻结策略"""
        if self.freeze_strategy == 'mv_mot_hybrid':
            self._apply_hybrid_freeze_strategy() # 🎯 推荐策略：混合冻结 - 专门为MV-MOT优化
        elif self.freeze_strategy == 'conservative':
            self._apply_conservative_freeze_strategy() # 保守策略：冻结更多层，适合数据量少的情况
        elif self.freeze_strategy == 'aggressive':
            self._apply_aggressive_freeze_strategy() # 激进策略：最少冻结，适合大数据量微调
        elif self.freeze_strategy == 'stage_adaptive':
            self._apply_stage_adaptive_freeze_strategy() # 分阶段自适应策略
        else:  # 'none' 或其他, 不冻结任何层
            print("⚠️  No freezing applied - full training mode")
            return
    
    def _apply_hybrid_freeze_strategy(self):
        """混合冻结策略 - MV-MOT专用（推荐）"""
        print("🎯 Applying MV-MOT Hybrid Freeze Strategy...")
        frozen_count = 0
        trainable_bn_count = 0
        for name, param in self.named_parameters():
            if self._is_early_conv_weight(name): # 1. 冻结底层特征提取权重（保持预训练特征）
                param.requires_grad = False
                frozen_count += 1
            elif self._is_batchnorm_param(name): # 2. 保持所有BatchNorm可训练（关键！适应多视角分布）
                param.requires_grad = True
                trainable_bn_count += 1
            elif 'bias' in name: # 3. 保持偏置可训练（提高表达能力）
                param.requires_grad = True
            elif self._is_multiview_critical_layer(name): # 4. FPN和适配器层始终可训练（多视角融合关键）
                param.requires_grad = True
            elif self._is_high_level_feature(name): # 5. 高层语义特征可训练（任务相关）
                param.requires_grad = True
            else:
                param.requires_grad = not self._should_freeze_middle_layer(name) # 6. 其他中层特征根据重要性决定
        
        print(f"✅ Hybrid strategy applied:")
        print(f"   - Frozen conv weights: {frozen_count}")
        print(f"   - Trainable BatchNorm: {trainable_bn_count}")
        print(f"   - Multi-view layers: Always trainable")
    
    def _apply_conservative_freeze_strategy(self):
        """保守冻结策略 - 适合小数据集"""
        print("🔒 Applying Conservative Freeze Strategy...")
        
        freeze_patterns = [
            "backbone.body.conv1",
            "backbone.body.layer1",
            "backbone.body.layer2.0",
            "backbone.body.layer2.1",
            "backbone.body.layer3.0"
        ]
        for name, param in self.named_parameters():
            if any(pattern in name for pattern in freeze_patterns): # 冻结指定层的权重，但保持BN可训练
                if 'weight' in name and not self._is_batchnorm_param(name):
                    param.requires_grad = False
                else:
                    param.requires_grad = True  # BN和bias保持可训练
            else:
                param.requires_grad = True
    
    def _apply_aggressive_freeze_strategy(self):
        """激进冻结策略 - 适合大数据集"""
        print("🚀 Applying Aggressive Freeze Strategy...")
        freeze_patterns = [ # 只冻结最早期的层
            "backbone.body.conv1",
            "backbone.body.layer1.0"
        ]
        for name, param in self.named_parameters():
            if any(pattern in name for pattern in freeze_patterns):
                if 'weight' in name and not self._is_batchnorm_param(name):
                    param.requires_grad = False
                else:
                    param.requires_grad = True
            else:
                param.requires_grad = True
    
    def _apply_stage_adaptive_freeze_strategy(self):
        """分阶段自适应冻结策略"""
        print(f"🔄 Applying Stage Adaptive Strategy - Stage: {self.training_stage}")
        if self.training_stage == 'initial':
            
            self._apply_conservative_freeze_strategy() # 初期：冻结更多层
        elif self.training_stage == 'intermediate':
            
            self._apply_hybrid_freeze_strategy() # 中期：部分解冻
        else:  # 'fine_tune'
            self._apply_aggressive_freeze_strategy() # 后期：最少冻结
    
    def _is_early_conv_weight(self, name):
        """判断是否为早期卷积层权重"""
        early_patterns = [
            "backbone.body.conv1.weight",
            "backbone.body.layer1.0.conv1.weight",
            "backbone.body.layer1.0.conv2.weight",
            "backbone.body.layer1.1.conv1.weight",
            "backbone.body.layer1.1.conv2.weight",
            "backbone.body.layer2.0.conv1.weight",
            "backbone.body.layer2.0.downsample.0.weight"
        ]
        return any(pattern in name for pattern in early_patterns)
    
    def _is_batchnorm_param(self, name):
        """判断是否为BatchNorm参数"""
        bn_patterns = ['bn', 'norm', 'downsample.1']
        return any(pattern in name for pattern in bn_patterns)
    
    def _is_multiview_critical_layer(self, name):
        """判断是否为多视角关键层"""
        mv_patterns = [
            'fpn_adapters',
            'mv_fusion', 
            'upsampling_layer',
            'depth_layer'
        ]
        return any(pattern in name for pattern in mv_patterns)
    
    def _is_high_level_feature(self, name):
        """判断是否为高层特征"""
        high_level_patterns = [
            "backbone.body.layer3.1",
            "backbone.body.layer3.2", 
            "backbone.fpn"
        ]
        return any(pattern in name for pattern in high_level_patterns)
    
    def _should_freeze_middle_layer(self, name):
        """判断中层是否应该冻结"""
        # 中层选择性冻结策略
        freeze_middle_patterns = [
            "backbone.body.layer2.1.conv1.weight",
            "backbone.body.layer2.2.conv1.weight"
        ]
        return any(pattern in name for pattern in freeze_middle_patterns)
    
    def _log_freeze_statistics(self):
        """记录冻结统计信息"""
        total_params = 0
        frozen_params = 0
        trainable_bn_params = 0
        
        for name, param in self.named_parameters():
            total_params += param.numel()
            if not param.requires_grad:
                frozen_params += param.numel()
            elif self._is_batchnorm_param(name):
                trainable_bn_params += param.numel()
        trainable_params = total_params - frozen_params
        self.freeze_stats = {
            'total_params': total_params,
            'frozen_params': frozen_params,
            'trainable_params': trainable_params,
            'trainable_bn_params': trainable_bn_params,
            'freeze_ratio': frozen_params / total_params,
            'strategy': self.freeze_strategy
        }
        print(f"\n📊 Freeze Statistics:")
        print(f"   Total parameters: {total_params:,}")
        print(f"   Frozen parameters: {frozen_params:,} ({frozen_params/total_params*100:.1f}%)")
        print(f"   Trainable parameters: {trainable_params:,} ({trainable_params/total_params*100:.1f}%)")
        print(f"   Trainable BN parameters: {trainable_bn_params:,}")
        print(f"   Strategy: {self.freeze_strategy}")
    
    def update_training_stage(self, stage):
        """更新训练阶段并重新应用冻结策略"""
        if stage in ['initial', 'intermediate', 'fine_tune']:
            self.training_stage = stage
            if self.freeze_strategy == 'stage_adaptive':
                print(f"🔄 Updating to {stage} stage...")
                self._apply_stage_adaptive_freeze_strategy()
                self._log_freeze_statistics()
        else:
            print(f"⚠️  Invalid stage: {stage}")
    
    def get_freeze_statistics(self):
        """获取冻结统计信息"""
        return self.freeze_stats
    
    def forward(self, x):
        if self.use_fpn:
            return self._forward_fpn(x)
        else:
            return self._forward_resnet(x)
    
    def _forward_fpn(self, x):
        """FPN前向传播 - 增强多视角处理"""
        fpn_features = self.backbone(x) # 提取FPN特征
        p2 = self.fpn_adapters['P2'](fpn_features['0'])  # 1/4 适配FPN特征
        p3 = self.fpn_adapters['P3'](fpn_features['1'])  # 1/8 适配FPN特征
        p4 = self.fpn_adapters['P4'](fpn_features['2'])  # 1/16 适配FPN特征
        if hasattr(self, 'mv_fusion'): # 多视角特征融合（如果需要）
            p3 = self.mv_fusion(p3)
        x = self.upsampling_layer1(p4, p3) # 上采样融合
        x = self.upsampling_layer2(x, p2)
        x = self.depth_layer(x)
        return x
    
    def _forward_resnet(self, x):
        """ResNet前向传播（兼容性方案）"""
        x0 = self.layer0(x)
        x1 = self.layer1(x0)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x = self.upsampling_layer1(x3, x2)
        x = self.upsampling_layer2(x, x1)
        x = self.depth_layer(x)
        return x


class Encoder_swin_t(nn.Module):
    def __init__(self, C):
        super().__init__()
        self.C = C
        swin_t = torchvision.models.swin_t(weights=torchvision.models.Swin_T_Weights.DEFAULT)
        freeze_bn(swin_t)
        self.layer0 = swin_t.features[0]
        self.layer1 = swin_t.features[1:3]
        self.layer2 = swin_t.features[3:5]
        # self.layer3 = swin_t.features[5:7]

        self.upsampling_layer1 = UpsamplingConcat(384 + 192, 384)
        self.upsampling_layer2 = UpsamplingConcat(384 + 96, 384)
        self.depth_layer = nn.Conv2d(384, self.C, kernel_size=1, bias=False)

    def forward(self, x):
        x0 = self.layer0(x)
        x1 = self.layer1(x0)
        x2 = self.layer2(x1)
        # x3 = self.layer3(x2)

        x = self.upsampling_layer1(x2.permute(0, 3, 1, 2), x1.permute(0, 3, 1, 2))
        x = self.upsampling_layer2(x, x0.permute(0, 3, 1, 2))
        x = self.depth_layer(x)

        return x


class Encoder_res101(nn.Module):
    def __init__(self, C):
        super().__init__()
        self.C = C
        resnet = torchvision.models.resnet101(weights=torchvision.models.ResNet101_Weights.DEFAULT)
        freeze_bn(resnet)
        self.backbone = nn.Sequential(*list(resnet.children())[:-4])
        self.layer3 = resnet.layer3

        self.depth_layer = nn.Conv2d(512, self.C, kernel_size=1, padding=0)
        self.upsampling_layer = UpsamplingConcat(1536, 512)

    def forward(self, x):
        x1 = self.backbone(x)
        x2 = self.layer3(x1)
        x = self.upsampling_layer(x2, x1)
        x = self.depth_layer(x)

        return x


class Encoder_res50(nn.Module):
    def __init__(self, C):
        super().__init__()
        self.C = C
        resnet = torchvision.models.resnet50(weights=torchvision.models.ResNet50_Weights.DEFAULT)
        freeze_bn(resnet)

        self.layer0 = nn.Sequential(*list(resnet.children())[:4])
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3

        self.upsampling_layer1 = UpsamplingConcat(1024 + 512, 512)
        self.upsampling_layer2 = UpsamplingConcat(512 + 256, 512)
        self.depth_layer = nn.Conv2d(512, self.C, kernel_size=1, bias=False)

    def forward(self, x):
        x0 = self.layer0(x)
        x1 = self.layer1(x0)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)

        x = self.upsampling_layer1(x3, x2)
        x = self.upsampling_layer2(x, x1)
        x = self.depth_layer(x)

        return x


class Encoder_res50_featDirection(nn.Module):
    def __init__(self, C):
        super().__init__()
        self.C = C
        resnet = torchvision.models.resnet50(weights=torchvision.models.ResNet50_Weights.DEFAULT)
        freeze_bn(resnet)
        conv1 = resnet.conv1
        resnet.conv1 = nn.Conv2d(6, conv1.out_channels, kernel_size=conv1.kernel_size,
                                 stride=conv1.stride, padding=resnet.conv1.padding, bias=conv1.bias)

        self.layer0 = nn.Sequential(*list(resnet.children())[:4])

        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3

        self.upsampling_layer1 = UpsamplingConcat(1024 + 512, 512)
        self.upsampling_layer2 = UpsamplingConcat(512 + 256, 512)
        self.depth_layer = nn.Conv2d(512, self.C, kernel_size=1, bias=False)

    def forward(self, x):
        x0 = self.layer0(x)
        x1 = self.layer1(x0)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)

        x = self.upsampling_layer1(x3, x2)
        x = self.upsampling_layer2(x, x1)
        x = self.depth_layer(x)

        return x


class Encoder_res34(nn.Module):
    def __init__(self, C):
        super().__init__()
        self.C = C
        resnet = torchvision.models.resnet34(weights=torchvision.models.ResNet34_Weights.DEFAULT)
        freeze_bn(resnet)

        self.layer0 = nn.Sequential(*list(resnet.children())[:4])
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3

        self.upsampling_layer1 = UpsamplingConcat(256 + 128, 256)
        self.upsampling_layer2 = UpsamplingConcat(256 + 64, 256)
        self.depth_layer = nn.Conv2d(256, self.C, kernel_size=1, bias=False)

    def forward(self, x):
        x0 = self.layer0(x)
        x1 = self.layer1(x0)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)

        x = self.upsampling_layer1(x3, x2)
        x = self.upsampling_layer2(x, x1)
        x = self.depth_layer(x)

        return x


class Encoder_res18(nn.Module):
    def __init__(self, C):
        super().__init__()
        self.C = C
        resnet = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.DEFAULT)
        freeze_bn(resnet)

        self.layer0 = nn.Sequential(*list(resnet.children())[:4])
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3

        self.upsampling_layer1 = UpsamplingConcat(256 + 128, 256)
        self.upsampling_layer2 = UpsamplingConcat(256 + 64, 256)
        self.depth_layer = nn.Conv2d(256, self.C, kernel_size=1, bias=False)

    def forward(self, x):
        x0 = self.layer0(x)
        x1 = self.layer1(x0)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)

        x = self.upsampling_layer1(x3, x2)
        x = self.upsampling_layer2(x, x1)
        x = self.depth_layer(x)

        return x
    

class Encoder_resnet18_fpn_v2(nn.Module):
    """改进版ResNet18-FPN编码器 - MV-MOD & MOT优化"""
    # '/home/s-jiang/Documents/Robust-MVMOT/fasterrcnn_wildtrack/model_resnet18_0/best_map_0.6436.pth'
    def __init__(self, C, pretrained_path='/home/s-jiang/Documents/Robust-MVMOT/fasterrcnn_wildtrack/model_resnet18_2/best_map_0.6822.pth', use_fpn=True, freeze_strategy='mv_mot_hybrid'):
        super().__init__()
        self.C = C
        self.use_fpn = use_fpn
        self.freeze_strategy = freeze_strategy
        self.training_stage = 'initial'  # 训练阶段控制
        self._load_pretrained_model(pretrained_path) # 加载预训练模型
        if use_fpn:
            print("🔄 Setting up FPN architecture for MV-MOT...")
            self._setup_fpn_architecture()
        else:
            print("🔄 Setting up ResNet architecture for compatibility...")
            self._setup_resnet_architecture()
        self._apply_mv_mot_freeze_strategy()
        self.freeze_stats = {}
        self._log_freeze_statistics()
    
    def _load_pretrained_model(self, pretrained_path):
        """加载预训练模型 - 处理键名不匹配问题"""
        # 🔧 首先创建基础模型
        self.resnet = torchvision.models.resnet18(weights=None)  # 修复deprecation warning
        
        if pretrained_path and os.path.exists(pretrained_path):
            print(f"🔄 Loading local model: {pretrained_path}")
            try:
                checkpoint = torch.load(pretrained_path, map_location='cpu')
                
                # 确定状态字典
                if 'model_state_dict' in checkpoint:
                    state_dict = checkpoint['model_state_dict']
                elif 'state_dict' in checkpoint:
                    state_dict = checkpoint['state_dict']
                else:
                    state_dict = checkpoint
                
                # 🔧 键名映射：从Faster R-CNN格式转换为ResNet格式
                converted_state_dict = {}
                resnet_state_dict = self.resnet.state_dict()
                
                for old_key, value in state_dict.items():
                    # 移除 'backbone.body.' 前缀
                    if old_key.startswith('backbone.body.'):
                        new_key = old_key.replace('backbone.body.', '')
                        
                        # 检查新键是否存在于ResNet模型中，且形状匹配
                        if new_key in resnet_state_dict and value.shape == resnet_state_dict[new_key].shape:
                            converted_state_dict[new_key] = value
                            print(f"✅ Mapped: {old_key} -> {new_key}")
                
                # 加载转换后的状态字典
                missing_keys, unexpected_keys = self.resnet.load_state_dict(converted_state_dict, strict=False)
                
                print(f"✅ Successfully loaded {len(converted_state_dict)} parameters")
                print(f"📊 Missing keys: {len(missing_keys)} (normal for partial loading)")
                print(f"📊 Unexpected keys: {len(unexpected_keys)} (should be 0 now)")
                
                if len(converted_state_dict) == 0:
                    print("⚠️ No matching parameters found, using ImageNet weights instead")
                    self.resnet = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.DEFAULT)
                
            except Exception as e:
                print(f"❌ Error loading pretrained model: {e}")
                print("🔄 Using ImageNet pretrained weights instead...")
                self.resnet = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.DEFAULT)
        else:
            print("🌐 Loading IMAGENET1K pretrained model")
            self.resnet = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.DEFAULT)
    
    def _setup_fpn_architecture(self):
        """设置FPN架构（推荐用于MV-MOT）"""
        # 🔧 修复：正确构建FPN backbone
        from torchvision.ops import FeaturePyramidNetwork
        from torchvision.models._utils import IntermediateLayerGetter
        
        # 定义要提取的层
        return_layers = {
            'layer1': '0',  # 1/4, 64 channels
            'layer2': '1',  # 1/8, 128 channels  
            'layer3': '2',  # 1/16, 256 channels
            'layer4': '3'   # 1/32, 512 channels
        }
        
        # 创建backbone特征提取器
        self.backbone_extractor = IntermediateLayerGetter(self.resnet, return_layers=return_layers)
        
        # 创建FPN
        in_channels_list = [64, 128, 256, 512]  # ResNet18各层通道数
        self.fpn = FeaturePyramidNetwork(
            in_channels_list=in_channels_list,
            out_channels=256
        )
        
        self.fpn_adapters = nn.ModuleDict({ # ✅ FPN特征适配器 - 专门为多视角设计
            'P2': nn.Sequential(
                nn.Conv2d(256, 128, 1),
                nn.BatchNorm2d(128),  # 保持可训练
                nn.ReLU(inplace=True),
                nn.Dropout2d(0.1)  # 增加正则化
            ),
            'P3': nn.Sequential(
                nn.Conv2d(256, 256, 1),
                nn.BatchNorm2d(256),  # 保持可训练
                nn.ReLU(inplace=True),
                nn.Dropout2d(0.1)
            ),
            'P4': nn.Sequential(
                nn.Conv2d(256, 512, 1),
                nn.BatchNorm2d(512),  # 保持可训练
                nn.ReLU(inplace=True),
                nn.Dropout2d(0.1)
            ),
        })
        self.mv_fusion = MultiViewFeatureFusion(256) # 多视角特征融合模块
        self.upsampling_layer1 = UpsamplingConcat(512 + 256, 512) # 改进的上采样结构 - 增强时序一致性
        self.upsampling_layer2 = UpsamplingConcat(512 + 128, 512)
        self.depth_layer = nn.Sequential( # 深度预测层 - 添加时序稳定性
            nn.Conv2d(512, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),  # 保持可训练
            nn.ReLU(inplace=True),
            nn.Conv2d(256, self.C, kernel_size=1, bias=False)
        )

    def _setup_resnet_architecture(self):
        """设置ResNet架构（兼容性方案）"""
        resnet = self.resnet
        self.layer0 = nn.Sequential(*list(resnet.children())[:4])
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.upsampling_layer1 = UpsamplingConcat(1024 + 512, 512) # 增强上采样模块
        self.upsampling_layer2 = UpsamplingConcat(512 + 256, 512)
        self.depth_layer = nn.Sequential( # 深度预测层
            nn.Conv2d(512, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),  # 保持可训练
            nn.ReLU(inplace=True),
            nn.Conv2d(256, self.C, kernel_size=1, bias=False)
        )

    def _apply_mv_mot_freeze_strategy(self):
        """应用MV-MOD & MOT优化的冻结策略"""
        if self.freeze_strategy == 'mv_mot_hybrid':
            self._apply_hybrid_freeze_strategy() # 🎯 推荐策略：混合冻结 - 专门为MV-MOT优化
        elif self.freeze_strategy == 'conservative':
            self._apply_conservative_freeze_strategy() # 保守策略：冻结更多层，适合数据量少的情况
        elif self.freeze_strategy == 'aggressive':
            self._apply_aggressive_freeze_strategy() # 激进策略：最少冻结，适合大数据量微调
        elif self.freeze_strategy == 'stage_adaptive':
            self._apply_stage_adaptive_freeze_strategy() # 分阶段自适应策略
        else:  # 'none' 或其他, 不冻结任何层
            print("⚠️  No freezing applied - full training mode")
            return
    
    def _apply_hybrid_freeze_strategy(self):
        """混合冻结策略 - MV-MOT专用（推荐）"""
        print("🎯 Applying MV-MOT Hybrid Freeze Strategy...")
        frozen_count = 0
        trainable_bn_count = 0
        for name, param in self.named_parameters():
            if self._is_early_conv_weight(name): # 1. 冻结底层特征提取权重（保持预训练特征）
                param.requires_grad = False
                frozen_count += 1
            elif self._is_batchnorm_param(name): # 2. 保持所有BatchNorm可训练（关键！适应多视角分布）
                param.requires_grad = True
                trainable_bn_count += 1
            elif 'bias' in name: # 3. 保持偏置可训练（提高表达能力）
                param.requires_grad = True
            elif self._is_multiview_critical_layer(name): # 4. FPN和适配器层始终可训练（多视角融合关键）
                param.requires_grad = True
            elif self._is_high_level_feature(name): # 5. 高层语义特征可训练（任务相关）
                param.requires_grad = True
            else:
                param.requires_grad = not self._should_freeze_middle_layer(name) # 6. 其他中层特征根据重要性决定
        
        print(f"✅ Hybrid strategy applied:")
        print(f"   - Frozen conv weights: {frozen_count}")
        print(f"   - Trainable BatchNorm: {trainable_bn_count}")
        print(f"   - Multi-view layers: Always trainable")
    
    def _apply_conservative_freeze_strategy(self):
        """保守冻结策略 - 适合小数据集"""
        print("🔒 Applying Conservative Freeze Strategy...")
        
        freeze_patterns = [
            "resnet.conv1",  # 🔧 修复：更新冻结模式匹配
            "resnet.layer1",
            "resnet.layer2.0",
            "resnet.layer2.1"
        ]
        for name, param in self.named_parameters():
            if any(pattern in name for pattern in freeze_patterns): # 冻结指定层的权重，但保持BN可训练
                if 'weight' in name and not self._is_batchnorm_param(name):
                    param.requires_grad = False
                else:
                    param.requires_grad = True  # BN和bias保持可训练
            else:
                param.requires_grad = True
    
    def _apply_aggressive_freeze_strategy(self):
        """激进冻结策略 - 适合大数据集"""
        print("🚀 Applying Aggressive Freeze Strategy...")
        freeze_patterns = [ # 只冻结最早期的层
            "resnet.conv1",  # 🔧 修复：更新冻结模式匹配
            "resnet.layer1.0"
        ]
        for name, param in self.named_parameters():
            if any(pattern in name for pattern in freeze_patterns):
                if 'weight' in name and not self._is_batchnorm_param(name):
                    param.requires_grad = False
                else:
                    param.requires_grad = True
            else:
                param.requires_grad = True
    
    def _apply_stage_adaptive_freeze_strategy(self):
        """分阶段自适应冻结策略"""
        print(f"🔄 Applying Stage Adaptive Strategy - Stage: {self.training_stage}")
        if self.training_stage == 'initial':
            
            self._apply_conservative_freeze_strategy() # 初期：冻结更多层
        elif self.training_stage == 'intermediate':
            
            self._apply_hybrid_freeze_strategy() # 中期：部分解冻
        else:  # 'fine_tune'
            self._apply_aggressive_freeze_strategy() # 后期：最少冻结
    
    def _is_early_conv_weight(self, name):
        """判断是否为早期卷积层权重"""
        early_patterns = [
            "resnet.conv1.weight",  # 🔧 修复：更新模式匹配
            "resnet.layer1.0.conv1.weight",
            "resnet.layer1.0.conv2.weight",
            "resnet.layer1.1.conv1.weight",
            "resnet.layer1.1.conv2.weight",
            "resnet.layer2.0.conv1.weight",
            "resnet.layer2.0.downsample.0.weight"
        ]
        return any(pattern in name for pattern in early_patterns)
    
    def _is_batchnorm_param(self, name):
        """判断是否为BatchNorm参数"""
        bn_patterns = ['bn', 'norm', 'downsample.1']
        return any(pattern in name for pattern in bn_patterns)
    
    def _is_multiview_critical_layer(self, name):
        """判断是否为多视角关键层"""
        mv_patterns = [
            'fpn_adapters',
            'mv_fusion', 
            'upsampling_layer',
            'depth_layer',
            'fpn'  # 🔧 添加FPN模块
        ]
        return any(pattern in name for pattern in mv_patterns)
    
    def _is_high_level_feature(self, name):
        """判断是否为高层特征"""
        high_level_patterns = [
            "resnet.layer3",  # 🔧 修复：更新模式匹配
            "resnet.layer4"
        ]
        return any(pattern in name for pattern in high_level_patterns)
    
    def _should_freeze_middle_layer(self, name):
        """判断中层是否应该冻结"""
        # 中层选择性冻结策略
        freeze_middle_patterns = [
            "resnet.layer2.1.conv1.weight",  # 🔧 修复：更新模式匹配
        ]
        return any(pattern in name for pattern in freeze_middle_patterns)
    
    def _log_freeze_statistics(self):
        """记录冻结统计信息"""
        total_params = 0
        frozen_params = 0
        trainable_bn_params = 0
        
        for name, param in self.named_parameters():
            total_params += param.numel()
            if not param.requires_grad:
                frozen_params += param.numel()
            elif self._is_batchnorm_param(name):
                trainable_bn_params += param.numel()
        trainable_params = total_params - frozen_params
        self.freeze_stats = {
            'total_params': total_params,
            'frozen_params': frozen_params,
            'trainable_params': trainable_params,
            'trainable_bn_params': trainable_bn_params,
            'freeze_ratio': frozen_params / total_params,
            'strategy': self.freeze_strategy
        }
        print(f"\n📊 Freeze Statistics:")
        print(f"   Total parameters: {total_params:,}")
        print(f"   Frozen parameters: {frozen_params:,} ({frozen_params/total_params*100:.1f}%)")
        print(f"   Trainable parameters: {trainable_params:,} ({trainable_params/total_params*100:.1f}%)")
        print(f"   Trainable BN parameters: {trainable_bn_params:,}")
        print(f"   Strategy: {self.freeze_strategy}")
    
    def update_training_stage(self, stage):
        """更新训练阶段并重新应用冻结策略"""
        if stage in ['initial', 'intermediate', 'fine_tune']:
            self.training_stage = stage
            if self.freeze_strategy == 'stage_adaptive':
                print(f"🔄 Updating to {stage} stage...")
                self._apply_stage_adaptive_freeze_strategy()
                self._log_freeze_statistics()
        else:
            print(f"⚠️  Invalid stage: {stage}")
    
    def get_freeze_statistics(self):
        """获取冻结统计信息"""
        return self.freeze_stats
    
    def forward(self, x):
        if self.use_fpn:
            return self._forward_fpn(x)
        else:
            return self._forward_resnet(x)
    
    def _forward_fpn(self, x):
        """FPN前向传播 - 修复版本"""
        # 🔧 修复：正确提取特征并通过FPN
        backbone_features = self.backbone_extractor(x)  # 提取ResNet特征
        fpn_features = self.fpn(backbone_features)       # 通过FPN处理
        
        # 🔍 调试信息（可选，调试完成后删除）
        # print(f"🔍 FPN features keys: {list(fpn_features.keys())}")
        
        # 适配FPN特征
        p2 = self.fpn_adapters['P2'](fpn_features['0'])  # 1/4 
        p3 = self.fpn_adapters['P3'](fpn_features['1'])  # 1/8 
        p4 = self.fpn_adapters['P4'](fpn_features['2'])  # 1/16 
        
        if hasattr(self, 'mv_fusion'): # 多视角特征融合（如果需要）
            p3 = self.mv_fusion(p3)
        x = self.upsampling_layer1(p4, p3) # 上采样融合
        x = self.upsampling_layer2(x, p2)
        x = self.depth_layer(x)
        return x
    
    def _forward_resnet(self, x):
        """ResNet前向传播（兼容性方案）"""
        x0 = self.layer0(x)
        x1 = self.layer1(x0)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x = self.upsampling_layer1(x3, x2)
        x = self.upsampling_layer2(x, x1)
        x = self.depth_layer(x)
        return x


class Encoder_res18_featDirection(nn.Module):
    def __init__(self, C):
        super().__init__()
        self.C = C
        resnet = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.DEFAULT)
        freeze_bn(resnet)
        conv1 = resnet.conv1
        resnet.conv1 = nn.Conv2d(6, conv1.out_channels, kernel_size=conv1.kernel_size,
                                 stride=conv1.stride, padding=resnet.conv1.padding, bias=conv1.bias)

        self.layer0 = nn.Sequential(*list(resnet.children())[:4])
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3

        self.upsampling_layer1 = UpsamplingConcat(256 + 128, 256)
        self.upsampling_layer2 = UpsamplingConcat(256 + 64, 256)
        self.depth_layer = nn.Conv2d(256, self.C, kernel_size=1, bias=False)


    def forward(self, x):
        x0 = self.layer0(x)
        x1 = self.layer1(x0)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)

        x = self.upsampling_layer1(x3, x2)
        x = self.upsampling_layer2(x, x1)
        x = self.depth_layer(x)

        return x


class Encoder_eff(nn.Module):
    def __init__(self, C, version='b4'):
        super().__init__()
        self.C = C
        self.downsample = 8
        self.version = version

        if self.version == 'b0':
            self.backbone = EfficientNet.from_pretrained('efficientnet-b0')
        elif self.version == 'b4':
            self.backbone = EfficientNet.from_pretrained('efficientnet-b4')
        self.delete_unused_layers()

        if self.downsample == 16:
            if self.version == 'b0':
                upsampling_in_channels = 320 + 112
            elif self.version == 'b4':
                upsampling_in_channels = 448 + 160
            upsampling_out_channels = 512
        elif self.downsample == 8:
            if self.version == 'b0':
                upsampling_in_channels = 112 + 40
            elif self.version == 'b4':
                upsampling_in_channels = 160 + 56
            upsampling_out_channels = 128
        else:
            raise ValueError(f'Downsample factor {self.downsample} not handled.')

        self.upsampling_layer = UpsamplingConcat(upsampling_in_channels, upsampling_out_channels)
        self.depth_layer = nn.Conv2d(upsampling_out_channels, self.C, kernel_size=1, padding=0)

    def delete_unused_layers(self):
        indices_to_delete = []
        for idx in range(len(self.backbone._blocks)):
            if self.downsample == 8:
                if self.version == 'b0' and idx > 10:
                    indices_to_delete.append(idx)
                if self.version == 'b4' and idx > 21:
                    indices_to_delete.append(idx)

        for idx in reversed(indices_to_delete):
            del self.backbone._blocks[idx]

        del self.backbone._conv_head
        del self.backbone._bn1
        del self.backbone._avg_pooling
        del self.backbone._dropout
        del self.backbone._fc

    def get_features(self, x):
        # Adapted from https://github.com/lukemelas/EfficientNet-PyTorch/blob/master/efficientnet_pytorch/model.py#L231
        endpoints = dict()

        # Stem
        x = self.backbone._swish(self.backbone._bn0(self.backbone._conv_stem(x)))
        prev_x = x

        # Blocks
        for idx, block in enumerate(self.backbone._blocks):
            drop_connect_rate = self.backbone._global_params.drop_connect_rate
            if drop_connect_rate:
                drop_connect_rate *= float(idx) / len(self.backbone._blocks)
            x = block(x, drop_connect_rate=drop_connect_rate)
            if prev_x.size(2) > x.size(2):
                endpoints['reduction_{}'.format(len(endpoints) + 1)] = prev_x
            prev_x = x

            if self.downsample == 8:
                if self.version == 'b0' and idx == 10:
                    break
                if self.version == 'b4' and idx == 21:
                    break

        # Head
        endpoints['reduction_{}'.format(len(endpoints) + 1)] = x

        if self.downsample == 16:
            input_1, input_2 = endpoints['reduction_5'], endpoints['reduction_4']
        elif self.downsample == 8:
            input_1, input_2 = endpoints['reduction_4'], endpoints['reduction_3']
        # print('input_1', input_1.shape)
        # print('input_2', input_2.shape)
        x = self.upsampling_layer(input_1, input_2)
        # print('x', x.shape)
        return x

    def forward(self, x):
        x = self.get_features(x)  # get feature vector
        x = self.depth_layer(x)  # feature and depth head
        return x
    
