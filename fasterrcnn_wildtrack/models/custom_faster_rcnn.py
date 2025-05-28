import torch
from collections import OrderedDict
from torchvision.models.detection.faster_rcnn import FasterRCNN
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor, TwoMLPHead
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.models.detection.transform import GeneralizedRCNNTransform
from torchvision.ops import MultiScaleRoIAlign

from models.custom_roi_heads import CustomRoIHeads
from models.custom_rpn import CustomRPNHead, CustomRegionProposalNetwork

class CustomFasterRCNN(FasterRCNN):
    def __init__(
        self,
        backbone,
        num_classes=2,
        min_size=800,
        max_size=1333,
        image_mean=None,
        image_std=None,
        rpn_anchor_generator=None,
        rpn_head=None,
        rpn_pre_nms_top_n_train=2000,
        rpn_pre_nms_top_n_test=1000,
        rpn_post_nms_top_n_train=2000,
        rpn_post_nms_top_n_test=1000,
        rpn_nms_thresh=0.7,
        rpn_fg_iou_thresh=0.7,
        rpn_bg_iou_thresh=0.3,
        rpn_batch_size_per_image=256,
        rpn_positive_fraction=0.5,
        box_roi_pool=None,
        box_head=None,
        box_predictor=None,
        box_score_thresh=0.05,
        box_nms_thresh=0.5,
        box_detections_per_img=100,
        box_fg_iou_thresh=0.5,
        box_bg_iou_thresh=0.5,
        box_batch_size_per_image=512,
        box_positive_fraction=0.25,
        bbox_reg_weights=None):
        
        if not hasattr(backbone, "out_channels"):
            raise ValueError("backbone should contain an out_channels attribute")
        # 1. 初始化RPN组件
        if rpn_anchor_generator is None:
            rpn_anchor_generator = AnchorGenerator(
                sizes=((32, 64, 128, 256, 512),),
                aspect_ratios=((0.5, 1.0, 2.0),))
        if rpn_head is None:
            rpn_head = CustomRPNHead(
                backbone.out_channels,
                rpn_anchor_generator.num_anchors_per_location()[0])
        # 2. 初始化RoI组件
        if box_roi_pool is None:
            box_roi_pool = MultiScaleRoIAlign(
                featmap_names=['0', '1', '2', '3'],
                output_size=7,
                sampling_ratio=2)
        if box_head is None:
            resolution = box_roi_pool.output_size[0]
            representation_size = 1024
            box_head = TwoMLPHead(
                backbone.out_channels * resolution ** 2,
                representation_size)
        if box_predictor is None:
            representation_size = 1024
            box_predictor = FastRCNNPredictor(
                representation_size,
                num_classes)
        # 3. 创建自定义RoI heads
        roi_heads = CustomRoIHeads(
            box_roi_pool=box_roi_pool,
            box_head=box_head,
            box_predictor=box_predictor,
            fg_iou_thresh=box_fg_iou_thresh,
            bg_iou_thresh=box_bg_iou_thresh,
            batch_size_per_image=box_batch_size_per_image,
            positive_fraction=box_positive_fraction,
            bbox_reg_weights=bbox_reg_weights,
            score_thresh=box_score_thresh,
            nms_thresh=box_nms_thresh,
            detections_per_img=box_detections_per_img)
        # 4. 创建RPN
        rpn_pre_nms_top_n = dict(
            training=rpn_pre_nms_top_n_train,
            testing=rpn_pre_nms_top_n_test)
        rpn_post_nms_top_n = dict(
            training=rpn_post_nms_top_n_train,
            testing=rpn_post_nms_top_n_test)
        rpn = CustomRegionProposalNetwork(
            rpn_anchor_generator, 
            rpn_head,
            rpn_fg_iou_thresh,
            rpn_bg_iou_thresh,
            rpn_batch_size_per_image,
            rpn_positive_fraction,
            rpn_pre_nms_top_n,
            rpn_post_nms_top_n,
            rpn_nms_thresh)
        # 5. 创建transform
        # 设置默认的image_mean和image_std
        if image_mean is None:
            image_mean = [0.485, 0.456, 0.406]
        if image_std is None:
            image_std = [0.229, 0.224, 0.225]
        # 确保转换为tensor
        image_mean = torch.tensor(image_mean)
        image_std = torch.tensor(image_std)
        transform = GeneralizedRCNNTransform(min_size, max_size, image_mean, image_std)
        # 6. 调用父类构造函数
        super().__init__(
            backbone,
            num_classes,
            rpn=rpn,
            roi_heads=roi_heads,
            transform=transform)
    def forward(self, images, targets=None):
        if self.training and targets is None:
            raise ValueError("In training mode, targets should be passed")
        # 记录原始图像尺寸
        original_image_sizes = []
        for img in images:
            val = img.shape[-2:]
            original_image_sizes.append((val[0], val[1]))

        # 预处理图像和目标
        images, targets = self.transform(images, targets)
        
        # 提取特征
        features = self.backbone(images.tensors)
        if isinstance(features, torch.Tensor):
            features = OrderedDict([("0", features)])

        # RPN处理
        proposals, proposal_losses = self.rpn(images, features, targets)

        # ROI处理 - 注意这里只传递必要的参数
        detections, detector_losses = self.roi_heads(
            features=features,
            proposals=proposals,
            image_shapes=images.image_sizes,
            targets=targets)  # targets中已包含masks信息

        # 后处理
        detections = self.transform.postprocess(
            detections, images.image_sizes, original_image_sizes)

        if self.training:
            losses = {}
            losses.update(detector_losses)
            losses.update(proposal_losses)
            return losses
        return detections