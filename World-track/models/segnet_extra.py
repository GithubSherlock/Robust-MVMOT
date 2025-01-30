import torch
import torch.nn as nn
import numpy as np
import utils.geom
import utils.vox
import utils.basic
from icecream import ic

from models.encoder import Encoder_res101, Encoder_res50, Encoder_res18_featDirection, Encoder_res18, Encoder_eff, \
    Encoder_swin_t
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
                 aggregate_mode='max_pool',  # mean,max_pool,2d_conv, 3d_conv

                 device=torch.device('cuda')):

        super(Segnet_e, self).__init__()
        assert (encoder_type in ['res101', 'res50', 'effb0', 'effb4', 'res18', 'res18_featDirection', 'vgg11',
                                 'swin_t'])

        self.Y, self.Z, self.X = Y, Z, X
        self.do_rgbcompress = do_rgbcompress
        self.rand_flip = rand_flip
        self.latent_dim = latent_dim
        self.encoder_type = encoder_type
        self.z_sign = z_sign
        self.num_cameras = num_cameras

        self.waighting_mode = waighting_mode
        self.feat_direction = feat_direction_mode
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
        elif encoder_type == 'swin_t':
            self.encoder = Encoder_swin_t(feat2d_dim)
        else:
            self.encoder = Encoder_eff(feat2d_dim, version='b4')

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
        elif self.aggregate_mode == 'mean':  # do not need a layer for mean
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

    def forward(self, rgb_cams, pix_T_cams, cams_T_global, vox_util, ref_T_global, prev_bev=None):
        """
        B = batch size, S = number of cameras, C = 3, H = img height, W = img width
        rgb_cams: (B,S,C,H,W)
        pix_T_cams: (B,S,4,4)
        cams_T_global: (B,S,4,4)
        vox_util: vox util object
        ref_T_global: (B,4,4)
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

        global_T_cams_ = torch.inverse(cams_T_global_)  # B*S,4,4
        ref_T_cams = torch.matmul(ref_T_global.repeat(S, 1, 1), global_T_cams_)  # B*S,4,4
        cams_T_ref_ = torch.inverse(ref_T_cams)  # B*S,4,4

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
            feat_cams_ = self.encoder(rgb_cams_)  # B*S,128,H/8,W/8

        if self.rand_flip:
            feat_cams_[self.rgb_flip_index] = torch.flip(feat_cams_[self.rgb_flip_index], [-1])
        _, C, Hf, Wf = feat_cams_.shape
        sy = Hf / float(H)
        sx = Wf / float(W)

        # image decoder
        out_image_dict = self.decoder_image(feat_cams_)

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
                                                               Y, Z, X
                                                               )
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

        elif self.aggregate_mode == 'mean':
            feat_mem = torch.mean(feat_mems, dim=1)

        elif self.aggregate_mode == 'max_pool':
            feat_mem, _ = torch.max(feat_mems, dim=1)  # maximum values, indices


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
        out_dict = {**out_BEV_dict, **out_image_dict}


        return out_dict
