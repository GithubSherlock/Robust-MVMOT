import os.path as osp
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import cv2, sys
current_dir = osp.dirname(osp.abspath(__file__))
parent_dir = osp.dirname(current_dir)
sys.path.append(parent_dir)
from datasets.pedestrian_datamodule import PedestrianDataModule
import utils
from utils import vox
import torch
from utils import geom
from extra_util.projections import *


def prepare_data(rgb_cams, pix_T_cams, cams_T_global, ref_T_global):
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
    # cams_T_ref_ = torch.inverse(ref_T_cams)  # B*S,4,4

    return rgb_cams_, pix_T_cams_, cams_T_global_, ref_T_cams


def plot(path):
    plt.rcParams['axes.facecolor'] = 'black'
    fig = plt.figure(figsize=(26, 12), dpi=200)
    fig.set_facecolor('black')
    plt.axis('off')
    predictions = np.genfromtxt(path, delimiter=",")
    predictions = predictions[:, (0, 1, 7, 8)]

    ids = np.unique(predictions[:, 1]).tolist()

    # print(len(list(mcolors.XKCD_COLORS.keys())))
    # for idx, id in enumerate(ids):
    #     id_data = data[data[:, 1] == id]
    #     color = mcolors.XKCD_COLORS[list(mcolors.XKCD_COLORS.keys())[idx]]
    #     plt.plot(id_data[:, 2], id_data[:, 3], linewidth=3, color=color)

    dataset_root = "/home/deep/Documents/Wildtrack"
    dataset = PedestrianDataModule(data_dir=dataset_root,
                                   resolution=[120, 2, 360],
                                   bounds=[0, 1440, 0, 480, 0, 200],
                                   train_cameras=[0, 1, 2, 3, 4, 5, 6], test_cameras=[0, 1, 2, 3, 4, 5, 6])
    dataset.setup('test')
    test_data = dataset.test_dataloader()

    Y, Z, X = [120, 1, 360]
    bounds = [0, 1440, 0, 480, 0, 200]
    scene_centroid = (0., 0., 0.)
    scene_centroid = torch.tensor(scene_centroid).reshape([1, 3])
    vox_util = vox.VoxelUtil(Y, Z, X, scene_centroid=scene_centroid, bounds=bounds)

    for batch_idx, (data, target) in enumerate(test_data):
        print(f"Batch {batch_idx + 1}")
        print(f"Data shape: {data.keys()}")
        print(f"Target shape: {target.keys()}")

        '''
        intrinsic_matrices
        [[1.74344788e+03 0.00000000e+00 9.34520203e+02]
         [0.00000000e+00 1.73515662e+03 4.44398773e+02]
         [0.00000000e+00 0.00000000e+00 1.00000000e+00]]
         
         extrinsic_matrices
        [[ 8.77676189e-01  4.78589416e-01  2.52333060e-02 -5.25894165e+02]
         [ 1.33892894e-01 -1.94308296e-01 -9.71759737e-01  4.54076347e+01]
         [-4.60170895e-01  8.56268942e-01 -2.34619483e-01  9.86723511e+02]]
        '''
        rgb_cams = data['img'],
        pix_T_cams = data['intrinsic'],
        cams_T_global = data['extrinsic'],
        ref_T_global = data['ref_T_global'],
        mem_T_ref = vox_util.get_mem_T_ref(1, Y, Z, X)
        ref_T_mem = vox_util.get_ref_T_mem(1, Y, Z, X)


        rgb_cams_, pix_T_cams_, cams_T_global_, ref_T_cams = prepare_data(rgb_cams[0], pix_T_cams[0], cams_T_global[0],
                                                                          ref_T_global[0])
        from numpy.linalg import inv
        for view in range(7):
            print('\n',view)
            print(pix_T_cams_[view])
            print(cams_T_global_[view])
        view = 0
        id_data = predictions[predictions[:, 1] == ids[1]]
        frame_data = predictions[predictions[:, 0] == 1800]

        # point_x, point_y = id_data[:, 2], id_data[:, 3]
        point_x, point_y = frame_data[:, 2], frame_data[:, 3]

        objectPoint = np.array([[point_x, point_y]], dtype=np.float32).T

        img_coor_list = []
        for point in objectPoint:
            get_imagecoord = get_imagecoord_from_worldcoord(point, pix_T_cams_[view], cams_T_global_[view], z=0)
            img_coor_list.append(get_imagecoord)

        img_coords = [tensor.cpu().numpy().reshape(2) for tensor in img_coor_list]
        img_coords = np.asarray(img_coords).T
        print(img_coords)

        imag_to_show = rgb_cams_[view].permute(1, 2, 0).numpy()

        print(imag_to_show.shape)
        plt.imshow(imag_to_show)
        plt.scatter(x=[img_coords[0]], y=[img_coords[1]], c='r', s=40)
        plt.show()
        exit()
        # print(ref_T_cams[0])
        # print(float(point_x), float(point_y))
        # print(geom.apply_4x4(ref_T_cams[0], torch.tensor([[[point_x, point_y, 0]]])))

        x, y = vox_util.BEV_T_image(rgb_cams_, pix_T_cams_, cams_T_global_, Y, Z, X)

        print(x[0])
        print(x.shape)
        id_data = predictions[predictions[:, 1] == ids[0]]
        point_x, point_y = id_data[0, 2], id_data[0, 3]
        print(point_x, point_y)

        temp_y = y[0].view(120, 360)
        temp_x = x[0].view(120, 360)

        difference = torch.abs(temp_x - point_x)
        flat_idx = torch.argmin(difference, dim=1)
        # Unflatten the index to get row and column indices
        row_idx_x = flat_idx // difference.shape[1]  # Integer division for row index
        col_idx_x = flat_idx % difference.shape[1]  # Modulo for column index

        difference = torch.abs(temp_y - point_y)
        flat_idx = torch.argmin(difference, dim=1)
        # Unflatten the index to get row and column indices
        row_idx_y = flat_idx // difference.shape[1]  # Integer division for row index
        col_idx_y = flat_idx % difference.shape[1]  # Modulo for column index

        temp_y[row_idx_y, col_idx_y] = 2000
        temp_x[row_idx_x, col_idx_x] = 2000
        plt.imshow(temp_x)
        plt.show()
        plt.imshow(temp_y)
        plt.show()
        # imag_to_show = rgb_cams_[0].permute(1, 2, 0)
        #
        # plt.imshow(imag_to_show)
        # plt.show()
        exit()

    '''
    rgb_cams=item['img'],
    pix_T_cams=item['intrinsic'],
    cams_T_global=item['extrinsic'],
    vox_util=self.vox_util,
    ref_T_global=item['ref_T_global'],
    '''
    # print(dataset.test_dataloader())
    # name = osp.basename(path)[:-4]
    # plt.savefig(f'plot_{name}.pdf', dpi=300, bbox_inches='tight')
    # plt.show()


if __name__ == '__main__':
    # path = '../../data/cache/mota_gt.txt'
    # path = '/media/rasho/Data 1/Arbeit/saved_models/EarlyBird_tests/Assessing_wild_02_mvdet_2_res18/mota_pred.txt'
    # path = '/media/rasho/Data 1/Arbeit/saved_models/EarlyBird_tests/Assessing_wild_02_mvdet_2_res18/mota_gt.txt'

    path = '/home/deep/PythonCode/EarlyBird/World_track-main/lightning_logs_EarlyBird/version_0/mota_pred.txt'
    plot(path)
