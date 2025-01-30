import os
import os.path as osp
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import cv2
from datetime import datetime

def plot(path, save_dir):
    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    plot_dir = osp.join(save_dir, 'vis_pred_' + current_time)
    os.makedirs(plot_dir, exist_ok=True)

    plt.rcParams['axes.facecolor'] = 'black'
    fig = plt.figure(figsize=(26, 12), dpi=200)
    fig.set_facecolor('black')
    plt.axis('off')
    data = np.genfromtxt(path, delimiter=",")
    data = data[:, (0, 1, 7, 8)]

    ids = np.unique(data[:, 1]).tolist()
    for idx, id in enumerate(ids):
        id_data = data[data[:, 1] == id]
        color = mcolors.XKCD_COLORS[list(mcolors.XKCD_COLORS.keys())[idx]]
        plt.plot(id_data[:, 2], id_data[:, 3], linewidth=3, color=color)

    name = osp.basename(path)[:-4]
    save_path = osp.join(plot_dir, f'plot_{name}.pdf')
    print(f"Saving plot to: {save_path}")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()


def plot_in_video(path, save_dir):
    data = np.genfromtxt(path, delimiter=",")
    data = data[:, (0, 1, 7, 8)]

    frames = np.unique(data[:, 0]).tolist()
    all_ids = np.unique(data[:, 1]).tolist()

    plt.rcParams['axes.facecolor'] = 'black'
    fig = plt.figure(figsize=(26, 12), dpi=100)
    fig.set_facecolor('black')
    plt.axis('off')
    plt.xlim([0, 1300])
    plt.ylim([0, 500])


    colors = [mcolors.XKCD_COLORS[list(mcolors.XKCD_COLORS.keys())[i]] for i in range(len(all_ids))]
    for frame in frames:
        last_index_of_frame = len(data[:, 0]) - 1 - data[:, 0][::-1].tolist().index(frame)
        untill_frame_data = data[:last_index_of_frame]
        ids = np.unique(untill_frame_data[:, 1]).tolist()
        for idx, id in enumerate(ids):
            id_data = untill_frame_data[untill_frame_data[:, 1] == id]
            color = colors[all_ids.index(id)] # mcolors.XKCD_COLORS[list(mcolors.XKCD_COLORS.keys())[idx]]
            
            plt.plot(id_data[:, 2], id_data[:, 3], linewidth=3, color=color)
        plt.savefig(os.path.join(save_dir, f'{frame}.png'))
        # plt.clf()
        # plt.show()

def images_to_video(path, name='output_video'):
    # Define video properties (adjust as needed)
    fps = 2 # Frames per second
    video_name = os.path.join(path,f"{name}.avi")

    # Get all image paths in the folder
    image_paths = [os.path.join(path, f) for f in os.listdir(path) if
                   os.path.isfile(os.path.join(path, f))]

    # Check if images exist
    if not image_paths:
        print("No images found in the folder!")
        exit()

    # Get image size from the first image (assuming all images have the same size)
    img = cv2.imread(image_paths[0])
    height, width, _ = img.shape

    # Create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'XVID')  # Adjust codec if needed
    video = cv2.VideoWriter(video_name, fourcc, fps, (width, height))

    # Read and write images to video
    for image_path in image_paths:
        img = cv2.imread(image_path)
        video.write(img)

    # Release resources
    video.release()
    cv2.destroyAllWindows()

    print(f"Video created: {video_name}")


if __name__ == '__main__':
    # # path = '../../data/cache/mota_gt.txt'
    # # path = '/media/rasho/Data 1/Arbeit/python_codes/World_space_tracking/EarlyBird/EarlyBird/lightning_logs/version_4/mota_pred.txt'
    # path = '/media/rasho/Data 1/Arbeit/saved_models/EarlyBird_tests/Assessing_wild_02_mvdet_2_res18/mota_pred.txt'
    # save_to_path = '/media/rasho/Data 1/Arbeit/saved_models/temp_saver/earlyBird_BEV_lines'
    # # plot(path)
    # plot_in_video(path,save_to_path)
    # images_to_video(save_to_path,name='output_video')
    
    save_dir='./plot_pred'
    os.makedirs(save_dir, exist_ok=True)
    dir_path = '/home/deep/PythonCode/EarlyBird/World_track-main/lightning_logs_EarlyBird/version_0'
    plot_flie = 'mota_pred.txt'
    # plot_flie = 'mota_gt.txt'
    path = osp.join(dir_path, plot_flie)
    # plot(path, save_dir)
    plot_in_video(path, save_dir)
    images_to_video(save_dir, name='output_video')