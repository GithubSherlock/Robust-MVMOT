import cv2
import os

def concatenate_images_and_create_video(folder1, folder2, output_video, fps):
    images1 = sorted([img for img in os.listdir(folder1) if img.endswith(".png")])  # Get all image filenames in folder1
    images2 = sorted([img for img in os.listdir(folder2) if img.endswith(".png")])  # Get all image filenames in folder2

    assert len(images1) == len(images2), "Number of images in both folders must be the same"

    video = cv2.VideoWriter(output_video, cv2.VideoWriter_fourcc(*'mp4v'), fps,
                            (2 * 640, 480))  # Adjust resolution as needed

    for img1, img2 in zip(images1, images2):
        image1_path = os.path.join(folder1, img1)
        image2_path = os.path.join(folder2, img2)

        # Read images
        image1 = cv2.imread(image1_path)
        image2 = cv2.imread(image2_path)

        # print(image1.shape)
        # print(image2.shape)
        # (1440, 1920, 3)
        # (1500, 6000, 3)
        # Resize images to the same dimensions if necessary
        image1 = cv2.resize(image1, (640, 480))
        image2 = cv2.resize(image2, (640, 480))

        # Concatenate images horizontally
        combined_image = cv2.hconcat([image1, image2])

        # Write frame to video
        video.write(combined_image)

    cv2.destroyAllWindows()
    video.release()


# Example usage:
folder1 = '/media/rasho/Data 1/Arbeit/python_codes/World_space_tracking/EarlyBird/EarlyBird/lightning_logs/vis_images/images'  # Replace with path to first folder containing images
folder2 = '/media/rasho/Data 1/Arbeit/python_codes/World_space_tracking/EarlyBird/EarlyBird/lightning_logs/vis_images/featurs'  # Replace with path to second folder containing images
output_video = '/media/rasho/Data 1/Arbeit/python_codes/World_space_tracking/EarlyBird/EarlyBird/lightning_logs/vis_images/test.mp4'  # Output video file name
fps = 2  # Frames per second (adjust as needed)

concatenate_images_and_create_video(folder1, folder2, output_video, fps)
