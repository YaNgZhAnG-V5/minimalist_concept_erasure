# plot images side by side by taking two folders

import os
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed

import cv2
import matplotlib.pyplot as plt
from alive_progress import alive_bar

import argparse


def process_image(image1, folder1, folder2, dest_folder, prefix1, prefix2, images2, bar):
    suffix1 = image1.split("_")[-1]
    image2 = f"{prefix2}_{suffix1}"
    if image2 in images2:
        img1 = cv2.imread(os.path.join(folder1, image1))
        img2 = cv2.imread(os.path.join(folder2, image2))
        fig, ax = plt.subplots(1, 2, figsize=(img1.shape[1] / 100, img1.shape[0] / 100))
        ax[0].imshow(cv2.cvtColor(img1, cv2.COLOR_BGR2RGB))
        ax[0].set_title(prefix1)
        ax[0].axis("off")
        ax[1].imshow(cv2.cvtColor(img2, cv2.COLOR_BGR2RGB))
        ax[1].set_title(prefix2)
        ax[1].axis("off")
        plt.savefig(os.path.join(dest_folder, f"merged_img_{suffix1}"), bbox_inches="tight", pad_inches=0)
        plt.close(fig)
    else:
        print(f"Image {image2} not found in folder2")
    bar()


def plot_imgs_side_by_side(folder1, folder2, dest_folder, replace_folder=False):
    # get all images in folder1
    images1 = [f for f in os.listdir(folder1) if f.endswith((".png", ".jpg", ".jpeg"))]
    images2 = [f for f in os.listdir(folder2) if f.endswith((".png", ".jpg", ".jpeg"))]

    prefix1 = "_".join(images1[0].split("_")[:-1])
    prefix2 = "_".join(images2[0].split("_")[:-1])

    # create destination folder if it doesn't exist
    if os.path.exists(dest_folder):
        if replace_folder:
            shutil.rmtree(dest_folder)
        else:
            raise ValueError(f"Destination folder {dest_folder} already exists")
    else:
        os.makedirs(dest_folder, exist_ok=True)

    # plot images side by side
    with alive_bar(len(images1)) as bar:
        with ThreadPoolExecutor(max_workers=8) as executor:
            futures = [
                executor.submit(process_image, image1, folder1, folder2, dest_folder, prefix1, prefix2, images2, bar)
                for image1 in images1
            ]
        for future in as_completed(futures):
            bar()


def argument_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--folder1", "-f1", type=str, default="./images/ring-a-bell-3-16/flux_unlearn")
    parser.add_argument("--folder2", "-f2", type=str, default="./images/ring-a-bell-3-16/original_image")
    parser.add_argument("--target-folder", "-t", type=str, default="./images/ring-a-bell-3-16/flux_unlearn_vs_original")
    parser.add_argument("--replace-folder", "-r", type=bool, default=True)
    args = parser.parse_args()
    return args


def main():
    args = argument_parser()
    folder1 = args.folder1
    folder2 = args.folder2
    dest_folder = args.target_folder
    replace_folder = args.replace_folder
    plot_imgs_side_by_side(folder1, folder2, dest_folder, replace_folder)


if __name__ == "__main__":
    main()
