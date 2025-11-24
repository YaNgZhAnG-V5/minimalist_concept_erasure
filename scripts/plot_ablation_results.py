import os
from concurrent.futures import ProcessPoolExecutor, as_completed

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from alive_progress import alive_bar

import argparse


def get_method_name(path):
    """read method name as all folders in the path folder"""
    return [folder for folder in os.listdir(path) if os.path.isdir(os.path.join(path, folder))]


def get_images_single(file, root, path, target_path, methods, methods_str, reference_method):
    image_path = os.path.join(root, file)
    fig, axes = plt.subplots(1, len(methods), figsize=(15, 5))
    if len(methods) == 1:
        axes = [axes]
    for ax, method, method_str in zip(axes, methods, methods_str):
        method_path = image_path.replace(reference_method, method)
        prompt = method_path.split("/")[-1].split("_")[0]
        folder_path = "/".join(method_path.split("/")[:-1])
        for another_root, dirs, files in os.walk(folder_path):
            for file in files:
                if prompt in file:
                    method_image_path = os.path.join(another_root, file)
        image = mpimg.imread(method_image_path)
        ax.imshow(image)
        ax.set_title(method_str)
        ax.axis("off")
    save_path = (
        method_image_path.replace(path, "").replace(method, "").replace("images", "").replace("///", "").split("/")
    )
    save_folder = save_path[-1].split("_")[0]
    save_name = "_".join(save_path[:-1])
    if not os.path.exists(os.path.join(target_path, save_folder)):
        os.makedirs(os.path.join(target_path, save_folder))
    plt.tight_layout()
    plt.savefig(os.path.join(target_path, save_folder, save_name), bbox_inches="tight")
    plt.close()


def get_images(methods, path, target_path, ablation_name="beta", max_workers: int = 32):
    methods_str = []
    for method in methods:
        parts = method.split("_")
        if ablation_name in parts:
            value = parts[parts.index(ablation_name) + 1]
            methods_str.append(f"{ablation_name}_{value}")
    # Extract the numeric value from each method_str and rank them
    method_values = [float(method.split("_")[1]) for method in methods_str]
    ranked_indices = sorted(range(len(method_values)), key=lambda i: method_values[i])
    methods_str = [methods_str[i] for i in ranked_indices]
    methods = [methods[i] for i in ranked_indices]
    reference_method = methods[0]
    reference_method_path = os.path.join(path, reference_method, "images")

    num_files = sum([len(files) for _, _, files in os.walk(reference_method_path)])
    with ProcessPoolExecutor(max_workers=max_workers) as executor, alive_bar(
        total=num_files, title="Plotting images"
    ) as bar:
        futures = [
            executor.submit(get_images_single, file, root, path, target_path, methods, methods_str, reference_method)
            for root, dirs, files in os.walk(reference_method_path)
            for file in files
        ]
        for future in as_completed(futures):
            bar()


def merge_images_single(file, root, prompt_folders, splits, target_path):
    fig, axes = plt.subplots(len(prompt_folders), figsize=(15, 3 * len(prompt_folders)))
    file_suffix = "_".join(file.split("_")[1:])
    for prompt in prompt_folders:
        new_root = root.replace(prompt_folders[0], prompt)
        for split in splits:
            file_name = f"{split}_{file_suffix}"
            if file_name in os.listdir(new_root):
                new_file = mpimg.imread(os.path.join(new_root, file_name))
                axes[prompt_folders.index(prompt)].imshow(new_file)
                axes[prompt_folders.index(prompt)].set_title(prompt)
                axes[prompt_folders.index(prompt)].axis("off")
    plt.tight_layout()
    plt.savefig(os.path.join(target_path, "merged", file_suffix), bbox_inches="tight")
    plt.close()


def merge_images(target_path, max_workers: int = 32):
    if not os.path.exists(os.path.join(target_path, "merged")):
        os.makedirs(os.path.join(target_path, "merged"))
    # get all prompt folders
    splits = ["train", "validation"]
    prompt_folders = [folder for folder in os.listdir(target_path) if os.path.isdir(os.path.join(target_path, folder))]
    prompt_folders.remove("merged")
    num_files = sum([len(files) for _, _, files in os.walk(os.path.join(target_path, prompt_folders[0]))])
    with ProcessPoolExecutor(max_workers=max_workers) as executor, alive_bar(
        total=num_files, title="Merging images"
    ) as bar:
        futures = [
            executor.submit(merge_images_single, file, root, prompt_folders, splits, target_path)
            for root, dirs, files in os.walk(os.path.join(target_path, prompt_folders[0]))
            for file in files
        ]
        for future in as_completed(futures):
            bar()


def args_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--path", type=str, default="/home/erjin/yang/Diffusion_Infringement_Resolver/results/flux_jan2_debug"
    )
    parser.add_argument(
        "--target_path",
        type=str,
        default="/home/erjin/yang/Diffusion_Infringement_Resolver/results/flux_jan2_debug/plot_images",
    )
    parser.add_argument("--get_images", action="store_true")
    parser.add_argument("--merge_images", action="store_true")
    parser.add_argument("--max_workers", type=int, default=32)
    return parser.parse_args()


def main():
    args = args_parser()
    path = args.path
    target_path = args.target_path
    method_names = get_method_name(path)
    ablation_name = "beta"
    if not os.path.exists(target_path):
        os.makedirs(target_path)
    if args.get_images:
        get_images(method_names, path, target_path, ablation_name, args.max_workers)
    if args.merge_images:
        merge_images(target_path, args.max_workers)


if __name__ == "__main__":
    main()
