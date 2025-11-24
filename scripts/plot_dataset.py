# this script is used to plot the dataset for training
import os

import matplotlib.pyplot as plt
from PIL import Image


def plot_dataset(dataset_path: str, per_row: int, save_path: str):
    print(f"collecting images from {dataset_path}")
    images_dict = {}
    for file_name in os.listdir(dataset_path):
        if file_name.endswith((".png", ".jpg", ".jpeg")):
            image_path = os.path.join(dataset_path, file_name)
            images_dict[file_name] = Image.open(image_path)

    # move deconcept samples in another dictionary
    deconcept_images = {}
    deconcept_filenames = []
    for file_name in images_dict.keys():
        if "deconcept" in file_name:
            deconcept_filenames.append(file_name)
    for file_name in deconcept_filenames:
        deconcept_images[file_name] = images_dict.pop(file_name)

    # generate the plot for neutral samples
    print("plotting neutral samples...")
    neutral_images = {
        file_name: image
        for file_name, image in sorted(images_dict.items(), key=lambda x: int(x[0].split(".")[0]))
        if int(file_name.split(".")[0]) % 2 != 0
    }
    rows = len(neutral_images) // per_row
    if len(neutral_images) % per_row != 0:
        rows += 1
    fig, ax = plt.subplots(rows, per_row, figsize=(per_row * 2, rows * 2))
    for i, (file_name, image) in enumerate(neutral_images.items()):
        ax[i // per_row, i % per_row].imshow(image)
        ax[i // per_row, i % per_row].set_title(file_name)
        ax[i // per_row, i % per_row].axis("off")
    plt.tight_layout()
    if not os.path.exists(save_path):
        os.makedirs(save_path, exist_ok=True)
    plt.savefig(os.path.join(save_path, "dataset_plot_neutral.png"), bbox_inches="tight")

    # generate the plot for concept and deconcept samples
    print("plotting deconcept samples...")
    rows = len(deconcept_images) // per_row
    if len(deconcept_images) % per_row != 0:
        rows += 1
    rows *= 2
    fig, ax = plt.subplots(rows, per_row, figsize=(per_row * 2, rows * 2))
    for i, (file_name, image) in enumerate(deconcept_images.items()):
        concept_image = images_dict[file_name.replace("deconcept_", "")]
        ax[i // per_row * 2, i % per_row].imshow(concept_image)
        ax[i // per_row * 2, i % per_row].set_title(file_name.replace("deconcept_", ""))
        ax[i // per_row * 2, i % per_row].axis("off")
        ax[i // per_row * 2 + 1, i % per_row].imshow(image)
        ax[i // per_row * 2 + 1, i % per_row].set_title(file_name)
        ax[i // per_row * 2 + 1, i % per_row].axis("off")
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, "dataset_plot_deconcept.png"), bbox_inches="tight")
    plt.close()
    print("plotting completed")


def main():
    path = (
        "datasets/flux/jan20_flux_inference_nsfw/model_flux_concept_"
        + "exposure front facing nude woman_eps_0.5_beta_0.1_sample_20_epochs_5_lr_00.80.8"
    )
    save_path = "images/dataset_plot"
    row_n = 4
    plot_dataset(path, row_n, save_path)


if __name__ == "__main__":
    main()
