# this script is for checking the output image with different denoiseing steps
# for unlearn concept training
# available models, flux-schnell, flux-dev


import os

import numpy as np
import torch
from matplotlib import pyplot as plt
from PIL import Image

import argparse
from diffsolver.utils import create_pipeline


def create_parser():
    parser = argparse.ArgumentParser("evaluate the model denoising latent performance")
    parser.add_argument("--model", type=str, default="flux", help="model type, available flux-dev, flux-schnell")
    parser.add_argument("--device", type=str, default="1", help="device to run the model")
    parser.add_argument("--seed", type=int, default=44, help="random seed")
    parser.add_argument("--num_intervention_steps", "-ni", type=int, default=5, help="number of intervention steps")
    parser.add_argument("--prompt", "-p", type=str, default="Superman flying over Metropolis at sunset")
    parser.add_argument(
        "--save_dir", type=str, default="./results/latent_visualization", help="directory to save the images"
    )
    return parser


def main(args):
    # configuration setting
    device = torch.device(args.device)
    torch_dtype = torch.bfloat16
    pipe = create_pipeline(args.model, device, torch_dtype)

    # create the save directory
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    images_list = []

    for ts in range(1, args.num_intervention_steps):
        g_cpu = torch.Generator(device).manual_seed(args.seed)
        image = pipe(prompt=args.prompt, num_inference_steps=ts, generator=g_cpu).images[0]
        # convert PIL image to smaller size
        im = np.array(image.resize((256, 256), Image.Resampling.BICUBIC))
        images_list.append(im)

    # Plot all images in a single figure
    num_images = len(images_list)
    columns = 4
    rows = (num_images // columns) + (num_images % columns > 0)
    fig, axes = plt.subplots(rows, columns, figsize=(20, 5 * rows))
    # Fill the subplot with images
    for idx, ax in enumerate(axes.flat):
        if idx < num_images:
            ax.imshow(images_list[idx])
            ax.axis("off")  # Turn off axis
        else:
            ax.axis("off")  # Turn off extra axes

    plt.tight_layout()
    plt.savefig(f"output_images_grid_model_{args.model}.png", bbox_inches="tight", pad_inches=0)
    plt.show()


if __name__ == "__main__":
    parser = create_parser()
    args = parser.parse_args()
    args.device = f"cuda:{args.device}"
    main(args)
