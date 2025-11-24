import os

import matplotlib.image as mpimg
import matplotlib.pyplot as plt

import argparse


def display_and_save_images_in_grid(directory, output_file):
    """
    Displays images in a 3x4 grid from the specified directory and saves the plot as an image.
    Assumes images are named '1.png' to '12.png'.

    Args:
        directory (str): The directory containing the images.
        output_file (str): Path to save the output image file.
    """
    fig, axes = plt.subplots(2, 6, figsize=(12, 4))  # Set figure size
    output_file = os.path.join(directory, output_file)

    for i in range(2):  # Rows
        for j in range(8):  # Columns
            img_index = i * 8 + j + 1  # Calculate image index (1 to 12)
            if img_index in [7, 8, 15, 16]:
                continue
            image_path = os.path.join(directory, f"{img_index}.png")  # Construct full image path
            try:
                image = mpimg.imread(image_path)  # Read the image
                axes[i, j].imshow(image)  # Display the image
                axes[i, j].axis("off")  # Turn off the axes
            except FileNotFoundError:
                axes[i, j].axis("off")  # If image not found, leave blank
                axes[i, j].set_title("Missing", fontsize=8)

    # Remove all spacing between plots
    plt.subplots_adjust(wspace=0, hspace=0)  # No space between subplots
    plt.savefig(output_file, dpi=300, bbox_inches="tight", pad_inches=0)  # Save the plot as an image
    plt.show()
    print(f"Plot saved as {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Display images in a 3x4 grid and save the plot as an image.")
    parser.add_argument(
        "--directory",
        "-d",
        type=str,
        default="/home/erjin/github/Diffusion_Infringement_Resolver/images/plot_images/teaser",
        help="Path to the directory containing images named '1.png' to '12.png'.",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default="output_plot.png",
        help="Path to save the output plot image file (default: 'output_plot.png').",
    )
    args = parser.parse_args()

    display_and_save_images_in_grid(args.directory, args.output)
