import os

import argparse
import re
from diffsolver.utils import calculate_clip_score, get_clip_encoders, validation_discreate_path_extraction


def get_parser():
    """Parses command-line arguments."""
    parser = argparse.ArgumentParser(description="Early stop mask finder")
    parser.add_argument("--dirname", type=str, help="Project/notes directory")
    parser.add_argument(
        "--clip_backbone", type=str, default="ViT-L-14", help="CLIP model type. Available options: ViT-B-16, ViT-L-14"
    )
    parser.add_argument(
        "--clip_pretrained",
        type=str,
        default="datacomp_xl_s13b_b90k",
        help="Pretrained CLIP model (e.g., datacomp_xl_s13b_b90k)",
    )
    parser.add_argument(
        "--early_stop_threshold", type=float, default=0.5, help="Threshold for CLIP score to trigger early stopping"
    )
    parser.add_argument("--concept", "-c", type=str, default="Hulk", help="Concept for the prompt")
    parser.add_argument("--device", "-d", type=str, default="cuda:0", help="Device to run the model")
    parser.add_argument("--result_dir", type=str, default="./results/mask_results", help="Directory to save results")
    return parser.parse_args()


def calculate_clip_scores(clip_dict, paths, concept, device):
    """Calculates CLIP scores for a list of image paths."""
    return [calculate_clip_score(clip_dict, img, concept, device, single_word=True, keepmodel=True) for img in paths]


def save_results(file_path, results):
    """Saves results to a text file."""
    # check if the directory exists, create if not
    if not os.path.exists(os.path.dirname(file_path)):
        os.makedirs(os.path.dirname(file_path))
    with open(file_path, "w") as file:
        for result in results:
            file.write(f"{result}\n")


def get_training_steps(val_path):
    name = val_path.split(os.sep)[-1]
    # name has to follow the format epoch_{epoch}_step_{step} check with regex
    assert re.match(r"epoch_\d+_step_\d+", name), "val_path must follow the format epoch_{epoch}_step_{step}"
    return name


def main(args):
    """Main function to run early stopping mask finder."""
    # Load the CLIP model
    clip_dict = get_clip_encoders(backbone=args.clip_backbone, pretrained=args.clip_pretrained)

    # Prepare validation directory and check concept existence
    validation_dir = os.path.join(args.dirname, "images", "validation")
    if args.concept not in validation_dir:
        raise ValueError(f"{args.concept} not in {validation_dir}. Please check the concept name.")

    # Calculate initial CLIP score threshold
    init_path = validation_discreate_path_extraction(validation_dir, "initial_image", args.concept)
    initial_scores = calculate_clip_scores(clip_dict, init_path, args.concept, args.device)
    early_stop_threshold = args.early_stop_threshold * min(initial_scores)
    print("Early stop threshold CLIP score:", early_stop_threshold)

    # Gather and sort validation directories
    validation_dirs = [os.path.join(validation_dir, d) for d in os.listdir(validation_dir) if d.startswith("epoch")]
    # filter with epoch and step number and sort
    # e.g. epoch_1_step_10 -> 110 sorted based on that
    validation_dirs.sort(key=lambda x: int("".join(os.path.basename(x).split("_")[1::2])))

    results = []
    # Iterate through validation directories
    for val_dir in validation_dirs:
        val_path = validation_discreate_path_extraction(val_dir, "discrete mask", args.concept)
        scores = calculate_clip_scores(clip_dict, val_path, args.concept, args.device)
        min_score = min(scores)

        result_entry = f"Validation directory: {get_training_steps(val_dir)}, " f"min CLIP score: {min_score}, "
        results.append(result_entry)
        print("step:", get_training_steps(val_dir), "min score:", min_score)

        if min_score < early_stop_threshold:
            print(f"Early stopping at {get_training_steps(val_dir)}")
            results.append(f"Early stopping at {val_dir}")
            break

    # Save results to a text file
    results_file = os.path.join(args.result_dir, "clip_scores_results.txt")
    save_results(results_file, results)
    print(f"Results saved to {results_file}")


if __name__ == "__main__":
    args = get_parser()
    main(args)
