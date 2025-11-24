import csv
import os
import shutil

import torch
import yaml
from matplotlib import pyplot as plt
from PIL import Image
from tqdm import tqdm

import argparse
from diffsolver.utils import calculate_clip_score, create_pipeline, get_clip_encoders, save_image_seed


def load_prompts_from_yaml(file_path):
    with open(file_path, "r") as file:
        prompts = yaml.safe_load(file)
    return prompts.get("prompts", [])


def create_parser():
    parser = argparse.ArgumentParser("binary lambda mask for quantitative analysis")

    parser.add_argument("--ckpt", default="", type=str, help="path to lambda ckpt path")
    parser.add_argument("--device", type=str, default="3", help="device to run the model")
    parser.add_argument("--seed", type=int, default=44, help="random seed")
    parser.add_argument("--prompt_file", "-p", type=str, default="prompts.yaml", help="YAML file containing prompts")
    parser.add_argument("--mix_precision", type=str, default="bf16", help="mixed precision, available bf16")
    parser.add_argument("--num_intervention_steps", "-ni", type=int, default=5, help="number of intervention steps")
    parser.add_argument("--model", type=str, default="sdxl", help="model type, available sdxl, sd2")
    parser.add_argument("--binary", action="store_true", help="whether to use binary mask")
    parser.add_argument(
        "--masking", type=str, default="binary", help="masking type, available binary, hard_discrete, sigmoid"
    )
    parser.add_argument("--scope", type=str, default="global", help="scope for lambda binary mask")
    parser.add_argument(
        "--ratio", type=float, nargs="+", default=[1, 0.45, 0.8], help="sparsity ratio for local global lambda mask"
    )
    parser.add_argument("--width", type=int, default=None)
    parser.add_argument("--height", type=int, default=None)
    parser.add_argument("--epsilon", "-e", type=float, default=0.0, help="epsilon for lambda")
    parser.add_argument("--lambda_threshold", "-lt", type=float, default=0.001, help="threshold for lambda")
    parser.add_argument("--keep_old", action="store_true", help="keep the old images")
    parser.add_argument("--output_dir", type=str, default="./results/")
    parser.add_argument(
        "--clip_backbone", type=str, default="ViT-L-14", help="clip model type, available ViT-B-16, ViT-L-14"
    )
    parser.add_argument(
        "--clip_pretrained",
        type=str,
        default="datacomp_xl_s13b_b90k",
        help="clip pretrained model, datacomp_xl_s13b_b90k",
    )
    parser.add_argument("--clip_eval", action="store_true", help="whether to use clip for evaluation")
    parser.add_argument("--sequence_length", type=int, default=64, help="sequence length for clip model")
    return parser


def setup_pipeline(args, device, torch_dtype):
    pipe = create_pipeline(args.model, device, torch_dtype)
    mask_pipe, hookers = create_pipeline(
        args.model,
        device,
        torch_dtype,
        args.ckpt,
        binary=args.binary,
        lambda_threshold=args.lambda_threshold,
        epsilon=args.epsilon,
        masking=args.masking,
        return_hooker=True,
        scope=args.scope,
        ratio=args.ratio,
    )
    return pipe, mask_pipe, hookers


def validate_hookers(hookers):
    for hooker in hookers:
        if hooker.binary:
            assert hooker.masking == "binary", "masking must be binary when using binary mask"
        else:
            assert hooker.masking != "binary", "masking must be not binary when using continuous mask"


def prepare_output_directory(args):
    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)
    dst = args.output_dir
    if not os.path.exists(dst):
        os.mkdir(dst)
    elif not args.keep_old:
        shutil.rmtree(dst)
        os.mkdir(dst)
    return dst


def perform_evaluation(args, pipe, mask_pipe, prompts, dst):
    for seed, prompt in tqdm([(s, p) for s in [args.seed] for p in prompts]):
        save_image_seed(
            pipe,
            prompt,
            args.num_intervention_steps,
            args.device,
            seed,
            width=args.width,
            height=args.height,
            save_path=os.path.join(dst, f"original_{prompt}_{seed}.png"),
        )
        # save latent
        original_latent = save_image_seed(
            pipe, prompt, args.num_intervention_steps, args.device, seed, output_type="latent"
        )
        torch.save(original_latent, os.path.join(dst, f"latent_{prompt}_{seed}.pt"))
        save_image_seed(
            mask_pipe,
            prompt,
            args.num_intervention_steps,
            args.device,
            seed,
            dst,
            width=args.width,
            height=args.height,
            save_path=os.path.join(dst, f"masked_{prompt}_{seed}.png"),
        )
        # save latent
        masked_latent = save_image_seed(
            mask_pipe, prompt, args.num_intervention_steps, args.device, seed, output_type="latent"
        )
        torch.save(masked_latent, os.path.join(dst, f"masked_latent_{prompt}_{seed}.pt"))


def binary_mask_eval(args):
    device = args.device
    torch_dtype = torch.bfloat16 if args.mix_precision == "bf16" else torch.float32
    prompts = load_prompts_from_yaml(args.prompt_file)
    pipe, mask_pipe, hookers = setup_pipeline(args, device, torch_dtype)

    validate_hookers(hookers)
    dst = prepare_output_directory(args)
    perform_evaluation(args, pipe, mask_pipe, prompts, dst)


def save_results_to_csv(results, output_file):
    with open(output_file, mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["prompt", "clip_score_original", "clip_score_mask", "mse"])
        writer.writerows(results)


def latent_heatmap(latent, output_file):
    # reshape the activate map
    heatmap = latent.view(1, 64, 64, 64).mean(dim=-1).squeeze(-1).to(torch.float32)
    heatmap = heatmap.cpu().detach().numpy()
    return heatmap


def load_image(image_path):
    return Image.open(image_path)


def vis_plot(mask_img, original_img, mask_heatmap, original_heatmap, clip_score, mse_score, prompt, args):
    fig, axes = plt.subplots(1, 4, figsize=(18, 6))

    images = [load_image(original_img), load_image(mask_img), original_heatmap[0], mask_heatmap[0]]

    title = ["Original Image", "Masked Image", "Original Latent Heatmap", "Masked Latent Heatmap"]
    for i, ax in enumerate(axes):
        ax.imshow(images[i])
        ax.axis("off")
        ax.set_title(title[i])

    fig.suptitle(
        f"Prompt: {prompt} \n Original CLIP Score: {clip_score[0]:.4f}"
        + f" Mask CLIP Score: {clip_score[1]:.4f}, MSE Score: {mse_score:.4f}",
        fontsize=16,
    )
    fig.tight_layout()

    save_path = os.path.join(args.output_dir, f"vis_{prompt}_{args.seed}.png")
    fig.savefig(save_path)
    plt.close(fig)


def quan_eval(args):
    clip_dict = get_clip_encoders(backbone=args.clip_backbone, pretrained=args.clip_pretrained)
    clip_model = clip_dict["clip_model"].to(args.device)
    transform = clip_dict["transform"]
    tokenizer = clip_dict["tokenizer"]

    # load prompts
    results = []
    prompts = load_prompts_from_yaml(args.prompt_file)

    for prompt in tqdm(prompts, desc="Processing Prompts"):
        # clip eval
        mask = os.path.join(args.output_dir, f"masked_{prompt}_{args.seed}.png")
        original = os.path.join(args.output_dir, f"original_{prompt}_{args.seed}.png")
        mask_score = calculate_clip_score(clip_model, mask, prompt, tokenizer, transform, args.device)

        # latent eval
        mask_latent = torch.load(os.path.join(args.output_dir, f"masked_latent_{prompt}_{args.seed}.pt"))
        original_latent = torch.load(os.path.join(args.output_dir, f"latent_{prompt}_{args.seed}.pt"))
        mse = (original_latent - mask_latent).pow(2).mean().item()

        original_score = calculate_clip_score(clip_model, original, prompt, tokenizer, transform, args.device)
        results.append([prompt, original_score, mask_score, mse])

        # save latent heatmap
        mask_latent = latent_heatmap(
            mask_latent, os.path.join(args.output_dir, f"masked_latentheatmap_{prompt}_{args.seed}.png")
        )
        original_latent = latent_heatmap(
            original_latent, os.path.join(args.output_dir, f"original_latentheatmap_{prompt}_{args.seed}.png")
        )

        # visulize the final results using matplotlib
        vis_plot(mask, original, mask_latent, original_latent, [original_score, mask_score], mse, prompt, args)

    output_file = os.path.join(args.output_dir, "clip_scores.csv")
    save_results_to_csv(results, output_file)


def main(args):
    print("Start evaluation, generating images ....")
    binary_mask_eval(args)
    print("Start evaluation, quantitative analysis ....")
    quan_eval(args)


if __name__ == "__main__":
    parser = create_parser()
    args = parser.parse_args()
    args.device = f"cuda:{args.device}"
    main(args)
