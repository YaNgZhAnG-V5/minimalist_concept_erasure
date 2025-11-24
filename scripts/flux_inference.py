import os
from os import path as osp

import matplotlib.pyplot as plt
import torch
from alive_progress import alive_it
from diffusers import FluxPipeline

import argparse

# import relative path from baselines
import sys
from diffsolver.data.adversarial_promptdatasets import (
    get_i2p_dataset,
    get_ip_characters,
    get_mma_dataset,
    get_nudity_concept_dataset,
    get_p4d_dataset,
    get_ring_a_bell_dataset,
)
from diffsolver.utils import create_pipeline

sys.path.append(osp.abspath(osp.join(__file__, "../../")))
from baselines.flowedit.flux_inference import flowedit_inference
from baselines.sld.sld import FluxPipelineForSLB


# Save the images 6 in a row using pyplot
def save_images_grid(images, filename="merged_images.png"):
    fig, axes = plt.subplots(
        len(images) // 6 + (1 if len(images) % 6 != 0 else 0), 6, figsize=(18, 3 * (len(images) // 6 + 1))
    )
    for idx, img in enumerate(images):
        row = idx // 6
        col = idx % 6
        axes[row, col].imshow(img)
        axes[row, col].axis("off")

    # Hide any unused subplots
    for idx in range(len(images), len(axes.flatten())):
        axes.flatten()[idx].axis("off")

    plt.tight_layout()
    plt.savefig(filename, bbox_inches="tight")
    plt.close()


def get_prompt_dataset(prompt: str):
    if prompt == "nudity_concept":
        return get_nudity_concept_dataset()
    elif prompt == "i2p":
        return get_i2p_dataset(nudity_threshold=50)
    elif "ring-a-bell" in prompt:
        threshold = int(prompt.split("-")[-2])
        length = int(prompt.split("-")[-1])
        return get_ring_a_bell_dataset(threshold=threshold, length=length)
    elif prompt == "p4d":
        return get_p4d_dataset()
    elif prompt == "mma":
        return get_mma_dataset()
    elif prompt == "ip":
        return get_ip_characters()
    else:
        raise ValueError(f"Prompt {prompt} not supported")


def main(args):
    assert args.baseline in [None, "flowedit", "sld"], "Baseline model not supported"
    torch.manual_seed(args.seed)
    device = f"cuda:{args.device}"
    prompts = get_prompt_dataset(args.prompt)
    if args.prompt == "ip":
        prompts = prompts[args.concept]

    print("Perform FLUX image generation")
    print(f"Generating images for {args.prompt}")
    print(f"Unlearned model path: \n{args.save_pt}")

    # get the normal flux model
    if args.baseline == "flowedit":
        pipe = FluxPipeline.from_pretrained(
            "black-forest-labs/FLUX.1-schnell",
            torch_dtype=torch.float16,
        )
    elif args.baseline == "sld":
        pipe = FluxPipelineForSLB.from_pretrained(
            "black-forest-labs/FLUX.1-schnell",
            torch_dtype=torch.bfloat16,
        )
    else:
        pipe = FluxPipeline.from_pretrained("black-forest-labs/FLUX.1-schnell", torch_dtype=torch.bfloat16)
    pipe.set_progress_bar_config(disable=True)
    pipe.to(device)

    # create save dir if not exists
    os.makedirs(osp.join(args.save_dir, args.prompt, "original_image"), exist_ok=True)
    os.makedirs(osp.join(args.save_dir, args.prompt, "flux_unlearn"), exist_ok=True)

    images = []
    print("Generating original images...")
    for idx, prompt in enumerate(alive_it(prompts)):
        img = pipe(
            prompt=prompt,
            guidance_scale=0.0,
            height=args.height,
            width=args.width,
            num_inference_steps=4,
            max_sequence_length=256,
        ).images[0]
        images.append(img)
        img.save(osp.join(args.save_dir, args.prompt, "original_image", f"image_{idx}.png"))

    if args.save_grid:
        save_images_grid(images, osp.join(args.save_dir, args.prompt, "original_image", "flux_images.png"))

    # get the unlearn flux model
    if not args.baseline:
        pipe = create_pipeline("flux", device, torch.bfloat16, save_pt=args.save_pt, binary=True, lambda_threshold=0.0)

    images = []
    print("Generating unlearn images...")
    for idx, prompt in enumerate(alive_it(prompts)):
        gen = torch.Generator(device=torch.device(f"cuda:{args.device}")).manual_seed(args.seed)
        if args.baseline == "flowedit":
            img = flowedit_inference(
                pipe,
                prompt,
                concept=args.concept,
                seed=args.seed,
                device=device,
                num_inference_steps=4,
                n_avg=1,
                src_guidance_scale=1.5,
                tar_guidance_scale=5.5,
                n_min=0,
                n_max=4,
            )
        elif args.baseline == "sld":
            img = pipe(
                prompt=prompt,
                generator=gen,
                guidance_scale=0.5,
                num_inference_steps=4,
                sld_guidance_scale=2000,
                sld_warmup_steps=2,
                sld_threshold=0.0025,
            ).images[0]
        else:
            img = pipe(
                prompt=prompt,
                generator=gen,
                guidance_scale=0.0,
                height=args.height,
                width=args.width,
                num_inference_steps=4,
                max_sequence_length=256,
            ).images[0]
        images.append(img)
        img.save(osp.join(args.save_dir, args.prompt, "flux_unlearn", f"flux_unlearn_{idx}.png"))

    if args.save_grid:
        save_images_grid(images, osp.join(args.save_dir, args.prompt, "flux_unlearn", "flux_unlearn_images.png"))


def arg_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", type=str, default="ring-a-bell-3-16")
    parser.add_argument(
        "--save_pt",
        type=str,
        default="/home/erjin/github/Diffusion_Infringement_Resolver/results/save_checkpoint/"
        + "NSFW_SOTA/model_flux_concept_nude couple_eps_0.5_beta_0.001_sample_20_epochs_6_lr_0.00.80.8/"
        + "epoch_1_step_24_ff.pt",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--device", type=str, default="0")
    parser.add_argument("--save_dir", type=str, default="images/1024x1024")
    parser.add_argument("--save_grid", action="store_true")
    # for ip evaluation
    parser.add_argument("--concept", "-c", type=str, default=None, help="Concept to evaluate for ip evaluate")
    parser.add_argument(
        "--baseline",
        type=str,
        default=None,
        help="Baseline model to compare with, if not provided, only the unlearned " "model will be used, flowedit, sld",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = arg_parse()
    main(args)
