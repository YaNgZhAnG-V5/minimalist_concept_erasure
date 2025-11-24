# generate datasets of different prompts and latents at different steps
import os
from typing import List

import torch
import yaml
from alive_progress import alive_bar
from diffusers import FluxPipeline
from utils import prepare_noise_addition, save_latent_to_image

import argparse


@torch.no_grad()
def generate_dataset(
    pipe: FluxPipeline,
    prompts: List[str],
    latents_per_prompt: int,
    num_inference_steps: int,
    device: torch.device,
    output_dir: str,
    verbose: bool = False,
    debug: bool = False,
):
    pipe.set_progress_bar_config(disable=True)
    with alive_bar(len(prompts) * latents_per_prompt * num_inference_steps) as bar:
        for prompt in prompts:
            output_path = os.path.join(output_dir, prompt)
            os.makedirs(output_path, exist_ok=True)
            for sample_idx in range(latents_per_prompt):
                # perform reverse process to get the final latent
                generator = torch.Generator(device="cuda:0").manual_seed(torch.randint(0, 1000000, (1,)).item())
                latents = pipe(
                    prompt, generator=generator, num_inference_steps=num_inference_steps, output_type="latent"
                ).images

                # perform forward process to get the noisy latent at each step
                for step in reversed(range(num_inference_steps)):
                    # Prepare timesteps for noise addition
                    timesteps = prepare_noise_addition(pipe, num_inference_steps, device, latents)
                    if verbose:
                        print(f"Adding noise for sample {sample_idx} at step {step}")
                    noise = torch.randn_like(latents)
                    pipe.scheduler._begin_index = 0
                    pipe.scheduler._step_index = step
                    latents = pipe.scheduler.scale_noise(latents, timesteps[step][None], noise)
                    pipe.scheduler._begin_index, pipe.scheduler._step_index = None, None

                    # Save the latent as a torch tensor
                    assert latents.shape[0] == 1
                    torch.save(latents, os.path.join(output_path, f"sample_{sample_idx}_latent_{step}.pt"))

                    # save the noise as a torch tensor
                    torch.save(noise, os.path.join(output_path, f"sample_{sample_idx}_noise_{step}.pt"))

                    if debug:
                        save_latent_to_image(
                            latents, pipe, os.path.join(output_path, f"sample_{sample_idx}_image_{step}")
                        )
                    bar()


def args_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--prompts-file", type=str, default="prompts.yaml")
    parser.add_argument("--concept", type=str, default="gun")
    parser.add_argument("--num-inference-steps", type=int, default=4)
    parser.add_argument("--latents-per-prompt", type=int, default=10)
    parser.add_argument("--output-dir", type=str, default="data/debug")
    parser.add_argument("--verbose", "-v", action="store_true")
    parser.add_argument("--debug", "-d", action="store_true")
    return parser.parse_args()


def main():
    args = args_parser()
    pipe = FluxPipeline.from_pretrained("black-forest-labs/FLUX.1-schnell", torch_dtype=torch.bfloat16)
    device = torch.device(f"cuda:{args.device}")
    pipe.to(device)
    torch.manual_seed(args.seed)
    with open(args.prompts_file, "r") as f:
        all_prompts = yaml.load(f, Loader=yaml.FullLoader)
    assert args.concept in all_prompts.keys(), f"Concept {args.concept} not found in prompts file"
    concept_train_prompts = all_prompts[args.concept]["train_prompts"]
    prompts = [prompt["prompt"] for prompt in concept_train_prompts]
    generate_dataset(
        pipe,
        prompts,
        args.latents_per_prompt,
        args.num_inference_steps,
        device,
        args.output_dir,
        args.verbose,
        args.debug,
    )


if __name__ == "__main__":
    main()
