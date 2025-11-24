import json
import os

import torch
from copyright_eval import calculate_clip_score

import argparse
from diffsolver.data.unlearn_template import SIMPLE_DECONCEPT_TEMPLATES, SIMPLE_DESTYLE_TEMPLATES
from diffsolver.utils import create_pipeline, get_clip_encoders
from diffsolver.utils.utils_deconcept import txt2txt


def create_parser():
    parser = argparse.ArgumentParser("deconcept prompt generation")
    parser.add_argument("--task", type=str, default="prompt_generate", help="prompt_generate, tempalte_generate")
    parser.add_argument("--num_prompts", type=int, default=40, help="number of prompts to generate")
    parser.add_argument("--concept", type=str, default="Captain America", help="concept for the prompt")
    parser.add_argument("--neutral_concept", type=str, default="man", help="neutral concept for the prompt")
    parser.add_argument("--device", type=str, default="3", help="device to run the model")
    parser.add_argument("--seed", type=int, default=44, help="random seed")
    parser.add_argument("--output_dir", type=str, default="./configs/prompts")
    # diffusion model config
    parser.add_argument("--model", type=str, default="flux", help="model type, available flux")
    parser.add_argument("--num_intervention_steps", "-ni", type=int, default=5, help="number of intervention steps")
    # clip config for filtering
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
    parser.add_argument("--threshold", type=float, default=0.15, help="threshold for clip score")
    parser.add_argument("--save_images", action="store_true", help="whether to save the images")
    parser.add_argument("--template_type", type=str, default="concept", help="concept or style")
    return parser


def deconcept_prompt_generation(args):
    input_text = (
        f"Give me {args.num_prompts} copyright prompts with only"
        + f"{args.concept} for image generation with 6 to 8 words"
    )
    output_text = txt2txt(
        input_text=input_text,
        model="gpt-4o",
        system_message="You are an AI assistant that helps people find information.",
        top_p=0,
        temp=0,
    )
    try:
        output_text_dict = json.loads(output_text)
    except Exception as e:
        print(f"Error: {e}, output_text: {output_text}")
        raise ValueError("output_text is not a valid json")
    assert isinstance(output_text_dict, dict), "output_text_dict must be a dictionary"
    return output_text_dict


@torch.no_grad()
def generate_image(args, pipe, prompt):
    g_cpu = torch.Generator(args.device).manual_seed(args.seed)
    preparation_phase_output = pipe.inference_preparation_phase(
        prompt,
        generator=g_cpu,
        num_inference_steps=args.num_intervention_steps,
        output_type="latent",
    )
    intermediate_latents = []
    timesteps = preparation_phase_output.timesteps
    for timesteps_idx, time in enumerate(timesteps):
        latents = pipe.inference_denoising_step(timesteps_idx, time, preparation_phase_output)
        preparation_phase_output.latents = latents
        intermediate_latents.append(latents)
    prompt_embeds = preparation_phase_output.prompt_embeds
    img = pipe.inference_aft_denoising(intermediate_latents[-1], prompt_embeds, g_cpu, "pil", True, args.device)
    return img[0][0]


def prompt_generation(args):
    # create output directory
    if not os.path.exists(args.output_dir):
        # create the output directory
        os.makedirs(args.output_dir)

    # load the clip model for eval
    clip_dict = get_clip_encoders(backbone=args.clip_backbone, pretrained=args.clip_pretrained)
    clip_model = clip_dict["clip_model"].to(args.device)
    transform = clip_dict["transform"]
    tokenizer = clip_dict["tokenizer"]

    pipe = create_pipeline(args.model, args.device, torch.bfloat16)
    prompt_dict = deconcept_prompt_generation(args)

    # convert string dict to dict with ast
    prompt_num = 0
    if not os.path.exists(os.path.join(args.output_dir, "image")):
        os.makedirs(os.path.join(args.output_dir, "image"))

    with open(os.path.join(args.output_dir, f"{args.concept}.yaml"), "w") as f:
        for key, prompt in prompt_dict.items():
            original_prompt = prompt
            prompt = prompt.replace(args.concept, args.neutral_concept)
            img = generate_image(args, pipe, prompt)
            torch.cuda.empty_cache()
            score = calculate_clip_score(clip_model, img, args.concept, tokenizer, transform, args.device)
            print(f"Prompt: {prompt}, Score: {score}")
            torch.cuda.empty_cache()
            if score > args.threshold:
                continue
            else:
                prompt_num += 1
                # save it in yaml file with format or idx: prompt
                if args.save_images:
                    img.save(os.path.join(args.output_dir, "image", f"{prompt}.png"))
                f.write(f"{prompt_num}: {original_prompt}\n")


def template_generation(args):
    if args.template_type == "concept":
        sample_template = SIMPLE_DECONCEPT_TEMPLATES
    elif args.template_type == "style":
        sample_template = SIMPLE_DESTYLE_TEMPLATES

    template_list = [t("concept") for t in sample_template]
    query_template = f"Give more 20 samples like this with c as concept, {template_list[:2]} "
    output_text = txt2txt(
        input_text=query_template,
        model="gpt-4o",
        system_message="You are an AI assistant that helps people find information.",
        top_p=0,
        temp=0,
    )

    refine_prompt = f"convert each sample in {output_text} as lambda function with lambda c: f'.... c ...' format"
    output_text = txt2txt(
        input_text=refine_prompt,
        model="gpt-4o",
        system_message="You are an AI assistant that helps people find information.",
        top_p=0,
        temp=0,
    )
    print(output_text)


def main(args):
    if args.task == "prompt_generate":
        prompt_generation(args)
    elif args.task == "template_generate":
        template_generation(args)
    else:
        raise ValueError("task must be prompt_generate or template_generate")


if __name__ == "__main__":
    parser = create_parser()
    args = parser.parse_args()
    args.device = f"cuda:{args.device}"
    main(args)
