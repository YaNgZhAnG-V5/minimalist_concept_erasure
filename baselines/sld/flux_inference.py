import torch
from sld import FluxPipelineForSLB
from torchvision import transforms
from torchvision.utils import save_image

import argparse


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--torch_dtype", type=str, default="bf16", choices=["bf16", "fp16"])
    parser.add_argument("--device", type=str, default="cuda:3")
    parser.add_argument("--model_name", type=str, default="schnell", choices=["schnell", "dev"])
    parser.add_argument("--disable_progress_bar", action="store_true")
    parser.add_argument("--seed", type=int, default=44)
    parser.add_argument("--num_interventions", "-ni", type=int, default=5)
    parser.add_argument("--guidance", "-g", type=float, default=0.5)
    parser.add_argument("--warmup_steps", "-ws", type=int, default=2)
    parser.add_argument("--sld_threshold", "-st", type=float, default=0.0025)
    parser.add_argument("--sld_guidance_scale", "-sgs", type=float, default=2000)
    parser.add_argument("--prompt", "-p", type=str, required=True)
    return parser.parse_args()


def get_precision(precision: str):
    assert precision in ["fp16", "fp32", "bf16"], "precision must be either fp16, fp32, bf16"
    if precision == "fp16":
        torch_dtype = torch.float16
    elif precision == "bf16":
        torch_dtype = torch.bfloat16
    elif precision == "fp32":
        torch_dtype = torch.float32
    elif precision == "fp64":
        torch_dtype = torch.float64
    return torch_dtype


def load_pipeline(torch_dtype: torch.dtype, disable_progress_bar: bool = False, model_name: str = "schnell"):
    if model_name == "schnell":
        pipe = FluxPipelineForSLB.from_pretrained("black-forest-labs/FLUX.1-schnell", torch_dtype=torch_dtype)
        pipe.set_progress_bar_config(disable=disable_progress_bar)
    else:
        pipe = FluxPipelineForSLB.from_pretrained("black-forest-labs/FLUX.1-dev", torch_dtype=torch_dtype)
        pipe.set_progress_bar_config(disable=disable_progress_bar)
    return pipe


def convert_pil_to_tensor(pil_img):
    if isinstance(pil_img, list):
        if len(pil_img) != 1:
            raise ValueError("check your prompts, one prompt per image is allowed")
        pil_img = pil_img[0]

    transform = transforms.ToTensor()
    return transform(pil_img)


def main(args):
    torch_type = get_precision(args.torch_dtype)
    device = torch.device(args.device)
    pipe = load_pipeline(torch_dtype=torch_type, model_name=args.model_name)
    pipe.to(device)
    gen = torch.Generator(device.type).manual_seed(args.seed)
    prompt = args.prompt
    out = pipe.inference(
        prompt=prompt,
        generator=gen,
        guidance_scale=args.guidance,
        num_inference_steps=args.num_interventions,
        sld_guidance_scale=args.sld_guidance_scale,
        sld_warmup_steps=args.warmup_steps,
        sld_threshold=args.sld_threshold,
    )
    gen = torch.Generator(device.type).manual_seed(args.seed)
    image_with_slb = out.images[0]
    out = pipe(
        prompt=prompt,
        generator=gen,
        guidance_scale=10,
        num_inference_steps=args.num_interventions,
    )
    image_without_slb = out.images[0]
    save_image(
        [convert_pil_to_tensor(image_without_slb), convert_pil_to_tensor(image_with_slb)],
        f"{prompt}_with_and_without_slb.png",
    )


if __name__ == "__main__":
    args = parse_args()
    main(args)
