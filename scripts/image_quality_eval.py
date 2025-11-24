# quality evaluation
# evaluation based on the following metrics
# FID, SSIM and CLIP Score not for classification

import glob
import json
import os
from collections import OrderedDict
from typing import Callable

import torch
import torchvision.transforms as transforms
from alive_progress import alive_bar
from PIL import Image
from torchmetrics.image import StructuralSimilarityIndexMeasure as SSIM
from torchmetrics.image.fid import FrechetInceptionDistance as FID

import argparse
from diffsolver.data import EvalDataset
from diffsolver.utils import (
    calculate_clip_score,
    create_pipeline,
    get_clip_encoders,
    get_precision,
    save_image_binarize_seed,
    save_image_seed,
)


def parse():
    parser = argparse.ArgumentParser(description="Image Quality Evaluation")
    parser.add_argument("--task", type=str, required=True, help="task including gen (generating dataset), eval")
    parser.add_argument("--data_dir", type=str, default="./datasets")
    parser.add_argument(
        "--dataset_name", "-dn", type=str, default="laion", help="dataset name, available laion, coco, flickr"
    )
    parser.add_argument("--result_dir", type=str, default="./images/laions_original_ds", help="save the tmp files")
    parser.add_argument("--image_size", type=int, default=1024)
    parser.add_argument("--device", type=str, default="cuda:2")
    parser.add_argument("--precision", type=str, default="bf16")
    parser.add_argument("--seed", type=int, default=44)
    parser.add_argument("--data_size", "-ds", type=int, default=10)
    # diffusion pipeline config
    parser.add_argument("--model", type=str, default="flux", help="model type, available flux")
    parser.add_argument("--num_intervention_steps", "-ni", type=int, default=5)
    parser.add_argument(
        "--binary", "-b", action="store_true", help="whether to use binary mask"
    )  # avoid using boolen type buggy
    parser.add_argument(
        "--masking", type=str, default="hard_discrete", help="masking type, available binary, hard_discrete, sigmoid"
    )
    parser.add_argument("--scope", type=str, default=None, help="scope for lambda binary mask")
    parser.add_argument(
        "--ratio", type=float, nargs="+", default=None, help="sparsity ratio for local global lambda mask"
    )
    parser.add_argument("--lambda_threshold", "-lt", type=float, default=0.001, help="threshold for lambda")
    parser.add_argument("--save_pt", type=str, help="checkpoint for the mask")
    parser.add_argument(
        "--clip_backbone", type=str, default="ViT-B-16", help="clip model type, available ViT-B-16, ViT-L-14"
    )
    parser.add_argument(
        "--clip_pretrained",
        type=str,
        default="datacomp_xl_s13b_b90k",
        help="clip pretrained model, datacomp_xl_s13b_b90k",
    )
    parser.add_argument("--clip_eval", action="store_true", help="whether to use clip for evaluation")
    parser.add_argument("--fid_features", "-ff", type=int, default=2048, help="feature size for FID")
    parser.add_argument("--only_mask_dm", action="store_true", help="only generate masked diffusion model images")
    return parser.parse_args()


def prompt_extraction(prompt_dict, path):
    # "0.png" format, get the index
    idx = os.path.basename(path)[:-4]
    return prompt_dict[idx]


def create_dir(dir_name, folder_name_list):
    assert isinstance(folder_name_list, list), "folder_name must be a list"
    folder_path_list = []
    for folder_name in folder_name_list:
        if not os.path.exists(dir_name):
            os.mkdir(dir_name)
        folder_path = os.path.join(dir_name, folder_name)
        if not os.path.exists(folder_path):
            os.mkdir(folder_path)
        folder_path_list.append(folder_path)
    return folder_path_list


def get_image_path(save_dir):
    output_list = glob.glob(os.path.join(save_dir, "*.png"))
    output_list.sort()
    return output_list


def pil_to_tensor(path, trans_func: Callable, size=None):
    image = Image.open(path)
    if size:
        image = image.resize((size, size), Image.Resampling.BICUBIC)
    # require 4D tensor B, C, H, W
    output = trans_func(image).unsqueeze(0)
    return output


def data_preparation(args, save_path_list, prompt_save_path):
    ds = EvalDataset(
        data_dir=args.data_dir,
        dataset_name=args.dataset_name,
        image_size=args.image_size,
        max_size=args.data_size,
    )

    # precision
    precision = get_precision(args.precision)

    # load diffusion pipeline same as in scripts/binary_mask_eval.py
    pipe = create_pipeline(args.model, args.device, precision)
    mask_pipe, hookers = create_pipeline(
        model_id=args.model,
        device=args.device,
        lambda_threshold=0,
        save_pt=args.save_pt,
        binary=args.binary,
        scope=args.scope,
        ratio=args.ratio,
        torch_dtype=torch.bfloat16,
        return_hooker=True,
    )

    # placeholder for saving prompt for generating CLIP score and validation
    prompt_dict = OrderedDict()

    # save the original images
    with alive_bar(args.data_size) as bar:
        for idx in range(args.data_size):
            data = ds[idx]
            image = data["image"]
            prompt = data["text"]

            # original_path, diff_path, mask_path
            path_list = [os.path.join(save_path, f"{idx}.png") for save_path in save_path_list]

            width, height = args.image_size, args.image_size
            image = image.resize((width, height), Image.Resampling.BICUBIC)
            if not args.only_mask_dm:
                image.save(path_list[0])
                save_image_seed(
                    pipe,
                    prompt,
                    args.num_intervention_steps,
                    args.device,
                    args.seed,
                    save_path=path_list[1],
                    width=width,
                    height=height,
                )
            save_image_binarize_seed(
                pipe=mask_pipe,
                prompts=prompt,
                hookers=hookers,
                steps=args.num_intervention_steps,
                device=args.device,
                seed=args.seed,
                save_path=path_list[2],
                width=width,
                height=height,
            )
            prompt_dict[idx] = prompt
            bar()

    # save prompt
    with open(prompt_save_path, "w") as f:
        json.dump(prompt_dict, f)


def evaluation(args, save_path_list, prompt_save_path, eval_result_path):
    """
    evaluation include the following metrics
    FID,
    SSIM,
    CLIP
    """

    # preapre the dataset
    ori_ds = get_image_path(save_path_list[0])
    diff_ds = get_image_path(save_path_list[1])
    mask_ds = get_image_path(save_path_list[2])

    trans_func = transforms.Compose([transforms.PILToTensor()])

    # prepare FID and SSIM metric
    diff_fid = FID(feature=args.fid_features).to(args.device)
    mask_fid = FID(feature=args.fid_features).to(args.device)
    mask_ssim = SSIM(data_range=1).to(args.device)

    # get prompt dict for CLIP score
    if args.clip_eval:
        prompt_dict = json.load(open(prompt_save_path, "r"))

    # prepare CLIP model, transform function for image, text tokenzier
    clip_dict = get_clip_encoders(backbone=args.clip_backbone, pretrained=args.clip_pretrained)
    diff_score_list = []
    mask_diff_score_list = []

    # FID SSIM
    print("Computing FID score...")

    with alive_bar(len(ori_ds)) as bar:
        for ori, diff, mask in zip(ori_ds, diff_ds, mask_ds):
            # only need it for laion dataset due to padding
            image_size = args.image_size if args.dataset_name == "laion" else None
            ori_img = pil_to_tensor(ori, trans_func, image_size).to(args.device)
            diff_img = pil_to_tensor(diff, trans_func, image_size).to(args.device)
            mask_img = pil_to_tensor(mask, trans_func, image_size).to(args.device)

            # detach from GPU
            try:
                diff_fid.update(ori_img, real=True)
                diff_fid.update(diff_img, real=False)

                mask_fid.update(ori_img, real=True)
                mask_fid.update(mask_img, real=False)
            except Exception as e:
                print("Error: ", e)
                continue

            # calculate SSIM
            try:
                mask_ssim.update(diff_img / 255, mask_img / 255)
            except Exception as e:
                continue

            if args.clip_eval:
                # calculate CLIP score
                diff_score = calculate_clip_score(
                    clip_dict=clip_dict, image=diff, text=prompt_extraction(prompt_dict, diff), device=args.device
                )

                mask_diff_score = calculate_clip_score(
                    clip_dict=clip_dict, image=mask, text=prompt_extraction(prompt_dict, mask), device=args.device
                )

                diff_score_list.append(diff_score)
                mask_diff_score_list.append(mask_diff_score)
            bar()

    diff_score = diff_fid.compute()
    mask_diff_score = mask_fid.compute()
    mask_ssim_score = mask_ssim.compute()

    print(
        "Start Image Quality Evaluation:\n"
        + "---------------------------------\n"
        + "model: {}\n".format(args.model)
        + "dataset: {}\n".format(args.dataset_name)
        + "data size: {}\n".format(args.data_size)
        + "CLIP backbone: {}\n".format(args.clip_backbone)
        + "---------------------------------\n"
    )

    print(f"FID diffusion model {args.model} images: {diff_score}")
    print(f"FID masked diffusion model {args.model} images: {mask_diff_score}")

    print("SSIM score: ", mask_ssim_score)

    if args.clip_eval:
        clip_diff_score = sum(diff_score_list) / len(diff_score_list)
        clip_mask_diff_score = sum(mask_diff_score_list) / len(mask_diff_score_list)
        print("CLIP score diffusion model: ", clip_diff_score)
        print("CLIP score masked diffusion model: ", clip_mask_diff_score)

    with open(eval_result_path, "w") as f:
        f.write(f"FID score for diffusion model: {diff_score}\n")
        f.write(f"FID score for masked diffusion model: {mask_diff_score}\n")
        f.write(f"mask diffusion SSIM: {mask_ssim_score}\n")
        if args.clip_eval:
            f.write(f"diffusion CLIP score: {clip_diff_score}\n")
            f.write(f"masked diffusion CLIP score: {clip_mask_diff_score}\n")


def main():
    args = parse()
    save_path_list = create_dir(args.result_dir, ["original", "diffusion", "masked"])
    prompt_save_path = os.path.join(args.result_dir, "prompt.json")
    eval_result_path = os.path.join(args.result_dir, "eval_result.txt")

    if args.task == "gen":
        # generate the dataset
        data_preparation(args, save_path_list, prompt_save_path)
    elif args.task == "eval":
        evaluation(args, save_path_list, prompt_save_path, eval_result_path)
    else:
        raise ValueError(f"task {args.task} not supported")


if __name__ == "__main__":
    main()
