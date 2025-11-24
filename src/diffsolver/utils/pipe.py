import pprint
from pathlib import Path

import torch
from diffusers import EulerDiscreteScheduler

import re
from diffsolver.hooks import HOOKNAMES, CrossAttentionExtractionHook, FeedForwardHooker, NormHooker
from diffsolver.models import (
    DiTPipelineForCheckpointing,
    FluxPipelineForCheckpointing,
    SD2PipelineForCheckpointing,
    SD3PipelineForCheckpointing,
    SDXLPipelineForCheckpointing,
)
from diffsolver.scheduler import ReverseDPMSolverMultistepScheduler


def load_pipeline(model_str: str, torch_dtype: torch.dtype, disable_progress_bar: bool):
    """load a diffusion pipeline"""
    if model_str == "sd1":
        pipe = SD2PipelineForCheckpointing.from_pretrained("CompVis/stable-diffusion-v1-4", include_entities=False)
    elif model_str == "sd2":
        model_id = "stabilityai/stable-diffusion-2-base"
        # Use the Euler scheduler here instead
        scheduler = EulerDiscreteScheduler.from_pretrained(model_id, subfolder="scheduler")
        pipe = SD2PipelineForCheckpointing.from_pretrained(
            model_id,
            scheduler=scheduler,
            torch_dtype=torch_dtype,
        )
    elif model_str == "sdxl":
        model_id = "stabilityai/stable-diffusion-xl-base-1.0"
        pipe = SDXLPipelineForCheckpointing.from_pretrained(
            model_id,
            torch_dtype=torch_dtype,
            use_safetensors=True,
        )
    elif model_str == "sdxl_turbo":
        model_id = "stabilityai/sdxl-turbo"
        pipe = SDXLPipelineForCheckpointing.from_pretrained(
            model_id,
            torch_dtype=torch_dtype,
            use_safetensors=True,
        )
        pipe.set_distilled()
    elif model_str == "sd3":
        pipe = SD3PipelineForCheckpointing.from_pretrained(
            "stabilityai/stable-diffusion-3.5-large", torch_dtype=torch_dtype
        )
    elif model_str == "dit":
        pipe = DiTPipelineForCheckpointing.from_pretrained("facebook/DiT-XL-2-256", torch_dtype=torch_dtype)
        pipe.scheduler = ReverseDPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    elif model_str == "flux":
        pipe = FluxPipelineForCheckpointing.from_pretrained("black-forest-labs/FLUX.1-schnell", torch_dtype=torch_dtype)
    elif model_str == "flux_dev":
        pipe = FluxPipelineForCheckpointing.from_pretrained("black-forest-labs/FLUX.1-dev", torch_dtype=torch_dtype)
    else:
        raise ValueError(f"Model {model_str} not supported")
    pipe.set_progress_bar_config(disable=disable_progress_bar)
    return pipe


# TODO need to remove once we have new saving, do it later
def get_save_pts(save_pt: str):
    """
    all checkpoints for each mask are save in the same directory
    as the following format epoch_{epoch}_step_{step}_{mask_name}.pt,
    mask name is ff, attn, norm

    Args:
        save_pt (str): path to the checkpoint file, can be ff.pt, attn.pt, norm.pt
    """
    global HOOKNAMES
    # convert to Path object to avoid os.sep issue
    path = Path(save_pt)
    basename, dirname = path.name, path.parent

    # check if basename is in the format of epoch_{epoch}_step_{step}_{mask_name}.pt
    # using regex for validation
    if not re.match(r"epoch_\d+_step_\d+_(ff|attn|norm).pt", basename):
        raise ValueError(f"Invalid checkpoint name {basename} format, should be epoch_<XXX>_step_<XXX>_<hookname>.pt")

    epoch, step = basename.split("_")[1], basename.split("_")[3]
    checkpoints_dict = {}
    for name in HOOKNAMES:
        path = Path.joinpath(dirname, f"epoch_{epoch}_step_{step}_{name}.pt")
        if path.exists():
            checkpoints_dict[name] = path
        else:
            continue
    return checkpoints_dict


def create_pipeline(
    model_id,
    device,
    torch_dtype,
    save_pt=None,
    lambda_threshold: float = 1,
    binary=False,
    masking="binary",
    return_hooker=False,
    scope=None,
    ratio=None,
    merge_mask=None,
):
    """
    create the pipeline and optionally load the saved mask
    """

    # return callable class hooker
    def get_hooker(hooker_name):
        if hooker_name == "attn":
            return CrossAttentionExtractionHook
        elif hooker_name == "ff":
            return FeedForwardHooker
        elif hooker_name == "norm":
            return NormHooker
        else:
            raise ValueError(f"hooker_name {hooker_name} is not supported")

    ratio: list
    pipe = load_pipeline(model_id, torch_dtype, disable_progress_bar=True)
    pipe.to(device)
    pipe.vae.requires_grad_(False)
    if hasattr(pipe, "unet"):
        pipe.unet.requires_grad_(False)
    else:
        pipe.transformer.requires_grad_(False)
    if save_pt:
        save_pts = get_save_pts(save_pt)

        if ratio:
            assert len(ratio) == len(
                save_pts
            ), "mismatch between number ratio and number of masks, check the ratio and save_pt"

        hookers = []
        for hook in HOOKNAMES:
            hooker_cls = get_hooker(hook)
            # only attn hook need the model name
            if hook == "attn":
                argument = [pipe, torch_dtype, 1, masking, save_pts[hook], ".*", 0, binary, False, model_id]
            else:
                argument = [pipe, ".*", torch_dtype, masking, save_pts[hook], 0, 0, False, binary]
            hooker = hooker_cls(*argument)
            hooker.add_hooks(init_value=1)
            hookers.append(hooker)

        g_cpu = torch.Generator(torch.device(device)).manual_seed(1)
        _ = pipe("abc", generator=g_cpu, num_inference_steps=1)

        for idx, (h, n) in enumerate(zip(hookers, HOOKNAMES)):
            if merge_mask:
                h.lambs = merge_mask[n]
            else:
                h.load(device=device, threshold=lambda_threshold)
                if scope == "local" or scope == "global" and ratio is not None:
                    h.binarize(scope, ratio[idx])

    if return_hooker:
        return pipe, hookers
    else:
        return pipe


def local_binarize(lambs, ratio):
    for i, lamb in enumerate(lambs):
        num_heads = lamb.size(0)
        num_activate_heads = int(num_heads * ratio)
        # Sort the lambda values with stable sorting to maintain order for equal values
        sorted_lamb, sorted_indices = torch.sort(lamb, descending=True, stable=True)
        # Find the threshold value
        threshold = sorted_lamb[num_activate_heads - 1]
        # Create a mask based on the sorted indices
        mask = torch.zeros_like(lamb)
        mask[sorted_indices[:num_activate_heads]] = 1.0
        # Binarize the lambda based on the threshold and the mask
        lambs[i] = torch.where(lamb > threshold, torch.ones_like(lamb), mask)
    return lambs


def merge_masks(masks_pt_list: list, ratio: list, device: str, verbose: bool = False):
    """
    Args:
        verbose: print the lambda values, the percentage of overlap
    """
    assert len(masks_pt_list) > 1, "need at least two masks to merge"
    save_pt_list = [get_save_pts(mask) for mask in masks_pt_list]
    lambs_dict_list = []

    if verbose:
        assert len(masks_pt_list) == 2, "only support two masks for verbose mode"
        stats_dict = {
            "attn": {"num": 0, "same": 0},
            "ff": {"num": 0, "same": 0},
            "norm": {"num": 0, "same": 0},
        }
    for save_pts in save_pt_list:
        lambs_dict = {}
        for i, hook in enumerate(save_pts):
            lambs = torch.load(save_pts[hook], weights_only=True, map_location=device)
            lambs = [torch.clamp(lamb, min=0.0) for lamb in lambs]
            lambs = local_binarize(lambs, ratio[i])
            lambs_dict[hook] = lambs
        lambs_dict_list.append(lambs_dict)

    output_lambs_dict = lambs_dict_list[0]
    for lambs_dict in lambs_dict_list[1:]:
        for hook in HOOKNAMES:
            temp_lambs = lambs_dict[hook]
            for i, lamb in enumerate(output_lambs_dict[hook]):
                output_lambs_dict[hook][i] = lamb * temp_lambs[i]
                if verbose:
                    stats_dict[hook]["num"] += torch.sum(lamb == 0)

                    # calculate the percentage of overlap
                    masking_overlap = (lamb == 0) & (temp_lambs[i] == 0)
                    stats_dict[hook]["same"] += torch.sum(masking_overlap).item()
    if verbose:
        pprint.pprint(stats_dict)
        print("Percentage of overlap")
        for hook in HOOKNAMES:
            print(f"{hook}: {stats_dict[hook]['same'] / stats_dict[hook]['num'] * 100:.2f}%")
    return output_lambs_dict
