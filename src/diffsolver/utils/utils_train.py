import glob
import os
from typing import Optional

import mmengine
import omegaconf
import torch
from torch.utils.data import Subset

import time
from diffsolver.data.unlearn_template import CON_DECON_DICT, SYNONYMS_DICT
from diffsolver.utils import update_con_decon
from .clip import calculate_clip_score, get_clip_encoders

CON_DECON_DICT = update_con_decon(CON_DECON_DICT, SYNONYMS_DICT)


def get_file_name(save_dir: str, prompt: str = None, seed: int = 44):
    # get current time
    timestr = time.strftime("%Y%m%d-%H%M%S")
    # get file name
    # if prompt is too long make the name shorter
    if len(prompt) > 30:
        prompt = prompt[:30]
    name = f"{prompt}_seed_{seed}_{timestr}.png"
    out_path = os.path.join(save_dir, name)
    return out_path


@torch.no_grad()
def save_image(
    pipe,
    prompts: str,
    g_cpu: torch.Generator,
    steps: int,
    seed: int,
    save_dir=None,
    save_path=None,
    width=None,
    height=None,
    output_type="pil",
):
    image = pipe(
        prompts, generator=g_cpu, num_inference_steps=steps, width=width, height=height, output_type=output_type
    )

    if save_path is not None:
        image["images"][0].save(save_path)
        return

    if save_dir is None:
        return image["images"]
    else:
        if isinstance(prompts, str):
            prompts = [prompts]
        for img, prompt in zip(image["images"], prompts):
            name = get_file_name(save_dir, prompt=prompt, seed=seed)
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            img.save(name)
        return None


def save_image_seed(
    pipe,
    prompts: str,
    steps: int,
    device: torch.device,
    seed: int,
    save_dir=None,
    save_path=None,
    width=None,
    height=None,
    output_type="pil",
    hookers: Optional[list] = None,
):
    assert hookers is None, "hookers is not required for this function"
    g_cpu = torch.Generator(device).manual_seed(seed)
    return save_image(
        pipe,
        prompts,
        g_cpu,
        steps,
        seed=seed,
        save_dir=save_dir,
        save_path=save_path,
        width=width,
        height=height,
        output_type=output_type,
    )


def save_image_binarize_seed(
    pipe,
    prompts: str,
    steps: int,
    device: torch.device,
    seed: int,
    save_dir=None,
    save_path=None,
    width=None,
    height=None,
    hookers: Optional[list] = None,
):
    assert hookers is not None, "hookers is required for this function"
    if not isinstance(hookers, list):
        hookers = [hookers]
    previous_masking = []
    for h in hookers:
        if h:
            previous_masking.append(h.masking)
            h.masking = "continues2binary"
    g_cpu = torch.Generator(device).manual_seed(seed)
    img = save_image(
        pipe, prompts, g_cpu, steps, seed=seed, save_dir=save_dir, save_path=save_path, width=width, height=height
    )
    for h, pm in zip(hookers, previous_masking):
        if h:
            h.masking = pm
    return img


def overwrite_debug_cfg(cfg):
    # overwrite the cfg for debug, use few image for training, and use few steps for generating image
    # get overwriteed cfg attribute list
    overwrite_list = cfg.debug_cfg
    for key, value in overwrite_list.items():
        key_list = key.split(".")
        attr = cfg[key_list[0]]
        if len(key_list) > 2:
            for k in key_list[1:-1]:
                attr = attr[k]
        attr[key_list[-1]] = value
    print("overwriting these config parameter:")
    print(overwrite_list)


def load_config(args):
    cfg = omegaconf.OmegaConf.load(args.cfg)

    if args.concept:
        cfg.data.concept = args.concept

    # override config with args
    cfg = mmengine.Config(omegaconf.OmegaConf.to_container(cfg))
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)
    cfg = omegaconf.OmegaConf.create(cfg.to_dict())

    if cfg.debug:
        overwrite_debug_cfg(cfg)

    return cfg


def is_float_with_fractional_part(number):
    return isinstance(number, float) and not number.is_integer()


def foreground_score_calculation(d, cfg, clip_dict):
    img_path = d["path"]
    # use only the concepts for evaluation, e.g. a photo of <concept>
    foreground_prompt = cfg.data.concept
    foreground_promptwnconcept = CON_DECON_DICT[cfg.data.concept]

    score = calculate_clip_score(clip_dict, img_path, foreground_prompt, cfg.trainer.device, single_word=True)
    score_wnconcept = calculate_clip_score(
        clip_dict,
        img_path,
        foreground_promptwnconcept,
        cfg.trainer.device,
        single_word=True,
    )
    # (negative score mean the image is not related to the concept)
    score = score - score_wnconcept
    # use score as penalty (1 meanes highest penalty)
    score = 1 if score < 0 else score
    return score


def get_all_concept_template_with_synonyms(concept):
    # TODO quick fix for now need to find a better way to get all concept template with synonyms
    syn_with_concept = []
    for k, v in SYNONYMS_DICT.items():
        if k in concept:
            concept_with_v = [concept.replace(k, v) for v in v]
            syn_with_concept = [concept] + concept_with_v
            break

    if len(syn_with_concept) == 0:
        raise ValueError(f"concept {concept} not found in SYNONYMS_DICT")
    else:
        return syn_with_concept


def background_score_calculation(d, cfg, clip_dict):
    img_path = d["path"]
    prompt = d["prompt"]
    # get the prompt w/o concept
    if cfg.data.with_synonyms:
        concept_with_synoyms = get_all_concept_template_with_synonyms(cfg.data.concept)
        for c in concept_with_synoyms:
            if c in prompt:
                background = prompt.replace(c, "")
                break
    else:
        background = prompt.replace(cfg.data.concept, "")
    score_wconcept = calculate_clip_score(clip_dict, img_path, background, cfg.trainer.device)

    # replace base name
    basename = "deconcept_" + os.path.basename(img_path)
    img_path = os.path.join(os.path.dirname(img_path), basename)
    score_woconcept = calculate_clip_score(clip_dict, img_path, background, cfg.trainer.device)
    return abs(score_wconcept - score_woconcept)


def dataset_filter(dataset, cfg, logger):
    # clearn cache to avoid OOM
    torch.cuda.empty_cache()

    if is_float_with_fractional_part(cfg.data.filter_ratio * cfg.data.size * 2):
        raise ValueError("filter_ratio * data.size must be an integer, change the filter_ratio or data.size")

    # prepare CLIP encoder
    clip_dict = get_clip_encoders()

    filter_score_list = []
    for idx in range(len(dataset)):
        d = dataset[idx]
        value = d["value"]
        # calculate filter score for image w concepts
        # calculate background score and foreground score (lower the better)
        if value:
            background_score = background_score_calculation(d, cfg, clip_dict)
            if cfg.data.with_fg_filter:
                foreground_score = foreground_score_calculation(d, cfg, clip_dict)
            else:
                foreground_score = 0
            score = background_score + foreground_score
            filter_score_list.append(score)
        else:
            filter_score_list.append(0)

    pos = [1 for _ in range(len(dataset))]
    # keep remove the lowest similarity score
    filter_score_list = torch.tensor(filter_score_list)
    threshold = torch.quantile(filter_score_list, cfg.data.filter_ratio)

    for idx, score in enumerate(filter_score_list):
        if score > threshold:
            pos[idx] = 0
            pos[idx + 1] = 0

    subset_pos = [i for i, p in enumerate(pos) if p == 1]
    logger.info(f"filter out {len(dataset) - len(subset_pos)} images")
    dataset = Subset(dataset, subset_pos)
    # update the dataset size in cfg
    cfg.data.size = len(subset_pos)
    # release GPU memory from clip model
    del clip_dict
    return dataset, cfg


def validation_discreate_path_extraction(val_dir, dir_name, concept: str):
    valid_paths = glob.glob(os.path.join(val_dir, dir_name, "*.png"))
    filter_paths = []
    for p in valid_paths:
        basename = os.path.basename(p)
        if concept in basename:
            filter_paths.append(p)
    return filter_paths
