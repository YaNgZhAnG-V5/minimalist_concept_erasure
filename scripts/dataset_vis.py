# This is script for visulization and validation of the dataset filter results
import os

import matplotlib.pyplot as plt
import mmengine
import torch
from accelerate.utils import set_seed
from PIL import Image

import argparse
from diffsolver.data import PromptImageDataset
from diffsolver.data.unlearn_template import CON_DECON_DICT, SYNONYMS_DICT
from diffsolver.utils import (
    calculate_clip_score,
    get_clip_encoders,
    get_precision,
    load_config,
    load_pipeline,
    update_con_decon,
)

CON_DECON_DICT = update_con_decon(CON_DECON_DICT, SYNONYMS_DICT)


def prepare_dataset(args):
    # initailize the config
    cfg = load_config(args)
    device = torch.device(cfg.trainer.device)

    # overwriete cfg concept with args concept
    cfg.data.concept = args.concept
    cfg.data.save_dir = args.save_dir

    if not os.path.exists(cfg.data.save_dir):
        os.makedirs(cfg.data.save_dir)

    # setup seed
    seed = cfg.trainer.seed
    set_seed(seed)  # use accelerate set_seed
    # set the precision, only support bf16, fp16, fp32
    torch_dtype = get_precision(cfg.trainer.precision)
    # initialize pipeline
    pipe = load_pipeline(cfg.trainer.model, torch_dtype, cfg.trainer.disable_progress_bar)
    pipe.to(device)

    # prepare for the datasets and dataloader
    train_dataset = PromptImageDataset(
        metadata=cfg.data.metadata,
        deconceptmeta=cfg.data.deconceptmeta,
        pipe=pipe,
        num_inference_steps=cfg.trainer.num_intervention_steps,
        save_dir=cfg.data.save_dir,
        device=device,
        seed=seed,
        size=cfg.data.size,
        concept=getattr(cfg.data, "concept", None),
        neutral_concept=getattr(cfg.data, "neutral_concept", None),
        only_deconcept_latent=getattr(cfg.data, "only_deconcept_latent", False),
        style=getattr(cfg.data, "style", True),
    )
    del pipe  # release VRAM
    return train_dataset, cfg


def main(args):
    dataset, cfg = prepare_dataset(args)

    # prepare CLIP encoder
    clip_dict = get_clip_encoders()

    # only visualization the unlearn prompts

    imgs = []
    captions = []
    score_list = []
    for idx in range(len(dataset)):
        d = dataset[idx]
        value = d["value"]
        if value:
            img_path = d["path"]
            prompt = d["prompt"]
            # get the prompt w/o concept
            background = prompt.replace(cfg.data.concept, "")

            # use only the concepts for evaluation, e.g. a photo of <concept>
            foreground_prompt = cfg.data.concept
            foreground_promptwnconcept = CON_DECON_DICT[cfg.data.concept]

            concept_score = calculate_clip_score(
                clip_dict,
                img_path,
                foreground_prompt,
                cfg.trainer.device,
                single_word=True,
                negative_word=foreground_promptwnconcept,
            )

            imgs.append(img_path)
            score_background = calculate_clip_score(clip_dict, img_path, background, cfg.trainer.device)
            captions.append(f"{prompt[:50]}\n concepts score: {(concept_score):.5f}")

            # replace base name
            basename = "deconcept_" + os.path.basename(img_path)
            deconcept_img_path = os.path.join(os.path.dirname(img_path), basename)

            neutral_score = calculate_clip_score(
                clip_dict,
                deconcept_img_path,
                foreground_promptwnconcept,
                cfg.trainer.device,
                single_word=True,
                negative_word=foreground_prompt,
            )

            score_background_woconcept = calculate_clip_score(
                clip_dict, deconcept_img_path, background, cfg.trainer.device
            )
            background_simil_score = abs(score_background - score_background_woconcept)
            captions.append(f"neutral score:{neutral_score:.5f}\n" + f"background score {background_simil_score:.5f}")
            imgs.append(deconcept_img_path)

            score_list = score_list + [background_simil_score] * 2

    def save_plot_images(imgs, captions, cfg, isfilter=False):
        # Create a figure with N rows and 4 columns
        n_images = len(imgs)
        n_rows, _ = divmod(n_images, 4)
        fig, axes = plt.subplots(nrows=n_rows, ncols=4, figsize=(16, 4 * n_rows))

        # Plot each image
        for i, ax in enumerate(axes.flat):
            # Load and display the image
            img = Image.open(imgs[i])
            ax.imshow(img)
            ax.axis("off")  # Turn off axis for better visualization
            ax.set_title(captions[i], fontsize=10)  # Add caption

        title = "after_filter" if isfilter else "before_filter"
        # Adjust layout
        print(f"Visualization {title}")
        plt.tight_layout()
        plt.savefig(
            os.path.join(os.path.dirname(cfg.data.save_dir), f"{cfg.data.concept.strip()}_ds_visual_{title}.png")
        )

    save_plot_images(imgs, captions, cfg, isfilter=False)

    # ignore the image with high score
    # sort score list and sort the image and caption as well
    num_image = len(score_list)
    filtered_num_image = num_image - int(num_image * args.ratio)
    sorted_indices = sorted(range(num_image), key=lambda k: score_list[k], reverse=True)
    # reordered the images and captions
    imgs = [imgs[i] for i in sorted_indices][filtered_num_image:]
    captions = [captions[i] for i in sorted_indices][filtered_num_image:]

    save_plot_images(imgs, captions, cfg, isfilter=True)


def get_parser():
    parser = argparse.ArgumentParser("Dataset Visualization and Filter Evaluation")
    parser.add_argument("--cfg", type=str, default="configs/flux.yaml", help="Config file to load all parameters")
    parser.add_argument("--save_dir", "-s", type=str, default="./results/datavis", help="Directory to save images")
    parser.add_argument("--task", "-t", type=str, default="general", help="Task to perform")
    parser.add_argument("--concept", "-c", type=str, default="nude woman", help="concept to visualize")
    parser.add_argument("--ratio", "-r", type=str, default=0.9, help="filtered ratio")
    parser.add_argument(
        "--cfg-options",
        "-o",
        nargs="+",
        action=mmengine.DictAction,
        help="Override the config entries with format xxx=yyy or xxx.zzz.qqq=yyy .",
    )
    args = parser.parse_args()
    args.save_dir = os.path.join(args.save_dir, args.concept)
    return args


if __name__ == "__main__":
    main(get_parser())
