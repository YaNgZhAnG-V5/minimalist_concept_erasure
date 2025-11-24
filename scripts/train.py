import logging
import os
import pprint

import mmengine
import omegaconf
import torch
import tqdm
from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs, ProjectConfiguration, set_seed
from diffusers.optimization import get_scheduler
from diffusers.utils.testing_utils import enable_full_determinism
from torch.utils.data import DataLoader

import argparse
import time
import wandb
from diffsolver.data import VALIDATION_PROMPT, PromptImageDataset  # noqa: F401
from diffsolver.hooks import init_hooker
from diffsolver.utils import (
    calculate_mask_sparsity,
    dataset_filter,
    get_precision,
    load_config,
    load_pipeline,
    save_image_binarize_seed,
    save_image_seed,
)

enable_full_determinism()


def setup_logging(cfg):
    logger = logging.getLogger(__name__)
    project_folder = os.path.join(cfg.logger.output_dir, cfg.logger.project, cfg.logger.notes)
    filename = os.path.join(project_folder, "report.log")
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.FileHandler(filename), logging.StreamHandler()],
    )
    return logger, project_folder


def initialize_wandb(cfg):
    config = omegaconf.OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
    timestr = time.strftime("%Y%m%d-%H%M%S")
    name = f"{cfg.logger.notes}_{timestr}"
    run = wandb.init(project=cfg.logger.project, notes=cfg.logger.notes, tags=cfg.logger.tags, config=config, name=name)
    return run


def load_validation_prompts(cfg):
    global VALIDATION_PROMPT
    concept = getattr(cfg.data, "concept", None)
    style = getattr(cfg.data, "style", "concept")
    validation_prompt = []
    prompts = VALIDATION_PROMPT[style]
    for t in prompts:
        if callable(t):
            validation_prompt.append(t(concept))
        else:
            validation_prompt.append(t)
    return validation_prompt


@torch.no_grad()
def forward_checkpointing(
    pipe,
    prompt,
    generator,
    num_inference_steps,
    output_type="latent",
    keep_last_latent=False,
    width=None,
    height=None,
):
    preparation_phase_output = pipe.inference_preparation_phase(
        prompt,
        generator=generator,
        num_inference_steps=num_inference_steps,
        output_type=output_type,
        width=width,
        height=height,
    )
    intermediate_latents = [preparation_phase_output.latents]
    timesteps = preparation_phase_output.timesteps
    for timesteps_idx, t in enumerate(timesteps):
        latents = pipe.inference_denoising_step(timesteps_idx, t, preparation_phase_output)
        # update latents in output class
        preparation_phase_output.latents = latents
        intermediate_latents.append(latents)
    # pop the last latents for backprop, only keep it for unlearn concept
    if not keep_last_latent:
        intermediate_latents.pop()
    latents.requires_grad = True
    return preparation_phase_output, intermediate_latents, timesteps, latents


def pruning_loss(
    reconstruction_loss_func,
    image_pt,
    image,
    cfg,
):
    if isinstance(image, dict):
        image = image["images"]
        loss_reconstruct = reconstruction_loss_func(image_pt, image)
    else:
        raise ValueError("image should be a dict")

    loss = loss_reconstruct * cfg.trainer.beta
    return loss, loss_reconstruct


def train(args):
    # inital config
    cfg = load_config(args)
    device = torch.device(cfg.trainer.device)

    # inital wandb
    if cfg.logger.type == "wandb":
        run = initialize_wandb(cfg)

    # inital logging
    logger, project_folder = setup_logging(cfg)
    logger.info("Running with config:")
    logger.info(pprint.pformat(omegaconf.OmegaConf.to_container(cfg)))

    # load validation prompts
    validation_prompts = load_validation_prompts(cfg)
    logger.info(f"Validation prompts: {validation_prompts}")

    # setup for accelerator
    accelerator_project_config = ProjectConfiguration(
        project_dir=cfg.logger.output_dir, logging_dir=cfg.logger.output_dir
    )
    kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)

    accelerator_log_with = "all"  # if cfg.logger.type == "csv" else cfg.logger.type
    accelerator = Accelerator(
        gradient_accumulation_steps=cfg.accelerator.accumulate_grad_batches,
        log_with=accelerator_log_with,
        project_config=accelerator_project_config,
        kwargs_handlers=[kwargs],
    )

    # setup seed
    seed = cfg.trainer.seed
    set_seed(seed)  # use accelerate set_seed
    g_cpu = torch.Generator(device.type).manual_seed(seed)
    # set the precision, only support bf16, fp16, fp32
    torch_dtype = get_precision(cfg.trainer.precision)
    # initialize pipeline
    pipe = load_pipeline(cfg.trainer.model, torch_dtype, cfg.trainer.disable_progress_bar)
    pipe.to(device)

    # set required_grad to False for all parameters
    # unet, vae, transformer (for sd3)
    pipe.vae.requires_grad_(False)
    if cfg.trainer.model in ["sd3", "dit", "flux", "flux_dev"]:
        pipe.transformer.requires_grad_(False)
    else:
        pipe.unet.requires_grad_(False)

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
        img_size=getattr(cfg.data, "img_size", None),
        with_synonyms=getattr(cfg.data, "with_synonyms", False),
    )

    logger.info(f"starting dataset filtering, current dataset size: {len(train_dataset)}")
    train_dataset, cfg = dataset_filter(train_dataset, cfg, logger)
    logger.info(f"filtered dataset, dataset size : {len(train_dataset)}")

    try:
        batch_size = cfg.data.batch_size
        logger.info(f"Batch size: {batch_size}")
    except Exception as e:
        logger.info(f"Error: {e}, setting batch size to 1")
        batch_size = 1
    dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)

    # save the original image
    path = os.path.join(project_folder, "images")
    train_path = os.path.join(path, "train", "initial_image") if cfg.logger.type != "wandb" else None
    val_path = os.path.join(path, "validation", "initial_image") if cfg.logger.type != "wandb" else None
    prompts = [validation_prompts, train_dataset[0]["prompt"]]
    all_imgs = []
    for path, prompt in zip([val_path, train_path], prompts):
        img = save_image_seed(pipe, prompt, cfg.trainer.num_intervention_steps, device, seed, save_dir=path)
        if img is not None:
            all_imgs += img
    if len(all_imgs) > 0:
        wandb.log({"initial image": [wandb.Image(i) for i in all_imgs]})

    # define loss for reconstruction
    if cfg.loss.reconstruct == 1:
        reconstruction_loss_func = torch.nn.L1Loss(reduction="mean")
    elif cfg.loss.reconstruct == 2:
        reconstruction_loss_func = torch.nn.MSELoss()
    else:
        raise ValueError(f"Reconstruction loss {cfg.loss.reconstruct} not supported")

    # make sure the mask precision is float32
    torch_dtype = torch.float32

    # initialize hooks
    hookers, hookers_tuple, lr_list = init_hooker(cfg, pipe, torch_dtype, project_folder)
    torch_dtype = get_precision(cfg.trainer.precision)

    # dummy generation to initialize the lambda
    logger.info(f"Initializing lambda to be {cfg.trainer.init_lambda}")
    g_cpu = torch.Generator(device.type).manual_seed(seed)
    _ = pipe(validation_prompts, generator=g_cpu, num_inference_steps=1)
    trainable_lambs = []
    for hooker in hookers:
        trainable_lambs += hooker.lambs

    # optimizer and scheduler
    params = []
    for hooker, lr in zip(hookers, lr_list):
        params.append({"params": hooker.lambs, "lr": lr})

    optimizer = torch.optim.AdamW(params)
    lr_scheduler = get_scheduler(
        cfg.lr_scheduler.type,
        optimizer=optimizer,
        num_warmup_steps=cfg.lr_scheduler.warmup_steps,
        num_training_steps=cfg.trainer.epochs * cfg.data.size,
        num_cycles=cfg.lr_scheduler.num_cycles,
        power=cfg.lr_scheduler.power,
    )

    # prepare with accelerator
    pipe, optimizer, lr_scheduler = accelerator.prepare(pipe, optimizer, lr_scheduler)
    logger.info("Start Training ...")

    # prepare logging metric placeholder list
    mean_loss_reconstruct, mean_intermediate_loss = 0, 0

    torch.cuda.empty_cache()
    optimizer.zero_grad()
    total_step = cfg.trainer.epochs * cfg.data.size // cfg.trainer.accumulate_grad_batches
    with tqdm.tqdm(total=total_step) as pbar:
        for i in range(cfg.trainer.epochs):
            for idx, data in enumerate(dataloader):
                # image_pt contains all latents, the denoise latent is the last one
                image_pt = data["image"]
                deconcept_image_pt = data["deconcept_image"]
                prompt = data["prompt"]
                value = data["value"].item()  # 1 for unlearn, 0 for the rest

                # use grad checkpointing to save memory
                g_cpu = torch.Generator(device.type).manual_seed(seed)
                (preparation_phase_output, intermediate_latents, timesteps, latents) = forward_checkpointing(
                    pipe,
                    prompt,
                    generator=g_cpu,
                    num_inference_steps=cfg.trainer.num_intervention_steps,
                    height=getattr(cfg.data, "img_size", None),
                    width=getattr(cfg.data, "img_size", None),
                )

                # backprop from loss to the last latents
                with torch.set_grad_enabled(True):
                    prompt_embeds = preparation_phase_output.prompt_embeds

                    image = pipe.inference_aft_denoising(latents, prompt_embeds, g_cpu, "latent", True, device)
                    # calculate loss
                    loss, loss_reconstruct = pruning_loss(
                        reconstruction_loss_func,
                        image_pt[:, -1, ...],  # last denoise latent
                        image,
                        cfg,
                    )

                    if value:  # w unlearn concept
                        # calculate the grad w.r.t. lambda in intermediate range (not z_0)
                        intermediate_loss = reconstruction_loss_func(
                            deconcept_image_pt[:, -1, ...], image[0]  # last denoise latent
                        )
                        mean_intermediate_loss += intermediate_loss.item() / cfg.trainer.accumulate_grad_batches
                        loss = intermediate_loss

                    (loss / cfg.trainer.accumulate_grad_batches).backward()
                    grad = latents.grad.detach()

                # backprop from the last latents to the first latents
                timesteps = preparation_phase_output.timesteps

                # update intermediate
                for timesteps_idx, t in enumerate(reversed(timesteps)):
                    current_latents = intermediate_latents[-(timesteps_idx + 1)].detach()
                    current_latents.requires_grad = True
                    timesteps_idx = len(timesteps) - timesteps_idx - 1
                    with torch.set_grad_enabled(True):
                        preparation_phase_output.latents = current_latents
                        # denoised latents t-1
                        latents = pipe.inference_denoising_step(
                            timesteps_idx,
                            t,
                            preparation_phase_output,
                            step_index=timesteps_idx,
                        )
                        # calculate grad w.r.t. lambda
                        lamb_grads = torch.autograd.grad(
                            latents,
                            trainable_lambs,
                            grad_outputs=grad,
                            retain_graph=True,
                        )

                        for lamb, lamb_grad in zip(trainable_lambs, lamb_grads):
                            if lamb.grad is None:
                                lamb.grad = lamb_grad
                            else:
                                lamb.grad += lamb_grad

                        grad = torch.autograd.grad(latents, current_latents, grad_outputs=grad)

                if (idx * batch_size) % cfg.trainer.accumulate_grad_batches == 0:
                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad()
                    # reset mean loss items
                else:
                    mean_loss_reconstruct += loss_reconstruct.item() / cfg.trainer.accumulate_grad_batches

                # START LOGGING
                if (idx * batch_size) % cfg.logger.plot_interval == 0:
                    loss_reconstruct = mean_loss_reconstruct
                    intermediate_loss = mean_intermediate_loss
                    # convert loss to float
                    loss = loss.to(torch.float).item()
                    if cfg.logger.type == "wandb":
                        wandb.log(
                            {
                                "loss_reconstruct": loss_reconstruct,
                                "intermediate_loss": intermediate_loss,
                                "loss": loss_reconstruct + intermediate_loss,
                                "lr": lr_scheduler.get_last_lr()[0],
                                "vram": torch.cuda.max_memory_allocated(device) / 1024**3,
                            },
                            commit=False,
                        )
                    # log images
                    path = os.path.join(project_folder, "images")
                    val_path = os.path.join(path, "validation", f"epoch_{i}_step_{idx}")
                    train_path = os.path.join(path, "train", f"epoch_{i}_step_{idx}")
                    img_continues_mask, img_discrete_mask = [], []
                    for path, prompts in zip([val_path, train_path], [validation_prompts, train_dataset[0]["prompt"]]):
                        for current_fn, current_hookers, current_path, all_imgs in zip(
                            [save_image_seed, save_image_binarize_seed],
                            [None, hookers],
                            ["continues mask", "discrete mask"],
                            [img_continues_mask, img_discrete_mask],
                        ):
                            if cfg.logger.type != "wandb":
                                actual_path = os.path.join(path, current_path)
                            else:
                                actual_path = None
                            imgs = current_fn(
                                pipe,
                                prompts,
                                cfg.trainer.num_intervention_steps,
                                device,
                                seed,
                                save_dir=actual_path,
                                hookers=current_hookers,
                            )
                            if imgs is not None:
                                all_imgs += imgs
                        torch.cuda.empty_cache()
                    for name, imgs in zip(["continues mask", "discrete mask"], [img_continues_mask, img_discrete_mask]):
                        if len(imgs) > 0:
                            wandb.log({f"image with {name}": [wandb.Image(i) for i in imgs]})

                    # log module sparsity
                    for hooker, name in hookers_tuple:
                        max_lamb = max([lamb.max().item() for lamb in hooker.lambs])
                        min_lamb = min([lamb.min().item() for lamb in hooker.lambs])
                        mean_lamb = sum([lamb.mean().item() for lamb in hooker.lambs]) / len(hooker.lambs)
                        logger.info(f"{name} max lambda: {max_lamb}, min lambda: {min_lamb}, mean lambda: {mean_lamb}")
                    # TODO: masking threshold should is hardcoded now
                    masking_threshold = 0
                    for hooker, name in hookers_tuple:
                        remain_head, total_head, sparsity = calculate_mask_sparsity(hooker, masking_threshold)
                        logger.info(
                            f"{name} sparsity for threshold {masking_threshold}: "
                            f"{remain_head}/{total_head}, {sparsity:.2%} \n"
                        )
                    logger.info(
                        f"loss_reconstruct: {loss_reconstruct}, "
                        + f"total_loss: {loss_reconstruct + intermediate_loss}, intermediate_loss: {intermediate_loss}"
                    )

                    # TODO should merge all checkpoint into one, or save it as dict
                    for hooker, name in hookers_tuple:
                        hooker.save(os.path.join("lambda", f"epoch_{i}_step_{idx}_{name}.pt"))
                    logger.info(f"epoch: {i}, step: {idx}: saving lambda")

                # reset loss
                if (idx * batch_size) % cfg.trainer.accumulate_grad_batches == 0:
                    mean_loss_reconstruct, mean_intermediate_loss = 0, 0
                    pbar.update()
            logger.info(f"epoch {i+1}/{cfg.trainer.epochs}")

        # save final image
        path = os.path.join(project_folder, "images")
        train_path = os.path.join(path, "train", "final_image") if cfg.logger.type != "wandb" else None
        val_path = os.path.join(path, "validation", "final_image") if cfg.logger.type != "wandb" else None
        prompts = [validation_prompts, train_dataset[0]["prompt"]]
        all_imgs = []
        for path, prompt in zip([val_path, train_path], prompts):
            img = save_image_binarize_seed(
                pipe,
                prompt,
                cfg.trainer.num_intervention_steps,
                device,
                seed,
                save_dir=path,
                hookers=hookers,
            )
            if img is not None:
                all_imgs += img
        if len(all_imgs) > 0:
            wandb.log({"final image": [wandb.Image(i) for i in all_imgs]})
        # END LOGGING
    logger.info(f"Training finished with cfg:{args.cfg}")


def get_parser():
    parser = argparse.ArgumentParser("Script to run unlearn concepts")
    parser.add_argument("--cfg", type=str, default="configs/flux.yaml", help="Config file to load all parameters")
    parser.add_argument("--notes", type=str, default="initial_run", help="Notes for wandb")
    parser.add_argument("--islaunch", action="store_true", help="Launch accelerator")
    parser.add_argument("--task", "-t", type=str, default="general", help="Task to perform")
    parser.add_argument("--load_lambda", "-l", action="store_true", help="Load lambda from checkpoint")
    parser.add_argument(
        "--cfg-options",
        "-o",
        nargs="+",
        action=mmengine.DictAction,
        help="Override the config entries with format xxx=yyy or xxx.zzz.qqq=yyy .",
    )
    parser.add_argument("--concept", "-c", type=str, default=None, help="Concept to evaluate for ip evaluate")
    return parser


def main():
    parser = get_parser()
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
