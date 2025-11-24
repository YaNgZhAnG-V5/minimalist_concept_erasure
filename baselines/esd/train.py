# Note: this script requires 2 80GB GPUs to run
# needs to load two models on two GPUs
import logging
import os

import torch
import yaml
from alive_progress import alive_it
from diffusers import FluxPipeline
from torch.utils.data import DataLoader, Dataset
from utils import prepare_noise_addition

import argparse


class NoisyLatentsDataset(Dataset):
    def __init__(self, file_path, with_noise=False):
        self.file_path = file_path
        self.with_noise = with_noise
        self.prompts = [f for f in os.listdir(file_path)]
        self.files = []
        for prompt in self.prompts:
            self.files.extend(
                [
                    os.path.join(file_path, prompt, f)
                    for f in os.listdir(os.path.join(file_path, prompt))
                    if f.endswith(".pt")
                ]
            )
        print(f"Loaded {len(self.files)} files as dataset")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        file_name = self.files[index]
        prompt = os.path.basename(os.path.dirname(file_name))
        time_step = int(os.path.basename(file_name).split("_")[-1].split(".")[0])
        data = torch.load(file_name)
        if self.with_noise:
            noise = torch.load(file_name.replace("latent", "noise"))
            return {"prompt": prompt, "time_step": time_step, "latents": data, "noise": noise}
        else:
            return {"prompt": prompt, "time_step": time_step, "latents": data}


def args_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--device1", "-d1", type=int, default=0)
    parser.add_argument("--device2", "-d2", type=int, default=3)
    parser.add_argument("--train-method", "-t", type=str, default="selfattn_one")
    parser.add_argument("--dataset-dir", "-d", type=str, default="data/nude")
    parser.add_argument("--num-inference-steps", "-n", type=int, default=4)
    parser.add_argument("--epochs", "-e", type=int, default=10)
    parser.add_argument("--validation-interval", "-v", type=int, default=50)
    parser.add_argument("--beta", "-b", type=float, default=1)
    parser.add_argument("--output-dir", "-o", type=str, default="esd_epoch_10_beta_1")
    parser.add_argument("--prompts-file", "-p", type=str, default="prompts.yaml")
    parser.add_argument("--concept", "-c", type=str, default="nudity")
    return parser.parse_args()


def main():
    args = args_parser()
    debug = args.debug
    device = f"cuda:{args.device1}"
    device2 = f"cuda:{args.device2}"
    train_method = args.train_method
    num_inference_steps = args.num_inference_steps

    # load prompts
    with open(args.prompts_file, "r") as f:
        all_prompts = yaml.load(f, Loader=yaml.FullLoader)
    assert args.concept in all_prompts.keys(), f"Concept {args.concept} not found in prompts file"
    prompts = all_prompts[args.concept]
    validation_prompt = prompts["validation_prompt"]
    beta = args.beta
    os.makedirs(args.output_dir, exist_ok=True)

    # Create a logger
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG if debug else logging.INFO)
    fh = logging.FileHandler(os.path.join(args.output_dir, "training.log"))
    fh.setLevel(logging.DEBUG if debug else logging.INFO)
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG if debug else logging.INFO)
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    ch.setFormatter(formatter)
    fh.setFormatter(formatter)
    logger.addHandler(ch)
    logger.addHandler(fh)
    logger.info("Logger initialized")

    # load noisy latent from dataset
    dataset = NoisyLatentsDataset(args.dataset_dir, with_noise=True)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

    # load model
    pipe = FluxPipeline.from_pretrained("black-forest-labs/FLUX.1-schnell", torch_dtype=torch.bfloat16)
    pipe.to(device)
    target_pipe = FluxPipeline.from_pretrained("black-forest-labs/FLUX.1-schnell", torch_dtype=torch.bfloat16)
    target_pipe.set_progress_bar_config(disable=True)
    target_pipe.to(device2)
    target_pipe.transformer.train()

    # get trainable parameters
    parameters = []
    for name, param in target_pipe.transformer.named_parameters():
        # train all layers except attns and time_embed layers
        if train_method == "noattn":
            if "attn" in name or "time_text_embed" in name:
                param.requires_grad = False
            else:
                logger.info(name)
                parameters.append(param)
        # train only self attention layers in transformer_blocks
        if train_method == "selfattn_one":
            if "transformer_blocks" in name and "attn" in name and "single_transformer_blocks" not in name:
                logger.info(name)
                parameters.append(param)
            else:
                param.requires_grad = False
        # train only self attention layers in single_transformer_blocks
        if train_method == "selfattn_two":
            if "single_transformer_blocks" in name and "attn" in name:
                logger.info(name)
                parameters.append(param)
            else:
                param.requires_grad = False
        # train only text attention layers
        if train_method == "textattn":
            if "add_k_proj" in name or "add_q_proj" in name or "add_v_proj" in name:
                logger.info(name)
                parameters.append(param)
            else:
                param.requires_grad = False
        # train all layers
        if train_method == "full":
            logger.info(name)
            parameters.append(param)
        # train all layers except time embed layers
        if train_method == "notime":
            if not ("time_text_embed" in name):
                logger.info(name)
                parameters.append(param)
            else:
                param.requires_grad = False

    # get optimizer
    optimizer = torch.optim.AdamW(parameters, lr=1e-5, betas=(0.9, 0.99), weight_decay=1e-04, eps=1e-08)
    criteria = torch.nn.MSELoss()

    # training loop
    loss_history = []
    for epoch_id in range(args.epochs):
        epoch_loss = []
        logger.info(f"### Epoch {epoch_id} ###")
        for sample_idx, sample in alive_it(enumerate(dataloader), total=len(dataloader)):
            prompt = sample["prompt"][0]
            time_step = sample["time_step"][0]
            latents = sample["latents"][0].to(device)
            noise = sample["noise"][0]

            # get anchor prompt
            for item in prompts["train_prompts"]:
                if item["prompt"] == prompt:
                    anchor_prompt = item["anchor_prompt"]
                    if args.debug:
                        print(f"Anchor prompt: {anchor_prompt}")
                        print(f"Concept prompt: {prompt}")
                    break

            # perform one step noise prediction using the anchor prompt
            with torch.no_grad():

                # encode the null prompt
                emb_0, pooled_emb_0, text_ids_0 = pipe.encode_prompt(
                    prompt="",
                    prompt_2="",
                    device=pipe.device,
                    num_images_per_prompt=1,
                    max_sequence_length=256,
                )
                # encode the concept prompt
                emb_p, pooled_emb_p, text_ids_p = pipe.encode_prompt(
                    prompt=[prompt],
                    prompt_2=[prompt],
                    device=pipe.device,
                    num_images_per_prompt=1,
                    max_sequence_length=256,
                )

                # encode the anchor prompt
                emb_n, pooled_emb_n, text_ids_n = pipe.encode_prompt(
                    prompt=[anchor_prompt],
                    prompt_2=[anchor_prompt],
                    device=pipe.device,
                    num_images_per_prompt=1,
                    max_sequence_length=256,
                )

                # Prepare timesteps for noise addition
                timesteps = prepare_noise_addition(pipe, num_inference_steps, device, latents)

                # prepare latent image ids
                height = 2 * (1024 // (pipe.vae_scale_factor * 2))
                width = 2 * (1024 // (pipe.vae_scale_factor * 2))
                latent_image_ids = pipe._prepare_latent_image_ids(1, height // 2, width // 2, device, latents.dtype)

                pipe.scheduler._begin_index, pipe.scheduler._step_index = 0, time_step.item()
                if debug:
                    logger.debug(f"Denoising for step {time_step} with anchor prompt")
                t = timesteps[time_step.item()]
                timestep = t.expand(latents.shape[0]).to(latents.dtype)
                e_p = pipe.transformer(
                    hidden_states=latents,
                    timestep=timestep / 1000,
                    guidance=None,
                    pooled_projections=pooled_emb_p,
                    encoder_hidden_states=emb_p,
                    txt_ids=text_ids_p,
                    img_ids=latent_image_ids,
                    # joint_attention_kwargs=pipe.joint_attention_kwargs,
                    return_dict=False,
                )[0]
                e_0 = pipe.transformer(
                    hidden_states=latents,
                    timestep=timestep / 1000,
                    guidance=None,
                    pooled_projections=pooled_emb_0,
                    encoder_hidden_states=emb_0,
                    txt_ids=text_ids_0,
                    img_ids=latent_image_ids,
                    # joint_attention_kwargs=pipe.joint_attention_kwargs,
                    return_dict=False,
                )[0]
                e_p.requires_grad = False
            # perform one step noise prediction using the concept prompt
            target_pipe.scheduler._begin_index, target_pipe.scheduler._step_index = 0, time_step.item()
            if debug:
                logger.debug(f"Denoising for step {time_step} with concept prompt")
            t = timesteps[time_step.item()]
            timestep = t.expand(latents.shape[0]).to(latents.dtype)
            e_n = target_pipe.transformer(
                hidden_states=latents.to(device2),
                timestep=timestep.to(device2) / 1000,
                guidance=None,
                pooled_projections=pooled_emb_n.to(device2),
                encoder_hidden_states=emb_n.to(device2),
                txt_ids=text_ids_n.to(device2),
                img_ids=latent_image_ids.to(device2),
                # joint_attention_kwargs=pipe.joint_attention_kwargs,
                return_dict=False,
            )[0]

            # compute loss
            loss = criteria(e_n, e_0.to(device2) - (beta * (e_p.to(device2) - e_0.to(device2))))
            logger.info(f"Loss: {loss.item()}")
            loss.backward()
            optimizer.step()
            epoch_loss.append(loss.item())
            loss_history.append(loss.item())
            if sample_idx % args.validation_interval == 0:
                # save images from trained model
                logger.info(f"Saving images from trained model for prompt: {validation_prompt}")
                target_pipe.transformer.eval()
                generator = torch.Generator(device=device2).manual_seed(0)
                with torch.no_grad():
                    images = target_pipe(
                        validation_prompt,
                        num_inference_steps=num_inference_steps,
                        guidance_scale=0.0,
                        generator=generator,
                    ).images[0]
                if not os.path.exists(os.path.join(args.output_dir, "validation_images")):
                    os.makedirs(os.path.join(args.output_dir, "validation_images"))
                images.save(f"{args.output_dir}/validation_images/epoch_{epoch_id}_sample_{sample_idx}.png")
                target_pipe.transformer.train()

        logger.info(f"Epoch loss: {sum(epoch_loss) / len(epoch_loss)}")

        # save trained model
        if not os.path.exists(args.output_dir):
            os.makedirs(args.output_dir)
        target_pipe.save_pretrained(f"{args.output_dir}/esd_{train_method}_{epoch_id}")


if __name__ == "__main__":
    main()
