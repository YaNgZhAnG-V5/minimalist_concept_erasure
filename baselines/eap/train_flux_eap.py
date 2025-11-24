"""
    @date:  2024.11.7  week45 星期四
    @func:  EAP(NeurIPS 2024)适配Flux
"""

import os

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
import os
import random

import matplotlib.pyplot as plt
import numpy as np
import torch
from diffusers import FluxPipeline
from diffusers.utils.torch_utils import randn_tensor
from gen_embedding import (
    create_prompt,
    learn_k_means_from_input_embedding,
    save_embedding_matrix,
    search_closest_tokens,
)
from PIL import Image
from torch.autograd import Variable
from torchvision import transforms
from tqdm import tqdm
from utils import flux_pack_latents, flux_unpack_latents, gumbel_softmax, save_to_dict

import argparse


def _prepare_latent_image_ids(batch_size, height, width, device, dtype):
    latent_image_ids = torch.zeros(height, width, 3)
    latent_image_ids[..., 1] = latent_image_ids[..., 1] + torch.arange(height)[:, None]
    latent_image_ids[..., 2] = latent_image_ids[..., 2] + torch.arange(width)[None, :]

    latent_image_id_height, latent_image_id_width, latent_image_id_channels = latent_image_ids.shape

    latent_image_ids = latent_image_ids.reshape(
        latent_image_id_height * latent_image_id_width, latent_image_id_channels
    )

    return latent_image_ids.to(device=device, dtype=dtype)


@torch.no_grad()
def latent_sample(
    flux_model,
    scheduler,
    batch_size,
    num_channels_latents,
    height,
    width,
    prompt_embeds,
    pooled_prompt_embeds,
    text_ids,
    guidance,
    timesteps,
):
    """
    Sample the model
    ESD quick_sample_till_t
    """

    height = int(height) // 8  # self.vae_scale_factor
    width = int(width) // 8  # self.vae_scale_factor
    shape = (batch_size, num_channels_latents, height, width)

    # (A) generate random tensor
    latents = randn_tensor(shape, generator=None, dtype=torch.bfloat16)
    latents = flux_pack_latents(latents, batch_size, num_channels_latents, height, width)
    latent_image_ids = _prepare_latent_image_ids(batch_size, height // 2, width // 2, flux_model.device, torch.bfloat16)

    # (B) generate prompt embed

    # (C) generate latents w.r.t text embedding
    scheduler.set_timesteps(timesteps, device=flux_model.device)
    timesteps = scheduler.timesteps

    latents = latents.to(flux_model.device).bfloat16()
    pooled_prompt_embeds = pooled_prompt_embeds.bfloat16()
    prompt_embeds = prompt_embeds.bfloat16()
    text_ids = text_ids.bfloat16()
    # Denoising loop
    for i, t in enumerate(timesteps):

        # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
        timestep = t.expand(latents.shape[0]).to(torch.bfloat16)

        # import pdb; pdb.set_trace()
        # self.transformer.config.guidance_embeds False => guidance = None
        noise_pred = flux_model(
            hidden_states=latents,
            timestep=timestep / 1000,
            guidance=None,
            pooled_projections=pooled_prompt_embeds,
            encoder_hidden_states=prompt_embeds,
            txt_ids=text_ids,
            img_ids=latent_image_ids,
            return_dict=False,
        )[0]

        # compute the previous noisy sample x_t -> x_t-1
        latents = scheduler.step(noise_pred, t, latents, return_dict=False)[0]

    return latents, latent_image_ids


def predict_noise(
    model,
    latent_code,
    prompt_embeds,
    text_ids,
    latent_image_ids,
    pooled_prompt_embeds,
    guidance,
    timesteps,
    CPU_only=False,
    dtype=torch.bfloat16,
):
    """
    ESD (apply_model)
    """

    if CPU_only:
        device = torch.device("cuda:0")
    else:
        device = torch.device("cuda:1")

    model = model.to(device).to(dtype)
    model_pred = model(
        hidden_states=latent_code.to(device).to(dtype),
        timestep=(timesteps / 1000).to(device).to(dtype),
        guidance=None,
        pooled_projections=pooled_prompt_embeds.to(device).to(dtype),
        encoder_hidden_states=prompt_embeds.to(device).to(dtype),
        txt_ids=text_ids.to(device).to(dtype),
        img_ids=latent_image_ids.to(device).to(dtype),
        return_dict=False,
    )[0]

    # print("20241108 predict noise e0 en ep", model_pred.device)

    model_pred = flux_unpack_latents(
        model_pred.to(dtype),
        height=512,
        width=512,
        vae_scale_factor=8,
    )

    return model_pred


def load_img(path, target_size=512):
    """Load an image, resize and output -1..1"""
    image = Image.open(path).convert("RGB")

    tform = transforms.Compose(
        [
            transforms.Resize(target_size),
            transforms.CenterCrop(target_size),
            transforms.ToTensor(),
        ]
    )
    image = tform(image)
    return 2.0 * image - 1.0


def moving_average(a, n=3):
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1 :] / n


def plot_loss(losses, path, word, n=100):
    v = moving_average(losses, n)
    plt.plot(v, label=f"{word}_loss")
    plt.legend(loc="upper left")
    plt.title("Average loss in trainings", fontsize=20)
    plt.xlabel("Data point", fontsize=16)
    plt.ylabel("Loss value", fontsize=16)
    plt.savefig(path)


# ESD Functions
def get_models():
    flux_orig = FluxPipeline.from_pretrained("black-forest-labs/FLUX.1-schnell", torch_dtype=torch.bfloat16)
    flux_model = FluxPipeline.from_pretrained("black-forest-labs/FLUX.1-schnell", torch_dtype=torch.bfloat16).to(
        "cuda:0"
    )
    model_orig = flux_orig.transformer.to("cuda:0")
    model = flux_model.transformer.to("cuda:1")

    del flux_orig
    return flux_model, model_orig, model


def train_eap(
    prompt,
    train_method,
    start_guidance,
    negative_guidance,
    iterations,
    lr,
    config_path,
    devices,
    seperator=None,
    image_size=512,
    ddim_steps=50,
    gumbel_k_closest=1000,
    gumbel_num_centers=100,
    gumbel_lr=1e-3,
    gumbel_temp=2,
    gumbel_hard=1,
):
    """
    Function to train diffusion models to erase concepts from model weights

    Parameters
    ----------
    prompt : str
        The concept to erase from diffusion model (Eg: "Van Gogh").
    train_method : str
        The parameters to train for erasure (ESD-x, ESD-u, full, selfattn).
    start_guidance : float
        Guidance to generate images for training.
    negative_guidance : float
        Guidance to erase the concepts from diffusion model.
    iterations : int
        Number of iterations to train.
    lr : float
        learning rate for fine tuning.
    config_path : str
        config path for compvis diffusion format.
    diffusers_config_path : str
        Config path for diffusers unet in json format.
    devices : str
        2 devices used to load the models (Eg: '0,1' will load in cuda:0 and cuda:1).
    seperator : str, optional
        If the prompt has commas can use this to seperate the prompt for
        individual simulataneous erasures. The default is None.
    image_size : int, optional
        Image size for generated images. The default is 512.
    ddim_steps : int, optional
        Number of diffusion time steps. The default is 50.

    Returns
    -------
    None

    """
    # PROMPT CLEANING
    word_print = prompt.replace(" ", "")

    if seperator is not None:
        words = prompt.split(seperator)
        erased_words = [word.strip() for word in words]
    else:
        erased_words = [prompt]
    print(erased_words)
    # MODEL TRAINING SETUP

    flux_model, model_orig, model = get_models()
    model = model.to("cuda:1")
    flux_model.vae.enable_slicing()
    flux_model.vae.enable_tiling()

    # TODO: choose parameters to train based on train_method
    parameters = []
    # aaa = [name for name, param in flux_model.transformer.named_parameters()]
    # with open("weights.txt", "a") as file:
    #     for item in aaa:
    #         file.writelines(item+"\n")
    # import pdb; pdb.set_trace()
    for name, param in model.named_parameters():
        # train all layers except attns and time_embed layers
        if train_method == "noattn":
            if "attn" in name or "time_text_embed" in name:
                pass
            else:
                print(name)
                parameters.append(param)
        # train only self attention layers in transformer_blocks
        if train_method == "selfattn_one":
            if "transformer_blocks" in name and "attn" in name and "single_transformer_blocks" not in name:
                print(name)
                parameters.append(param)
        # train only self attention layers in single_transformer_blocks
        if train_method == "selfattn_two":
            if "single_transformer_blocks" in name and "attn" in name:
                print(name)
                parameters.append(param)
        # train only text attention layers
        if train_method == "textattn":
            if "add_k_proj" in name or "add_q_proj" in name or "add_v_proj" in name:
                print(name)
                parameters.append(param)
        # train all layers
        if train_method == "full":
            print(name)
            parameters.append(param)
        # train all layers except time embed layers
        if train_method == "notime":
            if not ("time_text_embed" in name):
                print(name)
                parameters.append(param)
    # set model to train
    model.train()
    # create a lambda function for cleaner use of sampling code (only denoising till time step t)
    num_channels_latents = 16

    losses, losses_onehot = [], []
    print("Learning rate", lr)
    opt = torch.optim.AdamW(parameters, lr=lr, betas=(0.9, 0.99), weight_decay=1e-04, eps=1e-08)
    criteria = torch.nn.MSELoss()
    history = []
    history_dict = {}

    name = (
        f"Flux-EAP-word_{word_print}-method_{train_method}-sg_{start_guidance}-ng_{negative_guidance}-iter_{iterations}"
    )

    # Adversarial Prompt: EAP NeurIPS 2024
    # (a) generate embedding matrix
    if not os.path.exists("models/embedding_matrix_dict_EN3K_schnell.pt"):
        save_embedding_matrix(flux_model, model_name="Flux-schnell", save_mode="dict", vocab="EN3K")

    if not os.path.exists("models/embedding_matrix_array_EN3K_schnell.pt"):
        save_embedding_matrix(flux_model, model_name="Flux-schnell", save_mode="array", vocab="EN3K")

    # (b) similarities between tokens
    tokens_embedding = []
    all_sim_dict = dict()
    for word in erased_words:
        top_k_tokens, sorted_sim_dict = search_closest_tokens(word, flux_model, k=gumbel_k_closest)
        tokens_embedding.extend(top_k_tokens)
        all_sim_dict[word] = {key: sorted_sim_dict[key] for key in top_k_tokens}

    # (c) perserved
    if gumbel_num_centers > 0:
        assert (
            gumbel_num_centers % len(erased_words) == 0
        ), "Number of centers should be divisible by number of erased words"
    preserved_dict = dict()

    # (d) k means
    for word in erased_words:
        temp = learn_k_means_from_input_embedding(sim_dict=all_sim_dict[word], num_centers=gumbel_num_centers)
        preserved_dict[word] = temp

    history_dict = save_to_dict(preserved_dict, "preserved_set_0", history_dict)

    # (e) create a matrix of embeddings for the preserved set
    print("Creating preserved matrix")
    one_hot_dict = dict()
    preserved_matrix_dict = dict()
    for erase_word in erased_words:
        preserved_set = preserved_dict[erase_word]
        pbar = tqdm(preserved_set)
        for i, word in enumerate(pbar):
            if i == 0:
                preserved_matrix = create_prompt(word)
            else:
                preserved_matrix = torch.cat((preserved_matrix, create_prompt(word)), dim=0)
            pbar.set_description("Index: {0}, Word: {1}, Dimesion: {2}".format(i, word, preserved_matrix.shape))
        # preserved_matrix = torch.cat([create_prompt(word) for word in preserved_set], dim=0) # [n, 77, 768]
        preserved_matrix = preserved_matrix.flatten(start_dim=1)  # [n, 77*768]
        one_hot = torch.zeros((1, preserved_matrix.shape[0]), device=devices[0], dtype=preserved_matrix.dtype)  # [1, n]
        one_hot = one_hot + 1 / preserved_matrix.shape[0]
        one_hot = Variable(one_hot, requires_grad=True)
        print(one_hot.shape, preserved_matrix.shape)
        # print(one_hot)
        one_hot_dict[erase_word] = one_hot
        preserved_matrix_dict[erase_word] = preserved_matrix

    print("one_hot_dict:", one_hot_dict)
    history_dict = save_to_dict(one_hot_dict, "one_hot_dict_0", history_dict)

    # optimizer for all one-hot vectors
    opt_one_hot = torch.optim.Adam([one_hot for one_hot in one_hot_dict.values()], lr=gumbel_lr)

    # TRAINING CODE
    pgd_num_steps = 2
    pbar = tqdm(range(iterations * pgd_num_steps))

    for i in pbar:
        word = random.sample(erased_words, 1)[0]

        # get text embeddings for unconditional and conditional prompts
        emb_0, pooled_emb_0, text_ids_0 = flux_model.encode_prompt(
            prompt="",
            prompt_2="",
            device=flux_model.device,
            num_images_per_prompt=1,
            max_sequence_length=256,
        )
        emb_n, pooled_emb_n, text_ids_n = flux_model.encode_prompt(
            prompt=[word],
            prompt_2=[word],
            device=flux_model.device,
            num_images_per_prompt=1,
            max_sequence_length=256,
        )

        emb_r = torch.reshape(
            torch.matmul(
                gumbel_softmax(one_hot_dict[word].bfloat16(), temperature=gumbel_temp, hard=gumbel_hard),
                preserved_matrix_dict[word],
            ).unsqueeze(0),
            (1, 256, 4096),
        ).to(flux_model.device)
        assert emb_r.shape == emb_n.shape

        opt.zero_grad()
        ddim_steps = 4
        #         t_enc = torch.randint(ddim_steps, (1,), device=devices[0])
        #         # time step from 1000 to 0 (0 being good)
        #         og_num = round((int(t_enc)/ddim_steps)*1000)
        #         og_num_lim = round((int(t_enc+1)/ddim_steps)*1000)

        #         # 2024.11.04 week45 t_enc_ddpm [1000. 750. 500. 250.]
        #         t_enc_ddpm = torch.randint(og_num, og_num_lim, (1,), device=devices[0])

        tmp_index = np.random.choice([0, 1, 2, 3])
        if tmp_index == 0:
            t_enc_ddpm = torch.tensor([1000.0], device=devices[0])
        elif tmp_index == 1:
            t_enc_ddpm = torch.tensor([750.0], device=devices[0])
        elif tmp_index == 2:
            t_enc_ddpm = torch.tensor([500.0], device=devices[0])
        elif tmp_index == 3:
            t_enc_ddpm = torch.tensor([250.0], device=devices[0])

        # start_code = torch.randn((1, 4, 64, 64)).to(devices[0])
        # print("time enc", t_enc)

        with torch.no_grad():
            # generate an image with the concept from ESD model
            z, latent_image_ids = latent_sample(
                model_orig,
                flux_model.scheduler,
                1,
                num_channels_latents,
                512,
                512,
                emb_n.to(devices[0]),
                pooled_emb_n.to(devices[0]),
                text_ids_n.to(devices[0]),
                start_guidance,
                timesteps=4,
            )

            z_r, latent_image_ids_r = latent_sample(
                model_orig,
                flux_model.scheduler,
                1,
                num_channels_latents,
                512,
                512,
                emb_r.to(devices[0]),
                pooled_emb_n.to(devices[0]),
                text_ids_n.to(devices[0]),
                start_guidance,
                timesteps=4,
            )

            # get conditional and unconditional scores from frozen model at time step t and image z
            e_0_org = predict_noise(
                model_orig,
                z,
                emb_0,
                text_ids_0,
                latent_image_ids,
                pooled_emb_0,
                guidance=None,
                timesteps=t_enc_ddpm.to(devices[0]),
                CPU_only=True,
            )
            e_n_org = predict_noise(
                model_orig,
                z,
                emb_n,
                text_ids_n,
                latent_image_ids_r,
                pooled_emb_n,
                guidance=None,
                timesteps=t_enc_ddpm.to(devices[0]),
                CPU_only=True,
            )
            e_r_org = predict_noise(
                model_orig,
                z_r,
                emb_r,
                text_ids_n,
                latent_image_ids_r,
                pooled_emb_n,
                guidance=None,
                timesteps=t_enc_ddpm.to(devices[0]),
                CPU_only=True,
            )

        e_0_org.requires_grad, e_n_org.requires_grad, e_r_org.requires_grad = False, False, False

        # get conditional score from model.
        e_n_wo_prompt = predict_noise(
            model,
            z.to("cuda:1"),
            emb_n.to("cuda:1"),
            text_ids_n.to("cuda:1"),
            latent_image_ids.to("cuda:1"),
            pooled_emb_n.to("cuda:1"),
            guidance=None,
            timesteps=t_enc_ddpm.to("cuda:1"),
            CPU_only=False,
        )
        # import pdb; pdb.set_trace()
        e_r_wo_prompt = predict_noise(
            model,
            z_r.to("cuda:1"),
            emb_r.to("cuda:1"),
            text_ids_n.to("cuda:1"),
            latent_image_ids_r.to("cuda:1"),
            pooled_emb_n.to("cuda:1"),
            guidance=None,
            timesteps=t_enc_ddpm.to("cuda:1"),
            CPU_only=False,
        )

        # using Flow matching inversion to project the x_t to x_0
        # delta_sigma = 0.001 * 250  # 如果是4步? x 250?
        # https://github.com/huggingface/diffusers/blob/v0.31.0/src/diffusers/schedulers/scheduling_flow_match_euler_discrete.py#L297
        # https://github.com/tuananhbui89/Erasing-Adversarial-Preservation/blob/main/train_adversarial_gumbel.py#L396

        delta_sigma = 1.0  # ?
        z_n_wo_prompt_pred, z_r_wo_prompt_pred = delta_sigma * e_n_wo_prompt, delta_sigma * e_r_wo_prompt
        z_n_org_pred, z_0_org_pred, z_r_org_pred = (
            (delta_sigma * e_n_org),
            (delta_sigma * e_0_org),
            (delta_sigma * e_r_org),
        )

        # pgd_num_steps = 2
        if i % pgd_num_steps == 0:
            model.zero_grad()
            # for erased concepts, output aligns with target concept with or without prompt
            loss = 0
            loss += criteria(
                z_n_wo_prompt_pred,
                z_0_org_pred.to(devices[1])
                - (negative_guidance * (z_n_org_pred.to(devices[1]) - z_0_org_pred.to(devices[1]))),
            )
            loss += criteria(
                z_r_wo_prompt_pred, z_r_org_pred.to(devices[1])
            )  # for preserved concepts, output are the same without prompt

            # import pdb; pdb.set_trace()
            loss = loss.float()
            # update weights to erase the concept
            loss.backward()
            losses.append(loss.item())
            pbar.set_postfix({"ESD Loss": loss.item()})
            history_dict = save_to_dict(loss.item(), "loss", history_dict)
            opt.step()
            opt.zero_grad()
        else:
            # update the one_hot vector
            model.zero_grad()
            opt.zero_grad()
            # one_hot.grad = None
            loss = -criteria(z_r_wo_prompt_pred, z_r_org_pred.to(devices[1])).float()  # maximize the preserved loss
            losses_onehot.append(loss.item())
            pbar.set_postfix({"EAP Loss": loss.item()})
            loss.backward()
            preserved_set = preserved_dict[word]
            print(
                "index of one_hot before:",
                torch.argmax(one_hot_dict[word], dim=1),
                preserved_set[torch.argmax(one_hot_dict[word], dim=1)],
            )
            print("one_hot:", one_hot_dict[word])
            print("one_hot.grad:", one_hot_dict[word].grad)
            opt_one_hot.step()
            print(
                "index of one_hot after:",
                torch.argmax(one_hot_dict[word], dim=1),
                preserved_set[torch.argmax(one_hot_dict[word], dim=1)],
            )
            opt_one_hot.zero_grad()
            model.zero_grad()
            print("one_hot:", one_hot_dict[word])

        del loss
        torch.cuda.empty_cache()

        if i % 100 == 0:
            save_history(losses, name, word_print)
            save_history_onehot(losses_onehot, name, losses_onehot)
    model.eval()
    print("EAP Flux training finished!")
    save_model(model, name)
    save_history(losses, name, word_print)


def save_model(model, name):
    # SAVE MODEL

    folder_path = f"models/{name}"
    os.makedirs(folder_path, exist_ok=True)

    model_name = f"models/{name}_flux_transformer.pt"
    torch.save(model.state_dict(), model_name)


def save_history(losses, name, word_print):
    folder_path = f"models/{name}"
    os.makedirs(folder_path, exist_ok=True)
    with open(f"{folder_path}/loss.txt", "w") as f:
        f.writelines([str(i) for i in losses])
    plot_loss(losses, f"{folder_path}/loss.png", word_print, n=3)


def save_history_onehot(losses, name, word_print):
    folder_path = f"models/{name}"
    os.makedirs(folder_path, exist_ok=True)
    with open(f"{folder_path}/loss_onehot.txt", "w") as f:
        f.writelines([str(i) for i in losses])
    plot_loss(losses, f"{folder_path}/loss_onehot.png", word_print, n=3)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="TrainEAP Flux", description="Finetuning stable diffusion model to erase concepts using ESD method"
    )
    parser.add_argument("--prompt", help="prompt corresponding to concept to erase", type=str, required=True)
    parser.add_argument("--train_method", help="method of training", type=str, required=True)
    parser.add_argument(
        "--start_guidance", help="guidance of start image used to train", type=float, required=False, default=3
    )
    parser.add_argument(
        "--negative_guidance", help="guidance of negative training used to train", type=float, required=False, default=1
    )
    parser.add_argument("--iterations", help="iterations used to train", type=int, required=False, default=1000)
    parser.add_argument("--lr", help="learning rate used to train", type=float, required=False, default=1e-4)
    parser.add_argument(
        "--config_path",
        help="config path for stable diffusion v1-4 inference",
        type=str,
        required=False,
        default="configs/stable-diffusion/v1-inference.yaml",
    )
    parser.add_argument("--devices", help="cuda devices to train on", type=str, required=False, default="0,1")
    parser.add_argument(
        "--seperator",
        help="separator if you want to train bunch of words separately",
        type=str,
        required=False,
        default=None,
    )
    parser.add_argument("--image_size", help="image size used to train", type=int, required=False, default=512)

    parser.add_argument("--gumbel_lr", help="learning rate for prompt", type=float, required=False, default=1e-3)
    parser.add_argument("--gumbel_temp", help="temperature for gumbel softmax", type=float, required=False, default=2)
    parser.add_argument(
        "--gumbel_hard",
        help="hard for gumbel softmax, 0: soft, 1: hard",
        type=int,
        required=False,
        default=0,
        choices=[0, 1],
    )
    parser.add_argument(
        "--gumbel_k_closest", help="number of closest tokens to consider", type=int, required=False, default=10
    )
    parser.add_argument(
        "--gumbel_num_centers",
        help="number of centers for kmeans, if <= 0 then do not apply kmeans",
        type=int,
        required=False,
        default=100,
    )
    parser.add_argument(
        "--ddim_steps", help="ddim steps of inference used to train", type=int, required=False, default=50
    )
    args = parser.parse_args()

    prompt = args.prompt
    train_method = args.train_method
    start_guidance = args.start_guidance
    negative_guidance = args.negative_guidance
    iterations = args.iterations
    lr = args.lr
    config_path = args.config_path
    devices = [f"cuda:{int(d.strip())}" for d in args.devices.split(",")]
    seperator = args.seperator
    image_size = args.image_size
    ddim_steps = args.ddim_steps
    gumbel_k_closest = args.gumbel_k_closest
    gumbel_num_centers = args.gumbel_num_centers
    gumbel_lr = args.gumbel_lr
    gumbel_temp = args.gumbel_temp
    gumbel_hard = args.gumbel_hard

    train_eap(
        prompt=prompt,
        train_method=train_method,
        start_guidance=start_guidance,
        negative_guidance=negative_guidance,
        iterations=iterations,
        lr=lr,
        config_path=config_path,
        devices=devices,
        seperator=seperator,
        image_size=image_size,
        ddim_steps=ddim_steps,
        gumbel_k_closest=gumbel_k_closest,
        gumbel_num_centers=gumbel_num_centers,
        gumbel_lr=gumbel_lr,
        gumbel_temp=gumbel_temp,
        gumbel_hard=gumbel_hard,
    )
