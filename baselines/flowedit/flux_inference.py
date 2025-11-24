import os
from typing import Optional, Union

import numpy as np
import torch
from diffusers import FluxPipeline
from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion import retrieve_timesteps

import argparse


def scale_noise(
    scheduler,
    sample: torch.FloatTensor,
    timestep: Union[float, torch.FloatTensor],
    noise: Optional[torch.FloatTensor] = None,
) -> torch.FloatTensor:
    """
    Foward process in flow-matching

    Args:
        sample (`torch.FloatTensor`):
            The input sample.
        timestep (`int`, *optional*):
            The current timestep in the diffusion chain.

    Returns:
        `torch.FloatTensor`:
            A scaled input sample.
    """
    # if scheduler.step_index is None:
    scheduler._init_step_index(timestep)

    sigma = scheduler.sigmas[scheduler.step_index]
    sample = sigma * noise + (1.0 - sigma) * sample

    return sample


# for flux
def calculate_shift(
    image_seq_len,
    base_seq_len: int = 256,
    max_seq_len: int = 4096,
    base_shift: float = 0.5,
    max_shift: float = 1.16,
):
    m = (max_shift - base_shift) / (max_seq_len - base_seq_len)
    b = base_shift - m * base_seq_len
    mu = image_seq_len * m + b
    return mu


def calc_v_flux(pipe, latents, prompt_embeds, pooled_prompt_embeds, guidance, text_ids, latent_image_ids, t):
    timestep = t.expand(latents.shape[0])
    with torch.no_grad():
        noise_pred = pipe.transformer(
            hidden_states=latents,
            timestep=timestep / 1000,
            guidance=guidance,
            encoder_hidden_states=prompt_embeds,
            txt_ids=text_ids,
            img_ids=latent_image_ids,
            pooled_projections=pooled_prompt_embeds,
            joint_attention_kwargs=None,
            return_dict=False,
        )[0]

    return noise_pred


@torch.no_grad()
def FlowEditFLUX(
    pipe,
    scheduler,
    x_src,
    src_prompt,
    tar_prompt,
    negative_prompt="",
    T_steps: int = 28,
    n_avg: int = 1,
    src_guidance_scale: float = 1.5,
    tar_guidance_scale: float = 5.5,
    n_min: int = 0,
    n_max: int = 24,
):

    device = x_src.device
    orig_height, orig_width = x_src.shape[2] * pipe.vae_scale_factor // 2, x_src.shape[3] * pipe.vae_scale_factor // 2
    num_channels_latents = pipe.transformer.config.in_channels // 4

    pipe.check_inputs(
        prompt=src_prompt,
        prompt_2=None,
        height=orig_height,
        width=orig_width,
        callback_on_step_end_tensor_inputs=None,
        max_sequence_length=512,
    )

    x_src, latent_src_image_ids = pipe.prepare_latents(
        batch_size=x_src.shape[0],
        num_channels_latents=num_channels_latents,
        height=orig_height,
        width=orig_width,
        dtype=x_src.dtype,
        device=x_src.device,
        generator=None,
        latents=x_src,
    )
    x_src_packed = pipe._pack_latents(x_src, x_src.shape[0], num_channels_latents, x_src.shape[2], x_src.shape[3])
    latent_tar_image_ids = latent_src_image_ids

    # 5. Prepare timesteps
    sigmas = np.linspace(1.0, 1 / T_steps, T_steps)
    image_seq_len = x_src_packed.shape[1]
    mu = calculate_shift(
        image_seq_len,
        scheduler.config.base_image_seq_len,
        scheduler.config.max_image_seq_len,
        scheduler.config.base_shift,
        scheduler.config.max_shift,
    )
    timesteps, T_steps = retrieve_timesteps(
        scheduler,
        T_steps,
        device,
        timesteps=None,
        sigmas=sigmas,
        mu=mu,
    )

    num_warmup_steps = max(len(timesteps) - T_steps * pipe.scheduler.order, 0)
    pipe._num_timesteps = len(timesteps)

    # src prompts
    (
        src_prompt_embeds,
        src_pooled_prompt_embeds,
        src_text_ids,
    ) = pipe.encode_prompt(
        prompt=src_prompt,
        prompt_2=None,
        device=device,
    )

    # tar prompts
    pipe._guidance_scale = tar_guidance_scale
    (
        tar_prompt_embeds,
        tar_pooled_prompt_embeds,
        tar_text_ids,
    ) = pipe.encode_prompt(
        prompt=tar_prompt,
        prompt_2=None,
        device=device,
    )

    # handle guidance
    if pipe.transformer.config.guidance_embeds:
        src_guidance = torch.tensor([src_guidance_scale], device=device)
        src_guidance = src_guidance.expand(x_src_packed.shape[0])
        tar_guidance = torch.tensor([tar_guidance_scale], device=device)
        tar_guidance = tar_guidance.expand(x_src_packed.shape[0])
    else:
        src_guidance = None
        tar_guidance = None

    # initialize our ODE Zt_edit_1=x_src
    zt_edit = x_src_packed.clone()

    for i, t in enumerate(timesteps):
        import pdb

        pdb.set_trace()

        if T_steps - i > n_max:
            continue

        scheduler._init_step_index(t)
        t_i = scheduler.sigmas[scheduler.step_index]
        if i < len(timesteps):
            t_im1 = scheduler.sigmas[scheduler.step_index + 1]
        else:
            t_im1 = t_i

        if T_steps - i > n_min:

            # Calculate the average of the V predictions
            V_delta_avg = torch.zeros_like(x_src_packed)

            for k in range(n_avg):

                fwd_noise = torch.randn_like(x_src_packed).to(x_src_packed.device)

                zt_src = (1 - t_i) * x_src_packed + (t_i) * fwd_noise

                zt_tar = zt_edit + zt_src - x_src_packed

                # Merge in the future to avoid double computation
                Vt_src = calc_v_flux(
                    pipe,
                    latents=zt_src,
                    prompt_embeds=src_prompt_embeds,
                    pooled_prompt_embeds=src_pooled_prompt_embeds,
                    guidance=src_guidance,
                    text_ids=src_text_ids,
                    latent_image_ids=latent_src_image_ids,
                    t=t,
                )

                Vt_tar = calc_v_flux(
                    pipe,
                    latents=zt_tar,
                    prompt_embeds=tar_prompt_embeds,
                    pooled_prompt_embeds=tar_pooled_prompt_embeds,
                    guidance=tar_guidance,
                    text_ids=tar_text_ids,
                    latent_image_ids=latent_tar_image_ids,
                    t=t,
                )

                V_delta_avg += (1 / n_avg) * (Vt_tar - Vt_src)  # - (hfg-1)*( x_src))

            # propagate direct ODE
            zt_edit = zt_edit.to(torch.float32)

            zt_edit = zt_edit + (t_im1 - t_i) * V_delta_avg

            zt_edit = zt_edit.to(V_delta_avg.dtype)

        else:  # i >= T_steps-n_min # regular sampling last n_min steps

            if i == T_steps - n_min:
                # initialize SDEDIT-style generation phase
                fwd_noise = torch.randn_like(x_src_packed).to(x_src_packed.device)
                xt_src = scale_noise(scheduler, x_src_packed, t, noise=fwd_noise)
                xt_tar = zt_edit + xt_src - x_src_packed

            Vt_tar = calc_v_flux(
                pipe,
                latents=xt_tar,
                prompt_embeds=tar_prompt_embeds,
                pooled_prompt_embeds=tar_pooled_prompt_embeds,
                guidance=tar_guidance,
                text_ids=tar_text_ids,
                latent_image_ids=latent_tar_image_ids,
                t=t,
            )

            xt_tar = xt_tar.to(torch.float32)

            prev_sample = xt_tar + (t_im1 - t_i) * (Vt_tar)

            prev_sample = prev_sample.to(Vt_tar.dtype)
            xt_tar = prev_sample
    out = zt_edit if n_min == 0 else xt_tar
    unpacked_out = pipe._unpack_latents(out, orig_height, orig_width, pipe.vae_scale_factor)
    return unpacked_out


neutral_concepts = {
    "nude": [
        "nake" "topless",
        "shirtless",
        "bare",
        "unclothed",
        "exposed",
        "undressed",
        "disrobed",
        "uncovered",
        "stripped",
        "half-naked",
        "bare-chested",
        "skin-baring",
        "Nake" "Topless",
        "Shirtless",
        "Bare",
        "Unclothed",
        "Exposed",
        "Undressed",
        "Disrobed",
        "Uncovered",
        "stripped",
        "half-naked",
        "bare-chested",
        "skin-baring",
    ]
}


@torch.no_grad()
def flowedit_inference(
    pipe,
    p,
    concept,
    seed,
    device,
    num_inference_steps,
    n_avg=1,
    src_guidance_scale=1.5,
    tar_guidance_scale=5.5,
    n_min=0,
    n_max=4,
):
    # get the neural prompt
    synomyns_list = neutral_concepts.get(concept, [])
    if synomyns_list:
        for syn in synomyns_list:
            p = p.replace(syn, "")
        prompt_wo_concept = p
    else:
        prompt_wo_concept = p.replace(concept, "")
    g_cpu = torch.Generator(device).manual_seed(seed)
    img = pipe(prompt=p, generator=g_cpu, num_inference_steps=num_inference_steps)[0][0]
    img_src = pipe.image_processor.preprocess(img)
    img_src = img_src.to(device).half()
    with torch.autocast("cuda"), torch.inference_mode():
        x0_src_denorm = pipe.vae.encode(img_src).latent_dist.mode()
    x0_src = (x0_src_denorm - pipe.vae.config.shift_factor) * pipe.vae.config.scaling_factor
    x0_src = x0_src.to(device)
    scheduler = pipe.scheduler
    x0_tar = FlowEditFLUX(
        pipe,
        scheduler,
        x0_src,
        p,
        tar_prompt=prompt_wo_concept,
        T_steps=num_inference_steps,
        n_avg=n_avg,
        src_guidance_scale=src_guidance_scale,
        tar_guidance_scale=tar_guidance_scale,
        n_min=n_min,
        n_max=n_max,
    )

    x0_tar_denorm = (x0_tar / pipe.vae.config.scaling_factor) + pipe.vae.config.shift_factor
    img_tar = pipe.vae.decode(x0_tar_denorm, return_dict=False)[0]
    img_tar = pipe.image_processor.postprocess(img_tar)[0]
    return img_tar


def main(args):
    device = torch.device(args.device)
    # pipe = FluxPipeline.from_pretrained("black-forest-labs/FLUX.1-schnell", torch_dtype=torch.float16)
    pipe = FluxPipeline.from_pretrained(f"black-forest-labs/FLUX.1-{args.version}", torch_dtype=torch.float16)
    pipe = pipe.to(device)

    # create to save directory
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    # first generate the source image
    with torch.no_grad():
        for p in args.prompt:
            img_tar = flowedit_inference(
                pipe,
                p,
                args.concept,
                args.seed,
                device,
                args.num_inference_steps,
                args.n_avg,
                args.src_guidance_scale,
                args.tar_guidance_scale,
                args.n_min,
                args.n_max,
            )
            img_tar.save(f"{args.save_dir}/{p}.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=int, default=0, help="device number to use")
    parser.add_argument("--n_min", type=int, default=0, help="minimum number of steps")
    parser.add_argument("--n_max", type=int, default=4, help="minimum number of steps")
    parser.add_argument("--num_inference_steps", "-ni", type=int, default=5, help="number of inference steps")
    parser.add_argument("--n_avg", type=int, default=3, help="number of averaging steps")
    parser.add_argument("--src_guidance_scale", type=float, default=1.5, help="source guidance scale")
    parser.add_argument("--tar_guidance_scale", type=float, default=5.5, help="target guidance scale")
    parser.add_argument("--seed", type=int, default=42, help="random seed")
    parser.add_argument(
        "--prompt",
        type=str,
        nargs="+",
        default=["a front facing completely nude woman walking in the forest"],
    )
    parser.add_argument("--concept", type=str, default="nude")
    parser.add_argument("--save_dir", type=str, default="./demo", help="directory to save outputs")
    parser.add_argument("--version", type=str, default="schnell", help="version of the model, schnell and dev")
    args = parser.parse_args()
    args.device = f"cuda:{args.device}"
    main(args)
