from typing import List, Optional, Union

import numpy as np
import torch
from diffusers import FluxPipeline

import inspect


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


# Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.retrieve_timesteps
def retrieve_timesteps(
    scheduler,
    num_inference_steps: Optional[int] = None,
    device: Optional[Union[str, torch.device]] = None,
    timesteps: Optional[List[int]] = None,
    sigmas: Optional[List[float]] = None,
    **kwargs,
):
    r"""
    Calls the scheduler's `set_timesteps` method and retrieves timesteps from the scheduler after the call. Handles
    custom timesteps. Any kwargs will be supplied to `scheduler.set_timesteps`.

    Args:
        scheduler (`SchedulerMixin`):
            The scheduler to get timesteps from.
        num_inference_steps (`int`):
            The number of diffusion steps used when generating samples with a pre-trained model. If used, `timesteps`
            must be `None`.
        device (`str` or `torch.device`, *optional*):
            The device to which the timesteps should be moved to. If `None`, the timesteps are not moved.
        timesteps (`List[int]`, *optional*):
            Custom timesteps used to override the timestep spacing strategy of the scheduler. If `timesteps` is passed,
            `num_inference_steps` and `sigmas` must be `None`.
        sigmas (`List[float]`, *optional*):
            Custom sigmas used to override the timestep spacing strategy of the scheduler. If `sigmas` is passed,
            `num_inference_steps` and `timesteps` must be `None`.

    Returns:
        `Tuple[torch.Tensor, int]`: A tuple where the first element is the timestep schedule from the scheduler and the
        second element is the number of inference steps.
    """
    if timesteps is not None and sigmas is not None:
        raise ValueError("Only one of `timesteps` or `sigmas` can be passed. Please choose one to set custom values")
    if timesteps is not None:
        accepts_timesteps = "timesteps" in set(inspect.signature(scheduler.set_timesteps).parameters.keys())
        if not accepts_timesteps:
            raise ValueError(
                f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                f" timestep schedules. Please check whether you are using the correct scheduler."
            )
        scheduler.set_timesteps(timesteps=timesteps, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    elif sigmas is not None:
        accept_sigmas = "sigmas" in set(inspect.signature(scheduler.set_timesteps).parameters.keys())
        if not accept_sigmas:
            raise ValueError(
                f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                f" sigmas schedules. Please check whether you are using the correct scheduler."
            )
        scheduler.set_timesteps(sigmas=sigmas, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    else:
        scheduler.set_timesteps(num_inference_steps, device=device, **kwargs)
        timesteps = scheduler.timesteps
    return timesteps, num_inference_steps


def save_latent_to_image(latents, pipe, filename):
    latents = pipe._unpack_latents(latents, 1024, 1024, pipe.vae_scale_factor)
    latents = (latents / pipe.vae.config.scaling_factor) + pipe.vae.config.shift_factor
    image = pipe.vae.decode(latents, return_dict=False)[0]
    image = image.detach()
    image = pipe.image_processor.postprocess(image, output_type="pil")[0]
    image.save(f"{filename}.png")


def prepare_noise_addition(pipe, num_inference_steps, device, latents):
    sigmas = np.linspace(1.0, 1 / num_inference_steps, num_inference_steps)
    image_seq_len = latents.shape[1]
    mu = calculate_shift(
        image_seq_len,
        pipe.scheduler.config.get("base_image_seq_len", 256),
        pipe.scheduler.config.get("max_image_seq_len", 4096),
        pipe.scheduler.config.get("base_shift", 0.5),
        pipe.scheduler.config.get("max_shift", 1.16),
    )
    timesteps, num_inference_steps = retrieve_timesteps(
        pipe.scheduler,
        num_inference_steps,
        device,
        sigmas=sigmas,
        mu=mu,
    )
    return timesteps


@torch.no_grad()
def main():
    pipe = FluxPipeline.from_pretrained("black-forest-labs/FLUX.1-schnell", torch_dtype=torch.bfloat16)
    device = "cuda:2"
    pipe.to(device)
    num_inference_steps = 4

    prompt = "a beautiful woman with a gun"
    anchor_prompt = "a beautiful woman"

    emb_p, pooled_emb_p, text_ids_p = pipe.encode_prompt(
        prompt=[prompt],
        prompt_2=[prompt],
        device=pipe.device,
        num_images_per_prompt=1,
        max_sequence_length=256,
    )

    emb_n, pooled_emb_n, text_ids_n = pipe.encode_prompt(
        prompt=[anchor_prompt],
        prompt_2=[anchor_prompt],
        device=pipe.device,
        num_images_per_prompt=1,
        max_sequence_length=256,
    )

    # decode the latent vector to an image
    generator = torch.Generator(device="cuda:0").manual_seed(42)
    latents = pipe(prompt, generator=generator, num_inference_steps=num_inference_steps, output_type="latent").images
    save_latent_to_image(latents, pipe, f"image_{num_inference_steps}")

    # Prepare timesteps for noise addition
    timesteps = prepare_noise_addition(pipe, num_inference_steps, device, latents)

    # prepare latent image ids
    height = 2 * (1024 // (pipe.vae_scale_factor * 2))
    width = 2 * (1024 // (pipe.vae_scale_factor * 2))
    latent_image_ids = pipe._prepare_latent_image_ids(1, height // 2, width // 2, device, latents.dtype)

    # randomly add noise to the latent vector
    # add_noise_steps = random.randint(0, num_inference_steps)
    add_noise_steps = 4
    for i in reversed(range(add_noise_steps)):
        print(f"Adding noise for step {i}")
        noise = torch.randn_like(latents)
        pipe.scheduler._begin_index = 0
        pipe.scheduler._step_index = i
        latents = pipe.scheduler.scale_noise(latents, timesteps[i][None], noise)
        pipe.scheduler._begin_index, pipe.scheduler._step_index = None, None
        save_latent_to_image(latents, pipe, f"image_{i}")

    # perform noise prediction
    pipe.scheduler._begin_index, pipe.scheduler._step_index = 0, 0
    for i in range(add_noise_steps):
        print(f"Denoising for step {i}")
        t = timesteps[i]
        timestep = t.expand(latents.shape[0]).to(latents.dtype)
        noise_pred = pipe.transformer(
            hidden_states=latents,
            timestep=timestep / 1000,
            guidance=None,
            pooled_projections=pooled_emb_p,
            encoder_hidden_states=emb_p,
            txt_ids=text_ids_p,
            img_ids=latent_image_ids,
            joint_attention_kwargs=pipe.joint_attention_kwargs,
            return_dict=False,
        )[0]
        latents = pipe.scheduler.step(noise_pred, t, latents, return_dict=False)[0]
        save_latent_to_image(latents, pipe, f"denoised_image_{i}")


if __name__ == "__main__":
    main()
