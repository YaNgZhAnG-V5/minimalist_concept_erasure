from dataclasses import dataclass
from typing import Any, List, Optional, Tuple, Union

import torch
from diffusers import DiTPipeline, ImagePipelineOutput
from diffusers.utils import BaseOutput
from diffusers.utils.torch_utils import randn_tensor

from diffsolver.models.base import DiffusionModelForCheckpointing


@dataclass
class DiTDiffusionPreparaPhasePipelineOutput(BaseOutput):
    latents: torch.Tensor
    class_labels_input: Any
    guidance_scale: float
    latent_channels: int
    timesteps: Any
    prompt_embeds = None


class DiTPipelineForCheckpointing(DiTPipeline, DiffusionModelForCheckpointing):
    def inference(
        self,
        class_labels: List[int],
        guidance_scale: float = 4.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        num_inference_steps: int = 50,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        attn_res=None,
    ) -> Union[ImagePipelineOutput, Tuple]:
        batch_size = len(class_labels)
        latent_size = self.transformer.config.sample_size
        latent_channels = self.transformer.config.in_channels

        latents = randn_tensor(
            shape=(batch_size, latent_channels, latent_size, latent_size),
            generator=generator,
            device=self._execution_device,
            dtype=self.transformer.dtype,
        )
        latent_model_input = torch.cat([latents] * 2) if guidance_scale > 1 else latents

        class_labels = torch.tensor(class_labels, device=self._execution_device).reshape(-1)
        class_null = torch.tensor([1000] * batch_size, device=self._execution_device)
        class_labels_input = torch.cat([class_labels, class_null], 0) if guidance_scale > 1 else class_labels

        # set step values
        self.scheduler.set_timesteps(num_inference_steps)
        for t in self.scheduler.timesteps:
            if guidance_scale > 1:
                half = latent_model_input[: len(latent_model_input) // 2]
                latent_model_input = torch.cat([half, half], dim=0)
            latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)
            timesteps = t
            if not torch.is_tensor(timesteps):
                # TODO: this requires sync between CPU and GPU. So try to pass timesteps as tensors if you can
                # This would be a good case for the `match` statement (Python 3.10+)
                is_mps = latent_model_input.device.type == "mps"
                if isinstance(timesteps, float):
                    dtype = torch.float32 if is_mps else torch.float64
                else:
                    dtype = torch.int32 if is_mps else torch.int64
                timesteps = torch.tensor([timesteps], dtype=dtype, device=latent_model_input.device)
            elif len(timesteps.shape) == 0:
                timesteps = timesteps[None].to(latent_model_input.device)
            # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
            timesteps = timesteps.expand(latent_model_input.shape[0])
            # predict noise model_output
            noise_pred = self.transformer(
                latent_model_input, timestep=timesteps, class_labels=class_labels_input
            ).sample

            # perform guidance
            if guidance_scale > 1:
                eps, rest = noise_pred[:, :latent_channels], noise_pred[:, latent_channels:]
                cond_eps, uncond_eps = torch.split(eps, len(eps) // 2, dim=0)

                half_eps = uncond_eps + guidance_scale * (cond_eps - uncond_eps)
                eps = torch.cat([half_eps, half_eps], dim=0)

                noise_pred = torch.cat([eps, rest], dim=1)

            # learned sigma
            if self.transformer.config.out_channels // 2 == latent_channels:
                model_output, _ = torch.split(noise_pred, latent_channels, dim=1)
            else:
                model_output = noise_pred

            # compute previous image: x_t -> x_t-1
            latent_model_input = self.scheduler.step(model_output, t, latent_model_input).prev_sample

        if guidance_scale > 1:
            latents, _ = latent_model_input.chunk(2, dim=0)
        else:
            latents = latent_model_input

        if not output_type == "latent":
            latents = 1 / self.vae.config.scaling_factor * latents
            samples = self.vae.decode(latents).sample

            samples = (samples / 2 + 0.5).clamp(0, 1)

            # we always cast to float32 as this does not cause significant overhead and is compatible with bfloat16
            samples = samples.cpu().permute(0, 2, 3, 1).float().numpy()

            if output_type == "pil":
                samples = self.numpy_to_pil(samples)

            # Offload all models
            self.maybe_free_model_hooks()
        else:
            samples = latents

        if not return_dict:
            return (samples,)
        return ImagePipelineOutput(images=samples)

    def inference_preparation_phase(
        self,
        class_labels: List[int],
        guidance_scale: float = 4.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        num_inference_steps: int = 50,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
    ):
        batch_size = len(class_labels)
        latent_size = self.transformer.config.sample_size
        latent_channels = self.transformer.config.in_channels

        latents = randn_tensor(
            shape=(batch_size, latent_channels, latent_size, latent_size),
            generator=generator,
            device=self._execution_device,
            dtype=self.transformer.dtype,
        )
        latent_model_input = torch.cat([latents] * 2) if guidance_scale > 1 else latents

        class_labels = torch.tensor(class_labels, device=self._execution_device).reshape(-1)
        class_null = torch.tensor([1000] * batch_size, device=self._execution_device)
        class_labels_input = torch.cat([class_labels, class_null], 0) if guidance_scale > 1 else class_labels

        # set step values
        self.scheduler.set_timesteps(num_inference_steps)
        self.guidance_scale = guidance_scale

        return DiTDiffusionPreparaPhasePipelineOutput(
            latents=latent_model_input,
            class_labels_input=class_labels_input,
            guidance_scale=guidance_scale,
            latent_channels=latent_channels,
            timesteps=self.scheduler.timesteps,
        )

    def inference_denoising_step(
        self,
        i,
        t,
        preparation_phase_output: DiTDiffusionPreparaPhasePipelineOutput,
        step_index=None,
    ):
        guidance_scale = preparation_phase_output.guidance_scale
        latent_model_input = preparation_phase_output.latents
        class_labels_input = preparation_phase_output.class_labels_input
        latent_channels = preparation_phase_output.latent_channels

        if guidance_scale > 1:
            half = latent_model_input[: len(latent_model_input) // 2]
            latent_model_input = torch.cat([half, half], dim=0)

        latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)
        timesteps = t
        if not torch.is_tensor(timesteps):
            # TODO: this requires sync between CPU and GPU. So try to pass timesteps as tensors if you can
            # This would be a good case for the `match` statement (Python 3.10+)
            is_mps = latent_model_input.device.type == "mps"
            if isinstance(timesteps, float):
                dtype = torch.float32 if is_mps else torch.float64
            else:
                dtype = torch.int32 if is_mps else torch.int64
            timesteps = torch.tensor([timesteps], dtype=dtype, device=latent_model_input.device)
        elif len(timesteps.shape) == 0:
            timesteps = timesteps[None].to(latent_model_input.device)

        # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
        timesteps = timesteps.expand(latent_model_input.shape[0])
        # predict noise model_output
        noise_pred = self.transformer(latent_model_input, timestep=timesteps, class_labels=class_labels_input).sample

        # perform guidance
        if guidance_scale > 1:
            eps, rest = noise_pred[:, :latent_channels], noise_pred[:, latent_channels:]
            cond_eps, uncond_eps = torch.split(eps, len(eps) // 2, dim=0)

            half_eps = uncond_eps + guidance_scale * (cond_eps - uncond_eps)
            eps = torch.cat([half_eps, half_eps], dim=0)

            noise_pred = torch.cat([eps, rest], dim=1)

        # learned sigma
        if self.transformer.config.out_channels // 2 == latent_channels:
            model_output, _ = torch.split(noise_pred, latent_channels, dim=1)
        else:
            model_output = noise_pred

        # compute previous image: x_t -> x_t-1
        latent_model_input = self.scheduler.step(model_output, t, latent_model_input).prev_sample
        return latent_model_input

    def inference_aft_denoising(self, latents, prompt_embeds, generator, output_type, return_dict, device):
        if self.guidance_scale > 1:
            latents, _ = latents.chunk(2, dim=0)

        if not output_type == "latent":
            latents = 1 / self.vae.config.scaling_factor * latents
            samples = self.vae.decode(latents).sample
            samples = (samples / 2 + 0.5).clamp(0, 1)
            # we always cast to float32 as this does not cause significant overhead and is compatible with bfloat16
            samples = samples.cpu().permute(0, 2, 3, 1).float().numpy()
            if output_type == "pil":
                samples = self.numpy_to_pil(samples)
            # Offload all models
            self.maybe_free_model_hooks()
            if not return_dict:
                return (samples,)
        else:
            samples = latents

        return ImagePipelineOutput(images=samples)
