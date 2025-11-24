"""
    @date:  2024.10.28  week44
    @func:  flux pack&unpack.
"""

import torch


def flux_pack_latents(latents, batch_size, num_channels_latents, height, width):
    latents = latents.view(batch_size, num_channels_latents, height // 2, 2, width // 2, 2)
    latents = latents.permute(0, 2, 4, 1, 3, 5)
    latents = latents.reshape(batch_size, (height // 2) * (width // 2), num_channels_latents * 4)

    return latents


def flux_unpack_latents(latents, height, width, vae_scale_factor):
    batch_size, num_patches, channels = latents.shape

    height = height // vae_scale_factor
    width = width // vae_scale_factor

    latents = latents.view(batch_size, height // 2, width // 2, channels // 4, 2, 2)
    latents = latents.permute(0, 3, 1, 4, 2, 5)

    latents = latents.reshape(batch_size, channels // (2 * 2), height, width)

    return latents


def save_to_dict(var, name, dict):
    if var is not None:
        if isinstance(var, torch.Tensor):
            var = var.cpu().detach().numpy()
        if isinstance(var, list):
            var = [v.cpu().detach().numpy() if isinstance(v, torch.Tensor) else v for v in var]
    else:
        return dict

    if name not in dict:
        dict[name] = []

    dict[name].append(var)
    return dict


def gumbel_softmax(logits, temperature, hard, eps=1e-10):
    u = torch.rand_like(logits)
    gumbel = -torch.log(-torch.log(u + eps) + eps)
    y = logits + gumbel
    y = torch.nn.functional.softmax(y / temperature, dim=-1)
    if hard != 0:
        y_hard = torch.zeros_like(logits)
        y_hard.scatter_(-1, torch.argmax(y, dim=-1, keepdim=True), 1.0)
        y = (y_hard - y).detach() + y
    return y
