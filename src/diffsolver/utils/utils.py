# all utiles functions
import math
from typing import List, Optional

import torch
from diffusers.models.activations import GEGLU, GELU


def get_total_params(model, trainable: bool = True):
    return sum(p.numel() for p in model.parameters() if p.requires_grad == trainable)


def get_precision(precision: str):
    assert precision in ["fp16", "fp32", "bf16"], "precision must be either fp16, fp32, bf16"
    if precision == "fp16":
        torch_dtype = torch.float16
    elif precision == "bf16":
        torch_dtype = torch.bfloat16
    elif precision == "fp32":
        torch_dtype = torch.float32
    elif precision == "fp64":
        torch_dtype = torch.float64
    return torch_dtype


def calculate_mask_sparsity(hooker, threshold: Optional[float] = None):
    total_num_lambs = 0
    num_activate_lambs = 0
    binary = getattr(hooker, "binary", None)  # if binary is not present, it will return None for ff_hooks
    for lamb in hooker.lambs:
        total_num_lambs += lamb.size(0)
        if binary:
            assert threshold is None, "threshold should be None for binary mask"
            num_activate_lambs += lamb.sum().item()
        else:
            assert threshold is not None, "threshold must be provided for non-binary mask"
            num_activate_lambs += (lamb >= threshold).sum().item()
    return total_num_lambs, num_activate_lambs, num_activate_lambs / total_num_lambs


def linear_layer_masking(module, lamb):
    # perform masking on K Q V to see if it still works
    inner_dim = module.to_k.in_features // module.heads
    modules_to_remove = [module.to_k, module.to_q, module.to_v]
    for module_to_remove in modules_to_remove:
        for idx, head_mask in enumerate(lamb):
            module_to_remove.weight.data[idx * inner_dim : (idx + 1) * inner_dim, :] *= head_mask
            if module_to_remove.bias is not None:
                module_to_remove.bias.data[idx * inner_dim : (idx + 1) * inner_dim] *= head_mask

    # perform masking on the output
    for idx, head_mask in enumerate(lamb):
        module.to_out[0].weight.data[:, idx * inner_dim : (idx + 1) * inner_dim] *= head_mask
    return module


# create dummy module for skip connection
class SkipConnection(torch.nn.Module):
    def __init__(self):
        super(SkipConnection, self).__init__()

    def forward(*args, **kwargs):
        return args[1]


def linear_layer_pruning(module, lamb):
    """
    TODO still need to check if it is compatible with the other attention model
    linear layer pruning for the attention module
    support modules: AttnProcessor2_0, (TODO FluxAttnProcessor2_0, FluxSingleAttnProcessor2_0, JointAttnProcessor2_0)

    Detailed Steps (ignore the bia, normalize, etc.):
    1. Latent features will be feed into  nn.Linear module, .to_k, .to_q, .to_v
       with (cross_attn_dim, inner_kv_dim / inner_dim)
    2. inner features will be divided into head, q shape: [B, N, H, D]
        2.1 dim_hidden_feature = inner_dim * (unmasked heads // heads)
        2.2 to_q, to_k, to_v has shape [cross_attn_dim, inner_kv_dim / inner_dim]
            row has number of heads * inner_dim, only need to mask the rows
        2.3 "input channel remain unchanged"
    3. updated latent features after scaled dot product attention
    4. output projection layer, to_out from inner dim to original latent dim
        4.1 Masked dim should change from 0 to 1 , **output channel remain unchanged**

    """
    heads_to_keep = torch.nonzero(lamb).squeeze()
    if len(heads_to_keep.shape) == 0:
        # if only one head is kept, or none
        heads_to_keep = heads_to_keep.unsqueeze(0)

    modules_to_remove = [module.to_k, module.to_q, module.to_v]
    new_heads = int(lamb.sum().item())

    if new_heads == 0:
        return SkipConnection()

    for module_to_remove in modules_to_remove:
        # get head dimension
        inner_dim = module_to_remove.out_features // module.heads
        # place holder for the rows to keep
        rows_to_keep = torch.zeros(
            module_to_remove.out_features, dtype=torch.bool, device=module_to_remove.weight.device
        )

        for idx in heads_to_keep:
            rows_to_keep[idx * inner_dim : (idx + 1) * inner_dim] = True

        # overwrite the inner projection with masked projection
        module_to_remove.weight.data = module_to_remove.weight.data[rows_to_keep, :]
        if module_to_remove.bias is not None:
            module_to_remove.bias.data = module_to_remove.bias.data[rows_to_keep]
        module_to_remove.out_features = int(sum(rows_to_keep).item())

    # Also update the output projection layer if available, (for FLUXSingleAttnProcessor2_0)
    # with column masking, dim 1
    if getattr(module, "to_out", None) is not None:
        module.to_out[0].weight.data = module.to_out[0].weight.data[:, rows_to_keep]
        module.to_out[0].in_features = int(sum(rows_to_keep).item())

    # update parameters in the attention module
    module.inner_dim = module.inner_dim // module.heads * new_heads
    module.query_dim = module.query_dim // module.heads * new_heads
    module.inner_kv_dim = module.inner_kv_dim // module.heads * new_heads
    module.cross_attention_dim = module.cross_attention_dim // module.heads * new_heads
    module.heads = new_heads
    return module


def ffn_linear_layer_pruning(module, lamb):
    """
    TODO need to merge with linear_layer_pruning if possible, do it later
    """
    lambda_to_keep = torch.nonzero(lamb).squeeze()
    if len(lambda_to_keep) == 0:
        return SkipConnection()

    num_lambda = len(lambda_to_keep)

    if isinstance(module.net[0], GELU):
        # linear layer weight remove before activation
        module.net[0].proj.weight.data = module.net[0].proj.weight.data[lambda_to_keep, :]
        module.net[0].proj.out_features = num_lambda
        if module.net[0].proj.bias is not None:
            module.net[0].proj.bias.data = module.net[0].proj.bias.data[lambda_to_keep]

        update_act = GELU(module.net[0].proj.in_features, num_lambda)
        update_act.proj = module.net[0].proj
        module.net[0] = update_act
    elif isinstance(module.net[0], GEGLU):
        output_feature = module.net[0].proj.out_features
        module.net[0].proj.weight.data = torch.cat(
            [
                module.net[0].proj.weight.data[: output_feature // 2, :][lambda_to_keep, :],
                module.net[0].proj.weight.data[output_feature // 2 :][lambda_to_keep, :],
            ],
            dim=0,
        )
        module.net[0].proj.out_features = num_lambda * 2
        if module.net[0].proj.bias is not None:
            module.net[0].proj.bias.data = torch.cat(
                [
                    module.net[0].proj.bias.data[: output_feature // 2][lambda_to_keep],
                    module.net[0].proj.bias.data[output_feature // 2 :][lambda_to_keep],
                ]
            )

        update_act = GEGLU(module.net[0].proj.in_features, num_lambda * 2)
        update_act.proj = module.net[0].proj
        module.net[0] = update_act

    # proj weight after activation
    module.net[2].weight.data = module.net[2].weight.data[:, lambda_to_keep]
    module.net[2].in_features = num_lambda

    return module


# create SparsityLinear module
class SparsityLinear(torch.nn.Module):
    def __init__(self, in_features, out_features, lambda_to_keep, num_lambda):
        super(SparsityLinear, self).__init__()
        self.linear = torch.nn.Linear(in_features, num_lambda)
        self.out_features = out_features
        self.lambda_to_keep = lambda_to_keep

    def forward(self, x):
        x = self.linear(x)
        output = torch.zeros(x.size(0), self.out_features, device=x.device, dtype=x.dtype)
        output[:, self.lambda_to_keep] = x
        return output


def norm_layer_pruning(module, lamb):
    """
    Pruning the layer normalization layer for FLUX model
    """
    lambda_to_keep = torch.nonzero(lamb).squeeze()
    if len(lambda_to_keep) == 0:
        return SkipConnection()

    num_lambda = len(lambda_to_keep)

    # get num_features
    in_features = module.linear.in_features
    out_features = module.linear.out_features

    sparselinear = SparsityLinear(in_features, out_features, lambda_to_keep, num_lambda)
    sparselinear.linear.weight.data = module.linear.weight.data[lambda_to_keep]
    sparselinear.linear.bias.data = module.linear.bias.data[lambda_to_keep]
    module.linear = sparselinear
    return module


def hard_concrete_distribution(
    p, beta: float = 0.83, eps: float = 1e-8, eta: float = 1.1, gamma: float = -0.1, use_log: bool = False
):
    u = torch.rand(p.shape).to(p.device)
    if use_log:
        p = torch.clamp(p, min=eps)
        p = torch.log(p)
    s = torch.sigmoid((torch.log(u + eps) - torch.log(1 - u + eps) + p) / beta)
    s = s * (eta - gamma) + gamma
    s = s.clamp(0, 1)
    return s


def l0_complexity_loss(alpha, beta: float = 0.83, eta: float = 1.1, gamma: float = -0.1, use_log: bool = False):
    offset = beta * math.log(-gamma / eta)
    loss = torch.sigmoid(alpha - offset).sum()
    return loss


def calculate_reg_loss(
    loss_reg,
    lambs: List[torch.Tensor],
    p: int,
    use_log: bool = False,
    mean=True,
    reg=True,  # regularize the lambda with bounded value range
    reg_alpha=0.4,  # alpha for the regularizer, avoid gradient vanishing
    reg_beta=1,  # beta for shifting the lambda toward positive value (avoid gradient vanishing)
):
    if p == 0:
        for lamb in lambs:
            loss_reg += l0_complexity_loss(lamb, use_log=use_log)
        loss_reg /= len(lambs)
    elif p == 1 or p == 2:
        for lamb in lambs:
            if reg:
                lamb = torch.sigmoid(lamb * reg_alpha + reg_beta)
            if mean:
                loss_reg += lamb.norm(p) / len(lamb)
            else:
                loss_reg += lamb.norm(p)
        loss_reg /= len(lambs)
    else:
        raise NotImplementedError
    return loss_reg
