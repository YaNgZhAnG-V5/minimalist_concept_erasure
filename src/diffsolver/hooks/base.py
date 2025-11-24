import os
from abc import ABC, abstractmethod
from typing import List

import torch

import re
from ..utils.utils import hard_concrete_distribution

HOOKNAMES = ["attn", "ff", "norm"]


class BaseHooker(ABC):
    @abstractmethod
    def add_hooks(self, init_value: float):
        pass

    def clear_hooks(self):
        """clear all hooks"""
        for hook in self.hook_dict.values():
            hook.remove()
        self.hook_dict.clear()

    def save(self, name: str = None):
        if name is not None:
            dst = os.path.join(os.path.dirname(self.dst), name)
        else:
            dst = self.dst
        dst_dir = os.path.dirname(dst)
        if not os.path.exists(dst_dir):
            self.logger.info(f"Creating directory {dst_dir}")
            os.makedirs(dst_dir)
        torch.save(self.lambs, dst)

    def load(self, device, threshold):
        if os.path.exists(self.dst):
            self.logger.info(f"loading lambda from {self.dst}")
            self.lambs = torch.load(self.dst, weights_only=True, map_location=device)
            if self.binary:
                # set binary masking for each lambda by using clamp
                self.lambs = [(torch.relu(lamb - threshold) > 0).float() for lamb in self.lambs]
            else:
                self.lambs = [torch.clamp(lamb, min=0.0) for lamb in self.lambs]
            # self.lambs_module_names = [None for _ in self.lambs]
        else:
            self.logger.info("skipping loading, training from scratch")

    # TODO need to make it more efficient as utils, need to discuss
    def binarize(self, scope: str, ratio: float):
        """
        binarize lambda to be 0 or 1
        :param scope: either locally (sparsity within layer) or globally (sparsity within model)
        :param ratio: the ratio of the number of 1s to the total number of elements
        """
        assert scope in ["local", "global"], "scope must be either local or global"
        assert not self.binary, "binarization is not supported when using binary mask already"
        if scope == "local":
            for i, lamb in enumerate(self.lambs):
                num_heads = lamb.size(0)
                num_activate_heads = int(num_heads * ratio)
                # Sort the lambda values with stable sorting to maintain order for equal values
                sorted_lamb, sorted_indices = torch.sort(lamb, descending=True, stable=True)
                # Find the threshold value
                threshold = sorted_lamb[num_activate_heads - 1]
                # Create a mask based on the sorted indices
                mask = torch.zeros_like(lamb)
                mask[sorted_indices[:num_activate_heads]] = 1.0
                # Binarize the lambda based on the threshold and the mask
                self.lambs[i] = torch.where(lamb > threshold, torch.ones_like(lamb), mask)
        else:
            # Global binarization
            all_lambs = torch.cat([lamb.flatten() for lamb in self.lambs])
            num_total = all_lambs.numel()
            num_activate = int(num_total * ratio)
            # Sort all lambda values globally with stable sorting
            sorted_lambs, sorted_indices = torch.sort(all_lambs, descending=True, stable=True)
            # Find the global threshold value
            threshold = sorted_lambs[num_activate - 1]
            # Create a global mask based on the sorted indices
            global_mask = torch.zeros_like(all_lambs)
            global_mask[sorted_indices[:num_activate]] = 1.0
            # Binarize all lambdas based on the global threshold and mask
            start_idx = 0
            for i in range(len(self.lambs)):
                end_idx = start_idx + self.lambs[i].numel()
                lamb_mask = global_mask[start_idx:end_idx].reshape(self.lambs[i].shape)
                self.lambs[i] = torch.where(self.lambs[i] > threshold, torch.ones_like(self.lambs[i]), lamb_mask)
                start_idx = end_idx
        self.binary = True

    @property
    def get_lambda_block_names(self):
        return self.lambs_module_names

    @abstractmethod
    def masking_fn(self, **kwargs):
        pass

    @staticmethod
    def regular_expression_match(regexs: List[str], module_name: str):
        return any(re.match(regex, module_name) for regex in regexs)

    @staticmethod
    def base_masking_fn(**kwargs):
        lamb = kwargs["lamb"].view(1, 1, kwargs["lamb"].shape[0])
        if kwargs.get("masking", None) == "sigmoid":
            mask = torch.sigmoid(lamb)
        elif kwargs.get("masking", None) == "hard_discrete":
            use_log = kwargs.get("use_log", True)
            eps = kwargs.get("eps", 1e-8)
            mask = hard_concrete_distribution(lamb, use_log=use_log, eps=eps)
        elif kwargs.get("masking", None) == "binary":
            mask = lamb
        elif kwargs.get("masking", None) == "continues2binary":
            # TODO: this might cause potential issue as it hard threshold at 0
            mask = (lamb > 0).float()
        elif kwargs.get("masking", None) == "no_masking":
            mask = torch.ones_like(lamb)
        else:
            raise NotImplementedError
        return mask
