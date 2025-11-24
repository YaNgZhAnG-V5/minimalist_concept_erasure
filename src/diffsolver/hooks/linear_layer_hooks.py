import logging
from collections import OrderedDict
from functools import partial
from typing import List, Union

import torch
import torch.nn as nn

from diffsolver.hooks.base import BaseHooker


class LinearLayerHooker(BaseHooker):
    def __init__(
        self,
        pipeline: nn.Module,
        regex: Union[str, List[str]],
        dtype: torch.dtype,
        masking: str,
        dst: str,
        epsilon: float = 0.0,
        eps: float = 1e-6,
        use_log: bool = False,
        binary: bool = False,
    ):
        self.pipeline = pipeline
        self.net = pipeline.unet if hasattr(pipeline, "unet") else pipeline.transformer
        self.logger = logging.getLogger(__name__)
        self.dtype = dtype
        self.regex = [regex] if isinstance(regex, str) else regex
        self.hook_dict = {}
        self.masking = masking
        self.dst = dst
        self.epsilon = epsilon
        self.eps = eps
        self.use_log = use_log
        self.lambs = []
        self.lambs_module_names = []  # store the module names for each lambda block
        self.hook_counter = 0
        self.module_neurons = OrderedDict()
        self.binary = binary  # default, need to discuss if we need to keep this attribute or not

    def add_hooks_to_linear(self, hook_fn):
        """
        Add forward hooks to every feed forward layer matching the regex
        :param hook_fn: a callable to be added to torch nn module as a hook
        :return: dictionary of added hooks
        """
        total_hooks = 0
        for name, module in self.net.named_modules():
            if not isinstance(module, nn.Linear):
                continue
            if self.regular_expression_match(self.regex, name):
                hook_fn_with_name = partial(hook_fn, name=name)
                hook = module.register_forward_hook(hook_fn_with_name, with_kwargs=True)
                self.hook_dict[name] = hook
                self.module_neurons[name] = module.out_features
                self.logger.info(f"Adding hook to {name}, neurons: {self.module_neurons[name]}")
                total_hooks += 1
        self.logger.info(f"Total hooks added: {total_hooks}")
        return self.hook_dict

    def add_hooks(self, init_value=1.0):
        hook_fn = self.get_linear_masking_hook(init_value)
        self.add_hooks_to_linear(hook_fn)
        # initialize the lambda
        self.lambs = [None] * len(self.hook_dict)
        # initialize the lambda module names
        self.lambs_module_names = [None] * len(self.hook_dict)

    def masking_fn(self, hidden_states, **kwargs):
        hidden_states_dtype = hidden_states.dtype
        mask = self.base_masking_fn(**kwargs)
        epsilon = kwargs.get("epsilon", 0.0)
        hidden_states = hidden_states * mask + torch.randn_like(hidden_states) * epsilon * (1 - mask)
        return hidden_states.to(hidden_states_dtype)

    def get_linear_masking_hook(self, init_value=1.0):
        """
        Get a hook function to mask linear layer
        """

        def hook_fn(module, args, kwargs, output, name):
            # initialize lambda with acual head dim in the first run
            if self.lambs[self.hook_counter] is None:
                self.lambs[self.hook_counter] = (
                    torch.ones(self.module_neurons[name], device=self.pipeline.device, dtype=self.dtype) * init_value
                )
                self.lambs[self.hook_counter].requires_grad = True
                # load ff lambda module name for logging
                self.lambs_module_names[self.hook_counter] = name

            # perform masking
            output_dim = output.ndim
            output = self.masking_fn(
                output,
                masking=self.masking,
                lamb=self.lambs[self.hook_counter],
                epsilon=self.epsilon,
                eps=self.eps,
                use_log=self.use_log,
            )
            # if output dimension is greater than the original dimension, we need to reduce the dimension
            # happens only for norm layers
            if output.ndim > output_dim:
                output = output.squeeze(0)
            self.hook_counter += 1
            self.hook_counter %= len(self.lambs)
            return output

        return hook_fn
