import pytest
import torch
import torch.nn as nn

from diffsolver.hooks import LinearLayerHooker
from diffsolver.utils import load_pipeline


def test_regular_expression_match():
    assert LinearLayerHooker.regular_expression_match([".*transformer_block.*"], "transformer_block.0.norm1")
    assert not LinearLayerHooker.regular_expression_match([".*tranformer_block.*"], "transformer_block.0.ff")
    regex_list = [".*transformer_block.*", ".*attention.*", ".*ffn.*"]
    assert LinearLayerHooker.regular_expression_match(regex_list, "transformer_block.0.norm1")
    assert LinearLayerHooker.regular_expression_match(regex_list, "attention.1.proj")
    assert LinearLayerHooker.regular_expression_match(regex_list, "ffn.2.linear")
    assert not LinearLayerHooker.regular_expression_match(regex_list, "embedding.0.token")


@pytest.mark.parametrize("pipeline", ["flux", "sd3"])
def test_linear_layer_hooks(pipeline):
    pipe = load_pipeline(pipeline, torch.bfloat16, False)
    hooker = LinearLayerHooker(
        pipe, ".*transformer_block.*", torch.bfloat16, "softmax", "linear", 0.0, 1e-6, False, False
    )
    hooker.add_hooks(1.0)

    # check if the hooker is added correctly
    for name in hooker.hook_dict.keys():
        module = hooker.net.get_submodule(name)
        assert isinstance(module, nn.Linear)
        assert module.out_features == hooker.module_neurons[name]
