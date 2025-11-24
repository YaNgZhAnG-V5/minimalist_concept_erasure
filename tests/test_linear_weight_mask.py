import pytest
import torch
import torch.nn as nn

from diffsolver.hooks.linear_layer_weight_hook import WeightMaskedLinear


@pytest.mark.parametrize("in_features", [10, 20, 30])
@pytest.mark.parametrize("out_features", [5, 10, 20])
def test_linear_weight_layer(in_features, out_features):
    linear = nn.Linear(in_features, out_features)
    masked_linear = WeightMaskedLinear.from_linear(linear)
    # test shape
    assert masked_linear.weight.shape == linear.weight.shape
    assert masked_linear.bias.shape == linear.bias.shape

    # test zero mask
    mask = torch.zeros_like(masked_linear.weight)
    output = masked_linear(torch.randn(10, in_features), mask)
    assert output.shape == (10, out_features)
    # expand bias to match the output shape
    assert torch.allclose(output[0, :], linear.bias)

    # test non-zero mask
    mask = torch.ones_like(masked_linear.weight)
    input_tensor = torch.randn(10, in_features)
    output = masked_linear(input_tensor, mask)
    masked_weight = masked_linear.weight * mask
    assert torch.allclose(output, torch.matmul(masked_weight, input_tensor.T).T + linear.bias)


def test_linear_weight_hook():
    pass
