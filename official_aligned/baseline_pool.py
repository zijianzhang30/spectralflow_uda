"""Deterministic backward only; native official pooling forwards are preserved."""
import torch
from torch import nn
from torch.nn import functional as F

class _DeterministicGlobalPool(torch.autograd.Function):
    @staticmethod
    def forward(ctx, value):
        ctx.shape = value.shape
        return F.avg_pool3d(value, (1, value.shape[-2], value.shape[-1]))

    @staticmethod
    def backward(ctx, gradient):
        # Global, non-overlapping pooling: every input receives exactly one term.
        area = gradient.new_tensor(ctx.shape[-2] * ctx.shape[-1])
        return (gradient / area).expand(ctx.shape).contiguous()

class _DeterministicGlobalMaxPool(torch.autograd.Function):
    @staticmethod
    def forward(ctx, value):
        result, indices = F.adaptive_max_pool2d(value, 1, return_indices=True)
        ctx.shape = value.shape
        ctx.save_for_backward(indices)
        return result

    @staticmethod
    def backward(ctx, gradient):
        (indices,) = ctx.saved_tensors
        positions = torch.arange(ctx.shape[-2] * ctx.shape[-1], device=gradient.device)
        mask = positions.reshape(1, 1, *ctx.shape[-2:]) == indices
        return gradient * mask

def install_deterministic_pool(model):
    pool = model.feature_layers.avgpool
    assert isinstance(pool, nn.AvgPool3d)
    assert pool.kernel_size == (1, model.feature_layers.sz, model.feature_layers.sz)
    # Preserve the official forward kernel and state_dict; replace only backward.
    pool.forward = _DeterministicGlobalPool.apply
    max_pool = model.feature_layers.ca.max_pool
    assert isinstance(max_pool, nn.AdaptiveMaxPool2d) and max_pool.output_size == 1
    max_pool.forward = _DeterministicGlobalMaxPool.apply
    return model
