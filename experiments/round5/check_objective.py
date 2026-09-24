"""Check direction, class independence, and gradient isolation of reverse FM."""
import torch
from torch import nn

from flow import ReverseFlow, reverse_matching_loss, transport_target


def main():
    torch.manual_seed(23)
    source = torch.randn(32, 288, requires_grad=True)
    target = torch.randn(32, 288, requires_grad=True)
    labels = torch.arange(32) % 7
    q = torch.softmax(torch.randn(32, 7), dim=1).requires_grad_()
    flow = ReverseFlow()
    generator = torch.Generator().manual_seed(7)
    loss = reverse_matching_loss(flow, source, target, labels, q, generator)
    loss.backward()
    assert source.grad is None and target.grad is None and q.grad is None
    assert any(p.grad is not None and p.grad.abs().sum() > 0
               for p in flow.parameters())
    assert torch.isfinite(loss)

    # A constant velocity from target to source must integrate with the
    # positive sign. This also checks that inference needs no class label.
    constant = ReverseFlow()
    for module in constant.modules():
        if isinstance(module, nn.Linear):
            nn.init.zeros_(module.weight)
            nn.init.zeros_(module.bias)
    displacement = torch.arange(288, dtype=torch.float32) / 288
    constant.net[-1].bias.data.copy_(displacement)
    moved = transport_target(constant, target.detach(), steps=4)
    assert torch.allclose(moved, target.detach() + displacement, atol=1e-6)
    print(dict(reverse_fm_finite=True, endpoints_detached=True,
               class_condition_required=False, four_step_direction="target_to_source"))


if __name__ == "__main__":
    main()
