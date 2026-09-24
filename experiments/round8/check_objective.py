"""Check prototype mass changes only OT sampling, not reverse-FM gradients."""
import torch
from torch import nn

from flow import ReverseFlow, reverse_matching_loss, transport_target, ot_pairs


def main():
    torch.manual_seed(23)
    source = torch.randn(32, 288, requires_grad=True)
    target = torch.randn(32, 288, requires_grad=True)
    labels = torch.arange(32) % 7
    q = torch.full((32, 7), .001)
    q[:, 2] = .994
    q.requires_grad_()
    flow = ReverseFlow()
    generator = torch.Generator().manual_seed(7)
    si, ti = ot_pairs(source, target, labels, q, generator)
    assert len(si) == 32 and (labels[si] == 2).all()
    prototype_q = torch.softmax(torch.randn(32, 7), dim=1).requires_grad_()
    weighted_generator = torch.Generator().manual_seed(7)
    weighted_si, weighted_ti = ot_pairs(source, target, labels, q,
                                        weighted_generator, prototype_q)
    assert len(weighted_si) == len(si) == 32
    assert (labels[weighted_si] == 2).all()
    assert not torch.equal(weighted_ti, ti)
    loss, count = reverse_matching_loss(flow, source, target, labels, q, generator)
    assert count == 32
    loss.backward()
    assert source.grad is None and target.grad is None and q.grad is None
    assert any(p.grad is not None and p.grad.abs().sum() > 0
               for p in flow.parameters())
    assert torch.isfinite(loss)
    weighted_loss, weighted_count = reverse_matching_loss(
        flow, source, target, labels, q, generator, prototype_q)
    assert weighted_count == 32 and torch.isfinite(weighted_loss)
    weighted_loss.backward()
    assert prototype_q.grad is None and source.grad is None and target.grad is None
    empty_loss, count = reverse_matching_loss(flow, source, target, labels,
                                              torch.full((32, 7), 1/7), generator)
    assert count == 0 and empty_loss.item() == 0

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
               prototype_mass_changes_pairs=True, prototype_q_detached=True,
               class_condition_required=False, four_step_direction="target_to_source"))


if __name__ == "__main__":
    main()
