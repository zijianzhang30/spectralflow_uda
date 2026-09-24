"""Check pseudo-label selection and isolated target-classification gradients."""
import torch
from torch import nn
from torch.nn import functional as F
from flow import Flow, matching_loss, pseudo_label_mask


def main():
    torch.manual_seed(11)
    source = torch.randn(32, 288, requires_grad=True)
    target = torch.randn(32, 288, requires_grad=True)
    target_noise = torch.randn(32, 288, requires_grad=True)
    labels = torch.arange(32) % 7
    classifier = nn.Linear(288, 7)
    flow = Flow()
    q = torch.full((32, 7), .002)
    q[:, 3] = .988
    q[::4] = 1/7
    pseudo, control = pseudo_label_mask(flow, classifier, target, q,
                                        "target_consistency")
    _, gated = pseudo_label_mask(flow, classifier, target, q,
                                 "flow_gate_consistency")
    assert int(control.sum()) == 24
    assert torch.all(gated <= control)
    assert (pseudo[control] == 3).all()
    aux = F.cross_entropy(classifier(target_noise[control]), pseudo[control])
    generator = torch.Generator().manual_seed(8)
    fm = matching_loss(flow, source, target, labels, q, generator,
                       backprop_source=False)
    (fm + .5*aux).backward()
    assert source.grad is None and target.grad is None
    assert target_noise.grad is not None and target_noise.grad.abs().sum() > 0
    assert all(p.grad is not None for p in flow.parameters())
    assert classifier.weight.grad is not None
    print(dict(control_selected=int(control.sum()), flow_gate_selected=int(gated.sum()),
               target_noise_gradient=True, fm_encoder_gradient=False))


if __name__ == "__main__":
    main()
