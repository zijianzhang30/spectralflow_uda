"""Check the paired transport ablation's gradients and shared random path."""
import torch
from torch import nn
from flow import Flow, transport_objective


def run(variant):
    torch.manual_seed(37)
    source = torch.randn(32, 288, requires_grad=True)
    target = torch.randn(32, 288, requires_grad=True)
    labels = torch.arange(32) % 7
    q = torch.softmax(torch.randn(32, 7), dim=1).requires_grad_()
    flow = Flow()
    classifier = nn.Linear(288, 7)
    generator = torch.Generator().manual_seed(91)
    fm, aux = transport_objective(flow, classifier, source, target, labels, q,
                                  generator, variant)
    flow_grad = torch.autograd.grad(fm, list(flow.parameters()), retain_graph=True)
    aux_grad = torch.autograd.grad(aux, list(flow.parameters()), retain_graph=True,
                                   allow_unused=True)
    (fm + .1*aux).backward()
    assert source.grad is not None and source.grad.abs().sum() > 0
    assert target.grad is None and q.grad is None
    assert classifier.weight.grad is not None and classifier.weight.grad.abs().sum() > 0
    assert all(g is None for g in aux_grad)
    assert any(g.abs().sum() > 0 for g in flow_grad)
    return float(fm), float(aux), generator.get_state()


def main():
    fm_flow, ce_flow, rng_flow = run("flow_transport")
    fm_linear, ce_linear, rng_linear = run("linear_transport")
    assert fm_flow == fm_linear
    assert torch.equal(rng_flow, rng_linear)
    assert ce_flow != ce_linear
    print(dict(fm_equal=True, pair_rng_equal=True,
               source_and_classifier_grad=True, target_grad=False,
               aux_to_flow_grad=False, flow_ce=ce_flow, linear_ce=ce_linear))


if __name__ == "__main__":
    main()
