"""Gradient and paired-control checks for CE-to-Flow coupling."""
import torch
from torch import nn
from flow import Flow, transport_objective


def probe(variant):
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
    ce_to_flow = torch.autograd.grad(aux, list(flow.parameters()), retain_graph=True,
                                     allow_unused=True)
    (fm + .1 * aux).backward()
    assert source.grad is not None and source.grad.abs().sum() > 0
    assert classifier.weight.grad is not None and classifier.weight.grad.abs().sum() > 0
    assert target.grad is None and q.grad is None
    assert all(torch.isfinite(p.grad).all() for p in flow.parameters())
    active = any(g is not None and g.abs().sum() > 0 for g in ce_to_flow)
    assert active == (variant == "flow_transport_coupled")
    return float(fm), float(aux), generator.get_state(), active


def main():
    results = {v: probe(v) for v in ("linear_transport", "flow_transport_detached",
                                     "flow_transport_coupled")}
    assert len({r[0] for r in results.values()}) == 1
    assert all(torch.equal(next(iter(results.values()))[2], r[2]) for r in results.values())
    assert results["flow_transport_detached"][1] == results["flow_transport_coupled"][1]
    print({v: {"fm": r[0], "aux_ce": r[1], "ce_to_flow": r[3]} for v, r in results.items()})


if __name__ == "__main__":
    main()
