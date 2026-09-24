"""Numerical/gradient contracts for the two isolated mechanism interventions."""
import importlib.util
import json
from pathlib import Path
import torch
from flow import Flow, matching_loss, ot_pairs

ROOT = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("round1_flow", ROOT.parents[1]/"official_aligned/flow.py")
original = importlib.util.module_from_spec(spec)
spec.loader.exec_module(original)


def main():
    torch.set_num_threads(2)
    torch.manual_seed(31)
    source = torch.randn(14, 288, requires_grad=True)
    target = torch.randn(32, 288, requires_grad=True)
    q = torch.randn(32, 7).softmax(1).detach().requires_grad_()
    labels = torch.arange(14) % 7
    flow = Flow()
    params = [source, target, q]+list(flow.parameters())
    def calculate(fn, **kwargs):
        generator = torch.Generator().manual_seed(42)
        loss = fn(flow, source, target, labels, q, generator, backprop_source=True, **kwargs)
        gradients = torch.autograd.grad(loss, params, allow_unused=True)
        return loss.detach(), gradients, generator.get_state()
    old, og, rng = calculate(original.matching_loss)
    full, fg, rng2 = calculate(matching_loss)
    torch.testing.assert_close(old, full, rtol=0, atol=0)
    assert torch.equal(rng, rng2)
    for a,b in zip(og, fg):
        if a is None: assert b is None
        else: torch.testing.assert_close(a, b, rtol=0, atol=0)
    state, sg, rng3 = calculate(matching_loss, gradient_mode="state_only")
    torch.testing.assert_close(full, state, rtol=0, atol=0)
    assert torch.equal(rng, rng3)
    assert sg[1] is None and sg[2] is None
    assert not torch.equal(fg[0], sg[0])
    for a,b in zip(fg[3:], sg[3:]):
        torch.testing.assert_close(a, b, rtol=0, atol=0)
    pull, pg, rng4 = calculate(matching_loss, gradient_mode="ot_pull")
    assert torch.equal(rng, rng4)
    assert pg[0].norm()>0 and all(g is None for g in pg[1:])
    si,ti,_ = ot_pairs(source.detach(), target.detach(), labels, q.detach(),
                       torch.Generator().manual_seed(42))
    expected = (target.detach()[ti]-source[si]).square().mean()
    torch.testing.assert_close(pull, expected.detach(), rtol=0, atol=0)
    torch.testing.assert_close(pg[0], torch.autograd.grad(expected, source)[0], rtol=0, atol=0)
    detached = matching_loss(flow, source, target, labels, q,
                            torch.Generator().manual_seed(42), backprop_source=False)
    bg = torch.autograd.grad(detached, params, allow_unused=True)
    assert all(g is None for g in bg[:3]) and any(g is not None for g in bg[3:])
    report = dict(passed=True, original_loss_gradients_rng_exact=True,
                  state_only_same_value_and_flow_gradient=True,
                  target_and_q_detached=True, ot_pull_matches_pair_mse=True,
                  detached_B_preserved=True)
    (ROOT/"objective_checks.json").write_text(json.dumps(report, indent=2)+"\n")
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
