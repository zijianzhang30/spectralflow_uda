"""Focused checks for KL direction, weighting, and detached EMA teacher."""
import copy
import torch
from model import Backbone
from train import consistency_loss, update_teacher


def main():
    torch.manual_seed(41)
    student = Backbone()
    teacher = copy.deepcopy(student).eval()
    teacher.requires_grad_(False)
    weak = torch.randn(4, 48, 7, 7)
    strong = weak + .02*torch.randn_like(weak)
    _, student_logits = student(strong)
    with torch.no_grad():
        _, teacher_logits = teacher(weak)
    plain, ones, _ = consistency_loss(student_logits, teacher_logits, False)
    weighted, weights, q = consistency_loss(student_logits, teacher_logits, True)
    assert torch.isfinite(plain) and torch.isfinite(weighted)
    assert torch.all(ones==1)
    assert torch.all(weights>0) and torch.allclose(weights.mean(), torch.ones(()), atol=1e-6)
    assert not q.requires_grad and not weights.requires_grad
    assert torch.allclose(plain, torch.nn.functional.kl_div(
        student_logits.log_softmax(1), teacher_logits.softmax(1), reduction="batchmean"))
    weighted.backward()
    assert any(p.grad is not None and p.grad.abs().sum()>0 for p in student.parameters())
    assert all(p.grad is None for p in teacher.parameters())
    before = next(teacher.parameters()).detach().clone()
    with torch.no_grad():
        next(student.parameters()).add_(1)
    update_teacher(teacher, student)
    assert torch.allclose(next(teacher.parameters()), before*.99+(before+1)*.01)
    print("PASS: KL direction, mean-normalized positive weights, student-only gradients, EMA update")


if __name__ == "__main__":
    main()
