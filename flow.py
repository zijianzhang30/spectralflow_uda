"""Independent conditional flow and class-wise cosine Sinkhorn OT."""
import torch
from torch import nn
from torch.nn import functional as F


class Flow(nn.Module):
    def __init__(self, dim=288, classes=7):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(dim+classes+1, 288), nn.SiLU(),
                                 nn.Linear(288, 288), nn.SiLU(), nn.Linear(288, dim))

    def forward(self, z, t, classes):
        condition = F.one_hot(classes, 7).to(z)
        return self.net(torch.cat((z, t, condition), dim=1))


@torch.no_grad()
def ot_pairs(source, target, labels, q, generator, reg=.05, iterations=100):
    costs = 1 - F.normalize(source, dim=1) @ F.normalize(target, dim=1).T
    source_ids, target_ids, class_ids = [], [], []
    for c in range(7):
        indices = (labels == c).nonzero().flatten()
        if not len(indices):
            continue
        cost = costs[indices].double()
        a = torch.full((len(indices),), 1/len(indices), device=cost.device, dtype=cost.dtype)
        b = q[:, c].double().clamp_min(1e-12)
        b = b / b.sum()
        kernel = (-cost/reg).exp().clamp_min(1e-30)
        u, v = torch.ones_like(a), torch.ones_like(b)
        for _ in range(iterations):
            u = a / (kernel @ v).clamp_min(1e-30)
            v = b / (kernel.T @ u).clamp_min(1e-30)
        coupling = u[:, None] * kernel * v[None, :]
        if not torch.isfinite(coupling).all() or coupling.sum() <= 0:
            raise FloatingPointError("Invalid OT coupling")
        sampled = torch.multinomial(coupling.flatten().cpu(), 32, replacement=True,
                                    generator=generator).to(indices.device)
        source_ids.append(indices[sampled // len(target)])
        target_ids.append(sampled % len(target))
        class_ids.append(torch.full((32,), c, device=indices.device, dtype=torch.long))
    if not source_ids:
        raise ValueError("No labelled source classes in batch")
    return torch.cat(source_ids), torch.cat(target_ids), torch.cat(class_ids)


def matching_loss(flow, source, target, labels, target_q, generator, backprop_source):
    si, ti, classes = ot_pairs(source.detach(), target.detach(), labels,
                               target_q.detach(), generator)
    start = source[si] if backprop_source else source[si].detach()
    end = target[ti].detach()
    t = torch.rand((len(si), 1), generator=generator).to(start)
    state = (1-t)*start + t*end
    # In C both the interpolated state and displacement retain source gradients.
    velocity = end-start
    return F.mse_loss(flow(state, t, classes), velocity)
