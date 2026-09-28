"""Class-agnostic target-to-source field and frozen class-aware OT pairing."""
import torch
from torch import nn
from torch.nn import functional as F


class ReverseFlow(nn.Module):
    def __init__(self, dim=288):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(dim+1, 288), nn.SiLU(),
                                 nn.Linear(288, 288), nn.SiLU(),
                                 nn.Linear(288, dim))

    def forward(self, z, t):
        return self.net(torch.cat((z, t), dim=1))


@torch.no_grad()
def ot_pairs(source, target, labels, q, generator, reg=.05, iterations=100):
    costs = (1 - F.normalize(source, dim=1) @ F.normalize(target, dim=1).T).cpu()
    labels, q = labels.cpu(), q.cpu()
    confidence, pseudo = q.max(1)
    source_ids, target_ids = [], []
    for c in range(7):
        indices = (labels == c).nonzero().flatten()
        candidates = ((pseudo == c) & (confidence >= .95)).nonzero().flatten()
        if not len(indices) or not len(candidates):
            continue
        cost = costs[indices][:, candidates].double()
        a = torch.full((len(indices),), 1/len(indices), dtype=cost.dtype)
        b = q[candidates, c].double().clamp_min(1e-12)
        b = b/b.sum()
        kernel = (-cost/reg).exp().clamp_min(1e-30)
        u, v = torch.ones_like(a), torch.ones_like(b)
        for _ in range(iterations):
            u = a/(kernel @ v).clamp_min(1e-30)
            v = b/(kernel.T @ u).clamp_min(1e-30)
        coupling = u[:, None]*kernel*v[None, :]
        if not torch.isfinite(coupling).all() or coupling.sum() <= 0:
            raise FloatingPointError("Invalid OT coupling")
        sampled = torch.multinomial(coupling.flatten(), 32, replacement=True,
                                    generator=generator)
        source_ids.append(indices[sampled//len(candidates)])
        target_ids.append(candidates[sampled % len(candidates)])
    if not source_ids:
        empty = torch.empty(0, dtype=torch.long, device=source.device)
        return empty, empty
    return torch.cat(source_ids).to(source.device), torch.cat(target_ids).to(source.device)


def reverse_matching_loss(flow, source, target, labels, q, generator):
    si, ti = ot_pairs(source.detach(), target.detach(), labels, q.detach(), generator)
    if not len(si):
        return source.new_zeros(()), 0
    start = target[ti].detach()
    end = source[si].detach()
    t = torch.rand((len(si), 1), generator=generator).to(start)
    state = (1-t)*start + t*end
    return F.mse_loss(flow(state, t), end-start), len(si)


@torch.no_grad()
def transport_target(flow, target_features, steps=4):
    state = target_features
    for step in range(steps):
        t = torch.full((len(state), 1), (step+.5)/steps, device=state.device,
                       dtype=state.dtype)
        state = state + flow(state, t)/steps
    return state
