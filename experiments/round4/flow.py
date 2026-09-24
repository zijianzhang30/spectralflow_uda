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
def ot_pairs(source, target, labels, q, generator, reg=.05, iterations=100,
             solver_device="cpu"):
    if solver_device not in {"cpu", "same"}:
        raise ValueError(solver_device)
    # The 32x32 Sinkhorn problems are launch-bound on GPU. Compute feature
    # cosine costs on the original device, then solve the detached problems on
    # CPU, using the same float64 iterations and CPU sampling as round2.
    work_device = torch.device("cpu") if solver_device == "cpu" else source.device
    costs = (1 - F.normalize(source, dim=1) @ F.normalize(target, dim=1).T).to(work_device)
    work_labels, work_q = labels.to(work_device), q.to(work_device)
    source_ids, target_ids, class_ids = [], [], []
    for c in range(7):
        indices = (work_labels == c).nonzero().flatten()
        if not len(indices):
            continue
        cost = costs[indices].double()
        a = torch.full((len(indices),), 1/len(indices), device=cost.device, dtype=cost.dtype)
        b = work_q[:, c].double().clamp_min(1e-12)
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
    return tuple(x.to(source.device) for x in
                 (torch.cat(source_ids), torch.cat(target_ids), torch.cat(class_ids)))


def matching_loss(flow, source, target, labels, target_q, generator, backprop_source,
                  gradient_mode="full"):
    if gradient_mode not in {"full", "state_only", "ot_pull"}:
        raise ValueError(gradient_mode)
    si, ti, classes = ot_pairs(source.detach(), target.detach(), labels,
                               target_q.detach(), generator)
    start = source[si] if backprop_source else source[si].detach()
    end = target[ti].detach()
    t = torch.rand((len(si), 1), generator=generator).to(start)
    state = (1-t)*start + t*end
    # In C both the interpolated state and displacement retain source gradients.
    velocity = end-start
    if gradient_mode == "ot_pull":
        # Exact zero-velocity control: same OT pairs and MSE scaling, no learned field.
        return velocity.square().mean()
    if gradient_mode == "state_only":
        # Same numerical FM target; remove only its direct gradient to the encoder.
        velocity = velocity.detach()
    return F.mse_loss(flow(state, t, classes), velocity)


def transport_objective(flow, classifier, source, target, labels, target_q,
                        generator, variant):
    """Train the field on detached endpoints; classify source-labelled transported views.

    Both variants use the same OT pairs, FM objective, CE weight, and source-side
    gradient path. The only difference is the displacement used for the view.
    Target features and pseudo probabilities are never differentiated through.
    """
    if variant not in {"flow_transport", "linear_transport"}:
        raise ValueError(variant)
    si, ti, classes = ot_pairs(source.detach(), target.detach(), labels,
                               target_q.detach(), generator)
    start = source[si]
    end = target[ti].detach()
    t = torch.rand((len(si), 1), generator=generator).to(start)
    detached_start = start.detach()
    state = (1-t)*detached_start + t*end
    velocity = end-detached_start
    fm = F.mse_loss(flow(state, t, classes), velocity)

    t0 = torch.zeros((len(si), 1), device=start.device, dtype=start.dtype)
    if variant == "flow_transport":
        # Detaching the field here prevents the classification loss from making
        # the field a free classifier-specific shortcut around FM.
        displacement = flow(detached_start, t0, classes).detach()
    else:
        displacement = velocity.detach()
    transported = start + 0.5*displacement
    aux_ce = F.cross_entropy(classifier(transported), classes)
    return fm, aux_ce


@torch.no_grad()
def pseudo_label_mask(flow, classifier, target_features, target_q, variant,
                      threshold=.95):
    """Confidence-only control or confidence plus inverse-Flow agreement."""
    if variant not in {"target_consistency", "flow_gate_consistency"}:
        raise ValueError(variant)
    confidence, pseudo = target_q.max(dim=1)
    selected = confidence >= threshold
    if variant == "flow_gate_consistency":
        ones = torch.ones((len(target_features), 1), device=target_features.device,
                          dtype=target_features.dtype)
        inverse = target_features - flow(target_features, ones, pseudo)
        selected = selected & (classifier(inverse).argmax(dim=1) == pseudo)
    return pseudo, selected
