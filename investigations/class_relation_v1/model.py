"""Class anchored, source-only prototype retrieval before MLUDA's MBCA."""
from __future__ import annotations

import sys
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[2]
OFFICIAL = ROOT / 'official_aligned'
sys.path.insert(0, str(OFFICIAL / 'reference'))
sys.path.insert(0, str(OFFICIAL))
from net2 import DSANSS  # noqa: E402
from baseline_pool import install_deterministic_pool  # noqa: E402


class ClassRelationBeforeMBCA(nn.Module):
    def __init__(self, base: nn.Module, with_tokens: bool, dim: int = 288, classes: int = 7):
        super().__init__()
        self.base = base
        self.with_tokens = with_tokens
        self.register_buffer('prototypes', torch.zeros(classes, dim))
        self.register_buffer('reliability', torch.ones(classes))
        self.register_buffer('bank_ready', torch.tensor(False))
        self.delta = nn.Sequential(nn.Linear(2 * dim, dim), nn.ReLU(), nn.Linear(dim, dim))
        self.gate = nn.Linear(2 * dim, 1)
        nn.init.zeros_(self.gate.weight)
        nn.init.constant_(self.gate.bias, -3.89)  # sigmoid ~= 0.02
        if with_tokens:
            self.class_tokens = nn.Parameter(torch.zeros(classes, dim))
            nn.init.normal_(self.class_tokens, std=0.02)

    @torch.no_grad()
    def set_bank(self, features: torch.Tensor, labels: torch.Tensor):
        """All labeled source train samples; reliability is leave-one-out recall."""
        assert features.ndim == 2 and features.shape[1] == self.prototypes.shape[1]
        assert len(features) == len(labels) and len(features) > 0
        features = features.to(self.prototypes.device)
        labels = labels.to(self.prototypes.device).long()
        classes = self.prototypes.shape[0]
        counts = torch.bincount(labels, minlength=classes)
        assert bool(torch.all(counts > 1))
        sums = torch.stack([features[labels == c].sum(dim=0) for c in range(classes)])
        prototypes = F.normalize(sums / counts[:, None], dim=1)
        normalized = F.normalize(features, dim=1)
        scores = normalized @ prototypes.T
        own_loo = (sums[labels] - features) / (counts[labels, None] - 1)
        scores[torch.arange(len(labels), device=labels.device), labels] = (
            normalized * F.normalize(own_loo, dim=1)).sum(dim=1)
        prediction = scores.argmax(dim=1)
        reliability = torch.stack([(prediction[labels == c] == c).float().mean()
                                   for c in range(classes)]).clamp(min=0.25)
        self.prototypes.copy_(prototypes)
        self.reliability.copy_(reliability)
        self.bank_ready.fill_(True)

    def forward(self, source, target, initial_target_to_source, initial_source_to_target):
        assert source.ndim == target.ndim == 2
        if not bool(self.bank_ready):
            return self.base(source, target, initial_target_to_source,
                             initial_source_to_target)
        prototypes = self.prototypes.detach()
        similarity = F.normalize(target, dim=1) @ prototypes.T
        attention = (similarity / 0.2).softmax(dim=1)
        reliability_mass = attention @ self.reliability.detach()
        weighted = attention * self.reliability.detach()[None, :]
        content = prototypes
        if self.with_tokens:
            content = F.normalize(prototypes + self.class_tokens, dim=1)
        context = (weighted @ content) / reliability_mass[:, None].clamp_min(1e-6)
        pair = torch.cat((target, context), dim=1)
        update = self.delta(pair)
        gate = torch.sigmoid(self.gate(pair)) * reliability_mass[:, None]
        refined_target = target + gate * update
        return self.base(source, refined_target, source, refined_target)


def make_model(with_tokens: bool, device: str = 'cuda:0') -> DSANSS:
    model = install_deterministic_pool(DSANSS(48, 7, 7))
    rng_cpu = torch.get_rng_state()
    rng_cuda = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    model.feature_layers.atten = ClassRelationBeforeMBCA(model.feature_layers.atten,
                                                        with_tokens=with_tokens)
    torch.set_rng_state(rng_cpu)
    if rng_cuda is not None:
        torch.cuda.set_rng_state_all(rng_cuda)
    return model.to(device)
