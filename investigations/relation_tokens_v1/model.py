"""Minimal set-level relation tokens inserted before MLUDA's original MBCA."""
from __future__ import annotations

import sys
from pathlib import Path

import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[2]
OFFICIAL = ROOT / 'official_aligned'
sys.path.insert(0, str(OFFICIAL / 'reference'))
sys.path.insert(0, str(OFFICIAL))
from net2 import DSANSS  # noqa: E402
from baseline_pool import install_deterministic_pool  # noqa: E402


class RelationTokensBeforeMBCA(nn.Module):
    """Pool two unordered domain sets into latent tokens, then refine both sets."""

    def __init__(self, base_attention: nn.Module, dim: int = 288, count: int = 4):
        super().__init__()
        self.base = base_attention
        self.tokens = nn.Parameter(torch.empty(1, count, dim))
        self.source_type = nn.Parameter(torch.empty(1, 1, dim))
        self.target_type = nn.Parameter(torch.empty(1, 1, dim))
        nn.init.normal_(self.tokens, std=0.02)
        nn.init.normal_(self.source_type, std=0.02)
        nn.init.normal_(self.target_type, std=0.02)
        self.feature_norm = nn.LayerNorm(dim)
        self.pool_attention = nn.MultiheadAttention(dim, 4, batch_first=True)
        self.read_attention = nn.MultiheadAttention(dim, 4, batch_first=True)
        self.source_gate = nn.Linear(2 * dim, 1)
        self.target_gate = nn.Linear(2 * dim, 1)
        for gate in (self.source_gate, self.target_gate):
            nn.init.zeros_(gate.weight)
            nn.init.constant_(gate.bias, -2.0)

    def forward(self, source, target, initial_target_to_source, initial_source_to_target):
        del initial_target_to_source, initial_source_to_target
        assert source.ndim == target.ndim == 2 and source.shape == target.shape
        source_normalized = self.feature_norm(source).unsqueeze(0)
        target_normalized = self.feature_norm(target).unsqueeze(0)
        context = torch.cat((source_normalized + self.source_type,
                             target_normalized + self.target_type), dim=1)
        queries = self.tokens.expand(1, -1, -1)
        relation, _ = self.pool_attention(queries, context, context, need_weights=False)
        source_update, _ = self.read_attention(source_normalized, relation, relation,
                                               need_weights=False)
        target_update, _ = self.read_attention(target_normalized, relation, relation,
                                               need_weights=False)
        source_update = source_update.squeeze(0)
        target_update = target_update.squeeze(0)
        source_weight = torch.sigmoid(self.source_gate(torch.cat((source, source_update), dim=1)))
        target_weight = torch.sigmoid(self.target_gate(torch.cat((target, target_update), dim=1)))
        source_refined = source + source_weight * source_update
        target_refined = target + target_weight * target_update
        return self.base(source_refined, target_refined,
                         source_refined, target_refined)


def make_model(device: str = 'cuda:0') -> DSANSS:
    model = install_deterministic_pool(DSANSS(48, 7, 7))
    rng_cpu = torch.get_rng_state()
    rng_cuda = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    model.feature_layers.atten = RelationTokensBeforeMBCA(model.feature_layers.atten)
    torch.set_rng_state(rng_cpu)
    if rng_cuda is not None:
        torch.cuda.set_rng_state_all(rng_cuda)
    return model.to(device)
