"""Diagnostic controls only; not wired into frozen experiment trainers.

Shared learnable weights/BN affine parameters, separate source/target buffers.
Augmented views may use a domain bank without committing their updates.
"""
from contextlib import contextmanager
import numpy as np
import torch
from torch import nn

BUFFERS = ("running_mean", "running_var", "num_batches_tracked")


def spatial_flip(x):
    if x.ndim != 4:
        raise ValueError("Expected B,C,H,W patches")
    horizontal, vertical = np.random.random() > .5, np.random.random() > .5
    if horizontal:
        x = x.flip(-1)
    if vertical:
        x = x.flip(-2)
    return x


class DomainBNBuffers:
    """Owns buffers only. Save this state together with model.state_dict().

    Buffer objects are temporarily replaced, never copy_ restored after a
    gradient-recording forward: autograd may retain their version counters.
    Contexts cannot nest and require all controlled BN layers in the same mode.
    """
    def __init__(self, model):
        self.modules = {name: module for name,module in model.named_modules()
                        if isinstance(module, nn.modules.batchnorm._BatchNorm)}
        if not self.modules or any(not m.track_running_stats for m in self.modules.values()):
            raise ValueError("Requires BatchNorm layers with tracked running statistics")
        self.banks = {domain: {name: {key: getattr(module,key).detach().clone() for key in BUFFERS}
                              for name,module in self.modules.items()}
                      for domain in ("source", "target")}
        self.active = False

    def state_dict(self):
        if self.active:
            raise RuntimeError("Cannot checkpoint an active BN context")
        return {d: {n: {k: t.detach().clone() for k,t in buffers.items()}
                    for n,buffers in bank.items()} for d,bank in self.banks.items()}

    def load_state_dict(self, state):
        if self.active or set(state) != set(self.banks):
            raise ValueError("Invalid BN bank state or active context")
        prepared = {}
        for domain,bank in state.items():
            if set(bank) != set(self.modules):
                raise ValueError("BN module keys differ")
            prepared[domain] = {}
            for name,buffers in bank.items():
                if set(buffers) != set(BUFFERS):
                    raise ValueError("BN buffer keys differ")
                prepared[domain][name] = {}
                for key,value in buffers.items():
                    ref = getattr(self.modules[name],key)
                    if value.shape != ref.shape or value.dtype != ref.dtype or not torch.isfinite(value).all():
                        raise ValueError("Invalid BN buffer: " + name + "." + key)
                    if key != "running_mean" and (value < 0).any():
                        raise ValueError("Negative BN variance/count")
                    prepared[domain][name][key] = value.detach().to(ref.device).clone()
        self.banks = prepared

    @contextmanager
    def use(self, domain, update=False):
        if self.active or domain not in self.banks:
            raise ValueError("Invalid domain or nested BN context")
        modes = {m.training for m in self.modules.values()}
        if len(modes) != 1 or (update and modes != {True}):
            raise ValueError("Updates require all BN layers in training mode")
        previous = {name: {key: getattr(module,key) for key in BUFFERS}
                    for name,module in self.modules.items()}
        self.active = True
        try:
            for name,module in self.modules.items():
                for key in BUFFERS:
                    module._buffers[key] = self.banks[domain][name][key].to(previous[name][key].device).clone()
            yield
            if update:
                self.banks[domain] = {name: {key: getattr(module,key).detach().clone() for key in BUFFERS}
                                      for name,module in self.modules.items()}
        finally:
            for name,module in self.modules.items():
                for key in BUFFERS:
                    module._buffers[key] = previous[name][key]
            self.active = False
