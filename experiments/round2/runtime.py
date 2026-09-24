import contextlib
import hashlib
import json
import random
import numpy as np
import torch


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


@contextlib.contextmanager
def evaluation(model):
    modes = [(m, m.training) for m in model.modules()]
    np_state, py_state = np.random.get_state(), random.getstate()
    devices = sorted({p.device.index for p in model.parameters() if p.is_cuda})
    try:
        with torch.random.fork_rng(devices=devices), torch.no_grad():
            model.eval()
            yield
    finally:
        np.random.set_state(np_state)
        random.setstate(py_state)
        for m, mode in modes:
            m.training = mode


def tensor_hash(values):
    digest = hashlib.sha256()
    for v in values:
        v = v.detach().cpu().contiguous()
        digest.update(str((tuple(v.shape), str(v.dtype))).encode())
        digest.update(v.numpy().tobytes())
    return digest.hexdigest()


def atomic_json(path, value):
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False))
    tmp.replace(path)


def save_checkpoint(path, value):
    tmp = path.with_suffix(".tmp")
    torch.save(value, tmp)
    tmp.replace(path)


def scores(labels, predictions):
    cm = np.zeros((7, 7), dtype=np.int64)
    np.add.at(cm, (labels, predictions), 1)
    n = int(cm.sum())
    if n == 0:
        raise ValueError("Empty evaluation")
    acc = np.diag(cm)/np.maximum(cm.sum(1), 1)
    oa = float(np.trace(cm)/n)
    chance = float(cm.sum(0) @ cm.sum(1))/n**2
    return dict(oa=oa, aa=float(acc.mean()), kappa=(oa-chance)/(1-chance),
                per_class_accuracy=acc.tolist(), confusion_matrix=cm.tolist(), evaluated_n=n)
