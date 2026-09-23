"""Images and source labels only. Target labels are loaded in final_test.py."""
from pathlib import Path
import hashlib
import hdf5storage
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

DEFAULT_DATA = Path("/home/zhangzj26/TGRS_MLUDA-2024/datasets/Houston")


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def array_hash(array):
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def load_images(root, protocol, cache=None):
    source = hdf5storage.loadmat(str(root / "Houston13.mat"))["ori_data"]
    target = hdf5storage.loadmat(str(root / "Houston18.mat"))["ori_data"]
    if protocol == "official_ilda":
        if cache is None:
            raise ValueError("official_ilda requires an explicit --ilda-cache")
        with np.load(cache) as contents:
            s, t = contents["s"].copy(), contents["t"].copy()
        if s.shape != source.shape or t.shape != target.shape:
            raise ValueError("ILDA cache shape differs from raw images")
        source, target = s, t
    elif cache is not None:
        raise ValueError("raw protocol must not receive an ILDA cache")
    source, target = source.astype(np.float32), target.astype(np.float32)
    assert source.shape == target.shape == (210, 954, 48)
    assert np.isfinite(source).all() and np.isfinite(target).all()
    return source, target


def source_split(gt, seed):
    rng = np.random.RandomState(seed)
    train, val = [], []
    for c in range(1, 8):
        coordinates = np.argwhere(gt == c)
        rng.shuffle(coordinates)
        if len(coordinates) <= 180:
            raise ValueError(f"Source class {c} lacks held-out samples")
        train.append(coordinates[:180])
        val.append(coordinates[180:])
    train, val = np.concatenate(train), np.concatenate(val)
    rng.shuffle(train)
    rng.shuffle(val)
    train_ids = np.ravel_multi_index(train.T, gt.shape)
    val_ids = np.ravel_multi_index(val.T, gt.shape)
    assert not np.intersect1d(train_ids, val_ids).size
    assert len(train) + len(val) == np.count_nonzero(gt)
    return train, val


class Patches(Dataset):
    def __init__(self, cube, centers=None, labels=None):
        self.width = cube.shape[1]
        self.count = cube.shape[0] * cube.shape[1] if centers is None else len(centers)
        self.cube = np.pad(cube, ((3, 3), (3, 3), (0, 0)), mode="constant")
        self.centers, self.labels = centers, labels

    def __len__(self):
        return self.count

    def __getitem__(self, i):
        row, col = divmod(i, self.width) if self.centers is None else self.centers[i]
        x = torch.from_numpy(self.cube[row:row+7, col:col+7].transpose(2, 0, 1).copy())
        return (x, int(self.labels[i])) if self.labels is not None else x


def loaders(source, target, gt, seed):
    train, val = source_split(gt, seed)
    ty = gt[train[:, 0], train[:, 1]].astype(np.int64) - 1
    vy = gt[val[:, 0], val[:, 1]].astype(np.int64) - 1
    def make(ds, shuffle, drop, offset):
        return DataLoader(ds, batch_size=32, shuffle=shuffle, drop_last=drop,
                          num_workers=0, generator=torch.Generator().manual_seed(seed+offset))
    return (make(Patches(source, train, ty), True, True, 101),
            make(Patches(target), True, True, 202),
            make(Patches(source, val, vy), False, False, 303),
            dict(train_centers=train, train_labels=ty, val_centers=val, val_labels=vy))
