"""Generate a pinned shared ILDA input, using verbatim official functions."""
from pathlib import Path
import json
import numpy as np
import hdf5storage
import sklearn, cv2, skimage
from official_preprocessing import ILDA
from data import DEFAULT_DATA, file_hash
from runtime import atomic_json

root=Path(__file__).resolve().parent
out=root/"preprocessing"
out.mkdir(exist_ok=False)
np.random.seed(1341)
s=hdf5storage.loadmat(str(DEFAULT_DATA/"Houston13.mat"))["ori_data"]
t=hdf5storage.loadmat(str(DEFAULT_DATA/"Houston18.mat"))["ori_data"]
s,t=ILDA(s,t,2,0.009)
assert np.isfinite(s).all() and np.isfinite(t).all()
np.savez(out/"official_ilda.npz",s=s,t=t)
atomic_json(out/"manifest.json", dict(
    cache_sha256=file_hash(out/"official_ilda.npz"), preprocessing_seed=1341,
    pca_n=2, epsilon=0.009, guided_filter_radius=1,
    note="Official applies ILDA before seeding. We pin its random state for reproducibility.",
    sklearn=sklearn.__version__, cv2=cv2.__version__, skimage=skimage.__version__,
    official_code={p.name:file_hash(p) for p in (root/"reference").glob("*.py")},
    inputs={n:file_hash(DEFAULT_DATA/n) for n in ["Houston13.mat","Houston18.mat"]}))
print("ILDA cache ready",flush=True)
