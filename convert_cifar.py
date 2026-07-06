"""Converts the HuggingFace cifar10 parquet files into cifar_data.npz (uint8, classic CIFAR layout)."""
import numpy as np, io
import pyarrow.parquet as pq
from PIL import Image

def to_arrays(path):
    t = pq.read_table(path)
    d = t.to_pydict()
    imgs = d['img']; labels = np.array(d['label'], dtype=np.int64)
    X = np.zeros((len(imgs), 3072), dtype=np.uint8)
    for i, im in enumerate(imgs):
        a = np.array(Image.open(io.BytesIO(im['bytes'])))  # 32x32x3 RGB
        X[i] = a.transpose(2, 0, 1).reshape(-1)            # R|G|B planes, original CIFAR layout
    return X, labels

Xtr, ytr = to_arrays('hf_train.parquet')
Xte, yte = to_arrays('hf_test.parquet')
rng = np.random.default_rng(0)
p = rng.permutation(len(Xtr))
Xtr, ytr = Xtr[p], ytr[p]
np.savez_compressed('cifar_data.npz', Xtr=Xtr, ytr=ytr, Xte=Xte, yte=yte)
print("train:", Xtr.shape, "test:", Xte.shape, "classes:", np.bincount(ytr[:20000]))
print("CONVERT_OK")
