#!/usr/bin/env python3
"""Regenerate pca_sklearn_v1.json, the sklearn.decomposition.PCA oracle for aprender's PCA (#3148).

    uv run --with scikit-learn --with numpy python3 crates/aprender-core/tests/fixtures/gen_pca_sklearn_v1.py

Every X is rounded to float32 (aprender's PCA takes Matrix<f32>) and handed to sklearn
as float64, so both sides factor the same numbers. sklearn's components are already
sign-flipped by svd_flip(u_based_decision=False), the rule aprender applies.
`vectors` is how many leading components are well enough separated to be compared
element-wise; explained-variance ratios are compared for all of them.
"""
import json
import os

import numpy as np
import sklearn
from sklearn.decomposition import PCA

rng = np.random.default_rng(3148)


def orthonormal(n, k):
    q, r = np.linalg.qr(rng.standard_normal((n, k)))
    return q * np.sign(np.diag(r))


def scaled_gaussian(n, d):
    return rng.standard_normal((n, d)) * np.geomspace(10.0, 0.1, d)


def ill_conditioned(n, d, rank):
    # zero-mean left factor, so centering does not disturb the designed spectrum
    u = rng.standard_normal((n, rank))
    u -= u.mean(axis=0)
    u, _ = np.linalg.qr(u)
    v = orthonormal(d, rank)
    return u @ np.diag(10.0 ** -np.arange(rank)) @ v.T


def case(name, x, n_components, vectors):
    x = x.astype(np.float32).astype(np.float64)
    p = PCA(n_components=n_components, svd_solver="full").fit(x)
    return {
        "name": name,
        "rows": x.shape[0],
        "cols": x.shape[1],
        "x": x.ravel().tolist(),
        "n_components": n_components,
        "k": int(p.n_components_),
        "vectors": min(vectors, int(p.n_components_)),
        "mean": p.mean_.tolist(),
        "components": p.components_.ravel().tolist(),
        "explained_variance": p.explained_variance_.tolist(),
        "explained_variance_ratio": p.explained_variance_ratio_.tolist(),
        "singular_values": p.singular_values_.tolist(),
    }


cases = [
    case("gauss_50x8_k8", scaled_gaussian(50, 8), 8, 8),
    case("gauss_50x8_k3", scaled_gaussian(50, 8), 3, 3),
    case("gauss_60x10_ratio_0.9", scaled_gaussian(60, 10), 0.9, 10),
    case("wide_30x60_k5", scaled_gaussian(30, 60), 5, 5),
    case("illcond_200x12_k10", ill_conditioned(200, 12, 10), 10, 7),
]
out = {
    "generator": "gen_pca_sklearn_v1.py",
    "sklearn": sklearn.__version__,
    "numpy": np.__version__,
    "cases": cases,
}
path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "pca_sklearn_v1.json")
with open(path, "w") as f:
    json.dump(out, f, indent=0)
    f.write("\n")
print(path, sklearn.__version__, [c["k"] for c in cases])
