#!/usr/bin/env python3
"""Regenerate svd_scipy_v1.json, the scipy.linalg.svd oracle for trueno::Svd (#3147).

    python3 crates/aprender-compute/tests/fixtures/gen_svd_scipy_v1.py

Each case stores A (row-major), scipy's thin SVD with trueno's sign rule applied
(the largest-magnitude entry of each U column is positive, first on a tie), and
`vectors`: how many leading triplets are well separated enough (relative gap
>= 1e-3 on both sides, sigma above the rank tolerance) for their singular
vectors to be unique. Singular values are compared for every triplet.
"""
import json
import os
import sys

import numpy as np
import scipy
import scipy.linalg

rng = np.random.default_rng(3147)


def orthogonal(n):
    q, r = np.linalg.qr(rng.standard_normal((n, n)))
    return q * np.sign(np.diag(r))


def designed(m, n, sigma):
    r = min(m, n)
    u = orthogonal(m)[:, :r]
    v = orthogonal(n)[:, :r]
    return u @ np.diag(sigma) @ v.T


def canonical(u, vt):
    for j in range(u.shape[1]):
        col = u[:, j]
        i = int(np.argmax(np.abs(col)))  # argmax returns the first maximum
        if col[i] < 0:
            u[:, j] = -col
            vt[j, :] = -vt[j, :]
    return u, vt


def separated(s, m, n):
    tol = max(m, n) * np.finfo(float).eps * s[0]
    count = 0
    for j, x in enumerate(s):
        gaps = [abs(x - s[i]) / s[0] for i in (j - 1, j + 1) if 0 <= i < len(s)]
        if x <= tol or min(gaps, default=1.0) < 1e-3:
            break
        count += 1
    return count


CASES = [
    ("gaussian_tall_6x4", rng.standard_normal((6, 4))),
    ("gaussian_wide_4x6", rng.standard_normal((4, 6))),
    ("gaussian_square_5x5", rng.standard_normal((5, 5))),
    ("gaussian_tall_9x3", rng.standard_normal((9, 3)) * 100.0),
    ("gaussian_wide_2x9", rng.standard_normal((2, 9))),
    ("designed_7x7_decay", designed(7, 7, [10, 5, 2, 1, 0.5, 0.25, 0.125])),
    ("designed_8x5_cond1e9", designed(8, 5, np.logspace(0, -9, 5))),
    ("rank2_6x5", rng.standard_normal((6, 2)) @ rng.standard_normal((2, 5))),
    ("row_vector_1x5", rng.standard_normal((1, 5))),
    ("column_vector_5x1", rng.standard_normal((5, 1))),
]


def main():
    out = {
        "generator": "crates/aprender-compute/tests/fixtures/gen_svd_scipy_v1.py",
        "oracle": f"scipy.linalg.svd {scipy.__version__} (lapack_driver=gesdd, full_matrices=False), numpy {np.__version__}",
        "cases": [],
    }
    for name, a in CASES:
        m, n = a.shape
        u, s, vt = scipy.linalg.svd(a, full_matrices=False)
        u, vt = canonical(u.copy(), vt.copy())
        out["cases"].append({
            "name": name, "rows": m, "cols": n,
            "a": a.ravel().tolist(), "s": s.tolist(),
            "u": u.ravel().tolist(), "vt": vt.ravel().tolist(),
            "vectors": separated(s, m, n),
        })
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "svd_scipy_v1.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=1)
        f.write("\n")
    print(f"wrote {len(out['cases'])} cases to {path}", file=sys.stderr)


if __name__ == "__main__":
    main()
