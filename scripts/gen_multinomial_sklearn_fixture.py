#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "scikit-learn==1.9.0",
#     "numpy==2.3.5",
# ]
# ///
"""Generate the frozen scikit-learn reference fixture for `multinomial-head-v1`.

Run it with no setup at all:

    uv run scripts/gen_multinomial_sklearn_fixture.py

The PEP 723 header above is the whole environment specification. There is no venv
recipe to remember and no ambient install to hope for: `uv` resolves and caches
exactly the pinned versions from this file. That is deliberate — a reference fixture
whose provenance lives in someone's shell history is not a reference.

WHAT THIS PINS
--------------
The contracted relation between scikit-learn's inverse regularization strength `C`
and aprender's native `lambda`:

    sklearn:   (1/n) * sum_i NLL_i  +  (1/(n*C)) * (1/2) * ||W||_F^2
    aprender:  (1/n) * sum_i NLL_i  +  lambda    * ||W||_F^2
    =>         lambda = 1 / (2 * C * n)

The half lives INSIDE sklearn's r(W) = (1/2)||W||_F^2. Dropping it yields
lambda = 1/(C*n), which is exactly twice too much regularization. At n=24 that error
is far outside any usable tolerance, which is what makes it falsifiable.

`n` is the number of ROWS. Not pairs, not batches.

WHY THE CONVERGENCE ASSERT MATTERS
----------------------------------
scikit-learn does not fail on non-convergence — it emits a ConvergenceWarning and
returns the coefficients it happened to reach. Freezing those as "the reference" would
pin an arbitrary point on the optimizer's path. This script therefore treats
`n_iter_ >= max_iter` as a hard abort.
"""

from __future__ import annotations

import json
import sys

import numpy as np
import sklearn
from sklearn.linear_model import LogisticRegression

# --- Frozen experiment definition ---------------------------------------------
N_ROWS = 24  # 8-shot x 3 classes
N_FEATURES = 4
N_CLASSES = 3
C = 1.0

# Tolerance handed to scikit-learn. See the SOLVER TOLERANCE note in the plan's
# SUMMARY: this is deliberately tighter than sklearn's 1e-4 default so that the
# fixture pins the OPTIMUM of the contracted objective rather than wherever two
# different stopping rules happened to halt. Loosening the Rust-side assertion
# instead would have made the comparison a test of stopping rules.
TOL = 1e-10
MAX_ITER = 5000


def design_matrix() -> np.ndarray:
    """Deterministic design matrix from a closed-form integer fill.

    ``X[i][j] = ((7*i + 5*j) mod 25) / 8 - 1.5``, giving values in [-1.5, 1.5].

    No RNG is involved anywhere, seeded or otherwise: a fixture that depends on a
    generator depends on that generator's version.

    Two properties of the constants matter, and both were chosen after measuring:

    * **The divisor is a power of two.** Every value is then an eighth — exactly
      representable in BOTH f64 and f32. aprender's head takes f32 features, so a fill
      of thirds would have meant sklearn and aprender fitting subtly different data,
      with the comparison silently absorbing a conversion error it was never meant to
      test.
    * **The modulus 25 exceeds the row count, and 7 is coprime to it.** ``7*i mod 25``
      therefore takes 25 distinct values, so all 24 rows are distinct. A first draft
      used ``mod 13`` and silently repeated rows 0..10 as rows 13..23 — the same
      feature vector carrying two different labels, which is a noisier and weaker
      reference than the one this fixture claims to be.
    """
    return np.array(
        [
            [((7 * i + 5 * j) % 25) / 8.0 - 1.5 for j in range(N_FEATURES)]
            for i in range(N_ROWS)
        ],
        dtype=np.float64,
    )


def labels() -> np.ndarray:
    """Round-robin class assignment, so every class has exactly 8 rows."""
    return np.array([i % N_CLASSES for i in range(N_ROWS)], dtype=np.int64)


def max_abs_gradient(
    w: np.ndarray, b: np.ndarray, x: np.ndarray, y: np.ndarray, lam: float
) -> float:
    """Largest absolute entry of APRENDER's analytic gradient at ``(w, b)``.

    Mirrors ``SoftmaxNllProblem::gradient``:

        dJ/dW[k][j] = (1/n) sum_i (p_ik - [k == y_i]) x_ij + 2*lambda*W[k][j]
        dJ/db[k]    = (1/n) sum_i (p_ik - [k == y_i])

    with the same max-subtraction shift inside the softmax.
    """
    n = x.shape[0]
    z = x @ w.T + b
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    p = e / e.sum(axis=1, keepdims=True)
    onehot = np.zeros_like(p)
    onehot[np.arange(n), y] = 1.0
    resid = (p - onehot) / n
    grad_w = resid.T @ x + 2.0 * lam * w
    grad_b = resid.sum(axis=0)
    return float(max(np.abs(grad_w).max(), np.abs(grad_b).max()))


def rust_lit(v: float) -> str:
    """Shortest round-tripping decimal, as a valid Rust f64 literal.

    ``repr`` on a numpy scalar renders ``np.float64(...)``, which is not Rust; the
    explicit ``float()`` is what keeps the emitted constants compilable.
    """
    text = repr(float(v))
    return text if ("." in text or "e" in text or "E" in text) else text + ".0"


def rust_array_2d(name: str, values: np.ndarray, rows: int, cols: int) -> str:
    lines = [f"pub const {name}: [[f64; {cols}]; {rows}] = ["]
    for r in range(rows):
        row = ", ".join(rust_lit(values[r][c]) for c in range(cols))
        lines.append(f"    [{row}],")
    lines.append("];")
    return "\n".join(lines)


def rust_array_1d(name: str, values: np.ndarray) -> str:
    joined = ", ".join(rust_lit(v) for v in values)
    return f"pub const {name}: [f64; {len(values)}] = [{joined}];"


def main() -> int:
    x = design_matrix()
    y = labels()

    if sorted(np.bincount(y).tolist()) != [8, 8, 8]:
        print("ABORT: class counts are not 8/8/8", file=sys.stderr)
        return 1

    if len({tuple(row) for row in x.tolist()}) != N_ROWS:
        print(
            "ABORT: the design matrix contains duplicate rows. Pick a modulus larger "
            "than N_ROWS that is coprime to the row coefficient.",
            file=sys.stderr,
        )
        return 1

    # NOTE: `multi_class` is NOT passed. It was deprecated in scikit-learn 1.5 and
    # removed in 1.7; with the lbfgs solver and K > 2 the fit is multinomial anyway.
    model = LogisticRegression(C=C, tol=TOL, max_iter=MAX_ITER)
    model.fit(x, y)

    # Pitfall 7: a non-converged reference is not a reference.
    n_iter = int(np.max(model.n_iter_))
    if n_iter >= MAX_ITER:
        print(
            f"ABORT: sklearn did not converge (n_iter_={n_iter} >= max_iter={MAX_ITER}). "
            "The coefficients below would be an arbitrary point on the optimizer path, "
            "not the optimum of the contracted objective.",
            file=sys.stderr,
        )
        return 1

    proba = model.predict_proba(x)
    lam = 1.0 / (2.0 * C * N_ROWS)
    lam_wrong = 1.0 / (C * N_ROWS)

    # PROVE THE CONVENTION, do not assume it — on the reference side, before any Rust
    # is involved. sklearn 1.9 reports `penalty = "deprecated"` in get_params(), which
    # reads alarmingly like "no penalty at all", and a comparison of two norms is only
    # a hint. So instead: evaluate APRENDER's analytic gradient of
    #
    #     (1/n) sum_i NLL_i + lambda * ||W||_F^2      (intercept unpenalized)
    #
    # at sklearn's own converged solution. If aprender's objective is the same function
    # sklearn minimized, sklearn's optimum is a stationary point of it and the gradient
    # is ~0. If the factor of 2 is dropped, it is not.
    grad_ok = max_abs_gradient(model.coef_, model.intercept_, x, y, lam)
    grad_bad = max_abs_gradient(model.coef_, model.intercept_, x, y, lam_wrong)
    if not grad_ok < 1.0e-6:
        print(
            f"ABORT: sklearn's optimum is NOT a stationary point of aprender's "
            f"objective at lambda = 1/(2*C*n) = {lam!r}; max|grad| = {grad_ok!r}. "
            "The two objective conventions have diverged — re-derive the relation "
            "before freezing anything.",
            file=sys.stderr,
        )
        return 1
    if not grad_bad > 1.0e-3:
        print(
            f"ABORT: the factor-2 error is NOT separable on this data. "
            f"max|grad| at the WRONG lambda = 1/(C*n) = {lam_wrong!r} is "
            f"{grad_bad!r}, which is too close to zero for the falsification to bite.",
            file=sys.stderr,
        )
        return 1

    # Versions are read from the RUNNING interpreter, never echoed from the PEP 723
    # header above. If the resolved environment ever drifts from the pin, this output
    # shows the drift instead of hiding it.
    print("// ==========================================================================")
    print("// GENERATED by scripts/gen_multinomial_sklearn_fixture.py — DO NOT HAND-EDIT")
    print("//")
    print(f"// python       : {sys.version.split()[0]}")
    print(f"// scikit-learn : {sklearn.__version__}")
    print(f"// numpy        : {np.__version__}")
    print(f"// n_iter_      : {n_iter} (max_iter={MAX_ITER}) — converged")
    print("//")
    print("// get_params():")
    for k, v in sorted(model.get_params().items()):
        print(f"//   {k} = {json.dumps(v)}")
    print("//")
    print(f"// n = {N_ROWS} rows, d = {N_FEATURES}, K = {N_CLASSES}, C = {C!r}")
    print(f"// contracted lambda = 1/(2*C*n) = {lam!r}")
    print(f"// the factor-2 ERROR would be 1/(C*n) = {lam_wrong!r}")
    print("//")
    print("// Convention proof — aprender's analytic gradient AT sklearn's optimum:")
    print(f"//   max|grad| at lambda = 1/(2*C*n) : {grad_ok!r}   (stationary)")
    print(f"//   max|grad| at lambda = 1/(C*n)   : {grad_bad!r}   (NOT stationary)")
    print("// ==========================================================================")
    print()
    print(rust_array_2d("SKLEARN_X", x, N_ROWS, N_FEATURES))
    print(f"pub const SKLEARN_Y: [usize; {N_ROWS}] = [{', '.join(str(v) for v in y)}];")
    print()
    print(rust_array_2d("SKLEARN_COEF", model.coef_, N_CLASSES, N_FEATURES))
    print(rust_array_1d("SKLEARN_INTERCEPT", model.intercept_))
    print()
    print(rust_array_2d("SKLEARN_PROBA", proba, N_ROWS, N_CLASSES))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
