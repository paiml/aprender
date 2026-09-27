---
status: complete
ticket: GH-3147
github_issue: 3147
part: "E9 PR A (with GH-3148) on origin/main @aca6f2d7f6, branch 6b/e9a-svd-pca. Cop ruling 2026-09-27: E9 lands on main as 3 PRs, A = #3147+#3148."
kind: code
model: claude-opus-5-5 (author of the fold commits; the feature commits are cherry-picked from the E9 branch)
---
# implementation receipt: GH-3147, dense f64 SVD substrate

## READ FIRST

This PR folds two tickets, #3147 (SVD) and #3148 (PCA on SVD). #3148's receipt is
`impl-GH-3148-receipt.md`. This file covers the SVD commits only.

| commit | change |
|---|---|
| 34aa3c32e5 | `crates/aprender-compute/src/svd/`: Golub-Reinsch thin SVD (Householder bidiagonalization, then implicit-shift QR), randomized top-k (Halko p/q), sign canonicalization, `rank_tolerance()`; `contracts/svd-v1.yaml`; scipy fixture `tests/fixtures/svd_scipy_v1.json` |
| 6d1d565460 | svd-v1 restated in Σ symbols, plus a `valid_under` block, to pass PV-ONT-004 and the valid-under gate that landed after the branch was cut |
| 96ee11dfa4 | regenerated `contracts/census.json`, `contracts.nt` and README (shared with #3148) |

## Acceptance, criterion by criterion

| criterion | state | proof |
|---|---|---|
| Reconstruction ≤ 1e-10 on 200 matrices, all shape classes incl. cond ≥ 1e8 | MET | `svd/tests.rs::falsify_svd_001_reconstruction_over_200_matrices` (40 each: m<n, m=n, m>n, rank-deficient, cond ≥ 1e8, the condition measured with nalgebra), `acceptance_set_covers_every_shape_class` |
| Orthonormality proptest ≤ 1e-10 | MET | `falsify_svd_002_003_orthonormal_and_sorted`. Shapes go up to 9×9 only |
| σ ≥ 0 and descending, asserted | MET | `assert_well_formed`, run on every matrix |
| scipy parity 1e-9 | **PARTIAL** | `falsify_svd_004_scipy_oracle_parity`. The error is scaled by σ₀ (a norm-wise bound), not by each σ_j. Vectors are compared up to sign only on separated triplets. This is disclosed rather than tightened: a per-value relative bound on σ near 0 compares rounding noise |
| Randomized p=10, q=2 within 1% | **PARTIAL** | `falsify_svd_006_randomized_top_k_within_one_percent` asserts ≤ 1%, but on decaying spectra, under the precondition σ_{k+p+1}/σ_k ≤ 0.5, not on "rank-k plus noise". Outside that precondition it measured 1.8% (harmonic, k=20); recorded as finding FND-20260926-3147 |
| Rank-deficient: exact zeros or a documented tolerance | MET (tolerance) | `falsify_svd_005_rank_deficient_values_are_within_tolerance`. The tail is bounded by `max(m,n)·ε·σ₀` (`svd/mod.rs` doc, `rank_tolerance()`). Exact 0.0 is asserted only for the all-zero matrix |
| KernelContract YAML, `pv validate` | MET | `contracts/svd-v1.yaml` (SVD-INV-001 reconstruction, SVD-INV-002 orthonormality); `pv validate contracts/svd-v1.yaml` rc 0 |
| Mutation: revert bidiagonalization ⇒ reconstruction RED | MET (measured here) | see below |

### Mutation, measured 2026-09-27 on a detached worktree at 96ee11dfa4

The runner is `/mnt/nvme-raid0/tmp/6b-e9mut/run.sh`. The target is `svd::tests::falsify_svd_001`, with the mutation applied to `svd/golub_reinsch.rs::bidiagonalize`:

```
M0 unmutated                                        rc 0    1 passed
M1 apply_left_reflector(k, j) call removed          rc 101  panicked at svd/tests.rs:43 (reconstruction assert)
M2 right_reflector(k, &mut work) call removed       rc 101  panicked at svd/tests.rs:43
worktree after restore: git status --porcelain empty
```

## Measured at 96ee11dfa4 (private target `/mnt/nvme-raid0/cargo-targets/6b-e9main`)

```
cargo fmt --all -- --check                                   rc 0
pv lint (armed meet)                                         PASS (formal_prose 1464, baseline 1464)
pv extract contracts --check                                 rc 0
scripts/readme_sync.sh --check                               rc 0
cargo test -p aprender-compute --lib svd                     rc 0  (15 passed)
cargo clippy -p aprender-compute --lib --tests -- -D warnings rc 0
pv validate contracts/svd-v1.yaml                            rc 0
```
