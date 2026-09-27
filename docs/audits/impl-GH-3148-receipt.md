---
status: complete
ticket: GH-3148
github_issue: 3148
part: "E9 PR A (with GH-3147) on origin/main @aca6f2d7f6, branch 6b/e9a-svd-pca. Cop ruling 2026-09-27: E9 lands on main as 3 PRs, A = #3147+#3148."
kind: code
model: claude-opus-5-5 (author of the fold commits; the feature commit is cherry-picked from the E9 branch)
---
# implementation receipt: GH-3148, PCA on the SVD substrate, plus TruncatedSVD

## READ FIRST

PCA is built on #3147's SVD (`impl-GH-3147-receipt.md`), which is why the two ship as one PR.

| commit | change |
|---|---|
| e9006012ff | `preprocessing/pca.rs`: PCA fits by thin or randomized SVD of the centred data; the d×d covariance path is removed. Adds `TruncatedSVD`, `tests_pca_svd.rs`, the sklearn fixture `tests/fixtures/pca_sklearn_v1.json`, `contracts/pca-v1.yaml`, `contracts/beat-sklearn-pca-speed-v1.yaml`, `tests/beat_sklearn_pca_speed.rs`, the BEATS.md row, and **one additive step in `.github/workflows/beat-speed-nightly.yml`** (see below) |
| 4118c8f85d | PCA-INV sign convention restated in Σ symbols (PV-ONT-004: formal_prose stays at 1464) |
| 96ee11dfa4 | census/contracts.nt/README regenerated |
| (this commit) | BEATS.md "Why apr wins / loses" paragraph: PCA-SVD is removed from the named losses, which had contradicted the WON row |

### Workflow hunk: the cop must approve this

`.github/workflows/beat-speed-nightly.yml` gains one step, "Pillar-1 — apr vs scikit-learn PCA speed
(re-measurement, #3148)". It runs `cargo test -p aprender-core --release --test beat_sklearn_pca_speed
-- --ignored --nocapture` and appends to `$BEAT_LOG`. It mirrors the GMM step above it, and no other line of
the workflow changes. The two steps differ only in the step name and the test target; both open with
`set -o pipefail`, so `tee` cannot mask a failing test:

```
176  - name: Pillar-1 — apr vs scikit-learn GaussianMixture (GMM) speed beat      (existing, on main)
177    if: ${{ !cancelled() && steps.preflight.outcome == 'success' }}
178    run: |
179      set -o pipefail
180      cargo test -p aprender-core --release \
181        --test beat_sklearn_gmm_speed -- --ignored --nocapture 2>&1 | tee -a "$BEAT_LOG"
183  - name: Pillar-1 — apr vs scikit-learn PCA speed (re-measurement, #3148)     (added)
184    if: ${{ !cancelled() && steps.preflight.outcome == 'success' }}
185    run: |
186      set -o pipefail
187      cargo test -p aprender-core --release \
188        --test beat_sklearn_pca_speed -- --ignored --nocapture 2>&1 | tee -a "$BEAT_LOG"
```

A round-1 lane (gemini) FAILed this claim, saying the GMM step lacks `pipefail`. Line 179 above shows that it
does not lack it. The ticket needs it: "must be re-measured, not deleted". Without the step, the contract's
nightly measurement never runs.

## Acceptance, criterion by criterion

| criterion | state | proof |
|---|---|---|
| Covariance path gone: anchored guard plus case table | MET | `tests_pca_svd.rs::falsify_pca_svd_001_pca_source_forms_no_covariance_matrix` and `covariance_guard_case_table` (6 must-match, 8 must-not-match rows; comments are stripped first) |
| Ill-conditioned (σ spanning 1e0..1e-9): SVD matches to 1e-7 and the covariance path FAILS | MET | `falsify_pca_svd_002_…_where_covariance_fails`. It asserts `svd_err ≤ 1e-7` AND `cov_err > 1e-7`. The old route is kept in the test file only, as the foil |
| ratio sums to 1 within 1e-12 at full rank | MET | `falsify_pca_svd_003_ratio_sums_to_one_at_full_rank` (each solver, plus a wide matrix) |
| Two fits byte-identical | MET | `falsify_pca_svd_004_two_fits_are_byte_identical` (`to_bits`, each solver) |
| sklearn parity up to sign within 1e-8 | MET | `falsify_pca_svd_005_sklearn_parity` + fixture. Variance and ratio also match at 1e-10 relative |
| TruncatedSVD on TF-IDF beats random by an asserted margin | MET | `falsify_pca_svd_006_tfidf_lsa_nearest_neighbour_beats_random`: accuracy ≥ baseline + 0.5, baseline 19/59 |
| BEATS loss row retired and re-measured | MET | `beat-sklearn-pca-speed-v1.yaml`: new route tall 0.543, wide 0.279 (gate ≤ 0.80); old covariance route tall 0.498, wide 24.425. The row is WON. The measurement was taken on shared host lambda under load, so the nightly step is what keeps it honest |
| `readme_contract.rs` still passes | MET | see "Measured at" |

Honest note: the old route was already faster than sklearn on the TALL shape (0.498). The "loss" only ever
described wide data (d=1000), where d×d eigendecomposition dominates. The BEATS row says which shape it is.

## Measured at 96ee11dfa4 (private target `/mnt/nvme-raid0/cargo-targets/6b-e9main`)

```
cargo fmt --all -- --check                                        rc 0
pv lint (armed meet)                                              PASS (formal_prose 1464, baseline 1464)
pv extract contracts --check                                      rc 0
scripts/readme_sync.sh --check                                    rc 0
cargo test -p aprender-core --lib preprocessing                   rc 0
cargo test -p aprender-core --release --test beat_sklearn_pca_speed --no-run   rc 0 (the measurement is #[ignore], nightly)
cargo clippy -p aprender-core --lib --tests -- -D warnings        rc 0
pv validate contracts/{pca-v1,beat-sklearn-pca-speed-v1}.yaml  rc 0 / rc 0
cargo test -p aprender-core --test readme_contract                 rc 0  (16 passed)
```
