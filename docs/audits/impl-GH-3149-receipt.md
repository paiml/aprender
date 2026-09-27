---
status: complete
ticket: GH-3149
github_issue: 3149
part: "E9 PR B on origin/main, branch 6b/e9b-neighbor-index. Cop ruling 2026-09-27: E9 lands on main as 3 PRs, B = #3149."
kind: code
model: claude-opus-5-5 (author of the fold commits; the feature commit is cherry-picked from the E9 branch)
---
# implementation receipt: GH-3149, exact kd/ball/brute neighbor index

## READ FIRST

| commit | change |
|---|---|
| 55e3c7e63f | `crates/aprender-core/src/neighbors/`: `KdTree` (axis-aligned median splits), `BallTree`, `BruteForce`, metrics, `KBest`; KNN, DBSCAN and LOF query through the index (`classification/gaussian_nb.rs` holds KNN's `neighbor_index()`, `cluster/dbscan.rs` → `within_radius`, `cluster/lof.rs` → `k_nearest`); `contracts/neighbor-index-v1.yaml`; Kani harnesses KANI-NBR-001..004 |
| 5718fdf462 | neighbor-index-v1 obligations restated in Σ symbols. The fold onto stage had raised formal_prose from 1464 to 1471; it is back to 1464 (PV-ONT-004) |
| 7a38771f94 | census/contracts.nt/README regenerated |
| d142a6d053 | roadmap fragment GH-3149 |

An earlier quorum of this change (on the E9 branch, round by aprender-5d) returned 3/3 PASS. This round
judges the cherry-pick onto main.

## Acceptance, criterion by criterion

| criterion | state | proof |
|---|---|---|
| Exactness proptest, n 10..2000, d 1..30, identical k-nearest incl. ties | MET | `neighbors/tests.rs::falsify_nbr_001_knn_trees_equal_brute` (n in 10..2000, d in 1..30, k in 1..40; 96 cases). (distance, index) must match bit for bit. Ties come from integer-grid data and from queries on data points |
| Radius queries agree with brute force | MET | `falsify_nbr_002_radius_trees_equal_brute`. The radius is set to an existing neighbour distance, so the inclusive boundary is hit |
| Sub-linear visits across n = 1e3, 1e4, 1e5 (not wall-clock) | MET | `falsify_nbr_003_node_visits_sublinear` (not `#[ignore]`). It counts distance evaluations: e(1e5)/e(1e3) < 10, e(1e4) ≤ e(1e5), e(1e5) < 1000. Its control is brute force, which must grow more than 99×. The run is d=3, k=5, one seed |
| Degenerate inputs | MET | `falsify_nbr_004_degenerate_inputs`: n=0, n=1, k=0, k>n, 500 identical points (lowest index wins), duplicates at d=1 vs brute force, d=0. Every algorithm is covered |
| Metric axioms per metric | MET | `falsify_nbr_005_metric_axioms`: Euclidean, Manhattan, Chebyshev, Minkowski(1.5), Minkowski(3); cosine gets symmetry and self-distance only. Non-metrics are refused by the trees (`falsify_nbr_006_non_metric_distances_refuse_trees`) |
| DBSCAN / LOF / KNN identical before and after the rewire | **MET with one disclosed exception** | `tests_rewire.rs::falsify_nbr_007_rewire_changes_no_output`: 180 FNV digests pinned from origin/main a016cee94 before the rewire. 153 are bit-identical. The 27 distance-weighted KNN `predict_proba` rows are held to 1e-6 against an in-test legacy reference, which must itself reproduce the pinned digest. The old code summed 1/d in an unspecified partition order, so bit-identity there was never a property of the old code either (NBR-REWIRE-007). **The quorum should rule on whether it accepts this** |
| Contract YAML with the exactness equation | MET | `contracts/neighbor-index-v1.yaml`, NBR-EXACT-001 (tree ≡ brute, bitwise); `pv validate` rc 0 |
| Mutation: perturb the kd splitting plane ⇒ exactness RED | MET (re-measured here) | see below. The author's commit also records the prune `>=`→`>`, ball radius 0.9r and dropped tie-break mutants as RED |

### Mutation, measured 2026-09-27 on a detached worktree at 7a38771f94

The runner is `/mnt/nvme-raid0/tmp/6b-e9bmut/run.sh`, which runs `cargo test -p aprender-core --lib neighbors::tests::falsify_nbr_00`. The mutation is in `kd_tree.rs`, where the split `value` becomes the data coordinate + 0.5:

```
M0 unmutated                              rc 0    6 passed
M1 kd split value = coord + 0.5           rc 101  FAILED: falsify_nbr_001 (knn), falsify_nbr_002 (radius),
                                                  falsify_nbr_004 (degenerate, tests.rs:205)
```

## Non-test `expect`

`cluster/lof.rs`: `.expect("euclidean index over a well-formed matrix")`. It cannot fail, because Euclidean under
`Auto` is never refused, and the comment at the site gives that argument. No `unwrap()`, TODO or `panic!` was added.

## Measured at 7a38771f94 (private target `/mnt/nvme-raid0/cargo-targets/6b-e9main`)

```
cargo fmt --all -- --check                                   rc 0
pv lint (armed meet)                                         PASS (formal_prose 1464, baseline 1464)
pv extract contracts --check                                 rc 0
scripts/readme_sync.sh --check                               rc 0
pv validate contracts/neighbor-index-v1.yaml                 rc 0
cargo test -p aprender-core --lib neighbors                  rc 0  (29 passed, 1 ignored)
cargo test -p aprender-core --lib cluster                    rc 0  (238 passed)
cargo test -p aprender-core --lib gaussian_nb                rc 0  (31 passed)
cargo clippy -p aprender-core --lib --tests -- -D warnings   rc 0
cargo test -p aprender-core --test readme_contract           rc 0  (16 passed)
```
