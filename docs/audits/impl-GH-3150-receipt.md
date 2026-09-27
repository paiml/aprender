---
status: complete
ticket: GH-3150
github_issue: 3150
part: "E9 PR C on origin/main, branch 6b/e9c-ranked-tensor. Cop ruling 2026-09-27: E9 lands on main as 3 PRs, C = #3150."
kind: code
model: claude-opus-5-5 (author)
---
# implementation receipt: GH-3150, rank-typed tensors (`RankedTensor<D, L>`)

## READ FIRST

| commit | change |
|---|---|
| ae75729db9 | `crates/aprender-tensor/src/ranked.rs`: `RankedTensor<const D, L: Layout = RowMajor>` with a sealed `Layout` (RowMajor/ColMajor); `Matrix = RankedTensor<2>`; `new`/`zeros`/`reshape<D2>`/`get`; `from_dynamic`/`into_dynamic` (RowMajor only); `from_gguf`/`into_apr` (zero-copy `[ne0,ne1]` → `[ne1,ne0]`)/`to_row_major` (ColMajor rank 2 only); `Matrix::matmul`/`t`; `unsqueeze`/`squeeze` for ranks 0..=6. The dynamic `matmul` (the migrated hot path) now runs on `Matrix::matmul` instead of `einsum("ij,jk->ik")`. `tests/rank_typed_ui.rs` with 6 compile-fail cases |
| 59aada12f8 | `ci/explicit-test-commands.d/494-aprender-tensor-rank-typed-ui.cmd`: the trybuild target is wired into CI. Without it, a new `tests/*.rs` target is dark |
| cc20427c8e | closes the acceptance gaps below: `add`, 6 more compile-fail cases (12 total), a pass-suite, the `rank_typed_matmul` bench, and the CLAUDE.md line |

## Acceptance, criterion by criterion

| criterion | state | proof |
|---|---|---|
| ≥ 8 compile-fail cases incl. 2D-matmul on 3D, reshape to wrong rank, add of different rank, ColMajor where RowMajor required; stderr committed | MET | 12 in `tests/ui/`: `matmul_needs_rank_2`, `reshape_rank_is_typed`, `add_different_rank` (`expected 2, found 3`), `matmul_needs_row_major`, `matmul_col_major_argument`, `add_different_layout` (`expected RankedTensor<2, RowMajor>, found RankedTensor<2, ColMajor>`), `rank_mismatch_argument`, `squeeze_on_rank_0`, `unsqueeze_past_rank_6`, `into_apr_on_row_major`, `from_dynamic_col_major`, `layout_is_sealed`. Each `.stderr` was read and names the intended error, not an unrelated one |
| Matching pass-suite | MET | `tests/ui-pass/typed_pipeline.rs`: GGUF import → `into_apr` → `matmul` → `add` → `unsqueeze`/`squeeze` → `t` → `reshape` → dynamic round trip. It compiles AND runs (`t.pass`) |
| Zero runtime cost for the migrated path | MET, with one stated cost | see "Benchmark" below. The typed kernel does no rank checks at run time; its only check is the inner-dimension extent, which the untyped path also does. The migrated dynamic `matmul` keeps its old signature, so it pays one `from_dynamic` copy per operand at the boundary. That cost is measured and shown, not hidden |
| Byte-identical output before/after on the migrated path | MET (integer-valued inputs) | `tests_ranked.rs::matmul_matches_the_einsum_route` (proptest, m/k/n in 0..7, including zero-sized dims): typed `matmul` == `einsum("ij,jk->ik")` bit for bit. The inputs are half-integers, so every partial sum is exact; on arbitrary floats the two routes may round differently, because the summation order is not pinned by the old code |
| CLAUDE.md FORBIDDEN IMPORTS line + readme_contract passes | MET | CLAUDE.md, after the FORBIDDEN list: the ColMajor → `matmul` misuse is now unrepresentable in `aprender-tensor`. It also says the trueno `*_colmajor` imports are still reachable and still forbidden, because the type covers only this crate. `cargo test -p aprender-core --test readme_contract`: rc 0, 16 passed (it checks every path CLAUDE.md cites) |
| Mutation-verified in the new scope: a rank error in the migrated path ⇒ build RED | MET (measured here) | see below |

### Mutation, measured 2026-09-27 on a detached worktree at cc20427c8e

The runner is `/mnt/nvme-raid0/tmp/6b-e9cmut/run.sh`, which runs `cargo check -p aprender-tensor`. The mutations are in `src/einsum.rs::matmul`:

```
M0 unmutated                                                        rc 0
M1 lhs read as RankedTensor::<3>::from_dynamic(a)                   rc 101  E0599 no method `matmul` for RankedTensor<3>
M2 rhs built as RankedTensor::<2, ColMajor>::from_gguf(..)          rc 101  E0308 expected RankedTensor<_, RowMajor>, found RankedTensor<_, ColMajor>
worktree after restore: git status --porcelain empty
```

The `.stderr` pins also went RED once while the suite was built: a new case has no `.stderr` until
`TRYBUILD=overwrite`, and trybuild fails it.

### Benchmark (`cargo bench -p aprender-tensor --bench tensor_bench -- rank_typed_matmul`)

The run used `--warm-up-time 1 --measurement-time 3` on shared host lambda at load average about 30 to 40. Treat the numbers as ratios, not absolutes:

| n×n | einsum route (old) | dynamic `matmul` (migrated) | `Matrix::matmul` (typed) |
|---|---|---|---|
| 16 | 471 µs | 0.622 µs | 0.548 µs |
| 64 | 30.5 ms | 20.4 µs | 21.9 µs |
| 256 | 2.78 s | 1.147 ms | 1.067 ms |

The migrated path is roughly 750× to 2,400× faster than the einsum route it replaced. Across all three sizes it stays within about 14% of the typed kernel; at 16 and 256 that gap is the `from_dynamic` copy, and at 64 the two overlap within noise. Callers that hold a `Matrix` pay nothing.

## Measured at cc20427c8e (private target `/mnt/nvme-raid0/cargo-targets/6b-e9main`)

```
cargo fmt --all -- --check                                                   rc 0
cargo test -p aprender-tensor                                                rc 0  (48 lib, 1 trybuild target: 12 compile-fail + 1 pass)
cargo clippy -p aprender-tensor --lib --tests --benches -- -D warnings       rc 0
cargo test -p aprender-core --test readme_contract                           rc 0  (16 passed)
```
