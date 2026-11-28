# Agent rules: history and design detail

Moved verbatim from CLAUDE.md (#7444). These are design notes and history, not rules. The rules they support stay in CLAUDE.md.

### FFN Gate+Up Kernel Fusion (PMAT-FFN-FUSION)

The SwiGLU FFN block fuses gate and up projections into a single rayon dispatch via
`generic_fused_gate_up_matvec_into<F>` (`crates/aprender-serve/src/quantize/fused_gate_up.rs:63`). This halves
rayon spawn overhead (56→28 dispatches/token on 28-layer models) and improves L1/L2 cache
reuse by loading the activation vector once per midi-tile instead of twice.

- **Fused path**: Q4K, Q5K, Q6K when both gate+up weights share the same qtype and dims
- **Fallback**: `rayon::join` with two separate `fused_matmul_into` for mixed types
- **Q8K path**: Existing `fused_q4k_q8k_ffn_up_gate_into` still used when Q8K activations available
- **Key files**: `crates/aprender-serve/src/quantize/fused_gate_up.rs`,
  `crates/aprender-serve/src/gguf/inference/fused_matmul_into.rs` (`fused_gate_up_matmul_into`)


## Project overview: repo merge (moved from CLAUDE.md, slice 2)

20 repos (5 core + 15 satellites, per `docs/specifications/aprender-monorepo-consolidation.md`)
were merged in and archived 20/20.

## Derive the numbers: drift history (moved from CLAUDE.md, slice 2)

Every count in this file drifted. The published figures were wrong by 3.2× on the
test count and 29 minor versions on the release line before they were checked. So

## Derive the numbers: drift gate (moved from CLAUDE.md, slice 2)

`crates/aprender-core/tests/readme_contract.rs` is the drift gate for all three published
docs: README.md's crate/contract counts, `docs/BEATS.md` vs the beat contracts, and every
repo-relative file path cited in **this file**. A path here that does not exist fails the
build (FALSIFY-DOCS-CLAUDE-001) — that is what caught the six pre-monorepo `realizar/…`
and `src/format/…` paths this file still advertised. Counts it cannot check, you re-derive.

## Testing: coverage history (moved from CLAUDE.md, slice 2)

Target: 60% unit, 30% property, 10% integration. Coverage: **90.84% line** (849706/935347, measured 2026-09-24 by a manual `workflow_dispatch` of coverage-nightly (run 36065904363) on branch `PMAT-3839-cov-regex-scope` @ 237f10f8b, folded into batch/0.70.0 and NOT on `main` until that batch merges; `main`'s own nightly (49fe19c28, 2026-09-24) still measured 826969/941649 = 87.82% under the old regex (run 35937679357); target ≥95%, enforced floor 89% via COV_FLOOR). Before #3839 the exclude regex still described the pre-monorepo tree, so the 88.78% of 2026-07-29 measured a different set of files. The long-quoted "96.35%" predates the measurement ever working - the pipeline reported 0/0 until #2333.

## Verification Discipline (full text before slice 2) (moved from CLAUDE.md, slice 2)

## Verification Discipline (MANDATORY — read before reporting any result)

These are not style notes. Each cost real time, and in every case the general
principle was known and the specific instance still went wrong.

**1. Never read `$?` through a pipe.** It is the LAST command's status.
```text
cmd > /tmp/out.log 2>&1; rc=$?               # correct
cmd | tee /tmp/out.log; rc=${PIPESTATUS[0]}  # correct (bash)
cmd | grep -E "^error"; echo "exit=$?"       # WRONG — that is grep's status
```
This shipped twice — `qwen-story-daily` captured `tee`'s status so its fail-the-job
step was unreachable and three green runs proved nothing (#2336); `make publish`'s
POST-PUBLISH VERIFICATION did the same, so it could never report a broken published
crate (#2360). **When a result looks good, check how it was measured.**

**2. Never label a run by intent — prove the mechanism engaged.** A repro harness
printed `device: GPU` while built without `--features cuda`; three findings were
reported from CPU runs. `CUDA_VISIBLE_DEVICES` says what is *visible*, never what was
*used*. Cite a trace line, a version+SHA, or a behaviour delta.

**3. Pin the binary.** Never a bare `apr`, never a hardcoded absolute path — see
Step 0 above. Four `apr` binaries once coexisted here and a bare `apr` resolved to a
26-day-old one.

**4. Extending a guard's SCOPE requires re-mutating in the new scope.** The old proof
does not transfer. Extending `check_apr_bin_pinned.sh` to the Makefile found real
violations *and* was still blind to `\t@apr …`, the most common Makefile form — caught
only by re-running the RED-turning mutation there (#2360).

**5. A guard that does not scan the surface where the DECISION is made is theater.**
Enumerate the decision surfaces — release, publish, certify, gate — then check
coverage. Covering "CI" is not covering "the release".

**6. One failing input is an anecdote.** Vary it before naming a cause, and
especially before blocking a release. Four neighbouring prompts once inverted a
diagnosis from "GPU correctness defect" to "the gate sampled a near-tie" (#2359).

**7. Guard regexes ship a case table.** The `apr`-invocation patterns were wrong five
times; every one was caught by a must-match/must-not-match table, none by review.
Re-run the table rather than re-reading the pattern.

**8. A shadowed artifact is worse than a missing one** — edits look effective and
change nothing. `~/.local/bin/apr` shadowed a fresh install; `~/.claude/skills/dogfood/`
shadowed the repo's release-certifying skill so hardening it edited a file that never
ran (#2361). When a fix seems to have no effect, ask what else claims that name
(`type -aP <cmd>`; user-scope vs repo-scope skills; always set an explicit `name:`).

## Shell: sourced library exception (moved from CLAUDE.md, slice 2)

**Exception — a SOURCED library must be option-neutral.** `set` in a sourced file
mutates the *caller's* shell. `scripts/apr_bin.sh` opened with `set -euo pipefail`;
`qwen-story.sh` sources it and had deliberately chosen no `-e` so it could run every
beat and tally failures. The leak killed the nightly six lines in. Sourceable
libraries fail by **return status** instead: `. scripts/lib.sh || exit 1`. Enforced by
`scripts/check_sourced_libs_option_neutral.sh`.

## Publishing: check_package_includes vacuity (moved from CLAUDE.md, slice 2)

**`check_package_includes.sh` is currently vacuous — know this before trusting it.**
It greps `include!(` in the root `src/` only, and the root `src/` is a two-file facade
(`src/lib.rs`, `src/bin/`) with **zero** `include!()` directives. So it reports
`OK: All 0 include!() files are included in cargo package` and can never fail. The
real coverage against CB-510 today comes from `check_include_files.sh` (1771 files,
`src/` + `crates/`). Extending the package check to the 70+ publishable member crates
is open work — do not treat its green as evidence.

## Performance targets narrative (moved from CLAUDE.md, slice 2)

These are **targets**, not measurements. For what is actually measured against Ollama,
see `docs/BEATS.md`: GPU decode on RTX 4090 sm_89 is at **parity** (1.015–1.109×), and
`contracts/beat-ollama-decode-throughput-speed-v1.yaml` enforces `beat_threshold: 0.9000`
— a no-collapse floor. The old "apr beats Ollama 1.371×" headline is **withdrawn**.

## Linting: toolchain ceiling narrative (moved from CLAUDE.md, slice 2)

Without the ceiling gate, findings from newer clippy releases accumulate invisibly:
#2370 was a fresh mbp whose homebrew rustc is not rustup-managed (so the pin is
silently inert) running plain `make` and getting 28 errors out of `tier2` — the tree
had 107 findings across 12 lints by then, none of which any gate had ever run.
The ceiling gate refuses to pass vacuously (stale `stable`, missing clippy component,
broken version comparator) and ships a comparator case table that `tier3` re-runs.

Clippy's lint set is **not monotonic**: the #2370 tree is clean on 1.93/1.96/1.97 and
1.95 *alone* reports 8 `collapsible_match` findings. A green ceiling gate means
"clean on the pin and on current stable", never "clean on every release between".

## Modules: v0.4.0 and v0.7.x inventory (moved from CLAUDE.md, slice 2)

**v0.4.0 (TOP 10 ML):** LinearRegression, LogisticRegression, DecisionTree, RandomForest, GBM, NaiveBayes, KNN, SVM, KMeans, PCA + model selection + metrics

**v0.7.x (Advanced):** ARIMA time series, text processing (tokenizers, stop words, stemming, chat templates via minijinja), Bayesian inference (conjugate priors, BLR), GLMs (Poisson/Gamma/Binomial), ICA decomposition, graph algorithms (Dijkstra/A*/PageRank/community detection)

## PMAT scores provenance (moved from CLAUDE.md, slice 2)

**Scores.** Only one of these has a citable measurement, so only one is stated:

| Score | Value | Provenance |
|-------|-------|------------|
| Line coverage | **90.84%** (849706/935347) | coverage-nightly `workflow_dispatch` run 36065904363, 2026-09-24, on branch commit `237f10f8b` (monorepo scope, #3839; in batch/0.70.0, not yet `main`, whose scheduled nightly 35937679357 read 87.82%). The previous 88.78% (2026-07-29, `95145584f`) used the pre-monorepo exclude regex. The long-quoted "96.35%"/"96.94%" predates the pipeline ever working — it reported 0/0 until #2333 |
| Project score / TDG / mutation % | **re-derive** — `pmat rust-project-score`, `pmat tdg . --include-components`, `cargo mutants` | The previously published "124/134", "TDG 95.2/100" and "Mutation 85.3%" carried no date or commit and could not be reproduced from the tree |
