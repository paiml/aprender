# CRUX — TIRx Harness vs. the aprender stack (2026-10-06)

**Kind:** research (competitive research on user experience). **Feature it feeds:** `docs/specifications/KBENCH-001-agent-kernel-bench-loop.md`.
**Subject:** [mlc-ai/TIRx-harness](https://github.com/mlc-ai/tirx-harness) at `c3eac62` — "an open compiler harness for agentic GPU programming".
**Compared against:** aprender (`origin/main` `645dc34101`), pmat (paiml-mcp-agent-toolkit) and the private fleet harness.
**Provenance marks:** `[M]` read in the tree at the cited path · `[Q]` quorum verdict (§4) · `[A]` assumption.

## 1. What TIRx Harness is `[M]`

A Python package (`tirx_harness/`) plus an optimization-run driver (`evolution/`). An agent (Claude Code or Codex) writes one GPU kernel against one workload and iterates until it beats a baseline.

| Piece | What it does | Path |
|---|---|---|
| Workload contract | One YAML per task: math, shapes, `atol`/`rtol`/`matched_ratio`, baseline, `sota_baseline`, `bench_timeout_s`, `banned_paths`. Rendered into the prompt as **immutable**. | `evolution/tasks/*.yaml`, `evolution/prompts/PROMPT.md` |
| Locked adapter | The benchmark adapter is fixed. "Any modification invalidates the run." Speedup = baseline ÷ candidate, summarised as a geomean. | `evolution/benchmark/adapter.py`, `benchmark_common.py` |
| Anti-reward-hacking | After timing: perturb float inputs in place (0.25σ), re-run once, compare against a fresh reference (catches setup-time compute and cross-call caching). Also `check_after_timing`, `require_repeatable_outputs`, L2 flush plus median timing. | `benchmark_common.py` (`input_dependence_check`, `time_fn`) |
| Guards | Pre-tool hook blocks reads of `banned_paths`, and setup deletes them from the run worktree. A second hook blocks killing dangerous processes. | `evolution/preparation/guards.py` |
| Frontier | `frontier/<name>/solution.py` plus `frontier/index.json` (approach family, mechanism, ours vs. SOTA, why kept). The prompt asks for **diverse** approaches, not one winner. `scratch/` is for throwaway attempts. | `docs/optimization-runs.md` |
| Remote GPU | kcoral: each request packages its own code and inputs, then runs on a remote GPU (also compute-sanitizer, IKET). A tool timeout is a failed result (rc 124). | `evolution/remote/kcoral_*.py` |
| Inspection | CUDA/PTX/SASS dump and ptxas resource report. Sync, race and numerics checkers with a Rust engine. | `tirx_harness/src/.../dump_kernel.py`, `numsim/` |
| Skills | `tirx-wiki`, `tirx-debug-kernel` and `tirx-profile-kernel`, bundled in the wheel. External skills are pinned by commit. Rule: "reading a tool page is not evidence that the tool ran". | `skills/`, `skills/external.json` |

**Its weaknesses `[M]`:**
- No test CI; the workflows only build wheels, docs and PyPI uploads.
- Scoring exits 0 when workloads fail, and the docs say "check every row for PASS".
- The input recheck silently returns `None` for integer or index-only kernels and when inputs are copied.
- `banned_paths` matches text in shell commands, so Python can read around it.
- Tolerances are prose, never checked against the adapter.
- The "self-improvement loop" exists only in the README diagram.

## 2. CRUX — where the UX differs

| # | Dimension | TIRx Harness | aprender stack today | Edge |
|---|---|---|---|---|
| 1 | Task contract | One YAML per workload, injected as immutable | 1,329 YAML contracts with equations and Lean links (`contracts/`), not wired into any agent task loop | TIRx (UX), aprender (depth) |
| 2 | Perf feedback latency | Per candidate, minutes, on a remote GPU | Nightly only (`beat-speed-nightly.yml`, `nightly-bench.yml`). PR CI gates correctness only. The kernel microbenchmark is `DESIGNED, NOT ARMED` (PP-LLAMA-001 §7.3) | **TIRx** |
| 3 | Numeric reward-hack defense | Perturbation recheck, repeatability, after-timing check | Process-level defenses (receipts, quorum, orchestrator re-runs acceptance). No input-dependence check on a timed kernel | **TIRx** |
| 4 | Variant memory | `frontier/index.json`, kept diverse | Receipts only. Tiles are hard-coded, and `ptx/optimize/tile_validation` validates them but does not search | **TIRx** |
| 5 | Remote GPU execution | kcoral, with self-contained requests | GPU hosts reachable only as CI runners chosen by label | **TIRx** |
| 6 | Inspection surface | ptxas/SASS dump as a callable tool | Rich but scattered: `aprender-ptx-debug`, `aprender-cupti`, `aprender-cgp`, `launch_budget` (`aprender-gpu/src/driver/budget_query.rs`). pmat `cuda-tdg` is static only. None is exposed to agents through MCP | TIRx (reach), aprender (depth) |
| 7 | Kernel authoring | TIRx-lite DSL over TVM IR, Python | Rust emits PTX directly (`aprender-gpu/src/ptx/`). No nvcc, no Python | aprender |
| 8 | Verification rigor | Weak (§1) | Strong: `ci / gate`, falsifiable guards, planted-failure self-tests, spec conformance (PP-29) | **aprender** |
| 9 | Existing kernel CLI | — | `apr kernel parity` exists (tiled attention vs. naive, JSON, refuses a kernel it does not embed). `apr kernel` has no `bench` | aprender has the slot |

**In short:** TIRx has the agent loop and the stack has the rigor. The feature is TIRx's loop (contract, bench, recheck, frontier) built on aprender's rigor (receipts, planted failures, contracts as the single source of truth), in Rust, under `apr kernel`.

## 3. What is reusable, and where

| Pattern from TIRx | aprender | pmat | Fleet harness (RAH/`prah`) |
|---|---|---|---|
| Immutable task YAML + locked adapter | `apr kernel bench --task` (KBENCH K-1) | — | `kind=measurement` (refused today); adapter hash in the receipt |
| Input-perturbation recheck, fail-closed | KBENCH K-2, plus integer and index kernels | — | — |
| Tolerances from contracts, not prose | KBENCH K-3 (`contracts/*.yaml` → task YAML plus drift check) | — | — |
| ptxas/SASS/occupancy as a tool | KBENCH K-4 (`apr kernel inspect`, and an `aprender-mcp` tool) | Extend `cuda-tdg` from static to dynamic | — |
| Frontier of diverse variants | KBENCH K-6 (receipt-derived) | — | — |
| Kernel-workflow skill + "a tool page is not evidence" | KBENCH K-7 | — | — |
| Remote, self-contained GPU request | — | — | infra: thin ssh runner on gx10 first (transfer, APR-EPIC-001 rule 15) |
| `banned_paths` guard | — | — | Enforce at the file-access level (hook or sandbox), not by shell text match |
| Skills in the wheel, externals pinned by commit | `.claude/skills` pins | pmat skills | `install.sh` already records `install-receipt.json`; add commit pins for externals |

**Do not adopt:** TIRx-lite, TVM, or the Python harness. They violate the no-new-Python rule, and aprender already emits PTX from Rust. Copy the patterns, not the code.

## 4. Quorum verdict `[Q]`

Eleven draft recommendations (R1–R11) went to three independent lanes. The author was `claude-opus-5-5`, so no lane used that model id.

| Lane | Model | Backend |
|---|---|---|
| 1 | gemini-3.1-pro-high | agy |
| 2 | gpt-oss-120b-medium | agy |
| 3 | claude-sonnet-5-5 | `claude -p` |

| Rec | Summary | Verdict | Lane notes folded in |
|---|---|---|---|
| R1 | `prah` `kind=measurement` with locked adapter hash, nonzero exit on any failed workload | 3/3 (2 MODIFY) | Smallest slice first: adapter hash plus nonzero exit |
| R2 | `apr kernel bench` per-edit receipt loop | 3/3 | Start with PTX compile, CPU diff and median timing; PR gating later. Pin the CPU reference and put its hash in the receipt |
| R3 | Port the perturbation, repeatability and after-timing checks, fail closed on int/index | 3/3, **ranked first** | A planted-failure fixture for every check |
| R4 | Task YAML generated from contracts | 3/3 | Drift check that fails CI when the YAML and contract diverge |
| R5 | Kernel variant frontier | 3/3 (2 MODIFY) | Derived from receipts, not kept by hand. After R2. Foundation for tile autotuning |
| R6 | Remote GPU execution service | 2/3 MODIFY, 1 DROP | Thin ssh runner on gx10 before any service |
| R7 | ptxas/SASS/occupancy tool | 3/3 | Keep it separate from the static `cuda-tdg` score |
| R8 | `banned_paths` at the file-access level | 3/3 | Lower priority than R3 |
| R9 | Kernel-workflow skill | 3/3 | Tie "ran" to an R2 receipt so it is checkable |
| R10 | Add TIRx to the nightly-ux-crux competitor set | **0/3, dropped** | Different language and domain; no comparable artifact |
| R11 | Do not adopt TIRx-lite, TVM or Python | 3/3 | — |

Lane additions:
- Every new gate gets test CI and a planted-failure self-test (lane 3).
- Pin the reference implementation by hash (lane 3).
- Feed `launch_budget` limits into the agent's prompt (lane 1).
- Build a sync/race checker in Rust, like `numsim` (lane 1).
- Pin external skills by commit (lanes 2 and 3).

Vote counts for the top-5 ranking: R3 3, R7 3, R9 3, R2 2, R4 2.
