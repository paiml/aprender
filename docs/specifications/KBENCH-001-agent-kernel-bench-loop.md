# KBENCH-001 — an agent kernel loop: contract, bench, recheck, frontier

**Status:** plan. Nothing is applied: no issues are minted, nothing moves in `epics.yaml` or `roadmap.yaml` (APR-EPIC-001 rules 6, 13 and 17; the cop mints).
**Kind:** docs · **Research:** `docs/research/crux-tirx-harness-2026-10-06.md` (CRUX plus a three-lane quorum).
**Parent specs:**
- PP-LLAMA-001 §7.3 (the kernel microbenchmark, `DESIGNED, NOT ARMED`) and §12 row 13 (#4651).
- APR-EPIC-001 §2 (the epic set).

**Marks:** `[M]` measured at `origin/main` `645dc34101` by reading the cited path · `[Q]` quorum verdict (research §4) · `[A]` assumption · `[U]` unverified.

## §0 The feature in one paragraph

An agent working on a GPU kernel today has no single command that answers *"is my kernel still correct, did it get faster, and can I trust that number?"*. Perf is gated nightly. The one kernel microbenchmark the perf spec designs is not armed, and nothing records which variants were tried. TIRx Harness (mlc-ai) shows the agent UX: an immutable task YAML, one bench command, checks against reward hacking, and a frontier of diverse variants. KBENCH-001 builds that loop **in Rust, under `apr kernel`**, on aprender's existing rigor:
- receipts;
- planted-failure self-tests;
- contracts as the single source of tolerances;
- PP-6 (no comparator ratio at merge).

## §1 Ground truth `[M]`

| Fact | Where |
|---|---|
| `apr kernel` has exactly one subcommand, `parity` (tiled attention vs. naive, JSON, refuses `flash2` because it embeds no such kernel) | `crates/apr-cli/src/extended_commands.rs` `enum KernelCommands`; `crates/apr-cli/src/commands/kernel_parity.rs` |
| PTX is emitted from Rust. Kernels live in `aprender-gpu/src/kernels/{gemv,gemm,attention,quantize,fused,…}` | `crates/aprender-gpu/src/ptx/`, `crates/aprender-gpu/src/kernels/` |
| Device-queried register and shared-memory budget exists | `crates/aprender-gpu/src/driver/budget_query.rs`; `cuda_tests/launch_budget_hw.rs` |
| Kernel microbenchmark designed but not armed: Q4_K GEMV at `M ∈ {4,8}` vs. `M ∈ {16,32}`, `cargo bench`, self-baseline, `n ≥ 5` | PP-LLAMA-001 §7.3; §12 row 13 (#4651, expires 2026-10-15, carried to 0.71 by ruling C197) |
| Perf gates are nightly only; PR checks are `ci / gate` and `workspace-test` | `.github/workflows/beat-speed-nightly.yml`, `nightly-bench.yml` |
| No kernel autotuning; a TPE tuner exists for ML hyperparameters | `crates/aprender-core/src/automl/{tuner,tpe}.rs` |
| An MCP server crate exists | `crates/aprender-mcp/src/tools/` |
| Profilers exist but are not agent-callable | `crates/aprender-cupti`, `crates/aprender-cgp`, `crates/aprender-ptx-debug` |

## §2 Design

### §2.1 The task file — `kernels/tasks/<kernel>.yaml`

```yaml
id: q4k-gemv                     # one task per (quant × kernel family); shapes live inside it (K-5)
kernel: gemv::q4k_batched        # resolves to one symbol in aprender-gpu (refuse if 0 or >1)
contract: contracts/<kernel>-v1.yaml#<equation>   # tolerances are READ from here (K-3), never typed here
shapes: [{m: 4, k: 4096, n: 4096}, {m: 8, k: 4096, n: 4096}]
control: [{m: 16, k: 4096, n: 4096}, {m: 32, k: 4096, n: 4096}]   # PP-LLAMA-001 §7.3 control path
reference: cpu::q4k_gemv_ref     # pinned; its source hash goes into every receipt
dtype_inputs: {x: f32, w: q4_k}
timeout_s: 120                   # exceeding it is rc 1 with `failure: timeout`, never a skip
n: 5                             # PP-LLAMA-001 §4.3 floor
```

The task file is immutable within a run. Its sha256 and the bench binary's build sha go into the receipt, and a receipt whose task hash differs from the committed file is refused.

### §2.2 The command — `apr kernel bench`

`apr kernel bench --task kernels/tasks/q4k-gemv.yaml [--host local|<runner>] --json [--output <p>]`

1. Resolve the kernel, compile PTX, and check `launch_budget` against the device. An over-budget kernel is a FAIL before any launch.
2. Run the reference and the candidate on seeded inputs. Compare against `atol`/`rtol`/`matched_ratio` from the contract.
3. Time the candidate (and the control shapes; the reference is never timed): L2 flush, `n` runs, median plus spread. Compare each median with the self-baseline for this (task, shape, gpu) in `scripts/perf-matrix.yaml`.
4. Run the reward-hack checks from §2.3.
5. Emit the receipt from §2.4.

Exit codes: **0** PASS, **1** FAIL (any workload, any check, a ratchet regression, or a timeout, recorded as `failure: timeout`), **2** REFUSED (vacuous universe, unresolved symbol, unpinned reference, missing device, no perturbable input). There is no other exit code. This fixes TIRx's exit-0-on-failure weakness at the source.

### §2.3 Reward-hack checks (all mandatory, all fail-closed)

| Check | Defeats | Integer/index kernels |
|---|---|---|
| `input_dependence` — after timing, perturb inputs in place, re-run once, compare against a fresh reference | output computed in setup, cached across calls | **Not skipped** (TIRx skips them). Permute indices or flip low bits; a kernel with no perturbable input is REFUSED (rc 2), never PASS |
| `repeatable` — two runs on identical inputs are bit-identical (or within a declared nondeterminism bound from the contract) | nondeterministic shortcuts, races | same |
| `after_timing` — re-check correctness on the outputs of the timed run itself | a correct untimed path paired with a fast wrong timed path | same |
| `reference_pinned` — the reference's source hash equals the committed hash | a drifted reference that "agrees" with a wrong kernel | — |

Each check has a **planted-failure fixture** that must go RED: a kernel that caches, one that computes in setup, one with a timed-only path, and a drifted reference. This follows S1: vacuous PASS = 0.

### §2.4 Receipt — `evidence/kbench/<task-id>/<commit>.json`

`{task_id, task_sha256, kernel_symbol, ptx_sha256, reference_sha256, build_sha, host, gpu, driver, launch_budget:{regs,smem,limit}, shapes:[{shape, max_abs_diff, rel, matched_ratio, median_us, spread_us, n, baseline_us, delta, ratchet_verdict}], failure, checks:{input_dependence, repeatable, after_timing, reference_pinned}, verdict, rc}`

The receipt contains no comparator ratio, so PP-6 holds. The self-baseline is per (task, shape, gpu) and lives in `scripts/perf-matrix.yaml`, so PP-29 conformance covers it. Until it is armed under PP-LLAMA-001 §7.3 rules, `ratchet_verdict` is `REPORTING` and cannot fail. Once armed, a median past `baseline_us` × (1 + band) is rc 1.

## §3 Rows — one per epic, per APR-EPIC-001 rule 20

Each row becomes **one checklist line in one parent issue per epic** when the cop mints it (rules 1, 17 and 20). The train comes from the epic; scope slips and the date holds (rule 10).

| Row | Epic (train) | Item | done_when | First-green proof (must be able to fail) | Quorum |
|---|---|---|---|---|---|
| **K-1** | **E6 llama.cpp Parity (0.73)** | `apr kernel bench --task` (§2.1, §2.2, §2.4). The first task is PP-LLAMA-001 §12 row 13's Q4_K GEMV `M ∈ {4,8}` vs. `{16,32}`, **consumed, not duplicated**: if #4651 lands as `cargo bench` first, K-1 wraps the same kernel symbol. It arms per §7.3 (`n ≥ 5`, self-baseline, ratchet-down) | An armed self-baseline and a receipt on lambda (sm_89). The gx10 (sm_121) cell comes from the existing `cuda-nightly` gx10 runner, not from F-1. `apr kernel bench` is documented in `apr kernel --help` | A planted 20% slowdown in the GEMV moves the median past the ratcheted baseline, so the run is RED. A planted wrong output is RED at step 2 | R2 3/3 `[Q]` |
| **K-2** | **S1 Gates That Cannot Lie (standing)** | The four checks of §2.3, fail-closed, including integer and index kernels | 4/4 planted fixtures RED; 0 checks return "skipped" | Each fixture in a case table that `--self-test` runs (a backticked name is not evidence; PP-29 style) | R3 3/3, **ranked first** `[Q]` |
| **K-3** | **S2 Provable Contracts (standing)** | Tolerances come from `contracts/*.yaml` (the `contract:` pointer in §2.1). A drift guard fails `ci / gate` when a task names an absent equation or carries a typed tolerance | Every task resolves its tolerances from a contract; 0 typed tolerances | A task with a hand-typed `atol` is RED; a task pointing at a renamed equation is RED | R4 3/3 `[Q]` |
| **K-4** | **E6 llama.cpp Parity (0.73)** | `apr kernel inspect <kernel> --json`: ptxas resource report, SASS (when `cuobjdump` is present, else REFUSED with the reason), occupancy, `launch_budget` headroom. Exposed as an `aprender-mcp` tool. **Kept separate from pmat's static `cuda-tdg` score** | MCP tool listed by the server and returning the same JSON as the CLI | A kernel planted over the register budget reports negative headroom and rc 1 | R7 3/3 `[Q]` |
| **K-5** | **E7 Any Model (0.74)** | The task universe is **enumerated from the quant TRAITS table**, not by hand (S1: hand-enumerated universes = 0). One task per (quant × kernel family) the table dispatches | Σ tasks = Σ TRAITS rows × families, checked by a guard | Adding a TRAITS row with no task is RED. A row that cannot produce a valid task, or an empty enumeration, is rc 2, never PASS | R5 (universe) `[Q]` |
| **K-6** | **E9 Classical ML + AutoML (0.76)** | A frontier derived from receipts (`evidence/kbench/<task>/frontier.json`: approach family, mechanism, median, why kept, kept diverse). Tile and launch autotuning over the frontier uses the existing TPE tuner (`aprender-core/src/automl/tpe.rs`), with `apr kernel bench` as the objective | On ≥ 1 task, the tuner finds a configuration whose median is lower than the hand-picked tile's by more than the noise band, re-verified by an independent K-1 run. Until then the row stays open (a no-win receipt does not close it). The frontier keeps ≥ 2 approach families per task | Hand-editing `frontier.json` is RED (it must re-derive from receipts). A planted noisy objective must not "win" inside the band. An empty frontier (no receipts) is rc 2 | R5 3/3 (receipt-derived, after K-1) `[Q]` |
| **K-7** | **E6 llama.cpp Parity (0.73)** | `.claude/skills/kernel-loop/SKILL.md`: write → `inspect` → `bench` → frontier. It carries the rule "reading a tool's doc is not evidence it ran": a claim of a measurement must cite a K-1 receipt path, and `launch_budget` limits go into the agent's context up front | The skill's examples run in `scripts/dogfood_examples.sh` | A skill example citing a nonexistent receipt fails the dogfood row | R9 3/3 `[Q]` |

Order: K-1 → (K-2, K-3, K-4 in parallel) → K-7 → K-5 → K-6. K-6 never starts before K-1 has an armed self-baseline `[Q]`.

## §4 Foreign work (rule 15: transfer, don't park)

These came out of the same CRUX but their fix lives outside aprender. They are recorded as findings for the cop, not as aprender issues.

| Id | Repo | Item | Quorum |
|---|---|---|---|
| F-1 | infra | A thin ssh runner on gx10 so an agent can run self-contained `apr kernel bench` requests interactively (a timeout counts as failed). K-1 does not depend on it. A forjar-provisioned service only after it proves out | R6 2/3 MODIFY, 1 DROP `[Q]` |
| F-2 | fleet harness | `kind=measurement`, smallest slice first (locked adapter hash plus nonzero exit). Enforce `banned_paths` at the file-access level, not by shell text match | R1 3/3, R8 3/3 `[Q]` |
| F-3 | pmat | Extend `cuda-tdg` from static scoring to call `apr kernel inspect --json` when an `apr` binary is pinned | R7 lane note `[Q]` |

## §5 Non-goals

- No TIRx-lite, TVM, or Python (R11 3/3 `[Q]`; no new Python). Copy the patterns, not the code.
- No comparator wall-clock ratio at merge (PP-6). K-1 receipts are self-baseline only. Parity against llama.cpp stays in PP-LLAMA-001 L3.
- TIRx is not added to the nightly-ux-crux competitor set (R10 0/3 `[Q]`).
- No `roadmap.yaml`, `epics.yaml` or census edit in any KBENCH PR (APR-EPIC-001 rule 13).

## §6 Open questions (to a quorum, not the operator; C314)

1. When `M=4` lands (§9 #6), does the §7.3 control path stay in the task, or retire? `[U]`
