# APR-OBS-001: apr dogfood observability and the directed-improvement loop

**Spec id:** `APR-OBS-001` **v1.3** · **Rows:** `OBS-00..OBS-18` (OBS-16, OBS-17 are executed under TRACE-001) (the traffic cop mints one `pmat` ticket per row; the single-minter rule applies)
**Target repo:** `paiml/aprender` (`docs/specifications/APR-OBS-001-dogfood-observability.md`). Infra rows are filed, not done (§0.7).
**Runner:** the aprender traffic cop (`aprender-traffic-cop-prompt.md`).
**Launch:** from `~/src/aprender`, run `Implement docs/specifications/APR-OBS-001-dogfood-observability.md autonomously.`
**Supersedes:**
- APR-OBS-001 v1.0 (2026-09-26), v1.1 and v1.2 (2026-09-27), in place at the same path.
- `apr-dogfood-observability-improvement-plan.md` (cop, 2026-09-26).

**v1.1 inputs:**
- cop review `apr-serve-performance-measurement.md` (2026-09-27)
- two design reviews of v1.0 (2026-09-27)
- one design review of v1.1 (2026-09-27; rulings in §0.9 v1.2 table)
- APR-PERF-GATE-001 v2.2 §2
- PP-LLAMA-001 v3 §4–§5

**Related:**
- SRV-TIM-001 (serve timings)
- infra#1057 (per-binary perf ledger)
- infra#1088 (gx10 shadow lane)
- PRM-001 (Prometheus; §5.1 perf ratchet, §5.2 review ledger)
- ARBITER-001 / ARB-SELF-001 (self-improvement consumer)
- #3598 (0.71 "Verbs Are Fast")
- #3596 (load + TTFT reporting)
- #4539 KREG-001 (kernel registry)
- #4522 (memory receipts)
- #4551 PERF-HIST-001 (builds OBS-05)
- PVL-001 (proof credit only on discharge)

**Status:** spec, not implemented. Dated 2026-09-27.

**Provenance marks:** `[V]` verified at the cited sha/time · `[C]` computed (the command or derivation is given) · `[A]` asserted (operator directive or estimate) · `[U]` unverified/unmeasured · `[X]` third-party.

---

## §0.9 v1.1 change log (read this first)

v1.0 had the right parts: the identity block, `gpu_proof`, same-host ratio, append-only ledgers and writer ≠ subject. It measured the wrong band, though, and left its statistics without equations. v1.1 changes the scope and puts an equation, a proof obligation and a falsifier behind every rule (§10).

| # | Change | Where | Mechanism it fixes |
|---|---|---|---|
| C1 | Two tiers: `engine` (apr bench vs llama-bench) and `serve` (apr serve vs `llama-server -np c`, streaming) | §2.4, OBS-05, **OBS-13** | v1.0 measured only the engine at c=1, ctx 4096. That band cannot see the serve serialization defect (GATE-001 v2.2: η(16)=0.075) or the 32k decode regression (#4485: 0.80) |
| C2 | Serve-overhead ratio ω and scaling efficiency η(c) become gated series | §10 E5, E6 | G8 (4.4× end to end) and the 0.82× engine prefill cannot be joined today (S-4). ω is the instrument that splits the gap |
| C3 | ABBA interleaving, with a randomized order and a recorded seed; the paired log-ratio is the unit | §2.4, R-11, §10 T1 | "Host noise cancels" is false for blocked runs. ABBA cancels linear drift exactly |
| C4 | A recorder separate from the subject: `apr bench` emits raw samples only, and an independent recorder appends | §2.7, OBS-05, R-10 | The #4551 draft gave kreg "the apr bench row writer", which violates R-10 / S-3 |
| C5 | A forjar-declared host lease, explicit skip rows, and cancel-cause classification before any fix | §2.8, OBS-02, §10 E7 | v1.0 liveness conflicts with S-1 on shared gx10. "Isolation" without a classified cause is a guess |
| C6 | θ defined in log space with σ̂ = 1.4826·MAD, k = 3 | §4, §10 E3/T2 | v1.0's `3×MAD/median` equals about 2.02σ, giving a false-RED rate of ≥1.3×10⁻³ per series-night whenever the σ term binds |
| C7 | Baseline RED uses the median of the last 3 nights | §4, §10 T4 | A single-night comparison against the tag false-fires on noise |
| C8 | The v1.0 OBS-06 falsifier ("planted 2× threshold on one night") is replaced | §3, §10 T3 | It sits on the decision boundary: P(detect)=½ with noise, and `>` is false at equality without noise |
| C9 | Publish predicate requires `calibrated`; no calibration means no publish | §4, §10 E10 | v1.0 left undefined what happens if the 0.71 final lands before night 14 |
| C10 | Hash-chained ledgers plus an external anchor | §2.2, R-12, §10 T5 | "Never rewritten" was policy, not structure |
| C11 | Epochs with 3 bridging nights on any pin, driver or kernel change; a model change resets | §2.9, **OBS-14**, §10 E9 | A comparator or driver bump silently shifts the ratio |
| C12 | Backfill goes to a separate ledger, never the primary | §2.2, OBS-14 | "0 incomplete rows" contradicted the admission of backfill |
| C13 | `quiescence_proof` per row; otherwise `host_unproven` | §2.6 | Host contention was assumed, not proven |
| C14 | `kernel_path` per row; KREG-001 schema is arch-generic, and an entry needs a host receipt | §2.10, **OBS-15** | A "wrong kernel silently used" regression can't be attributed today |
| C15 | Plateau rule is per declared, calibrated metric, with a guardrail on every gated series | §5.3, §10 E11 | A blanket exemption for memory PRs is a loophole. v1.0 counted only traced layers |
| C16 | Every scored rule carries a falsifier corpus of {positive, noise, step, drift, adversarial} | §3, §0.10 | A gate calibrated only on bad specimens can't recognise a good one (learnings 2026-09-18) |

**Withdrawn review findings (already in v1.0):**
- RED-issue dedupe per (host, backend, metric) is in v1.0 §4.6.
- Baseline RED already had a θ tolerance (v1.0 §4.4).
- θ was already computed over 14 nights (v1.0 §4.2).

The cop's summary had dropped these, and the reviews repeated that omission.

**Rejected:** registering KREG entries for aarch64 or macOS before a parity receipt exists on that host. "Registered" must imply "verified".

### v1.3 changes (milestones)

| # | Change | Where |
|---|---|---|
| C24 | Every row is assigned to a milestone (`0.70.0` … `0.73.0`), an epic and, where one applies, an APR-LOOKAHEAD-001 §2a exit criterion, with merge-by constraints | §7.1 |
| C25 | Blocking starts at a **release defined by calibration**, not by a version number: release B = the first final release tagged ≥ 14 admissible calibration nights after the clock starts. Operator ruling **D-1** requested; default stated | §4 |
| C26 | OBS-16 (external serve witness) and OBS-17 (one trace schema) registered; executed as TRACE-001 TR-07 and TR-09 | §7, §7.1 |
| C27 | OBS-18 added: extend the nightly to the 0.73 hosts and backends (mini Metal, gx10 aarch64 CPU, intel wgpu) | §7, §7.1 |

### v1.2 changes (review of v1.1)

| # | Change | Where | Mechanism it fixes |
|---|---|---|---|
| C17 | **Provisional mode** after a failed bridge: the gate stays armed with θ widened by the measured uncertainty of σ̂ and Δ_e, instead of fail-closing for 14 nights | §2.9, §10 E9 | v1.1 S-11 had ¬calibrated ⇒ ¬publish_ok, so one noisy pin bump could freeze releases for 14 nights. The review described this as the gate being *lost*. It is the opposite (fail-closed), but the cost is real |
| C18 | **Epoch-change freeze** on gated hosts inside T−14 nights of a scheduled final (S-12) | §2.9, §8 | Heijunka: don't change the instrument while it is gating |
| C19 | `rustc` removed from epoch triggers | §2.9 | A toolchain bump changes the apr binary. That is signal to catch, not instrument drift to bridge. v1.1 would have hidden rustc regressions behind a bridge offset |
| C20 | **Per-block leases** instead of one ~78-min monolith; the ABBA block (~6 min serve) is the atomic unit | §2.8, §10 E1 | Bounds preemption loss to one block. T1 cancels drift within a block, so blocks may sit in separate leases without bias. A two-lease split (engine/serve) still leaves a 68-min serve lease |
| C21 | **Anchor catch-up by deterministic replay**: a missed RED-job night is recovered by verifying the chain from the last anchor, cross-checking the journal receipts, replaying the RED rules over the gap, then anchoring the head | §2.2, §4.5, §10 E8, E13 | v1.1 left a transient RED-job outage as a release block with no recovery. It also ignored that a missed RED night means the rules *did not run*, so the anchor alone is not the loss |
| C22 | T1b `blocked_bias` proof term supplied | §10 T1 | It was a named `sorry`. Credit still waits for `pv discharge run` |
| C23 | T5 threat model stated: the chain detects post-write mutation. It does **not** detect a lying writer (that is covered by E12, identity and falsifiers) | §10 T5 | The v1.1 review read T5 as "impossible to rewrite history"; the scope was unstated |

**Rejected from the v1.1 review:**
- **A 5-night expedited calibration as the fix.** The MAD-based σ̂ has relative standard error ≈ 1.166/√n `[C]` (asymptotic Gaussian efficiency of MAD 0.368): 0.52 at n=5 versus 0.31 at n=14. A 5-night θ can be half the true value, which produces a false-RED storm. C17 keeps the gate armed and prices that uncertainty into θ explicitly.
- **"T1b must be discharged before KANI-OBS-ABBA can be trusted."** The dependency is not real. KANI-OBS-ABBA checks the recorder against E1, and T1 (`abba_cancel`) is what makes E1 sound. T1b only quantifies the bias of the *rejected* blocked design, motivating R-11. It is proved in v1.2 anyway (C22).

---

## §0 Operating assumptions

1. **Purpose.** apr runs many times a day and records almost nothing. This spec does three things:
   - Every real apr invocation leaves an identity-bearing timing row.
   - Regressions turn RED without anyone looking.
   - The *cause* of a slow run is one flag away.
2. **Why the order matters (design basis).** A self-improving loop needs four things, in this order (Zhang, Yuan & Zhang, arXiv 2607.04277v2, §3.5.2):
   - a **self-model** (a record of its own behaviour),
   - **self-simulation** (replaying chosen inputs),
   - **self-evaluation** (comparing expected against actual),
   - **directed modification** (changing *where* the evaluation points).

   Change not directed by self-knowledge is the degenerative regime (ibid. §A.8 Remark).

   | Requirement | Row |
   |---|---|
   | Self-model | OBS-01 (lane rows), OBS-05 / OBS-13 (nightly rows) |
   | Self-simulation | OBS-05 / OBS-13 (fixed workload replay) |
   | Self-evaluation | OBS-06, OBS-07 (RED rules, fixed baseline) |
   | Directed modification | OBS-03 (prefill/decode split), OBS-09 (per-layer trace), OBS-11 (trace per perf PR), OBS-15 (kernel path) |

   No row may consume speed signals for self-improvement before the rows above it are GREEN (OBS-10).
3. **The evaluator is outside what it evaluates.** Ledger writers, RED rules and the baseline live in forjar-declared units and CI. Neither apr nor arbiter holds write access to them, and this is enforced by OS permission, not convention (§2.7; ibid. Corollary A1; doctrine "producer is never the gate").
4. **Self-reports are cross-checked.** Server-reported timings (OBS-03) are the system describing itself. They are admissible only where they agree with an independent client measurement (OBS-04).
5. **The release train wins.** Every job here yields when the fleet-wide `train-active` flag is set on its pool. It holds `/tmp/apr-gpu.lock` around the binary only, never around cargo. Nothing here takes a session or CUDA host from an active release cut. A yield is recorded as a skip row (§2.8), never as a gap.
6. **Genchi genbutsu.** Every §1 value is re-measured at HEAD in Phase 0. A row whose premise is false at HEAD is `premise-falsified` and skipped.
7. **Cross-repo work is filed, not done.** Forjar declarations, timers, leases and dashboards in `paiml/infra` are filed as issues with checkable acceptance criteria. Dependent rows are `NotRun{NoDeclaredExecutor}` until they land.
8. **The operator has no pre-steps.** Anything only Noah can do is a §8 STOP.
9. **Two tiers, one identity.** `engine` isolates kernels. `serve` is what users run. Neither substitutes for the other, and their quotient ω is a gated series (§10 E5).
10. **Every scored rule owes a falsifier corpus**: one positive specimen (a genuine improvement stays GREEN), one noise specimen (stays GREEN), one step, one drift and one adversarial specimen (each turns RED).
11. **Provable model.** Every RED rule, gate predicate and ledger invariant is an equation in §10. It has:
    - a proof obligation: a Lean theorem for pure arithmetic, or a Kani harness for the Rust implementation;
    - a planted falsifier.

    Proof credit is given only when discharged (PVL-001: a theorem in `discharge.json` with `lake_exit == 0`). Statistical properties (T2, T3) are **simulation receipts `[C]`, not proofs**, and are labelled so.

**Toyota mapping:**

| Principle | Mechanism here |
|---|---|
| Jidoka | Publish predicate E10: no calibration, no publish; a RED stops the line |
| Andon | §4 RED rules open or comment on one issue per series key |
| Poka-yoke | Recorder uid (§2.7), hash chain (§2.2), ABBA order (R-11), identity FATAL (S-4) |
| Heijunka | One host lease per perf window (§2.8); rotating non-gated bands (§2.4) |
| Genchi genbutsu | Cancel-cause classification before any liveness fix (OBS-02); §1 re-measured at HEAD |
| Kaizen | Tag-baseline ratchets (OBS-07); per-metric plateau rule (§5.3) |
| Five whys | Terminal cause of untrustworthy numbers: measured and measuring systems not separated. Fixed by C4 |

---

## §1 Ground truth (baseline, frozen 2026-09-27; never quote as current)

| # | Fact | Value | Mark | Source of truth |
|---|---|---|---|---|
| G1 | Review-lane logging | `apr60-shadow-51725` (lambda, CPU, v0.69.3, Qwen3.5-4B Q4_K_M) records nothing per request | [V] cop, 2026-09-26 | lane unit journal |
| G2 | gx10 shadow serve lane (:8091) | stopped, held | [V] | infra#1088 |
| G3 | Scheduled model runs | `qwen-story-daily.yml` cancelled ×3; `cuda-nightly.yml` failure, cancelled, success (last 3) | [V] cop, 2026-09-27 | `gh run list -w <wf>.yml -L 3` |
| G4 | Per-binary perf ledger | no data file exists | [V] | infra#1057 |
| G5 | Serve timings | `timings` null on every path except CUDA chat | [V] | SRV-TIM-001 branch (2e) |
| G6 | Serve tracing | `X-Trace-Level` returns a single wall total; `inference_trace` never runs in serve | [V] | `pmat query "X-Trace-Level"` |
| G7 | `apr profile` on Qwen 3.5 GPU | fixed on a branch only | [V] | branch 9f1e456e0 (60) |
| G8 | End-to-end gap | apr serve ≈ 4.4× slower than llama.cpp, 200 prompts, gx10, n=1 | [V] n=1 | PRM-S1 v2 |
| G9 | Oracle pin | llama.cpp `d1d3c3396` | [V] 2026-09-20; [U] at HEAD | `scripts/llama_pin.toml` `build_commit` |
| G10 | Disk path in draft plan | `/mnt/nvme-raid0/...` is intel's; the lane runs on lambda | [V] | project guide §4.1 |
| G11 | Stale-binary incident | a bare `apr` resolved to a 26-day-old build | [V] | cop plan §5 |
| G12 | Serve serialization | η_apr(c) = 0.310 / 0.152 / 0.075 at c = 4 / 8 / 16; ς(16) = 16.02; η_llama(16) = 0.415 | [C] from GATE-001 v2.2 §2.1 | `APR-PERF-GATE-001` §2.1 table |
| G13 | Long-context decode | DP4A path 0.80× llama at 32k ctx; 1.05–1.09× at 2k/8k | [A] worker, 2026-09-27 | #4485 |
| G14 | Engine prefill | pp1006 0.82× llama-bench (0.71× same morning), release branch only; host `[U]` | [A] | cop inbox 2026-09-27 |
| G15 | Existing perf nightlies | `nightly-bench.yml` states it does not measure any apr-vs-llama ratio | [V] | workflow output |
| G16 | #4551 writer assignment | kreg owns "the `apr bench` row writer" | [A] | #4551 body; conflicts with R-10 |
| G17 | OBS-09 | implemented on `feat/4496-obs09-trace-serve` (aprender-76), no PR | [A] | cop inbox 2026-09-27 |

**Join prohibition:** G8 (serve, gx10) and G14 (engine, host `[U]`) do not share identity. No quotient of them may be computed or quoted (S-4). The first matched ω row comes from OBS-13.

---

## §2 Design

### §2.1 Row identity block (shared by every ledger)

Every row in every ledger carries this block. A row missing any field is **inadmissible** and counts as a missing row. An explicit `null` with a `*_reason` string is allowed only where marked.

| Field | Meaning | Source |
|---|---|---|
| `schema` | ledger schema id + version | constant |
| `ts` | UTC RFC 3339 | clock |
| `host` | forjar machine name (not `$(hostname)`: lambda is `noah-Lambda-Vector`) | forjar fact |
| `apr_version`, `apr_tag` | release tag or `dev-<sha>` / `rc.N` | `scripts/apr_bin.sh` / fleet pin |
| `crate_tarball_sha256` | cross-host identity | APR-PERF-GATE-001 §4.2.2 |
| `binary_sha256` | host-local anti-substitution | sha256 of the resolved binary |
| `build_identity` | `rustc -vV`, triple, **features read from the binary**, `uname -a`, accelerator + driver | APR-PERF-GATE-001 §4.2.2 |
| `comparator` | `{name, build_commit, binary_sha256, flags}` for llama-bench / llama-server | `scripts/llama_pin.toml` |
| `model_id`, `model_sha256` | weights identity | sha256 of the GGUF |
| `backend` | `cpu` / `cuda` / `wgpu` / `metal` | request + proof (§2.5) |
| `gpu_proof` | trace line or used-GPU probe; `null` only when `backend=cpu` | §2.5 |
| `quiescence_proof` | §2.6 | recorder |
| `kernel_path` | §2.10; `null` + `kernel_path_reason` allowed until OBS-09 merges | trace / KREG |
| `tier` | `engine` / `serve` | recorder |
| `band` | `{c, ctx, workload_id}` | §2.4 |
| `order` | `{design: "ABBA", seed, blocks: [ "ABBA" \| "BAAB" ]}` | recorder RNG |
| `epoch_id` | §2.9 | recorder |
| `lease_id` | §2.8 | forjar lease |
| `prev_row_sha256` | §2.2 chain | recorder |
| `request_id` | UUIDv7; joins OBS-01 ↔ OBS-03 rows (lane/serve ledgers) | client header |

Cross-host comparison, or any ratio or quotient across mismatched identity, is **FATAL** (refused; S-4). Within-host history with mismatch is annotated and splits the epoch (§2.9).

### §2.2 Ledgers

| Ledger | Schema | Writer (sole) | Cadence | Path |
|---|---|---|---|---|
| Lane | `apr-lane-row-v1` | review-lane client | every lane call | per-host forjar-declared `fleet_perf_dir` `[U]`; synced to almacén |
| Serve | `apr-serve-timing-v1` | `apr serve` structured log → journal → recorder | every request | same dir |
| Nightly | `apr-perf-ledger-v1` | `apr-perf-recorder` (§2.7) | nightly per host × backend × tier | same dir |
| Trace | `apr-trace-v1` | `apr serve` on `X-Trace-Level` → recorder | on demand + one per nightly series | same dir, by path + sha256 |
| Backfill | `apr-perf-backfill-v1` | OBS-14 importer | once | same dir; **never** read by RED, baseline or view series |
| Anchor | `apr-ledger-anchor-v1` | OBS-06 RED job (a different writer from the recorder) | nightly | same dir + committed in the RED job receipt |

**Append discipline:**
- One JSON object per line, canonical JSON (sorted keys, no insignificant whitespace), `O_APPEND`.
- Line ≤ 4096 bytes; longer payloads go to a side file by path + sha256.
- A single writer per ledger, holding `flock(LOCK_EX)` for the write. `O_APPEND` alone is not relied on across NFS.
- No ledger is rewritten in place.

**Chain (R-12):**
- The genesis row carries `prev_row_sha256 = h₀`.
- Every row k carries `prev_row_sha256 = h_{k−1}`, where `h_k = sha256(canonical_json(row_k))`.
- The anchor ledger records `(ledger, n_rows, h_n)` nightly.
- The recorder also emits `(ledger, n_rows, h_n)` to the systemd journal after every append. This is an independent witness in a store neither the recorder nor apr can rewrite.
- **Anchor catch-up (C21).** A RED-job run that finds its last anchor A_m older than the previous night does all of the following, in order (§10 E13):
  1. verifies the chain from `A_m` to the head;
  2. checks head `n_rows ≥` the maximum journal witness count;
  3. replays every RED rule over each gap night;
  4. writes one anchor `{catch_up: true, gap_nights: g, from: A_m}`.
- A catch-up with `g > 3` `[A]` is RED on the RED job's own liveness series. The evaluator being down for more than 3 nights is itself the abnormality.
- Invariants and proof: §10 T5, E13.

### §2.3 Lane row (`apr-lane-row-v1`), beyond §2.1

`prompt_n, completion_n, prompt_sha256, wall_ms, ttft_ms, ttft_reason, prefill_ms_per_tok, decode_ms_per_tok, ok, error, load1, cold`

- `ttft_ms` is measured only when the client streams. Otherwise `ttft_ms: null` and `ttft_reason: "non-streaming"`. **Never** copy `wall_ms` into `ttft_ms`.
- `cold: true` for the first request after unit start.
- Lane rows are **operational data, not benchmark data**. No apr/llama ratio is ever computed from them.

### §2.4 Nightly workload (OBS-05 engine, OBS-13 serve)

Reuse the APR-PERF-GATE-001 §4.3 / PP-LLAMA-001 W1 corpus. **No second prompt set is invented.** The long-context prompt is derived deterministically from W1.

| Field | Value | Mark |
|---|---|---|
| model | Qwen3.5-4B Q4_K_M, sha pinned | [A] (#3558 may change it; a model change **resets** the epoch, §2.9) |
| W1 corpus | `prompts-w1.jsonl`, `prompt_tokens = 512 ± 8`, ≥ 8 prompts | [A] |
| W5 long-context | the concatenation of W1 prompts in file order, cycled, truncated to 30 720 tokens | [A] |
| generation | `max_tokens = 128`, greedy, `seed = 0`, ignore-EOS; `completion_tokens == 128` on every retained sample or the band is fatal (PP-28) | [A] |
| order | ABBA/BAAB blocks, `B = 3` blocks per band → 6 runs per arm; block order ~ Bernoulli(½) from a recorded seed (R-11) | [A] |
| engine tier (OBS-05) | `apr bench` vs `llama-bench -p 512 -n 128`; W1; ctx 4096 | [A] |
| serve tier (OBS-13) | `apr serve … --stream` vs `llama-server` relaunched per band with `-np c`, `-c (c · n_ctx_slot)`, `n_ctx_slot ≥ 640`, pinned defaults (PP-LLAMA §5.3); window 60 s, warmup 15 s, cooldown 10 s | [C] PP-LLAMA §5.1 |
| gated bands (every night) | `(c=1, ctx 4096, W1)`, `(c=16, ctx 4096, W1)`, `(c=1, ctx 32768, W5)` | [A] |
| rotating bands | `(c=4)` on even nights, `(c=8)` on odd nights; report-only (heijunka, to bound the lease) | [A] |
| ladder | `c ≤ min(slots_admitted_apr, slots_admitted_llama)` read from both servers (PP-24); a band above a reported ceiling is `NA` with the budget recorded | [C] PP-LLAMA |
| reported per run | load ms, TTFT, pp tok/s, per-user decode tok/s, aggregate tok/s, wall; raw samples in a side file | #3596 |

**Nightly cost estimate `[C]`:**
- Serve band = 12 runs × 85 s = 17 min.
- 3 gated + 1 rotating band = 68 min.
- Engine tier ≈ 10 min `[U]`.
- Total ≈ 78 min per host × backend, excluding model load.
- OBS-02 replaces this with a measured p95 after 3 nights.

The atomic scheduling unit is **one ABBA block**: 4 × 85 s ≈ 5.7 min for serve (§2.8), not the night.

### §2.5 GPU proof

A row claiming `backend ∈ {cuda, wgpu, metal}` must carry `gpu_proof`: a trace line naming the device kernel path, or a used-GPU probe sampled during the run. `CUDA_VISIBLE_DEVICES`, a flag, or a feature list is not proof. A GPU claim without proof is recorded as `backend_unproven` and excluded from GPU series.

### §2.6 Quiescence proof

Each row carries `quiescence_proof`:
- `lease_id`
- `foreign_gpu_procs` (`nvidia-smi --query-compute-apps=pid,process_name`, excluding the subject and comparator pids)
- `load1_pre`
- `cpu_governor`
- `gpu_sm_clock_mhz` and `gpu_throttle_reasons` (sampled at window start and end)
- `train_active` (read at start)

The row is `host_unproven` and excluded from series if any of these hold:
- no lease is held,
- `foreign_gpu_procs ≠ ∅`,
- `train_active = true`.

`load1_pre` and clock ceilings have **no threshold** until 14 nights exist. They are then set by the §10 E3 procedure on their own series (instrument first, R-9).

### §2.7 Recorder ≠ subject (R-10, made structural)

- `apr bench --emit raw-samples-v1` and `apr serve` structured logs **emit** only; the subject never opens a ledger.
- `apr-perf-recorder` is a Rust binary in the aprender workspace (R-5). It runs as a forjar-declared service user distinct from the subject's user. It:
  - validates each sample file with `pv`,
  - computes §10 E1–E2,
  - checks identity (S-4),
  - appends one row.
- `fleet_perf_dir` is owned by the recorder user, mode `0750`. The subject's systemd unit or invocation carries `ReadOnlyPaths=<fleet_perf_dir>`.
- **Ownership split for #4551:**
  - kreg owns the schema, the `pv` contract and the `--emit raw-samples-v1` emitter.
  - infra-8d owns the recorder unit, the timer and the arbiter ingest.
- Falsifier: a planted apr build that opens a ledger for append gets `EACCES`. The recorder records `subject_write_attempt` and the night is RED.

### §2.8 Host lease (heijunka)

- Each perf host (gx10, lambda) declares an `apr-perf` **nightly window** in `machines/<host>/forjar.yaml`:
  - window start `[U]`, chosen by OBS-02 from measured occupancy;
  - width = 1.25 × measured p95 nightly wall `[A]`.
- Within the window, the recorder acquires **one lease per ABBA block** (C20):
  - lease length = 1.25 × measured p95 block wall `[A]`;
  - mutually exclusive with `train-active`, release cuts and every other perf workflow on that host.
- Blocks are scheduled in priority order: gated serve bands, then the engine tier, then rotating bands.
- The three blocks of one band need not be contiguous. T1 cancels drift within a block; between blocks, time separation adds variance, not bias.
- A block whose quiescence proof fails at start or end (e.g. `train-active` toggled mid-block) is **voided** and retried if window time remains. A voided block never enters E2.
- A band with fewer than 3 valid blocks by the window end writes `{state: "skipped", reason, lease_holder, blocks_valid}` for that band's series, never a gap.
- Preemption loss is bounded by one block (≈ 5.7 min serve) instead of the night.
- Liveness (§10 E7):
  - a skip is never admissible;
  - more than 1 skip in 14 nights is RED;
  - 2 consecutive skips are RED.

  A permanently skipping control logged as ok is the gate-that-cannot-fire class (learnings 2026-09-15).

### §2.9 Epochs

An epoch is a maximal run of nights with constant:
- `(model_sha256, comparator.build_commit, comparator.flags)`,
- the accelerator driver version,
- the §2.4 workload definition.

On a change:

| Change | Rule |
|---|---|
| Model or workload | New epoch, fresh 14-night calibration, no carried baseline |
| Comparator or driver | 3 bridging nights measuring old and new side by side; offset Δ_e per §10 E9 |
| Bridge passes (`bridge_ok`) | Baseline carried as L_tag + Δ_e; θ carried; gate fully armed |
| Bridge fails (`¬bridge_ok`) | **Provisional mode** (C17, §10 E9): gate armed with θ_prov(n_e), which shrinks as nights accrue; full calibration at n_e = 14 replaces it. Recorded as S-11 in the §9 report, **not a stop of the gate** |
| apr binary, **including a rustc bump** | **Not** an epoch change: that is the signal being measured (C19) |

**Where the old configuration can run alongside the new one** (comparator pin: both llama.cpp binaries are pinned and installable), the old epoch keeps gating during the bridge and provisional nights. The new epoch is measured in parallel within the same leases. Driver changes cannot run side by side (one driver per host); provisional mode is their only path.

**Freeze (C18):** an epoch change on a gated host inside T−14 nights of a scheduled 0.71+ final is refused by the forjar declaration check and is a STOP (S-12). A security-driven driver change is an operator ruling.

### §2.10 Kernel path

`kernel_path = {source: "trace" | "kreg", entries: [{op, kernel_id, qtype, layout, arch, shape_class, precision}]}`:
- From the OBS-09 per-layer trace until KREG-001 (#4539) lands.
- Then from KREG registry tuple ids.

A night-over-night ratio change with a changed `kernel_path` is auto-attributed in the RED issue.

**KREG-001 schema requirement (filed on #4539 by OBS-15):**
- `(arch, isa_features, backend)` are key dimensions, with no x86- or CUDA-only fields, so 0.73 (aarch64, mini, wgpu) needs no schema migration.
- An entry is admitted only with a parity receipt from a host of that arch.

---

## §3 Contracts and falsifiers

Each contract ships in the same PR as its row. Each falsifier is planted and shown RED→GREEN in the PR body. Every scored rule carries the §0.10 corpus.

| Contract | Row | Falsifiers (each must turn RED) | Positive / noise specimens (must stay GREEN) |
|---|---|---|---|
| `apr-lane-row-v1` | OBS-01 | client with the write step removed; row missing `model_sha256`; `ttft_ms == wall_ms` on a non-streaming call | a real streaming call |
| `apr-nightly-liveness-v1` | OBS-02 | workflow cancelled 2 nights running reported GREEN; 2 consecutive skip rows GREEN; 2 skips in 14 GREEN; a skip counted toward ≥ 13/14 | 1 skip in 14 with a lease-holder reason |
| `apr-serve-timing-v1` (SRV-TIM-001 F1–F5) | OBS-03 | F1 null timings on any backend; F2 token counts ≠ usage; F3 prefill + decode > wall; F4 planted `PhaseTimings::default()`; F5 log line ≠ response timings | one request per backend |
| `apr-selfreport-agreement-v1` | OBS-04 | planted +20% skew in server timings passes | real joined rows |
| `apr-perf-ledger-v1` | OBS-05, OBS-13 | missing host row; empty ledger; missing version/sha; GPU row without `gpu_proof`; ratio across mismatched identity; blocked-order (non-ABBA) receipt; row without `quiescence_proof`; row written by the subject's uid; chain break (mid-file edit); tail truncation below anchor; voided block (quiescence failed mid-block) entering E2 | a real rc.1 night; a night assembled from 3 non-contiguous valid blocks |
| `apr-perf-red-rules-v1` | OBS-06 | single-night step δ = 2θ + 3σ_D stays GREEN; two-night step δ = θ + 1.633σ_D not RED by night 2 in ≥ 90% of seeded trials; noise-free step δ = 2θ + 10⁻⁶ stays GREEN | noise series (σ̂ from calibration) stays GREEN in ≥ 99.9% of seeded trials; a +2θ improvement stays GREEN |
| `apr-perf-baseline-v1` | OBS-07 | planted 3%/week slowdown (s = −ln 0.97 / 7) for 28 nights not RED by night 13 (§10 T4); baseline row edited after write | a flat series stays GREEN |
| `apr-perf-epoch-v1` | OBS-14 | pin change without 3 bridge nights accepted; bridge with dispersion > σ̂ carried as `bridge_ok`; provisional θ below θ_prov(n_e); backfill row read by a series; a rustc-only change opening an epoch; an epoch change accepted inside T−14 of a scheduled final | a clean pin bump with a passing bridge; a failed bridge that stays armed in provisional mode and graduates at n_e = 14 |
| `apr-ledger-anchor-v1` | OBS-06 | catch-up accepted over a chain break in the gap; catch-up accepted with head `n_rows` below the journal witness; a replayed verdict ≠ the live verdict on the same ledger prefix; gap of 4 nights not RED | a 1-night RED-job outage recovered by catch-up with identical verdicts |
| `apr-trace-v1` | OBS-09 | trace returned with provenance `Measured` when the tracer did not run; per-layer sum > request wall | a traced request |
| `apr-perf-pr-trace-v1` | OBS-11 | 0.71 perf PR merged without before/after trace; perf PR with undeclared or uncalibrated metric; PR regressing a gated series by > θ merged | a real perf PR with a gain > θ on its declared metric |
| `apr-kernel-path-v1` | OBS-15 | GPU row after the OBS-09 merge with `kernel_path: null`; KREG entry for an arch with no host receipt | a gx10 row naming its kernels |
| `apr-publish-gate-v1` | OBS-06/07 | publish allowed with any gated series uncalibrated, chain-invalid, anchor-mismatched, baseline-RED, liveness-RED or without first-green | a calibrated, clean tag |

---

## §4 RED rules (OBS-06, OBS-07): definitions in §10

**Signal:** the log paired ratio `L_n(κ)` of apr to the comparator, same host, same run, ABBA-interleaved (§10 E1–E2), with higher meaning apr is faster. Raw apr values are reported, never used for regression.

1. **Calibration (report-only).** Until 14 admissible nights exist in the *first* epoch for series κ, every rule on κ is report-only, and κ is `calibrated = false`. In a later epoch entered through a failed bridge, κ is `provisional` (§2.9, §10 E9): rules are armed with θ_prov, and `provisional` satisfies the E10 `calibrated` conjunct.
2. **Threshold:** `θ_κ = max(θ_min, 3·σ̂_κ)`, with `θ_min = −ln 0.95 = 0.05129` and `σ̂ = 1.4826·MAD` over the calibration window (§10 E3). It is recomputed every 30 nights, logged, and never lowered within an epoch by less than the bridge rule allows.
3. **Rolling RED:** `D_n < −θ ∧ D_{n−1} < −θ`, or `D_n < −2θ`, where `D_n = L_n − median(L_{n−7..n−1})` (§10 E4).
4. **Baseline RED (OBS-07):** `L_tag − median(L_{n−2}, L_{n−1}, L_n) > θ`. The tag baseline is written once per tag by the release train (single writer) and never edited (§10 T4).
5. **Liveness RED:** §10 E7. It also fires on:
   - an empty ledger,
   - a lane with calls but 0 rows in 24 h,
   - a nightly workflow cancelled or failed 2 nights running,
   - a chain or anchor failure (§10 T5).

   A missed RED-job night is **not** an anchor failure if the next run's catch-up (§2.2, §10 E13) succeeds. The replayed verdicts for the gap nights are the verdicts of record. A catch-up gap > 3 nights `[A]` is RED on the RED job's liveness series.
6. **Model-falsified STOP (S-9):**
   - At night 14, the recorded false-RED count across all series must be consistent with the §10 T2 simulated rate at the 99% Poisson upper bound.
   - If it is not, the Gaussian noise model is falsified, and θ switches to the empirical rule `θ_κ = max(θ_min, q_{0.999}(|D|))` from the calibration window.
   - This is logged, and the operator is notified via §9.
7. **RED output:** opens (or comments on) **one** aprender issue per series key `(host, backend, tier, metric, band)`, with both receipts, the latest trace (OBS-09) and the `kernel_path` diff attached. Never a Slack-only alert.

**Series gated from release B** (defined below): operator ruling requested; §8 S-7 applies until it is given. The proposed default:

| Series | Rule |
|---|---|
| serve TTFT, c=1, ctx 4096 | Baseline RED blocks publish |
| serve per-user decode, c=1, ctx 4096 | Baseline RED blocks publish |
| η_apr(16), comparator-free | Never below tag baseline by > θ (ratchet) |
| serve decode, c=1, ctx 32768 | Report-only through release B; gated from the first final release after B |
| ω (serve overhead) | Report-only |
| engine pp512 / tg128 | Report-only |

**Blocking policy:**
- **Release B** is the first *final* release whose tag is cut after every gated series has ≥ 14 admissible calibration nights (the clock starts per §7 "Order for tonight", item 3).
- Before B: report-only. RED events appear in the release notes and the §9 report; they never block.
- From B: the §10 E10 predicate gates the crates.io publish. Release candidates warn only.
- **D-1 (operator ruling requested).** Default (a): B is defined by calibration as above, whatever its version number. Alternative (b): the 0.71 final waits until calibration completes. Under trains every 2–3 days, 14 nights spans about 5 trains `[C]`, so (b) would hold 0.71 for roughly two weeks. Until ruled: S-7 applies to every blocking rule; report-only continues.
- Every blocking rule needs a first-green run on a real tag before it can block.
- `calibrated = false` on any gated series **blocks** (jidoka). This does not relax the gate.

---

## §5 The directed-improvement loop

### §5.1 Admission (OBS-10)

A consumer (PRM-001 REX-10 perf ratchet, ARB-SELF-001, any autotuner) may act on speed signals only when **all** of these hold:
- OBS-01, OBS-03, OBS-04, OBS-05 and OBS-13 are GREEN.
- The consumer's host × backend has ≥ 14 consecutive admissible nights in the current epoch.
- The consumer holds **no write access** to any ledger, anchor, RED rule, baseline or this spec's contracts. This is checked by a CI job the consumer cannot modify.

### §5.2 Directed modification (OBS-09, OBS-11, OBS-15)

- `X-Trace-Level: step|layer` runs `inference_trace` for that request. The response carries per-step and per-layer timings with provenance `Measured`. It is off by default.
- The nightly runs one traced request per series. A RED night therefore already has its per-layer table and `kernel_path` attached.
- **Every 0.71 performance PR** carries a before/after `apr-trace-v1` diff for the layers it claims to change, measured on the same host at the same identity.

### §5.3 Plateau rule, per declared metric (§10 E11)

- A PR is a **perf PR** only if it declares exactly one *calibrated* primary metric m ∈ {a gated or report-only ratio series κ, η(16), ω, `mem_peak`}.
  - `mem_peak` becomes declarable once its series (from the #4522 `resources{}` field and the infra#1234 cgroup receipt) has 14 nights and a θ.
  - Until then, memory PRs are non-perf PRs.
- **Plateau:** the last 5 perf PRs declaring the same m each showed Δ_m ≤ θ_m. The owning epic then stops and re-plans from traces.
- **Guardrail (every PR, perf or not):** no gated series may regress by more than θ. Otherwise the PR is blocked.

This keeps memory work free of the plateau counter without exempting it from the guardrail.

---

## §6 Hard rules

- **R-1 Pinned binaries only.** Resolve apr via `scripts/apr_bin.sh` or the fleet pin; never a bare `apr` on `PATH`; never aprender HEAD. The comparator resolves via `scripts/llama_pin.toml` with `scripts/check_llama_pin.sh` green.
- **R-2 Empty is RED.** An empty ledger, a missing host row or a missing identity field is never a pass. A skip is never a pass.
- **R-3 Varied input.** Never a single prompt; ≥ 8 per workload.
- **R-4 No `$?` through a pipe** in any script (`set -o pipefail`, or capture per stage). Shell passes `bashrs`.
- **R-5 No Python** in any file this spec creates. Harness, recorder and analysis are Rust (aprender workspace crate or xtask).
- **R-6 Declared executors only.** Timers, units, users, leases, ledger paths and sync are forjar-declared (`forjar apply -f machines/<host>/forjar.yaml`). No ad-hoc SSH, no hand-installed apr.
- **R-7 GPU lock around the binary, never cargo; `train-active` yields**, and the yield writes a skip row.
- **R-8 Retention.** Raw rows are kept forever in almacén; no roll-up replaces raw rows (≈ 11 MB/year at 100 rows/day × 300 B `[C]`).
- **R-9 No invented thresholds.** Every number cites its measurement or derivation, or is marked `[A]`/`[U]`.
- **R-10 Writer ≠ subject**, enforced by uid and `ReadOnlyPaths` (§2.7). apr and arbiter never write ledgers, anchors, RED rules, baselines or contracts.
- **R-11 Interleave.** Every comparator measurement uses ABBA/BAAB blocks with a recorded seed. A blocked-order receipt is inadmissible.
- **R-12 Chain.** Every ledger row carries `prev_row_sha256`. The view (OBS-08) and the RED job verify the chain and the anchor before reading.
- **R-13 No cross-tier or cross-identity quotients** except ω under the E5 identity precondition.
- **R-14 Never hand-cancel** a perf run with started jobs (learnings 2026-09-16). Liveness fixes go through the lease and concurrency declarations.
- **R-15 Verdicts are pure.** Every RED verdict, θ and baseline value is a function of the ledger prefix (and the tag-baseline ledger) only. No wall clock, network, environment or randomness at evaluation time. This is what makes catch-up replay exact (§10 E13).

---

## §7 Tickets (EV-ordered)

`K̂` is in minutes `[A]`, recalibrated after the first three rows. **No row starts while a release cut holds the session or host it needs (§0.5).** Dependencies are in `deps`; EV order is value.

| EV | Row | Work | Contract | Done when (all must hold) | deps | K̂ |
|---|---|---|---|---|---|---|
| 0 | **OBS-00** schemas + model | Commit this spec and the §2.1 identity block. Add all ledger schemas (incl. backfill, anchor) as pv contracts. §10 obligations are declared as Lean theorems (`Theorems/Obs/*.lean` under the PVL-001 lean dir `[U]`) and Kani harness stubs. File the infra issue for per-host `fleet_perf_dir`, the recorder user and almacén sync | all §3 schemas | schemas lint; §10 obligations listed by `pv` with `status: declared`; infra issue URL recorded with acceptance criteria | — | 120 |
| 1 | **OBS-01** lane rows (owner 60) | Lane client appends `apr-lane-row-v1` per call and sends the `request_id` header | `apr-lane-row-v1` | rows = lane calls over 24 h, 0 gaps > 24 h; §3 falsifiers RED→GREEN | 00 | 120 |
| 1 | **OBS-02** liveness + lease | (a) Classify every non-success of `cuda-nightly`, `qwen-story-daily` and `beat-speed-nightly` in the trailing 30 days by mechanism: `cancel-in-progress` supersession, slow-dispatch concurrency hold, timeout, manual cancel, runner unavailable, `train-active` yield. (b) Fix per class (no fix without a class). (c) File the §2.8 lease for gx10 and lambda on infra with acceptance criteria. (d) Skip rows. (e) Extend `NIGHTLY-RED` to cancelled runs and skip rules. (f) Per-block lease acquisition and voiding (§2.8) | `apr-nightly-liveness-v1` | 0 `unknown`-class non-successes in 30 days; root cause written as a mechanism; 0 cancelled perf runs over the next 7 nights; lease issue URL recorded; measured block-wall p95 committed | 00 | 180 |
| 2 | **OBS-03** serve timings (owner 2e; SRV-TIM-001) | Every backend fills `timings`; one structured log line per request; `/metrics` histograms for prefill, decode, TTFT | `apr-serve-timing-v1` | F1–F5 RED→GREEN on every backend; lands after the 0.70 publish | 00 | 240 |
| 2 | **OBS-09** serve tracing (owner 76; branch exists) | Open the PR from `feat/4496-obs09-trace-serve`; the nightly attaches one traced request per series | `apr-trace-v1` | PR merged; provenance `Measured` only when the tracer ran; per-layer sum ≤ wall | 00 | 90 |
| 2 | **OBS-04** self-report agreement | Join OBS-01 and OBS-03 rows by `request_id`; check server total vs client wall | `apr-selfreport-agreement-v1` | ≥ 99% of joined rows within max(5%, 50 ms) `[A]`; planted +20% skew RED | 01, 03 | 90 |
| 3 | **OBS-05** recorder + engine tier (#4551; kreg: schema + emitter; infra-8d: recorder unit + timer + ingest; closes infra#1057) | §2.7 recorder and `apr bench --emit raw-samples-v1`. Engine tier on gx10 CUDA and lambda CPU under the lease, with ABBA, chain and anchor. The first row is the rc.1 binary, marked `calibration` | `apr-perf-ledger-v1` | ≥ 13 of 14 nights per host × backend; subject-write falsifier `EACCES`; chain, anchor and ABBA falsifiers RED→GREEN | 00, 02 (lease) | 180 |
| 3 | **OBS-13** serve tier | §2.4 serve bands (c=1, c=16, ctx 32768; rotating c=4/8), llama-server per band, η(c), ω. Record measured nightly wall p95 → lease width | `apr-perf-ledger-v1` | ≥ 13 of 14 nights per gated band; η_llama(16) recorded; first matched ω row committed | 03, 05, 09 | 240 |
| 3 | **OBS-06** RED rules | §4 rules 1–7 as a CI job over the ledgers (pure, R-15); anchor writer with catch-up replay (§2.2, E13); T2/T3 simulation receipts (seeded, ≥ 10⁷ series-nights) | `apr-perf-red-rules-v1`, `apr-publish-gate-v1`, `apr-ledger-anchor-v1` | calibration log committed at night 14; §3 corpus RED/GREEN as specified; T2/T3 receipts `[C]` committed; a planted 1-night RED-job outage recovered with identical verdicts | 05 | 210 |
| 4 | **OBS-07** release-tag baseline | Release train writes the tag baseline at T-0 (single writer); §4 rule 4 | `apr-perf-baseline-v1` | baseline row exists for the first tag after OBS-05; drift falsifier RED by night 13 | 06 | 90 |
| 4 | **OBS-14** epochs + backfill | §2.9 epoch detection, bridge, provisional mode, side-by-side comparator epochs, T−14 freeze check; `apr-perf-backfill-v1` importer for `evidence/perf*` receipts with complete identity only | `apr-perf-epoch-v1` | §3 falsifiers; 0 backfill rows in any series | 05 | 120 |
| 5 | **OBS-08** one view (owner infra-8d) | Page rendered from ledgers only, after chain and anchor verify: ratio over time per series, η, ω, lane p50/p95 per token, nightly and skip status, latest trace link | — | 0 hand-edited values; refuses to render on a chain failure; filed as infra issue | 06 | 90 |
| 6 | **OBS-10** loop admission | CI gate encoding §5.1 | extends §3 | a consumer with a planted write path is refused; a consumer before night 14 is refused | 06 | 90 |
| 6 | **OBS-11** perf-PR trace + plateau | §5.2 trace diff; §5.3 declared metric, plateau counter and guardrail | `apr-perf-pr-trace-v1` | 100% of 0.71 perf PRs carry a trace and a declared calibrated metric; §3 falsifiers | 06, 09 | 90 |
| 6 | **OBS-15** kernel path | `kernel_path` from the trace; file the §2.10 KREG-001 schema requirement on #4539 as acceptance criteria | `apr-kernel-path-v1` | 100% of GPU rows after the OBS-09 merge carry `kernel_path`; #4539 comment URL recorded | 09 | 60 |
| 7 | **OBS-12** hourly probe decision | After 14 nights, report whether any RED went undetected > 12 h between nightlies; operator rules | — | report committed; probe built only if the operator approves | 06 | 30 |
| 4 | **OBS-16** external serve witness | Executed as TRACE-001 **TR-07** (budget counted there) | `serve-external-witness-v1` | per TRACE-001 | 03, 05 | 0 |
| 2 | **OBS-17** one trace schema | Executed as TRACE-001 **TR-09** (budget counted there) | `apr-trace-v1` | per TRACE-001 | 09 | 0 |
| 8 | **OBS-18** 0.73 hosts | Extend OBS-05/OBS-13 to mini (Metal), gx10 (aarch64 CPU backend), intel (wgpu, AMD). Lease (§2.8) declared on each host; `gpu_proof` for Metal and wgpu defined (a device-kernel trace line) before any row is admitted | `apr-perf-ledger-v1` | ≥ 13 of 14 nights per new host × backend; Metal/wgpu rows without `gpu_proof` = 0 | 05, 13, 15 | 180 |

**K̂ = 2220 `[A]` · K = 2470 · andon at 1976.**

### §7.1 Milestones

Themes and exit criteria are APR-LOOKAHEAD-001 §2a (operator ruling 2026-09-27). Epics: 0.71 → **#3598** `[V]`; 0.72 → **#4000** and 0.73 → **#3999** `[A]` (cop report 2026-09-27; confirm at HEAD, LA-01 re-creates milestone `0.73.0` from `backlog`). The cop mints one ticket per row as a sub-issue of the epic, one parent issue per PR (APR-EPIC-001), fragment in `docs/roadmaps/entries/<TICKET>.yaml`.

| Milestone | Rows | Exit criterion served | Merge-by constraint |
|---|---|---|---|
| **`0.70.0`** (rc.1, 2026-09-27) | none | — | Only #4445 (already on the train) and the first rc.1 row written by #4551. Rows written before the calibration clock starts are backfill (OBS-14), not calibration |
| **`0.71.0`** Verbs Are Fast | OBS-00, 01, 02, 03, 04, 05, 06 (report-only), 07 (records the 0.71 tag baseline), 09, 11, 13, 14, 15, 17 | **V1** TTFT ≤ 2× llama.cpp at APR-OBS identity ← OBS-05, 13, 14. **V2** load + TTFT reported ← OBS-03, 04. **V5** KREG live ← OBS-15 (`kernel_path`, schema requirement on #4539). **V6** per-layer serve tracing before the first perf PR ← OBS-09, 17, 11 | OBS-09 merges **before** the first 0.71 performance PR (V6). OBS-00 and the §2.7 recorder separation merge before any calibration night counts. OBS-07 writes the 0.71 tag baseline even though it does not block (it is the baseline release B compares against) |
| **`0.72.0`** Train What You Serve | OBS-06 **blocking switch** (only if release B falls in 0.72 under D-1 default), 08, 10, 12, 16 | **T3** Prometheus B2 is a speed-signal consumer ← OBS-10 admission must be GREEN before it acts on speed data | OBS-10 before any REX-10 or ARB-SELF-001 consumer reads the ledger |
| **`0.73.0`** Runs Everywhere | OBS-18 | **E1/E2** parity and ≥ 0.5× speed per backend ← measured by OBS-18 rows. **E5** KREG covers every dispatched key ← entries admitted only with a host receipt (§2.10). **E6** mini-metal and intel-wgpu cells admissible ← OBS-18 liveness ≥ 13/14 | New hosts start a new epoch (§2.9) and a fresh 14-night calibration; their series are report-only until calibrated |

A row that misses its train's cut moves to the next milestone with `slipped_from:` (train rule 7). If it serves an exit criterion, the move needs an operator ruling; exit criteria are never waived (APR-LOOKAHEAD-001 §2a).

**Order for tonight (rc.1):**
1. #4445 lands.
2. OBS-05 writes the first engine-tier `calibration` row from the rc.1 binary.
3. The 14-night calibration clock for any series starts only when OBS-00, the OBS-05 recorder separation (§2.7) and R-11 are merged.

Rows written before that are backfill (OBS-14), not calibration.

---

## §8 STOP conditions (stop, write the §9 report, do not work around)

- **S-1** A row would take a session, runner or CUDA host from an active release cut, or run while `train-active` is set on its pool.
- **S-2** An executing host resolves apr (or llama.cpp) to an undeclared or unpinned binary.
- **S-3** A ledger writer, anchor, RED rule or baseline would be placed where apr or arbiter can write it (R-10).
- **S-4** A ratio or quotient would be computed across mismatched identity (§2.1, R-13).
- **S-5** A harness hook refuses the session's writes for a reason other than a missing ticket.
- **S-6** Budget K reached, or andon crossed with more than one row incomplete.
- **S-7** A rule would become release-blocking without an operator ruling on the §4 gated-series table.
- **S-8** Two consecutive non-passing quorums on the same PR.
- **S-9** The noise model is falsified at night 14 (§4.6). The empirical θ is applied and the operator is notified; blocking waits for the ruling.
- **S-10** The §2.8 lease cannot be declared on gx10 or lambda (infra refuses, or occupancy leaves no window ≥ the measured p95 × 1.25).
- **S-11** An epoch bridge fails its dispersion test (§10 E9). This is a *report* stop for the session: the gate stays armed in provisional mode, the session writes the §9 report with the bridge receipts, and the operator is notified. It never disarms the gate.
- **S-12** An epoch change (comparator or driver) would land on a gated host inside T−14 nights of a scheduled 0.71+ final (§2.9 freeze).

---

## §9 Final report schema (one per session)

```yaml
spec: APR-OBS-001
spec_version: 1.3
milestone: 0.70.0 | 0.71.0 | 0.72.0 | 0.73.0
session: {date_utc, host, tree: {head, origin_main, worktree}, model}
selected_row: OBS-NN
ticket: PMAT-NNNN
outcome: merged | stopped | already-done | premise-falsified
stop: {id: S-n, evidence: "..."}
baseline_remeasured: [{row: G#, value, mark, command}]
cancel_causes: [{workflow, run_id, class, evidence}]
ledgers:
  - {ledger, host, backend, tier, rows_24h, admissible_pct, skipped_14, last_row_ts, empty: bool, chain_ok: bool, anchor_ok: bool}
red:
  calibration: {epoch_id, nights, per_series: [{key, median_L, mad, sigma_hat, theta, calibrated}]}
  model_check: {false_red_observed, false_red_expected_T2, poisson_p99_upper, falsified: bool}
  events: [{key, rule, L_n, D_n, baseline, kernel_path_changed: bool, issue_url}]
serve:
  eta: [{host, backend, c, eta_apr, eta_llama}]
  omega: [{host, backend, omega, identity_match: true}]
epochs: [{epoch_id, cause, bridge_nights, delta_e, bridge_ok, mode: calibrated|provisional, n_e, theta_prov}]
anchors: [{ledger, last_anchor_ts, catch_up: bool, gap_nights, journal_witness_max, replay_verdicts_identical: bool}]
leases: [{host, night, blocks_scheduled, blocks_valid, blocks_voided, preempted_by}]
proofs: [{obligation, kind: lean|kani|simulation, status: declared|discharged, receipt}]
selfreport_agreement: {joined_rows, within_tolerance_pct}
loop_admission: [{consumer, admitted: bool, reason}]
issues_filed: [{repo, url, purpose}]
scoreboard_moved: [{metric, before, after, command}]
budget: {k_hat_row, k_actual_row, cumulative, andon_crossed: bool}
next_row: OBS-NN
```

### §9.1 Scoreboard (targets and probes only; state is rendered, never written here)

| Metric | Target | Rendered by |
|---|---|---|
| Lane rows ÷ lane calls, 24 h | **1.0**, 0 gaps > 24 h | OBS-01 |
| Unclassified perf-nightly non-successes, trailing 30 d | **0** | OBS-02 |
| Cancelled or failed perf nightlies, trailing 7 nights | **0** | OBS-02 |
| Skip rows per series, trailing 14 | **≤ 1**, never 2 consecutive | OBS-02 |
| Serve rows with non-null prefill + decode, every backend | **100%** | OBS-03 |
| Joined rows with server/client agreement | **≥ 99%** | OBS-04 |
| Nights with a row for every gated series, trailing 14 | **≥ 13** | OBS-05, OBS-13 |
| Rows with a missing identity field, blocked order, or no quiescence proof admitted | **0** | receipt lint |
| GPU rows without `gpu_proof` / without `kernel_path` (after OBS-09) | **0 / 0** | OBS-05, OBS-15 |
| Ledger rows written by the subject's uid | **0** | OBS-05 |
| Chain or anchor failures | **0** | OBS-06 |
| RED-job catch-up gaps > 3 nights | **0** | OBS-06 |
| Replayed verdicts ≠ live verdicts on the same prefix | **0** | OBS-06 |
| Gated series neither `calibrated` nor `provisional`, after the first epoch's night 14 | **0** nights | OBS-06, OBS-14 |
| Epoch changes on gated hosts inside T−14 of a final | **0** | OBS-14 |
| Voided blocks entering E2 | **0** | OBS-05 |
| Backfill rows in any series | **0** | OBS-14 |
| RED events without an issue carrying receipts + trace + kernel diff | **0** | OBS-06 |
| Observed false REDs at night 14 vs T2 prediction | within the Poisson p99 bound | OBS-06 |
| 0.71 perf PRs without trace or declared calibrated metric | **0** | OBS-11 |
| Consumers acting on speed signals without admission | **0** | OBS-10 |
| Gated series (§4 table) vs tag baseline, from release B (§4) | never down by > θ | OBS-07 |
| η_apr(16) | ratchet up from the first tag baseline | OBS-07, OBS-13 |
| ω on the primary host | measured nightly; first matched value committed | OBS-13 |
| Python lines in files this spec touches | **0** | `grep -rl python3` over the diff |

---

## §10 Provable model

Pure arithmetic claims are **Lean theorems** (credited L4 only when `pv discharge run` lists them with `lake_exit == 0`). Rust implementations of the same arithmetic carry **Kani harnesses**. Statistical claims are **seeded simulation receipts `[C]`**, never labelled proofs. Every theorem is `status: declared` until discharged.

### §10.1 Notation

- Series key κ = (host, backend, tier, metric, band, epoch).
- Per-run performance **x > 0, oriented higher-is-better**: tok/s for throughput metrics, and 1/ms for TTFT, load and wall.
- Arm A = apr, arm B = comparator. All statistics are in **log space**, ℓ = ln(x_A / x_B), where ℓ > 0 means apr is faster.
- `med_k` is the median of k values (k odd). Φ is the standard normal CDF.

### E1: ABBA block statistic

A block occupies four consecutive slots t, t+w, t+2w, t+3w. Its order is `ABBA` (u=0) or `BAAB` (u=1), with u ~ Bernoulli(½) from the recorded seed. Pairing is by adjacency:

$$
\ell_b=\tfrac12\Big[\ln\tfrac{x_{A}(t_{A,1})}{x_{B}(t_{B,1})}+\ln\tfrac{x_{A}(t_{A,2})}{x_{B}(t_{B,2})}\Big]
$$

- ABBA: (A@t, B@t+w) and (A@t+3w, B@t+2w).
- BAAB: (A@t+w, B@t) and (A@t+2w, B@t+3w).

### E2: nightly statistic

$$
L_n(\kappa)=\operatorname{med}_{b=1..B}\,\ell_b,\qquad B=3
$$

Each run is one pass of the full prompt set; its x is the median over prompts. Raw per-prompt samples go to the side file.

**T1 (linear-drift cancellation).** Model host drift as a shared log-linear term: ln x_A(t) = α + s·t and ln x_B(t) = β + s·t.

- For **both** block orders, ℓ_b = α − β **exactly**, for all s, t, w.
- A blocked design (n runs of A, then n runs of B, spacing w) has bias −s·n·w.

```lean
-- Theorems/Obs/Abba.lean
theorem abba_cancel (α β s t w : ℝ) :
  ((α + s*t) - (β + s*(t+w)) + ((α + s*(t+3*w)) - (β + s*(t+2*w)))) / 2 = α - β := by ring

theorem baab_cancel (α β s t w : ℝ) :
  ((α + s*(t+w)) - (β + s*t) + ((α + s*(t+2*w)) - (β + s*(t+3*w)))) / 2 = α - β := by ring

theorem blocked_bias (α β s t w : ℝ) (n : ℕ) (hn : 0 < n) :
  (∑ i ∈ Finset.range n, (α + s*(t + i*w))) / n
    - (∑ i ∈ Finset.range n, (β + s*(t + (n+i)*w))) / n = α - β - s*n*w := by
  have hn' : (n : ℝ) ≠ 0 := Nat.cast_ne_zero.mpr hn.ne'
  rw [div_sub_div_same, ← Finset.sum_sub_distrib]
  have h : ∀ i ∈ Finset.range n,
      (α + s*(t + i*w)) - (β + s*(t + (n+i)*w)) = α - β - s*n*w := by
    intro i _; ring
  rw [Finset.sum_congr rfl h, Finset.sum_const, Finset.card_range, nsmul_eq_mul]
  field_simp [hn']
```

- Kani harness `KANI-OBS-ABBA`: the recorder's fixed-point i64 implementation of E1 returns exactly α−β for all bounded (α, β, s, t, w), for both orders.
- R-11 is the operational consequence of T1.

### E3: threshold

Over the calibration window W_κ (the first 14 admissible nights of the epoch):

$$
\tilde L=\operatorname{med}_{n\in W}L_n,\quad
\hat\sigma=1.4826\cdot\operatorname{med}_{n\in W}\lvert L_n-\tilde L\rvert,\quad
\theta_\kappa=\max(\theta_{\min},\,k\hat\sigma),\quad \theta_{\min}=-\ln 0.95=0.05129,\ k=3
$$

- 1.4826 makes σ̂ a consistent estimator of σ under Gaussian noise, so k is in σ units.
- **Relation to v1.0 `[C]`:** to first order MAD(ln r) ≈ MAD(r)/med(r), so v1.0's `3·MAD/median` is k ≈ 3/1.4826 = **2.02σ**. The consequence is quantified in T2.

### E4: rolling statistic

$$
D_n=L_n-\operatorname{med}_7(L_{n-7},\dots,L_{n-1});\qquad
\text{RollingRED}_n \iff (D_n<-\theta\wedge D_{n-1}<-\theta)\ \vee\ D_n<-2\theta
$$

**T2 (false-RED rate under the noise model; simulation `[C]`).** Assume H₀: L_n i.i.d. N(μ, σ²) and θ = kσ (the σ term binds). Then D_n ~ N(0, σ_D²), with σ_D = σ√(1+v₇) and v₇ = Var(med₇)/σ² ≈ π/14 = 0.2244 (asymptotic) `[C]`.

| Arm | k = 3 (v1.1) | k = 2.02 (v1.0) |
|---|---|---|
| single night, P(D_n < −2θ) = Φ(−2k/√1.2244) | Φ(−5.42) ≈ 3.0×10⁻⁸ | Φ(−3.65) ≈ 1.3×10⁻⁴ |
| one night of the pair, p₁ = Φ(−k/√1.2244) | Φ(−2.71) ≈ 3.4×10⁻³ | Φ(−1.83) ≈ 3.4×10⁻² |
| two-night arm, bounds [p₁², p₁] | [1.1×10⁻⁵, 3.4×10⁻³] | [1.1×10⁻³, 3.4×10⁻²] |

- The pair's correlation (shared median window) places the true two-night rate inside the bound.
- OBS-06 commits the exact rate from ≥ 10⁷ seeded series-nights.
- With S gated series, the expected false REDs per night ≈ S·p.
- When θ_min binds (the common case for stable series), both rates are lower still.
- §4.6 falsifies this model empirically at night 14.

**T3 (detection power; simulation `[C]`).** For a step regression of size δ > 0 at night n₀:

- Single-night arm: P(RED at n₀) = Φ((δ − 2θ)/σ_D).
- Two-night arm by n₀+1: the reference med₇ contains at most 1 post-step value, and a median of 7 moves by at most one order statistic. So P ≈ Φ((δ − θ)/σ_D)².
- Minimum detectable effect at 90% by night 2: **MDE = θ + 1.633·σ_D** (solving Φ(x)² = 0.9 gives x = 1.633).
- **Corollary (C8):** a planted single-night step of exactly 2θ has P = Φ(0) = ½ with noise, and fails the strict `<` without noise. The v1.0 falsifier was ill-posed. v1.1 plants 2θ + 3σ_D (P ≥ 0.9987) with noise, and 2θ + 10⁻⁶ without.

**T4 (rolling blindness ⇒ baseline necessity).** Take a noise-free linear slowdown L_n = L_tag − s·n with s > 0.

1. D_n = −4s for all n ≥ 7, because the median of 7 terms of an arithmetic progression is its middle term.
2. Hence **∀ s ≤ θ/4, RollingRED never fires**, for every n.
3. Baseline statistic: B_n = L_tag − med₃(L_{n−2}, L_{n−1}, L_n) = s(n−1). So BaselineRED fires first at

$$
N^\ast=\big\lfloor \theta/s \big\rfloor+2 .
$$

**Worked `[C]` (the OBS-07 falsifier):** 3%/week means s = −ln(0.97)/7 = 0.0043513/night.

- With θ = θ_min = 0.051293: rolling needs s > θ/4 = 0.012823, so it is **blind**.
- N* = ⌊11.788⌋ + 2 = **13 nights** ≤ 28, so the falsifier fires inside its window.

This proves OBS-07 is necessary, not redundant.

```lean
-- Theorems/Obs/Drift.lean
theorem rolling_linear (c s : ℝ) (n : ℕ) :
  (c - s*n) - (c - s*(n-4)) = -4*s := by ring        -- middle of the 7-term AP window

theorem baseline_linear (Lt s : ℝ) (n : ℕ) :
  Lt - (Lt - s*(n-1)) = s*(n-1) := by ring           -- middle of the 3-term AP window

theorem rolling_blind (s θ : ℝ) (h : s ≤ θ/4) : ¬ (-4*s < -θ) := by
  intro h'; linarith
```

Kani harness `KANI-OBS-MED`: the recorder's `med_k` for k ∈ {3, 7} returns the middle element of the sorted input, and for any arithmetic progression returns its middle term.

### E5: serve overhead

Define ω only when both rows share (host, backend, binary_sha256, model_sha256, epoch) and the same night:

$$
\ln\omega_n=L^{\text{serve,dec},c=1}_n-L^{\text{engine,tg128}}_n
\quad\Longleftrightarrow\quad
\omega=\frac{x^{\text{serve}}_{\text{apr}}/x^{\text{bench}}_{\text{apr}}}{x^{\text{serve}}_{\text{llama}}/x^{\text{bench}}_{\text{llama}}}
$$

ω is the ratio of each system's own serve efficiency. ω < 1 means apr's serve layer loses more to overhead than llama-server does.

**Identity:** ln ρ_serve = ln ρ_engine + ln ω.

**Hypothesis H-S** (tested by the first matched rows, OBS-13): the G8 gap is dominated by the serve layer, i.e. |ln ω| > |ln ρ_engine|. No value is claimed before a matched row exists (§1 join prohibition).

### E6: concurrency

$$
\eta(c)=\frac{\text{agg}(c)}{c\cdot\text{agg}(1)},\qquad
\varsigma(c)=\frac{\text{dec\_ratio}(c)}{\text{agg\_ratio}(c)}
$$

**L6 (serial-server signature).** If a server admits one request at a time, agg(c) = agg(1), so η(c) = 1/c and c·η(c) = 1.

- Measured `[C]` from G12: c·η(c) = 4×0.310 = 1.24, 8×0.152 = 1.22, 16×0.075 = 1.20. This is flat at ≈ 1.2, consistent with serial admission plus small overlap.
- η_apr(16) is gated comparator-free (ratchet).
- η_llama(16) is recorded. Divergence between the η series and the ratio series indicts the pin, not the build (GATE-001 v2.2).

```lean
theorem serial_eta (a : ℝ) (c : ℕ) (ha : 0 < a) (hc : 0 < c) :
  a / (c * a) * c = 1 := by
  have hc' : (c : ℝ) ≠ 0 := Nat.cast_ne_zero.mpr hc.ne'
  field_simp [ha.ne', hc']
```

### E7: liveness

For each series κ and lease night n: state(n) ∈ {admissible, skipped, missing, unproven}.

$$
\text{LiveRED}_n\iff \text{missing}(n)\ \vee\ \#\{\text{skipped}\}_{[n-13,n]}>1\ \vee\ (\text{skipped}(n)\wedge\text{skipped}(n-1))
$$

The ≥ 13/14 target counts `admissible` only.

**Invariant:** `skipped` and `unproven` are never admissible.

Kani harness `KANI-OBS-LIVE`: over all state sequences of length 14, the admissible count never includes a skipped or unproven night, and two consecutive skips always yield RED.

### E8: chain

$$
h_0=\mathrm{sha256}(\texttt{schema}\,\Vert\,\texttt{host}\,\Vert\,\texttt{backend}\,\Vert\,\texttt{epoch\_start}),\quad
\text{row}_k.\texttt{prev}=h_{k-1},\quad h_k=\mathrm{sha256}(\mathrm{canon}(\text{row}_k))
$$

Anchor A_n = (n_rows, h_n) is written by the RED job, a distinct writer.

**T5 (tamper evidence).** Assume sha256 is collision-resistant. Then:

- Any alteration, deletion or reordering of row j < n fails verification at row j+1.
- Alteration or truncation of the tail beyond the last anchor is detected by the anchor check `ledger[n_rows_anchor].h = A.h`.
- **The chain alone cannot detect tail truncation. The anchor is necessary.**
- **Threat model (C23).** T5 covers mutation *after* a row is written, by any principal including root. It does **not** cover a writer that appends false rows with a valid chain. That class is closed by:
  - E12 (the subject cannot write),
  - the identity block (S-4),
  - the quiescence and GPU proofs,
  - the §3 falsifiers.
- The journal witness (§2.2) is a second store written at append time, so post-hoc truncation must defeat both the anchor and journald.

Kani harness `KANI-OBS-CHAIN`: model the hash as an uninterpreted injective function. For all ledgers of length ≤ 6, any single-row mutation at index j < n−1 makes `verify` return false, and any truncation below the anchor count makes `verify_anchor` return false.

### E9: epoch bridge

On a comparator or driver change e → e′, measure both configurations for 3 bridging nights:

$$
\Delta_e=\operatorname{med}_{j=1..3}\big(L^{e'}_j-L^{e}_j\big),\quad
\text{bridge\_ok}\iff 1.4826\cdot\operatorname{MAD}_j\big(L^{e'}_j-L^{e}_j\big)\le\hat\sigma_e,\quad
L^{e'}_{\text{tag}}=L^{e}_{\text{tag}}+\Delta_e
$$

If `¬bridge_ok`, the series enters **provisional mode** (C17). With n_e admissible nights in the new epoch (n_e ≥ 3, counting bridge nights):

$$
\theta_{\text{prov}}(n_e)=\max\!\Big(\theta_{\min},\;k\,\hat\sigma_{e'}(n_e)\big(1+\tfrac{2\cdot 1.166}{\sqrt{n_e}}\big)\Big)+u_\Delta,\qquad
u_\Delta=k\cdot\frac{1.2533\,\hat\sigma_{\text{bridge}}}{\sqrt 3}
$$

- σ̂_{e′}(n_e) is E3 computed over the new epoch's nights so far.
- The factor (1 + 2·1.166/√n_e) is the approximate 2-SE upper bound of the MAD-based σ̂. Its relative SE is ≈ 1.166/√n, from the MAD's asymptotic Gaussian efficiency of 0.368 `[C]`.
- u_Δ is k standard errors of the bridge-offset median (SE of a median ≈ 1.2533σ/√n) `[C]`.
- The baseline is carried as L_tag + Δ_e, with the offset's uncertainty priced into u_Δ.
- At n_e = 14, E3 replaces θ_prov and the mode becomes `calibrated`.

**Property (`prov_ge`):** θ_prov(n_e) ≥ max(θ_min, kσ̂) for the same σ̂. Provisional mode is never *more* sensitive than full calibration (no false-RED storm), yet it stays armed.

Worked `[C]`: at n_e = 3 the σ̂ factor is 1 + 2.332/1.732 = 2.35. With a 5-night reset (the rejected alternative) and no factor, θ would carry a ±52% (1 SE) error in either direction.

```lean
-- Theorems/Obs/Epoch.lean
theorem prov_ge (θmin k σ f u : ℝ) (hk : 0 ≤ k) (hσ : 0 ≤ σ) (hf : 1 ≤ f) (hu : 0 ≤ u) :
    max θmin (k*σ) ≤ max θmin (k*σ*f) + u :=
  le_add_of_le_of_nonneg (max_le_max le_rfl (le_mul_of_one_le_right (mul_nonneg hk hσ) hf)) hu
```

Model or workload changes never bridge. They recalibrate under the first-epoch rule (§4.1, report-only; E10 blocks until n_e = 14).

A model or workload change never bridges.

### E10: publish predicate (from release B, §4; subject to S-7 and D-1)

$$
\text{publish\_ok}(tag)\iff\bigwedge_{\kappa\in K_{\text{gate}}}\Big(\text{calibrated}(\kappa)\wedge\text{chain\_ok}(\kappa)\wedge\text{anchor\_ok}(\kappa)\wedge\neg\text{BaselineRED}(\kappa,tag)\wedge\neg\text{LiveRED}(\kappa)\wedge\text{first\_green}(\kappa)\Big)
$$

- **Property:** publish_ok is a conjunction, so it is monotone. Any conjunct false ⇒ false. In particular, ¬calibrated ⇒ ¬publish_ok (jidoka).
- Kani harness `KANI-OBS-GATE`: over all Boolean assignments to the conjuncts for |K_gate| ≤ 6, publish_ok = true iff every conjunct is true.
- The predicate is evaluated in the base-owned `sovereign-ci.yml` path, never in the PR head (producer is never the gate).
- It sits after the clean-room gate on the tag, which remains the first hard gate for any crates.io publish.

### E11: plateau and guardrail

PR p declares m(p) ∈ M_decl (calibrated metrics only). Its improvement is measured before/after on the same host and identity:

$$
\Delta_m(p)=L^{\text{after}}_m-L^{\text{before}}_m\quad(\text{for }\texttt{mem\_peak}:\ \ln(\text{mem}_{\text{before}}/\text{mem}_{\text{after}}))
$$

$$
\text{Plateau}(m)\iff\forall j\in\text{last 5 perf PRs with }m(p_j)=m:\ \Delta_m(p_j)\le\theta_m;\qquad
\text{Guard}(p)\iff\forall\kappa\in K_{\text{gate}}:\ \Delta_\kappa(p)\ge-\theta_\kappa
$$

- ¬Guard(p) blocks the merge.
- A PR declaring an uncalibrated m is refused as a perf PR.

### E13: replay determinism (anchor catch-up)

Let V(P) be the vector of RED verdicts computed from ledger prefix P, together with the tag-baseline ledger prefix at the same night. By R-15, V is a pure function.

**Property (prefix stability):** for any ledger P and extension P′ = P ‖ Q, the verdicts for nights ≤ last(P) are identical whether computed on P or on P′. This holds because rows are ordered by night, and every E2–E9 statistic for night n reads only rows with night ≤ n. Replayed verdicts therefore equal what the live job would have produced.

Catch-up acceptance, with W_j the journal witness counts in the gap and m the gap length in nights:

$$
\text{catchup\_ok}\iff \text{chain\_ok}(A_m\to\text{head})\ \wedge\ n_{\text{head}}\ge\max_j W_j\ \wedge\ m\le 3
$$

- Kani harness `KANI-OBS-REPLAY`: for all ledgers of ≤ 6 nights × 1 series with bounded integer L, the verdicts for nights ≤ k computed on the prefix equal those computed on the full ledger.
- **Residual risk (stated):** tampering inside the gap that also rewrites the journald witness is undetectable. The window is bounded by m ≤ 3 nights.

### E12: writer ≠ subject (capability invariant)

$$
\forall P\in\text{ledgers}\cup\text{anchors}\cup\text{baselines}:\ \neg W(\text{uid}_{\text{apr}},P)\wedge\neg W(\text{uid}_{\text{arbiter}},P)
$$

This is enforced by forjar-declared ownership, `0750` mode and `ReadOnlyPaths`.

- Falsifier: a planted append from the subject gets `EACCES`.
- This is a system invariant, checked by the falsifier, not a theorem.

### §10.2 Obligation register

| Id | Kind | Statement | Status |
|---|---|---|---|
| T1 `abba_cancel`, `baab_cancel` | Lean | E1 linear-drift cancellation | declared (`ring`) |
| T1b `blocked_bias` | Lean | blocked-design bias −s·n·w | declared (proof term supplied v1.2; credit on discharge) |
| T4 `rolling_linear`, `baseline_linear`, `rolling_blind` | Lean | E4 blindness; E4/OBS-07 detection time | declared |
| L6 `serial_eta` | Lean | serial signature c·η = 1 | declared |
| E9 `prov_ge` | Lean | provisional θ is never below the E3 rule | declared |
| `KANI-OBS-ABBA` | Kani | recorder E1 exactness | declared |
| `KANI-OBS-MED` | Kani | med₃, med₇ correctness and AP middle | declared |
| `KANI-OBS-LIVE` | Kani | E7 invariants | declared |
| `KANI-OBS-CHAIN` | Kani | T5 under an injective hash | declared |
| `KANI-OBS-GATE` | Kani | E10 monotone conjunction | declared |
| `KANI-OBS-REPLAY` | Kani | E13 prefix stability | declared |
| T2, T3 | simulation `[C]` | false-RED rate, power, MDE | receipt owed by OBS-06 |
| E12 | falsifier | uid capability | planted by OBS-05 |

No obligation carries `sorry` as of v1.2. The Lean terms were written without a local `lake build`: any that fail are reported by `pv discharge run` and fixed in OBS-00, never marked proved by hand.
