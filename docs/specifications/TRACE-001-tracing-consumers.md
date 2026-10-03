# TRACE-001: stop false telemetry, give every tracer a consumer

**Spec id:** `TRACE-001` **v1.2** · **Rows:** `TR-01..TR-16` (the traffic cop mints one `pmat` ticket per row; single-minter rule)
**Target repo:** `paiml/aprender` → `docs/specifications/TRACE-001-tracing-consumers.md`
**Launch:** from `~/src/aprender`, run `Implement docs/specifications/TRACE-001-tracing-consumers.md autonomously.`
**Related:** APR-OBS-001 v1.2 (TR-07 is its third timing witness; TR-09 depends on OBS-09), APR-PERF-GATE-001 v2.2 §1.1 (CF-5) and §3 (CRUX research), EXT-001 §10 (CRUX binding rules), `contracts/crux-competitive-research-ux-v1.yaml` (CRUX master), PVL-001, `scripts/dogfood.sh`.
**v1.2:** every row assigned to a milestone (§4.1); TR-11 split into define (0.71) and record (TR-15, 0.72); TR-14 folded into EXT-001 EXT-25 (no second gate); TR-16 added for 0.73 platform arms.
**v1.1:** adds CRUX tooling (§1.4, §2.4, TR-10..TR-14); v1.0 bound no tracing surface to a competitor, comparator or `gap_effect`. Also folds in one design review (rulings in §0.9).
**Status:** spec, not implemented. Dated 2026-09-27.
**Provenance:** `[V]` verified at `origin/main` **`aca6f2d7f`** (2026-09-27 10:04 UTC) by `git show`/`git grep` against that ref · `[C]` computed · `[A]` asserted · `[U]` unverified.

---

## §0.9 Review rulings (v1.1)

| Review item | Ruling | Change |
|---|---|---|
| ptrace overhead distorts the TR-07 witness; set a `< 5%` ceiling, fall back to eBPF | **Mechanism adopted; the 5% and eBPF are rejected.** The witness checks *agreement* between three views of the same traced execution, so a uniform slowdown does not bias agreement. What it can bias is behaviour under concurrency (ptrace stops serialise the server). 5% is an invented number (R-9 of APR-OBS), and no in-tree eBPF tracer exists to fall back to `[U]` | §2.3: witness runs only in dedicated c=1 blocks; rows carry `traced: true` and are **never** admitted to any perf series; the syscall filter must stop only on the listed calls (stop count measured vs total syscalls). The overhead ratio stays recorded, not gated |
| Linux-only renacer leaves macOS blind | **Partly adopted.** No syscall tracing on macOS is planned (renacer is ptrace-based; `compile_error!` off Linux). But blindness must be explicit | §2.2: every registry entry carries `platforms`. A surface run on an unsupported platform is `NotRun{UnsupportedPlatform}`, counted in the report, never silent. In-process `Measured` timings (TR-01) are platform-neutral and feed the registry on every OS |
| TR-06 has no noise tolerance; absolute budgets flake | **Adopted.** | §2.5: golden gate compares **syscall counts only**, per class, in log space, with θ derived from ≥ 10 baseline runs (APR-OBS E3 form) and a resolution floor of one syscall. Timing budgets in existing `renacer.toml` files (`max_time_ms`) are host-dependent: report-only, and timing moves to the APR-OBS ledger |
| TR-02 moves the `976.0` magic number into a file | **Adopted.** | TR-02 removes the efficiency arm entirely. Escalation keys on CV (dimensionless, computed from the run's own samples) or on an APR-OBS RED for that series |

---

## §0 Operating assumptions

1. **Jidoka first.** A tracer that prints a cause or a number it did not measure is worse than no tracer: it is cited, then acted on (CF-5). Phase A removes every such output before anything new is wired.
2. **A tracer without a consumer is inventory.** Every tracing, profiling or metrics surface names a consumer: a ledger, a gate, or a runbook step. Otherwise it is deprecated. This is the terminal cause from the five-whys: tracers were built as commands for people, never as instruments with a declared reader.
3. **In-tree only.** renacer is `crates/aprender-profile` (`[[bin]] name = "renacer"`). Never `cargo install renacer` from crates.io, never a bare `renacer` on `PATH`. Same rule as trueno.
4. **Linux for syscall tracing.** `crates/aprender-profile` is `compile_error!` on non-Linux by design (`ci.yml:300-305` `[V]`). Every row that runs the renacer binary runs on a Linux clean-room runner.
5. **New gates start report-only** and block only after a first-green on a real target plus 7 consecutive green nights `[A]`.
6. **Genchi genbutsu.** Phase 0 re-runs every §1 command at HEAD. A row whose premise is false at HEAD is `premise-falsified` and skipped.
7. **The operator has no pre-steps.** Anything only Noah can do is a §7 STOP.
8. **CRUX binding (EXT-001 §10).** Every surface this spec creates or depends on has a CRUX contract: category, surface, competitor, comparator, metric (same tool for every arm), `gap_effect`. Existing contracts are **reused and extended, never duplicated**. New ones use the existing `contracts/crux-<L>-<NN>-v1.yaml` schema. If the category enum is owned elsewhere, file an issue; do not fork it.
9. **Dogfood our own tracer in our own gates.** Where a CRUX falsifier needs a syscall trace, the in-tree renacer is the instrument and the competitor (strace) is an oracle arm, once parity is proven (TR-12).

---

## §1 Ground truth at `aca6f2d7f`

### §1.1 Size

| # | Fact | Value | Mark | Command |
|---|---|---|---|---|
| G1 | Rust lines, whole repo | 4,049,505 | [V] | `git grep -c '' origin/main -- '*.rs' \| awk -F: '{s+=$NF}END{print s}'` |
| G2 | renacer (`aprender-profile` + `-profile-core`) | 82,841 + 2,532 | [V] | same, path-scoped |
| G3 | `aprender-cgp` (bin `cgp`) + `aprender-cupti` | 11,292 + 1,776 | [V] | same |
| G4 | Other `.rs` files named `*trac*\|*profil*\|*brick*\|*span*\|*telemetr*\|*metrics*` | 1,181 files, 690,932 lines (upper bound: includes tests and contracts) | [V] | `git ls-tree -r --name-only origin/main \| grep -iE …` |
| G5 | Share of the repo | ≈ 0.79 M lines ≈ 19.5% (upper bound) | [C] | (G2+G3+G4)/G1 |

### §1.2 Consumers today

| # | Fact | Mark | Evidence |
|---|---|---|---|
| G6 | The only CI job that touches renacer runs its **library tests**; no workflow runs the renacer **binary** | [V] | `guards-nightly.yml:178` `cargo test -p aprender-profile --lib`; the other `.github` hits are a book grep and a mac `--exclude` |
| G7 | 6 `renacer.toml` files with budgets and assertions (`aprender-compute`, `-db`, `-distribute`, `-orchestrate`, `-train`, `-profile/examples`); **0** `.renacer/` baseline directories | [V] | `git ls-tree -r --name-only origin/main \| grep -E '(^\|/)\.renacer/'` → 0 |
| G8 | `dogfood.sh:1091` gate is `renacer validate --baseline .renacer -- "$BINPATH" --version`: it traces only `--version`, against a baseline that exists nowhere (G7) | [V] | `scripts/dogfood.sh:1091` |
| G9 | `scripts/capture_golden_traces.sh:23` runs `cargo install renacer --version 0.6.2` (a pre-monorepo crates.io release, not the in-tree crate) and writes to `golden_traces/`, which does not exist at HEAD | [V] | file + `git ls-tree origin/main golden_traces` → empty |
| G10 | `Makefile` target `profile:` runs `renacer --function-time --source -- cargo bench` from bare `PATH` | [V] | Makefile |
| G11 | apr-cli uses renacer only as `BrickTracer`, via `bench.rs:33` / `qa.rs:72` (`visualization` is a default feature) and `benchmark.rs:143` `TracerImpl::new_local()`. No call site configures OTLP, so no span ever leaves the process | [V] | `git grep -n TracerImpl` |

### §1.3 False telemetry (each is an output that claims something unmeasured)

| # | Site | What it claims | What is true | Mark |
|---|---|---|---|---|
| F1 | `aprender-profile/src/brick_tracer.rs:395` doc: "Captures syscalls during execution"; `:440` `SyscallBreakdown { compute_us: duration_us, ..Default::default() }` | A syscall breakdown with 100% of the time attributed to compute | No syscalls are captured (`:437` "we don't have syscall capture integrated here"). It is an `Instant` stopwatch whose attribution is invented | [V] |
| F2 | `apr-cli/src/commands/cbtop_measure_batch.rs:144,153` | "BrickTracer: Enabled for syscall breakdown" | `let _tracer = BrickTracer::new_local();` is built and dropped; nothing is traced | [V] |
| F3 | `cbtop_measure_batch.rs:132` `efficiency = tokens_per_sec / 976.0 * 100.0` | An efficiency percentage that triggers escalation | 976.0 is an undeclared literal with no `basis=` (R-9 of APR-OBS) | [V] |
| F4 | `apr-cli/src/commands/serve/handler_apr_cpu_completion.rs:397` and `:747` | `"layers": 28` in the `X-Trace-Level` payload | Hard-coded for every model. `state.num_layers` exists (`handlers.rs:37`). The payload carries no provenance, while the sibling `aprender-serve` `build_trace_data` (`mod_create_demo.rs:482`) marks the same data `WallClockTotal` | [V] |
| F5 | `aprender-serve/tests/modality_matrix/common.rs:14` `pub mod renacer { … }` | "renacer-compatible tracing API for PARITY-112 compliance"; test QA-A08 "renacer::capture() API works correctly" | A local mock named `renacer`. The test passes against the mock, not against renacer | [V] |

`F1`–`F5` are the CF-5 class: a diagnostic emitting a claim it never measured.

### §1.4 CRUX coverage of tracing (`contracts/crux-*.yaml`, 318 files, categories A–O)

| # | Fact | Mark | Evidence |
|---|---|---|---|
| G12 | Tracing-relevant CRUX contracts that exist: **F-02** `apr trace` layer-by-layer (active), **F-05** roofline (active), **F-07** GPU memory Chrome trace (partial), **F-09** gradient-norm telemetry (partial), **F-16** nsys/nvprof export (draft), **K-07** Prometheus `/metrics` (partial), **K-08** OpenTelemetry traces (partial), **E-07** latency P50/P95/P99 (partial), **E-16** TTFT (draft), **E-18** throughput (draft), **J-16** event log (mentions renacer via `pmat query` only). Competitor on every F-row is `pytorch` | [V] | `head -1` + `status`/`competitor` of each file |
| G13 | **No CRUX contract exists** for: syscall tracing (renacer), golden-trace regression (`renacer validate`), GPU/SIMD kernel profiling by `cgp` against nsys/ncu, per-op time attribution (BrickTracer), the external serve witness | [V] | title grep over all 318 contracts |
| G14 | F-02 and F-05 are `status: active` with **no** `discharge_status` recorded on any falsifier. F-05's instrument is `apr profile --roofline`, the tool CF-5 invalidated | [V] | `grep discharge_status` → empty for both |
| G15 | **K-08 is blocked on something we already have.** Its header: "Full discharge blocks on a live `apr serve` OTLP exporter … tracked as BLOCKER-UPSTREAM-MISSING". renacer ships `src/otlp_exporter.rs` + `otlp_types.rs` in the same workspace; `BrickTracer::new(endpoint)` already takes an OTLP endpoint (F1 file, `:299`) | [V] | `crux-K-08-v1.yaml:1-10`; `git ls-tree crates/aprender-profile/src` |
| G16 | Our own CRUX falsifiers trace syscalls with **external `strace`** in 8 contracts (A-05, A-07, A-08, A-15, A-20, C-27, G-10, G-15); renacer is the instrument in **0**. `ncu` appears in 18 | [V] | `git grep -l -F strace -- 'contracts/crux-*.yaml'` |
| G17 | Category labels drifted from contents: **B** is labelled "Inspection & Debugging" but its 20 rows are conversion/quantization; **I** is labelled "Observability & Metrics" but its 15 rows are tool-calling/MCP. The observability rows live in F and K | [V] | `category:` field vs titles |
| G18 | CRUX harness tooling is largely Python (`scripts/lib/crux_*.py`, `scripts/crux_hf`, `scripts/crux_vllm` with `uv.lock`). New CRUX tooling from this spec is Rust (R-3) | [V] | `git ls-tree scripts` |

---

## §2 Design

### §2.1 Honest provenance on every timing output

Every trace, profile or timing value that leaves a process carries `provenance ∈ {Measured, WallClockTotal, NotInstrumented}` (the enum already exists as `TraceProvenance` in `aprender-serve`; hoist it to `aprender-profile-core` so apr-cli and renacer share it).

- `Measured` requires the instrument to have run for that value (e.g. ptrace events present, CUPTI records present).
- A breakdown with no measured components is `NotInstrumented` and its component fields are `null`, never zero-filled and never set equal to the total.

### §2.2 Consumer registry

`contracts/trace-consumers-v1.yaml`, one entry per surface:

```yaml
- surface: "renacer validate"          # CLI verb, flag value, route, MCP tool, or library API
  crate: aprender-profile
  consumer: {kind: gate, ref: ".github/workflows/guards-nightly.yml#renacer-golden"}
  status: active                      # active | report-only | deprecated
  first_green: {run_url, sha}         # required before status: active
```

- `kind ∈ {ledger, gate, runbook}`. A `runbook` consumer names the file and step that tells a person when to run it and what decision it feeds.
- The surface list is derived from the **built binaries** by `scripts/dogfood_surfaces.sh` (never grep; SKILL G1), filtered to trace/profile/metrics surfaces.
- Gate (`pv` contract `trace-consumers-v1`): a derived surface absent from the registry is RED; a `deprecated` surface still present after its `remove_by` version is RED; an `active` entry without `first_green` is RED.
- Each entry carries `platforms: [linux, macos, …]`. Running a surface on a platform outside that list yields `NotRun{UnsupportedPlatform}`, which is counted in the §8 report and is never a pass or a silent skip.

### §2.5 Golden-trace thresholds (TR-06)

For each baselined workload w and syscall class s, capture N ≥ 10 `[A]` baseline runs on the clean-room runner. With c_i(s) the count in run i:

$$
\tilde\ell_s=\operatorname{med}_i \ln\big(1+c_i(s)\big),\quad
\hat\sigma_s=1.4826\cdot\operatorname{med}_i\big\lvert \ln(1+c_i(s))-\tilde\ell_s\big\rvert,\quad
\theta_s=\max\!\Big(\ln\big(1+\tfrac{1}{1+\tilde c_s}\big),\ 3\hat\sigma_s\Big),\quad \tilde c_s=e^{\tilde\ell_s}-1
$$

- The floor is the log-distance of **one syscall** from the median count, so a fully deterministic class (σ̂ = 0) tolerates exactly one-call jitter and nothing invented.
- RED when |ln(1 + c(s)) − ℓ̃_s| > θ_s for any s on 2 consecutive runs, or > 2θ_s on one run (APR-OBS E4 form).
- New syscall classes absent from the baseline are always RED (a new `fsync` in the decode loop is the defect this gate exists for).
- Timing is **not** gated here; `max_time_ms` budgets in `renacer.toml` are report-only.
- Falsifier: +10 000 `write` RED on the first run; the positive specimen (unchanged build) stays GREEN in ≥ 99% of 100 seeded reruns `[C]`, committed as a receipt.

### §2.3 renacer as the external serve witness (feeds APR-OBS OBS-04)

Run the serve process under renacer (`renacer -f -e trace=accept4,recvfrom,read,sendto,write,writev --format json -- <pinned apr> serve …`, exact flags confirmed against `crates/aprender-profile/src/cli.rs` at HEAD). Per client connection fd *k*:

$$
\text{TTFT}^{\text{ext}}_k = t\big(\text{first } \texttt{write/sendto/writev} \text{ on } k \text{ after the request is fully read}\big) - t\big(\texttt{accept4} \to k\big)
$$

$$
\text{wall}^{\text{ext}}_k = t(\text{last write on } k) - t(\texttt{accept4} \to k)
$$

- The first definition holds for streaming responses. For non-streaming, only wall^ext is defined; TTFT^ext is `null` with `ttft_reason: "non-streaming"` (same rule as APR-OBS §2.3).
- The witness measures from outside the subject's address space, so it satisfies APR-OBS §0.3 more strongly than the server's own `timings`.
- Its own overhead is measured, not assumed: the same workload is run with and without renacer, and the overhead ratio is recorded per night.
- Agreement with server `timings` and client wall uses APR-OBS OBS-04's tolerance, `max(5%, 50 ms)` `[A]`. A planted +20% skew in server timings must go RED against the external witness alone.
- **Isolation from perf data (review ruling).** Witness runs happen only in dedicated c=1 lease blocks, never inside a gated perf block. Every row from a traced process carries `traced: true` and is excluded from every APR-OBS ratio, η and ω series. Under ptrace the server is perturbed; the witness is valid for agreement, not for speed.
- **Filter efficiency.** The tracer must stop the tracee only on the listed syscalls (seccomp-filtered tracing if renacer supports it at HEAD `[U]`). Measured per run: `stops / total_syscalls`. If renacer stops on every syscall, that is a renacer defect ticket, and the witness stays report-only.

**Obligation (Kani `KANI-TRACE-WITNESS`):** for any event sequence of ≤ 8 syscalls on one fd with monotone timestamps, the parser returns TTFT^ext ≤ wall^ext, and returns `null` TTFT when no write follows a completed read.

### §2.4 CRUX binding for every tracing surface

**Coverage matrix** (TR-10 emits it as `crux-trace-bind-receipt.json`; a surface with no row is RED):

| Surface | CRUX contract | Competitor arms (pinned) | Metric (same tool for every arm) |
|---|---|---|---|
| `apr trace --layers`, `apr run --trace-level` | **F-02** (extend: re-discharge on `apr-trace-v1`) | PyTorch profiler | per-layer set equality; Σ layers ≤ wall |
| `apr profile --roofline` | **F-05** (extend: status → `partial` until PERF-016 clears CF-5) | PyTorch profiler, ncu roofline | arithmetic intensity per op vs ncu on the same kernel |
| Chrome trace export | **F-07** (reuse) | PyTorch profiler | schema validity; no negative/NaN durations |
| `/metrics` | **K-07** (reuse; OBS-03 discharges it) | vLLM | metric-name set; histogram counts = requests |
| OTLP spans from `apr serve` | **K-08** (unblock, TR-13) | vLLM, TGI (pinned images) | `apr otlp-lint --require-apr-span --require-genai-attrs` on a captured export |
| renacer syscall tracing | **new** `crux-F-NN-syscall-trace` | strace (pinned), `perf trace`, bpftrace | E-S1, E-S2 below |
| `renacer validate` golden traces | **new** `crux-F-NN-golden-trace` | strace `-c` summary diff (the practical baseline; no packaged competitor `[U]`) | planted-regression detection rate; false-RED rate on unchanged builds |
| `cgp profile` (cuda, simd, wgpu) | **new** `crux-F-NN-kernel-profile` (F-16 stays export-only) | nsys, ncu (pinned) | E-S3 below |
| BrickTracer per-op attribution | **new** `crux-F-NN-op-attribution` | PyTorch profiler; `perf record` on CPU | Σ components ≤ wall; `NotInstrumented` when unmeasured (R-1) |
| External serve witness (TR-07) | none: an instrument, not a product claim | — | governed by APR-OBS OBS-04 |

**E-S1 (syscall parity, renacer vs strace).** Two ptrace tracers cannot attach to one process, so parity is measured across interleaved runs (ABBA, APR-OBS T1) of a deterministic workload. With n_t(s) the count of syscall s in a run traced by tool t:

$$
D(t_1,t_2)=\frac{\sum_s \lvert n_{t_1}(s)-n_{t_2}(s)\rvert}{\sum_s n_{t_2}(s)}
$$

The gate compares renacer against strace's own run-to-run noise, so no threshold is invented:

$$
\text{parity\_ok}\iff \operatorname{med}_{\text{runs}} D(\text{renacer},\text{strace}) \le q_{0.95}\big(D(\text{strace},\text{strace})\big)
$$

**E-S2 (overhead).** ρ_t = wall_traced / wall_untraced per tool on the same workload; `gap_effect` = ln(ρ_renacer / ρ_strace), where below 0 means renacer is cheaper. Reported every release; the ratchet arms after 3 records (EXT-001 R-15).

**E-S3 (kernel profile agreement, cgp vs nsys).** On one run captured by nsys with cgp's own records alongside (or interleaved runs where both cannot co-attach):

$$
J=\frac{\lvert K_{\text{cgp}}\cap K_{\text{nsys}}\rvert}{\lvert K_{\text{cgp}}\cup K_{\text{nsys}}\rvert}=1,\qquad
\forall k:\ \lvert\ln(\tau^{\text{cgp}}_k/\tau^{\text{nsys}}_k)\rvert\le q_{0.95}\big(\lvert\ln(\tau^{\text{nsys},1}_k/\tau^{\text{nsys},2}_k)\rvert\big)
$$

J = 1 means cgp sees exactly the kernels nsys sees. The time bound is again nsys's own run-to-run dispersion.

**Obligations:** `KANI-CRUX-D`: D ≥ 0, D(t,t) = 0 on identical count vectors, and D is invariant to syscall ordering. `KANI-CRUX-J`: J ∈ [0,1], and J = 1 iff the kernel sets are equal.

---

## §3 Contracts and falsifiers

| Contract | Row | Falsifiers (must turn RED) | Positive specimen (must stay GREEN) |
|---|---|---|---|
| `brick-tracer-provenance-v1` | TR-01 | a traced run with no ptrace events reporting `compute_us == duration_us`; any `SyscallBreakdown` component non-null under `NotInstrumented` | a run with real events reports `Measured` with non-null components |
| `no-false-escalation-v1` | TR-02 | a message containing "Enabled for syscall breakdown" printed with no tracer run; an efficiency computed from an undeclared literal | the escalation message printed with `provenance` and the actual action taken |
| `serve-trace-payload-v1` | TR-03 | `X-Trace-Level` response on a model whose `num_layers ≠ 28` reporting 28; payload without `provenance` | Qwen3.5-4B response reporting its real layer count and `WallClockTotal` |
| `no-mock-named-real-v1` | TR-04 | a module or type named after a real workspace crate (`renacer`, `trueno`, `realizar`, …) defined outside that crate | a mock named `mock_trace` |
| `tool-resolution-v1` | TR-05 | any script or Makefile target that runs `cargo install renacer` or a bare `renacer` | `scripts/renacer_bin.sh` resolving the in-tree binary and proving it built from HEAD |
| `renacer-golden-v1` | TR-06 | a `renacer.toml` without a committed baseline; a planted budget breach (extra 10 000 `write` calls) staying GREEN; a new syscall class staying GREEN; a gate that traces only `--version` | an unchanged crate GREEN in ≥ 99% of 100 seeded reruns (§2.5) |
| `serve-external-witness-v1` | TR-07 | planted +20% server-timing skew passing; witness TTFT on a non-streaming request not `null`; a `traced: true` row admitted to any APR-OBS perf series | a real streaming W1 request with all three timings in tolerance |
| `trace-consumers-v1` | TR-08 | an unregistered trace surface; `active` without `first_green`; `deprecated` past `remove_by` | the registry at HEAD after TR-08 lands |
| `crux-F-NN-syscall-trace-v1` | TR-11, TR-15 | planted renacer build that drops `openat` events passing E-S1; overhead unreported | renacer and strace on the W1 workload with parity_ok |
| `crux-F-NN-golden-trace-v1` | TR-11, TR-15 | planted +10 000 `write` regression undetected | unchanged build GREEN on 3 runs |
| `crux-F-NN-kernel-profile-v1` | TR-11, TR-15, TR-16 | planted cgp build missing one kernel (J < 1) passing | gx10 W1 decode with J = 1 |
| `crux-F-NN-op-attribution-v1` | TR-11, TR-15 | Σ components > wall; zero-filled components under `NotInstrumented` | a CPU run with `perf record` agreement |
| CRUX-K-08 (existing) | TR-13 | `apr serve` with `OTEL_EXPORTER_OTLP_ENDPOINT` set emitting no `apr.inference` span; missing `gen_ai.*` attrs | `apr otlp-lint --require-apr-span --require-genai-attrs` GREEN on a captured export |
| CRUX strace→renacer (A-05, A-07, A-08, A-15, A-20, C-27, G-10, G-15) | TR-12 | a contract switched to renacer before E-S1 parity is recorded | each contract's falsifier outcome identical under renacer and strace |
| `apr-trace-v1` (owned by APR-OBS OBS-09) | TR-09 | `apr run --trace-level layer` and serve `X-Trace-Level: layer` producing different schemas for the same request | both emitting `apr-trace-v1` |

---

## §4 Tickets (EV-ordered)

`K̂` in minutes `[A]`. Phase A removes false outputs and is small, mechanical and high-value, so it goes first.

| EV | Row | Work | Contract | Done when (all must hold) | deps | K̂ |
|---|---|---|---|---|---|---|
| 1 | **TR-01** BrickTracer honesty | Hoist `TraceProvenance` to `aprender-profile-core`. `trace_with_reason` returns `NotInstrumented` with `null` breakdown components when no syscall events were captured; delete the false doc line. Callers in `bench*.rs`, `benchmark.rs`, `qa.rs` print the provenance | `brick-tracer-provenance-v1` | falsifiers RED→GREEN; `git grep -n "compute_us: duration_us"` → 0 | — | 60 |
| 1 | **TR-02** cbtop escalation | Delete the unused tracer and the false message. Either run a real escalation (re-exec the measured step under the in-tree renacer and attach the trace path) or print `escalation: not traced`. **Remove the efficiency arm and its `976.0`**; escalate on CV from the run's own samples, or on an APR-OBS RED for the matching series | `no-false-escalation-v1` | falsifiers RED→GREEN; `git grep -n '976\.0' crates/apr-cli` → 0 | — | 45 |
| 1 | **TR-03** CPU serve trace payload | Replace both hand-built payloads (`handler_apr_cpu_completion.rs:397,747`) with `aprender_serve::api::build_trace_data(level, latency_us, prompt, completion, state.num_layers)` | `serve-trace-payload-v1` | a request on a model with ≠ 28 layers returns its real count; `git grep -n '"layers": 28'` → 0 | — | 45 |
| 2 | **TR-04** mock rename | Rename `tests/modality_matrix/common.rs` `mod renacer` to `mod mock_trace`; relabel QA-A08 as a mock self-test; add the `no-mock-named-real-v1` lint to `pv lint` or a guard script (Rust, no Python) | `no-mock-named-real-v1` | lint RED on a planted `mod trueno` in a test; 0 hits at HEAD | — | 45 |
| 2 | **TR-05** tool resolution | `scripts/renacer_bin.sh` (sourced, `return` never `exit`, mirrors `apr_bin.sh`) builds and resolves `target/release/renacer` from HEAD. Rewrite `capture_golden_traces.sh` and `Makefile profile:` to use it; delete the `cargo install` line | `tool-resolution-v1` | `git grep -nE 'cargo install renacer\|^\s*renacer ' -- scripts Makefile` → 0; resolver selftest (target absent → caller survives) | — | 60 |
| 3 | **TR-06** golden gate made real | For each of the 6 `renacer.toml` crates: capture a baseline over a **real workload** named in the toml (an example or bench binary, not `--version`), commit `.renacer/`, or delete the toml if no workload exists. New nightly job `renacer-golden` in `guards-nightly.yml` on a Linux clean-room runner, report-only. `dogfood.sh:1091` traces the same workload | `renacer-golden-v1` | 0 tomls without a baseline; planted breach RED; 3 green nights recorded; first-green run URL in the registry | 05 | 180 |
| 4 | **TR-07** external serve witness | Rust parser (`aprender-profile` subcommand or xtask) from renacer JSON to `apr-serve-witness-v1` rows per §2.3; overhead ratio measured. Filed on APR-OBS as **OBS-16**, joined to OBS-01/OBS-03 rows by connection and `request_id` | `serve-external-witness-v1` | Kani `KANI-TRACE-WITNESS` declared; planted skew RED from the witness alone; overhead ratio committed `[V]` | 05, APR-OBS OBS-03 | 180 |
| 5 | **TR-08** consumer registry | `contracts/trace-consumers-v1.yaml` seeded from the built-binary surface derivation; every trace/profile/metrics surface gets `consumer` or `deprecated` + `remove_by`; gate in `ci / gate` report-only | `trace-consumers-v1` | registry covers 100% of derived surfaces; unregistered-surface falsifier RED | 01–06 | 150 |
| 6 | **TR-09** one trace schema | `apr run --trace-level layer\|chrome` becomes a renderer over `apr-trace-v1` emitted by `InferenceTracer`; chrome output is a format of the same data. Filed on APR-OBS as **OBS-17** | `apr-trace-v1` | same request through `apr run` and serve yields byte-identical `apr-trace-v1` modulo timestamps | APR-OBS OBS-09 | 120 |

| 3 | **TR-10** CRUX bind | Enumerate existing `contracts/crux-*.yaml` via `pmat query` over the loader (not grep). Emit `crux-trace-bind-receipt.json` (§2.4 matrix: surface × contract × pinned arms). Re-check G14: F-02/F-05 `active` without a discharge record → set to `partial` with the missing discharge named. File the category-drift issue (G17) on the CRUX master owner; do not rename IDs | reuse | every tracing surface has a row or `NotRun{reason}`; F-05 no longer `active` while CF-5 stands; issue URL recorded | 01, 03 | 60 |
| 4 | **TR-11** new CRUX contracts (define) | Four contracts in the existing schema (syscall-trace, golden-trace, kernel-profile, op-attribution) with equations E-S1..E-S3, `gap_effect: unmeasured`, Kani obligations `KANI-CRUX-D`, `KANI-CRUX-J`, every competitor arm pinned (version + digest) or `Refused{reason}`, falsifiers planted | four `crux-F-NN-*` | `pv validate` counts the new obligations; 0 anonymous obligations; each planted falsifier RED | 05, 10 | 120 |
| 5 | **TR-15** CRUX first records | Rust harness (no Python, R-3) runs every arm of the four contracts on the clean-room runner (gx10 for kernel-profile): E-S1 parity, E-S2 overhead, E-S3 agreement, op-attribution sums. Commit the first record per contract; set `gap_effect` from it; ratchets arm after 3 records (EXT-001 R-15) | four `crux-F-NN-*` | first record committed per contract; `gap_effect ≠ unmeasured`; E-S1 `parity_ok` recorded (unblocks TR-12) | 06, 11 | 120 |
| 4 | **TR-13** K-08 unblocked | Wire renacer's `otlp_exporter` into `apr serve` behind `OTEL_EXPORTER_OTLP_ENDPOINT`: one `apr.inference` span per request with `gen_ai.*` attributes and W3C trace-context propagation. Off by default. Remove the BLOCKER-UPSTREAM-MISSING note | CRUX-K-08 | K-08 falsifiers discharged FULL against a live `apr serve`; vLLM arm pinned or `Refused{reason}` | 01 | 150 |
| 5 | **TR-12** dogfood renacer in CRUX | After TR-15 records E-S1 parity: switch the 8 strace-based CRUX falsifiers (G16) to the in-tree renacer via `scripts/renacer_bin.sh`, keeping strace as a recorded oracle arm | 8 existing contracts | each falsifier's outcome identical under both tools on 3 runs; renacer is the instrument in 8/8 | 15 | 90 |
| 3 | **TR-14** CRUX coverage for tracing | **No new gate.** Add the tracing surfaces from TR-10's bind receipt to EXT-001 **EXT-25**'s CRUX coverage check in `ci / gate` (report-only first, R-6). When TR-08 lands, the check reads the registry instead of the receipt | extends EXT-25 | planted tracing surface with no contract RED in EXT-25's check | 10, EXT-25 | 45 |
| 7 | **TR-16** 0.73 platform arms | Extend `crux-F-NN-kernel-profile` with wgpu (intel AMD) and Metal (mini) arms against each platform's native profiler, pinned or `Refused{reason}`; registry `platforms` updated; macOS surfaces report `NotRun{UnsupportedPlatform}` where syscall tracing does not exist | `crux-F-NN-kernel-profile-v1` | every 0.73 backend has an arm or a recorded refusal; 0 silent platform skips | 11, 15, APR-OBS OBS-18 | 90 |

**K̂ = 1560 `[A]` · K = 1730 · andon at 1384.**

### §4.1 Milestones

Themes and exit criteria are APR-LOOKAHEAD-001 §2a. Epics: 0.71 → **#3598** `[V]`; 0.72 → **#4000**, 0.73 → **#3999** `[A]` (cop report 2026-09-27; confirm at HEAD). One ticket per row, minted by the cop as a sub-issue of the epic; one parent issue per PR (APR-EPIC-001).

| Milestone | Rows | Exit criterion served | Merge-by constraint |
|---|---|---|---|
| **`0.70.0`** (rc.1, 2026-09-27) | none | — | The current train carries release-critical scope only |
| **`0.71.0`** Verbs Are Fast | TR-01, 02, 03, 04, 05, 09, 10, 11, 14 | **V6** per-layer serve tracing ← TR-03 (honest serve trace payload) and TR-09 (= APR-OBS OBS-17, one schema). **V7** EXT-001 CRUX gates defined ← TR-10, TR-11, TR-14. **V1/V2** trust in reported timings ← TR-01, TR-02 remove fabricated attribution from `apr bench`/`qa`/`cbtop` | TR-01..TR-04 first (jidoka: no instrument is wired while it can still lie). TR-03 and TR-09 merge before the first 0.71 performance PR, together with APR-OBS OBS-09 (V6). TR-10, TR-11, TR-14 merge before the 0.71 freeze (V7) |
| **`0.72.0`** Train What You Serve | TR-06, 07, 08, 12, 13, 15 | Hardening that keeps 0.71's speed claims honest while training verbs land: golden gate (TR-06), external witness (TR-07 = APR-OBS OBS-16), registry (TR-08), CRUX records (TR-15), renacer dogfooded in CRUX (TR-12), K-08 OTLP discharged (TR-13) | TR-15 before TR-12 (parity first, S-8). TR-06 gate becomes blocking only after R-6 (first-green + 7 green nights) |
| **`0.73.0`** Runs Everywhere | TR-16 | **E5/E6** kernel profiling and registry coverage on wgpu and Metal ← TR-16 with APR-OBS OBS-18 | After OBS-18 has admissible rows on the new hosts |

A row that misses its cut moves to the next milestone with `slipped_from:`. If it serves an exit criterion (V1, V2, V6, V7), the move needs an operator ruling; exit criteria are never waived.

---

## §5 Hard rules

- **R-1** No output may attribute time it did not measure. Unmeasured is `null` + `NotInstrumented`, never zero and never the total.
- **R-2** In-tree tools only (`scripts/renacer_bin.sh`, `scripts/apr_bin.sh`); never crates.io, never bare `PATH`.
- **R-3** No Python in any file this spec creates. Parsers and guards are Rust.
- **R-4** No `$?` through a pipe; shell passes `bashrs`.
- **R-5** A mock is named as a mock.
- **R-6** New gates are report-only until first-green plus 7 green nights `[A]`.
- **R-7** Workflow-file PRs are merged by the agent, never assigned to the operator.

---

## §6 Scoreboard (targets; state is rendered, never written here)

| Metric | Baseline `[V]` at `aca6f2d7f` | Target |
|---|---|---|
| False-telemetry sites (F1–F5) | 5 | **0** |
| `renacer.toml` without committed baseline | 6 / 6 | **0** |
| CI jobs running the renacer **binary** | 0 | **≥ 1** nightly, blocking after R-6 |
| Scripts/targets resolving renacer from crates.io or bare `PATH` | 2 | **0** |
| Mock modules named after real crates | 1 | **0** |
| Trace/profile surfaces with a registered consumer | not measured | **100%**; `deprecated` count shrinks every release |
| Serve requests on the nightly with an external witness row | 0 | **1 per APR-OBS series per night** |
| Per-layer trace schemas | 2 | **1** |
| Tracing surfaces with no CRUX contract (G13) | 5 | **0** |
| CRUX `active` contracts with no discharge record, among tracing rows (G14) | 2 | **0** |
| CRUX falsifiers using external strace as the instrument (G16) | 8 / 8 | **0 / 8** (strace kept as oracle arm) |
| K-08 discharge | PARTIAL, blocked | **FULL** |
| E-S1 parity renacer vs strace | not measured | recorded every release; `parity_ok` |

---

## §7 STOP conditions

- **S-1** A row would take a runner or CUDA host from an active release cut, or run while `train-active` is set.
- **S-2** renacer cannot trace the pinned apr binary on the Linux clean-room runner (ptrace denied by the container profile). Write the §8 report with the exact error; the fix is an infra issue, not a workaround.
- **S-3** A `renacer.toml` names no workload that exists at HEAD **and** its crate owner is unknown: list it for an operator ruling (delete or assign).
- **S-4** Deleting a surface (TR-08 `deprecated`) would remove a public CLI verb or route in a published crate: operator ruling on `remove_by`.
- **S-5** A harness hook refuses writes for a reason other than a missing ticket.
- **S-6** Budget K reached, or andon crossed with more than one row incomplete.
- **S-7** A competitor arm cannot be pinned hermetically (no digest, no version): mark it `comparator: unrunnable-hermetically`, report-only, never cited (EXT-001 S-12).
- **S-8** E-S1 parity fails: renacer disagrees with strace beyond strace's own noise. TR-12 does not start; the diff goes to a renacer defect ticket.
- **S-9** The CRUX category enum is closed and owned outside this repo: file the issue and use the nearest existing category; never fork the enum.

---

## §8 Final report schema (one per session)

```yaml
spec: TRACE-001
session: {date_utc, host, tree: {head, origin_main, worktree}, model}
selected_row: TR-NN
ticket: PMAT-NNNN
outcome: merged | stopped | already-done | premise-falsified
stop: {id: S-n, evidence: "..."}
baseline_remeasured: [{row: G#|F#, value, mark, command}]
false_telemetry_sites: {before, after, list: [{file, line}]}
golden: [{crate, workload, baseline_path, status: report-only|active, first_green_url}]
witness: {rows, overhead_ratio, stops_per_syscall, agreement_pct, traced_rows_in_perf_series: 0}
platform_notrun: [{surface, platform, count}]
registry: {surfaces_derived, registered, deprecated, unregistered}
milestone: 0.71.0 | 0.72.0 | 0.73.0
crux: {bind_receipt, surfaces_without_contract, new_contracts: [id], k08_discharge, strace_to_renacer: "n/8", parity: {D_rs_median, D_ss_q95, ok}, overhead: {rho_renacer, rho_strace}}
falsifiers: [{contract, planted, red_run_url, green_run_url}]
budget: {k_hat_row, k_actual_row, cumulative, andon_crossed: bool}
next_row: TR-NN
```
