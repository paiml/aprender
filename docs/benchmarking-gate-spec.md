# APR-PERF-GATE-001 — Benchmarking, Gating, Dogfood and Repo Migration

**Status:** DRAFT FOR TEAM REVIEW · **Date:** 2026-08-25 · **Refs:** #2693 #2695 #2696 #2697

> **Operator ruling:** this is the most important gate in the project. Not following
> it could kill adoption. *"If this is not addressed the entire project shouldn't
> exist."*

This document is the single place to review four things that turned out to be one
thing: what we measure, what blocks a release, how we dogfood on hardware, and
what we assimilate from prior work before archiving it.

---

## 0. Read this first — four verified adoption killers

Each is independently sufficient for a new user to leave and not come back. All
were measured on 2026-08-25, on a quiet box, against llama.cpp pinned at
`39173bcac`.

| # | Defect | Evidence |
|---|---|---|
| 1 | **`cargo install aprender` is CPU-only and says nothing.** `apr serve run --gpu` accepts the flag and runs on CPU. A user with an idle RTX 4090 gets **15.7 tok/s and 7.5 s to first token** vs llama.cpp's 158.9. | published `aprender-0.64.0/Cargo.toml` read from static.crates.io: `default = ["cli"]`, `cuda` opt-in. #2696 |
| 2 | **apr does not batch.** Aggregate throughput is flat at ~110 tok/s at every concurrency — **0.097× llama.cpp at c=16**. | band sweep, §2 |
| 3 | **`--batch` hangs** on four concurrent chat requests. Advertised as "2X+ throughput". | probe timed out at 9m50s; benchmark reported 0.5 tok/s aggregate |
| 4 | **The book publishes a number no harness produced** — "851.8 tok/s = 2.93× Ollama" from a run where Ollama never executed. | `book/src/examples/showcase-benchmark.md:17,22`; `book/src/tools/apr-cli.md:1396,1493,1498` |

**Our gate caught none of them.** It measured c=1 only, per-user decode only, on a
local `--features cuda` build, and never read the book.

### 0.1 The instrument was lying too

`apr profile --granular` prints `"Large non-kernel overhead — investigate sampling
sync (gpu_argmax D2H)"`. That is a **hardcoded string** fired on a threshold
(`crates/apr-cli/src/commands/profile_print_hotspot.rs:155-162`); the tool never
inspects sampling. The 90.3% it attaches is arithmetically invalid —
`crates/apr-cli/src/commands/kernel.rs:531-542` subtracts a **16-token** kernel sum
from a **32-token** wall time, and the "kernel" term is CPU `Instant` timing, not
GPU time. It was quoted in this investigation as evidence for a root cause before
being caught. Neither number nor cause may be cited.

---

## 1. Why one metric was not enough

The two metrics move in **opposite directions** as concurrency rises. Measured on
lambda (RTX 4090), comparator pinned `39173bcac`, quiet box:

| band | llama agg | apr agg | **agg ratio** | llama dec | apr dec | dec ratio |
|---|---|---|---|---|---|---|
| c=1 | 168.9 | 90.2 | 0.534× | 171.5 | 100.7 | 0.587× |
| c=4 | 484.7 | 111.9 | 0.231× | 123.3 | 113.8 | 0.923× |
| c=8 | 650.5 | 109.6 | 0.169× | 83.0 | 112.2 | **1.352×** |
| c=16 | 1120.8 | 108.4 | **0.097×** | 71.2 | 110.6 | **1.554×** |

apr's aggregate is flat because it **serialises**. Per-user decode *rises* to
1.554× purely because each request gets the whole GPU in turn while llama.cpp
shares it sixteen ways.

**A gate reading only per-user decode scores c=16 a comfortable PASS while a
sixteen-user deployment runs at a tenth of llama.cpp.** That is the cannot-fail
shape, in the one gate that decides whether anyone adopts this.

`scripts/llama_pin.toml` said `http_concurrency = 1`. That single line is why every
parity number this project has published measured the worst band and called it the
answer.

---

## 2. Competitive research (CRUX)

Quorum across **six projects, 129 practices**: llama.cpp, vLLM, Ollama,
TGI/SGLang, Rust runtimes (candle, mistral.rs, tract), and non-LLM perf-rigor
projects (rustc-perf, ClickHouse, SQLite, Chromium).

### 2.1 What the field does that we did not

| Practice | Who | Blocking there | Our status |
|---|---|---|---|
| Concurrency sweep is the **unit** of measurement, not a loop the caller writes | vLLM (`max_concurrency` list), llama.cpp (`-npl 1,2,4,8,16,32`), TGI (30 rates), SGLang, mistral.rs (c∈{1,8,16}) | SGLang: release | **was absent** — now banded |
| Aggregate throughput gated **jointly** with per-request latency | SGLang (`output_throughput>3800` **and** `median_ttft_ms<86`), vLLM (TTFT≤3000 **and** TPOT≤100), mistral.rs (`{8:450, 16:500}`) | yes | **absent**: only `let threshold = 10.0;` (`crates/apr-cli/src/commands/bench.rs:255`) — passes at 110 tok/s while we lose 10.3× on aggregate |
| `completed == requested` before reading throughput | SGLang | yes | **was absent** — adopted |
| Hardware/config identity as a **JOIN KEY** so cross-host comparison is impossible | llama.cpp `compare-llama-bench.py` (25 properties must agree) | n/a (tool) | ours agreed on 4, none naming host or workload — adopted as REPORTING |
| Fail-closed when the fast path is unreachable in this build | ollama `server/sched_test.go`, `ml/device_test.go` — hardware as data, GPU-less runner | **every PR** | **absent** — this is defect #1 exactly |
| Self-referential dispatch assert: picked path vs best measured in the same run | tract `hwbench --assert --tolerance 20` | the one perf gate in 20 projects trusted to fail a build | **absent** |
| Threshold derived from the metric's own history, `needs_history` for new cells | tract `.travis/bench-thresholds.toml` (k=3.0 × historical dispersion) | advisory | **absent** — hardcoded `10.0` |
| Comparator pinned by SHA + literal build flags, captured into the result | mistral.rs `capture_metadata.sh`; ClickHouse pins baseline binary **and** checks out `tests/performance` at the baseline SHA | ClickHouse: yes | landed on branch (`scripts/llama_pin.toml`), **not on main** |

### 2.2 The finding that reframes the problem

**On claim-drift, all twenty surveyed projects abstain.** llama.cpp publishes *zero*
quantitative performance claims. vLLM says "state-of-the-art serving throughput" and
gives no number. Ollama, candle, SGLang, TGI: nothing.

Nobody has a guard because **nobody makes the claim**. We make claims, so we need
the guard — but *make fewer claims* is the cheaper half of the answer, and it is
llama.cpp's actual strategy.

### 2.3 Where we are ahead — do not regress this

**No surveyed project installs from a package registry and measures the installed
artifact.** `scripts/check_multiplatform_dogfood.sh` does, with a `-lt 4` host floor
and its comparand read from protected `origin/main` so the matrix can grow and never
shrink. That is the correct answer to *"any state the author writes and the gate
reads can be moved in the same commit"*, and it is the foundation everything else
bolts onto.

---

## 3. The gate, specified

**Entry point:** `scripts/perf_gate.sh --host <name>`, host resolved from a committed
`perf-matrix.yaml`, runnable verbatim on a dev box.

### 3.1 What is measured

The **published crates.io binary**, installed by the step that already writes
`evidence/dogfood/<version>/<host>.json`:

```
cargo install aprender --version X.Y.Z --locked --force
. scripts/apr_bin.sh || exit 1     # never a bare apr, never a hardcoded path
```

Never a workspace build. Never `target/release`. The receipt records `apr --version`
read from the running binary plus `binary_sha256` — not the CI commit that launched
the job.

> **Why this is stated so emphatically.** On 2026-08-25 `~/.cargo/bin/apr` was
> silently replaced by a local build of main at 06:58, where the day before it was
> the genuine crates.io artifact. A conclusion about "the published binary" drawn
> from a path would have been wrong. Read the crate from static.crates.io.

### 3.2 Bands and metrics

| Band | Client | Required metrics | Blocking | Reporting |
|---|---|---|---|---|
| c=1 | external HTTP, 1 stream | decode tok/s, TTFT, `compute_class` | ratio ∈ **[0.80, 1.50]** | bootstrap 95% CI vs cell history |
| c=4 | 4 concurrent | aggregate tok/s, p95 TTFT, p95 ITL, `completed`, `timeouts` | ratio ∈ [0.80, 1.50] **AND** `timeouts == 0` **AND** `completed == requested` | as above |
| c=8 | 8 concurrent | same | same | as above |
| c=16 | 16 concurrent | same | same | as above |

**Both metrics, every band.** A decode-only band cannot be expressed in the schema.

- **Floor 0.80** — below it a user is better served by llama.cpp. That is the
  adoption question.
- **Ceiling 1.50** — a ratio above it is likelier a measurement error than a win and
  must be explained before it is believed.

### 3.3 Hosts and comparators

| Host | Silicon | c=1 | c=4 | c=8 | c=16 | llama.cpp | ollama | vLLM |
|---|---|---|---|---|---|---|---|---|
| lambda-4090 | x86_64, sm_89, 24 GB | ✅ | ✅ | ✅ | ✅ | ✅ CUDA | ✅ | ✅ |
| gx10 | aarch64, GB10 sm_121, 120 GB unified | ✅ | ✅ | ✅ | ✅ | ⚠️ must be *proven* buildable | ⚠️ | ❌ no aarch64/sm_121 wheel |
| intel | x86_64 CPU, AVX-512 | ✅ | ✅ | ✅ | ✅ | ✅ CPU | ✅ | ❌ NOT_APPLICABLE |
| mini | arm64 macOS, M4 Metal | ✅ | ✅ | ⚠️ unified-memory bound | ⚠️ likely UNMEASURED | ✅ Metal | ✅ | ❌ |

Host list is `HOSTS="lambda intel gx10 mini"`
(`check_multiplatform_dogfood.sh:36`), floored against `origin/main`.

**intel needs a dedicated single-agent label.** `intel-clean-room-{1..16}` is 16
org-scoped agents on one box; a perf run must not share it.

### 3.4 When a band is impossible on a host

Cell status is **three-valued, never two** — this is what keeps the gate from being
either vacuously green or permanently red:

```json
{ "host":"gx10", "band":16, "comparator":"vllm", "status":"NOT_APPLICABLE",
  "reason":"vLLM publishes no aarch64/sm_121 wheel; source build unsupported",
  "permanent":true, "decided_by":"perf-matrix.yaml", "decided_on":"2026-08-25" }

{ "host":"mini", "band":16, "comparator":"llamacpp", "status":"UNMEASURED",
  "reason":"16 GB unified memory — OOM at c=16 with Q4_K",
  "permanent":false, "expires":"2026-11-25", "owner":"@noah" }
```

`NOT_APPLICABLE` is permanent and **excluded from the geomean denominator**.
`UNMEASURED` is dated, owned, expiring, and counts as a FAIL after its expiry.

### 3.5 Invariants

1. The expected cell set is enumerated from committed `perf-matrix.yaml`; the verdict
   job asserts every expected cell is present.
2. `provenance.compute_class` is the dispatch path **taken**, read from the running
   process — not the hardware present.
3. No `ratio` is representable without a `baseline` object that itself passes every
   receipt rule.
4. Raw samples are retained on every cell. A receipt carrying only summary statistics
   cannot be resampled and is rejected.
5. `timeouts > 0` on any band is fatal to that host's ratio.
6. **No wall-clock ratio is a required per-PR check.** Eleven have failed here; one
   blocked all nine open PRs. Blocking belongs at the release cut.
7. Serialize: one global `concurrency: perf-gate`, `cancel-in-progress: false`. Two
   concurrent full runs have twice starved this box.
8. The comparator pin carries an expiry and annotates when stale.
9. **Never retry a perf assertion to green.** Record both attempts; red on median-of-N.

---

## 4. Toyota Way mapping

| Concept | Here | Mechanism / file |
|---|---|---|
| **Jidoka** — the product stops itself | `--gpu` on a build with no GPU backend must fail, not run on CPU | `crates/apr-cli/src/commands/serve/mod.rs::ensure_accelerator_available` — **landed**, exit 9 with a remedy that works |
| **Andon** — one visible signal | one `compute_class()` feeds the serve banner, `/health`, and `provenance.compute_class` | pending |
| **Poka-yoke** — unwriteable, not detected | a lane must carry `bands`; a band must carry **both** metrics; a ratio must be derived from that band's samples; a lane can never be greener than its worst band | `scripts/lib/bench_receipt.py` — **landed** |
| **Genchi Genbutsu** — go and see | the gemba is the crates.io binary on four hosts; the receipt is emitted by the process that installed it | `scripts/check_multiplatform_dogfood.sh` — partly landed |
| **Standardized work** | one entrypoint, one receipt schema, one verdict — and the other six harnesses are **deleted**, not deprecated | §6 |
| **Kaizen** — a ratchet that cannot slip | per-cell baseline whose comparand is read from protected `origin/main`; baselines may only shrink | `scripts/claim_literal_baseline.txt` — **landed** |

---

## 5. Five Whys, twice

### 5.1 Failure #2 — no batching, 0.097× aggregate at c=16

1. **Why is aggregate 0.097× at c=16?** The published binary serves one request at a
   time.
2. **Why did nobody notice?** Every number we publish is measured at concurrency 1.
3. **Why has the one sweep harness never run?** `scripts/benchmark-matrix.sh` accepts
   `--batch-sizes 1,8,16,32` and `git grep -l` across `.github/`, `Makefile`, `*.toml`
   returns **zero hits**.
4. **Why is an unwired harness invisible?** `unwired_guards_baseline.txt` enumerates
   *guards*, not measurement harnesses.
5. **Why does no schema require a claim to name the workload it is a claim about?**
   `PROVENANCE_REQUIRED` had no workload field.

**Countermeasure:** a required `workload` object (concurrency, model, quantization,
prompt profile) in `scripts/lib/bench_receipt.py`, so a number that does not say what
it measured is unwriteable.

### 5.2 Failure #4 — 2.93× Ollama from a harness that never ran Ollama

1. **Why does the book publish it?** The ratio was computed against a default
   baseline constant.
2. **Why is there a default baseline?** So the harness would run on hosts without
   Ollama installed.
3. **Why is unmeasured output indistinguishable from measured?** The ratio is computed
   from a bare scalar with no provenance.
4. **Why did it reach the book?** `readme_contract.rs` is wired and covers
   `README.md` — the claim lives in `book/`.
5. **Why do claims and receipts live in disjoint universes?** No index maps a
   published number to the receipt that produced it.

**Countermeasure:** `scripts/check_perf_claims_cite_receipts.sh`, scanning
`README.md`, `book/`, and `docs/`; every performance number must cite an
`evidence/` receipt or be deleted.

---

## 6. Repo migration — assimilate `qwen-coder-deploy`, then archive

`~/src/qwen-coder-deploy` (public, last pushed 2026-03-29) is the **existing**
methodology: 5 runtimes × 5 hosts, forjar deploy/bench/**teardown** isolation,
`docs/specifications/benchmarking-v2.md` (which *designed* `probador llm bench`),
`perf-parity-spec.md`, and 942 result JSONs.

We had **more of its code than its practice**: `crates/aprender-test-lib/src/llm/`
already implements tail percentiles, jitter CV, drift and scoring anchors, and
`git grep "llm load\|llm score"` across `scripts/`, `.github/`, `Makefile` returns
**zero callers**.

### 6.1 Placement decisions

| # | Decision | Authority |
|---|---|---|
| **P1** | **No `git subtree`** — curated copy, source SHA recorded per landed file. Subtreeing 1,214 files / 312 MB / 475 commits puts every dropped blob in every clone forever. History is preserved by the archived read-only repo, which is what archiving is *for*. | deviation from APR-MONO Phase 2, flagged |
| **P2** | Specs → `docs/specifications/aprender-serve/`, components → `.../sub/` | precedent `docs/specifications/aprender-compute/sub/` |
| **P3** | Contracts → flat `contracts/` | APR-MONO F5 |
| **P4** | Evidence → `evidence/qwen-coder-showdown-2026-03-29/` + `findings.json`, **dated by source HEAD** so the 5-month staleness is legible in the path | Convention 8 |
| **P5** | Reproducible corpus → `crates/aprender-serve/benchmarks/qwen-coder/`. No root `benchmarks/`. | Convention 9 |
| **P6** | **forjar host descriptors → `paiml/infra` `machines/<host>/`, NOT into aprender** | infra Policy 2 + APR-MONO Appendix B |

**P6 is the one to argue about in review.** The forjar files are the uniform hardware
dogfood mechanism the operator asked for, but per-host state belongs in `paiml/infra`.
The narrow exception is a recipe provisioning only one crate's own toolchain.

### 6.2 Archive procedure

1. Land P1–P6; record source SHA in each landed file.
2. `README.md` of the old repo → pointer to the monorepo paths.
3. GitHub: **Archive** (read-only), do not delete — it is the history P1 relies on.
4. The 942 provenance-free receipts become `UNMEASURED` cells. **Do not backfill.**

---

## 7. What must be DELETED

Standardized work means the other ways *go*. Deprecation is not enough.

**Harnesses** — each a competing protocol with its own defaults. Verified
2026-08-25: all three **still exist on `origin/main`** and are **already deleted on
branch `feat/2692-apr-probar-llm`** (PR #2682, "delete 1,086 lines that fabricate
their comparator"). The deletion is therefore *staged, not shipped* — it lands only
when the stack merges.

- `scripts/benchmark-2x-ollama.sh` — three default baselines (291/120/15)
- `scripts/gpu_2x_benchmark.sh`
- `scripts/benchmark-matrix.sh` — accepts `--batch-sizes 1,8,16,32` and has **zero
  references** in `.github/` or `Makefile`; fold into `apr test llm bench --concurrency`
- `scripts/bench.sh` — still present on the branch, **audit outstanding**

**Fabrication sites** — the pattern `OLLAMA_BASELINE="${OLLAMA_BASELINE:-291}"` is
now referenced only inside `scripts/check_no_fabricated_baselines.sh`, as the case it
bans. Still outstanding: the `"Using default Ollama baseline (318 tok/s from spec)"`
branch, and 15+ `225.0 // Ollama parity` literals in
`crates/aprender-serve/src/gguf/tests/parity*.rs`.

**Rule going forward:** the comparator is a **required argument**. Its absence is an
error, never a default.

**Claims — delete, do not soften (llama.cpp's move):**
`book/src/examples/showcase-benchmark.md:17,22` · `book/src/tools/apr-cli.md:1396,
1493,1498`

**Also open:** whether `scripts/qwen-story.sh` + `.github/workflows/qwen-story-daily.yml`
is a second methodology or a genuinely different subject (correctness vs performance).
**Reviewer input wanted.**

---

## 8. Implementation status

### Landed (branch `feat/2692-apr-probar-llm`, **not on main**)

| Item | File |
|---|---|
| `apr test llm bench` — the harness, English naming | `crates/apr-cli/src/commands/test_llm.rs` |
| Banded protocol, floor/ceiling | `scripts/llama_pin.toml` |
| Band rules, poka-yoke | `scripts/lib/bench_receipt.py` |
| 23-case table, mutation-verified | `scripts/check_parity_receipt.sh` |
| Producer sweeps bands; records `accel-absent` so the gate stays satisfiable | `scripts/parity_host_receipt.sh`, `scripts/lib/parity_block.py` |
| `--gpu` fails loudly on a CPU-only build (Jidoka) | `crates/apr-cli/src/commands/serve/mod.rs` |
| Zero-token and `completed == requested` preconditions | `crates/apr-cli/src/commands/test_llm.rs` |
| Claim-literal guard + 8-case selftest + shrink-only baseline | `scripts/check_no_claim_literals.sh` |
| llama.cpp pin resolver (3 defects fixed; sourcing it used to exit the shell) | `scripts/llama_bin.sh` |

**Proof the gate now works:** fed the real 2026-08-25 data, every band FAILs
including c=8 (decode 1.352×) and c=16 (decode 1.553×) — the two the old gate would
have passed.

### Not started

`perf-matrix.yaml` · `scripts/perf_gate.sh` · `check_perf_claims_cite_receipts.sh` ·
ollama-style fail-closed device test · tract-style dispatch assert · history-derived
thresholds · **the batching fix itself** · gx10/intel/mini wiring (gx10 has a pinned
llama.cpp and a cuda apr built; intel and mini do not)

---

## 9. Questions for review

1. **P6** — forjar descriptors to `paiml/infra`, or in-tree for uniformity? The
   operator asked for one dogfood mechanism; the conventions say per-host state
   leaves the monorepo.
2. **Claims strategy** — adopt llama.cpp's abstention (publish no quantitative
   claims) or keep claims and carry the guard? Abstention is cheaper and is what the
   field does.
3. **0.80 floor vs the prior art's `decode ≥ 1.0×`.** `perf-parity-spec.md` sets a
   stricter bar (decode ≥ llama.cpp, TTFT ≤ 2×, ITL ≤ 1.5×). Which governs?
4. **qwen-story** — one methodology or two subjects?
5. **Release posture for #2696** — ship CUDA in the published artifact, or state
   plainly that crates.io is CPU-only? Until this is decided, the accel lane cannot
   be produced on 3 of 4 hosts.
