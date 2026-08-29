# PERF-054 — measurement campaign scope for APR-PERF-GATE-001 (#2706)

**Reconnaissance only. No cell was measured, `scripts/perf-matrix.yaml` was not modified.**

Probed 2026-08-29 against `origin/main` = `c00ba00cb`. Every claim below cites the
command that produced it. Remote hosts were read-only: nothing was installed,
built, or started on them.

---

## 0. The headline: 0 of 8 cells are measurable today, and the reason is not CUDA

The four CUDA defects (#2767, #2770, #2771, #2774) block **one** cell, `lambda/W1`,
above c=1. They are not what stops the other seven.

What stops all eight is that **no conformant receipt producer exists**. The
§4.4.6/§4.4.7/§4.4.9 receipt emitter was written and landed — and has no caller:

```
$ git grep -ln "ReceiptInput\|TokenizationBlock" origin/main -- crates/ scripts/
crates/aprender-test-lib/src/perf_gate/mod.rs
crates/aprender-test-lib/src/perf_gate/receipt.rs
$ git grep -rn "perf_gate" origin/main -- crates/apr-cli/     # (no output)
```

Both type names resolve only inside `perf_gate/` itself. Nothing in `apr-cli`
references the module. The measurement harness that `scripts/llama_pin.toml`
declares — `apr test llm bench` — emits a different document entirely
(`evidence/parity-http/bands/apr-c1.json`): it has `request_details[]`,
`itl_p50_ms`, `decode_tok_per_sec`, and **no `drain_ms`, no `tokenization`
block, no §4.4.9 scheduler block**.

`scripts/perf_gate.sh:42` fails any receipt whose `drain_ms` is absent. So a
measurement taken today on any of the four hosts produces a document the gate
refuses. This is the mirror of the defect `perf_gate/mod.rs`'s own header
documents (PERF-026) — the emitter was built, the wire was not.

**This is one piece of work and it unblocks four cells at once.** It should be
the first thing the campaign does, before any host burns compute.

---

## 1. Cell table

`E` = estimate (derivation in §5); `M` = grounded in a measured number.

| Cell | Measurable today | Blocked by | Est. wall-clock (both lanes, N=3) |
|---|---|---|---|
| `lambda/W1` | No | **PERF-054-A** (receipt bridge) + **#2774** — caps it at c=1, so Arm A has no numerator | **~1 h** `M` |
| `gx10/W1` | No | **PERF-054-A** only. Everything else is present and conformant | **~1 h** `M` |
| `intel/W1` | No | **PERF-054-A**, + `apr` build at commit-under-test, + **comparator is the wrong build and incomplete** (§3) | **~11 h** `E` |
| `mini/W1` | No | **PERF-054-A**, + `apr` build, + **no llama.cpp and no cmake** (§3), + 16 GB ceiling (§4) | **~8 h** `E`, and c=8/16 probably impossible |
| `lambda/W2` | No | Should not be measured — see §6 | — |
| `gx10/W2` | No | Should not be measured — see §6 | — |
| `intel/W2` | No | Should not be measured — see §6 | — |
| `mini/W2` | No | Should not be measured — see §6 | — |

Recommended order once PERF-054-A lands: **gx10 → lambda(c=1) → intel → mini.**
gx10 is the only host where every prerequisite except the bridge is already
satisfied, so it is also the cheapest place to prove the bridge works.

---

## 2. Host identity — one correction

The four SSH targets are **not** four machines:

```
$ for h in gx10 intel mini mac-server; do ssh $h 'hostname; uname -m -s'; done
gx10        gx10-a5b5   aarch64 Linux
intel       mac-server  x86_64  Linux
mini        mini        arm64   Darwin
mac-server  mac-server  x86_64  Linux
```

`intel` and `mac-server` are aliases for the same host. `lambda` is **this box** —
`noah-Lambda-Vector`, x86_64, RTX 4090 (24564 MiB), 48 cores, 125 GB RAM,
303 G free on `/`, load 9.6. The matrix's four rows map to three remote targets
plus the local box.

Note `docs/specifications/APR-PERF-GATE-001-v2.2.md` §4.9 marks lambda-4090
*"not a CI runner — retired 2026-05-10, do-not-revive"*. That is a statement about
CI enrolment, not about the machine, which is alive and idle-GPU (885 MiB used).
Measuring on it by hand is consistent with §4.9.1, which exists precisely because
the gate cannot be a job that runs *on* the measuring host.

---

## 3. What each host is missing

Model: **all four hosts hold the W1 model, byte-identical.**

```
509287f78cb4d4cf6b3843734733b914b2c158e43e22a7f4bf5e963800894d3c
  lambda /home/noah/models/qwen2.5-coder-7b-instruct-q4_k_m.gguf
  gx10   /home/noah/models/qwen2.5-coder-7b-instruct-q4_k_m.gguf
  intel  /home/noah/models/qwen2.5-coder-7b-instruct-q4_k_m.gguf
  mini   /Users/noahgift/models_perf012_7b_q4km.gguf     (renamed, same bytes)
```

That is the one prerequisite that is fully satisfied fleet-wide. Everything else:

| | `apr` binary | comparator (`llama.cpp` @ `39173bcac`) | build toolchain |
|---|---|---|---|
| **lambda** | none on PATH; use `. scripts/apr_bin.sh` | ✅ `/home/noah/src/llama.cpp/build/bin/` — `version: 7746 (39173bcac)`, `cuobjdump` shows `sm_86 sm_89 sm_120`, init reports *"NVIDIA GeForce RTX 4090, compute capability 8.9"* | cargo present, 303 G free |
| **gx10** | 0.64.0 — **too old** | ✅ `/home/noah/src/llama.cpp-pin/build/bin/` — `version: 7746 (39173bcac)`, `sm_121` present | cargo 1.95.0, 20 cores, 346 G free, load 0.02 |
| **intel** | 0.64.0 — **too old** | ❌ `/home/noah/src/llama.cpp/build/bin/` reports **`version: 1 (35bee03)`** — not the pin. Directory holds **only `llama-server`**: no `llama-cli`, no `llama-bench` | cargo 1.97.1, 32 cores, 345 G free (91% used), load 11.7 |
| **mini** | 0.64.0 — **too old** | ❌ **no llama.cpp at all** | cargo 1.96.0, 10 cores, Xcode CLT present, **no cmake** anywhere |

### 3.1 Every host needs an `apr` rebuild — this is not optional

```
$ ssh gx10 'apr test llm bench --help'
error: unrecognized subcommand 'llm'
```
Same on `intel` and `mini`. All three run `apr 0.64.0`. `crates/apr-cli/src/commands/test_llm.rs`
was **added by #2705** (`ce712eae0`, the v2.2 landing commit); `v0.64.0` is dated
2026-08-23 and `git show v0.64.0:crates/apr-cli/src/commands/mod.rs | grep -c test_llm`
returns `0`. The declared harness does not exist in any installed binary.

This is required anyway: I-10 demands `receipt.commit ⊇ commit-under-test`, so the
binary must be built from the measured commit on every host regardless.

### 3.2 Which arms actually need a comparator

`arms.B1` and `arms.B2` carry `comparator_required: true`. **A, C, D and E do not.**
Arm A is the PRIMARY release-blocking arm and is comparator-free. So `intel/W1` and
`mini/W1` can produce a **complete Arm A/C/D/E cell** with no llama.cpp at all,
marking Arm B `UNMEASURED` per §4.7.2 — the matrix's own `cell_exceptions` block
already models exactly this for `gx10/band-16/vllm`.

That is the single largest scope reduction available: it removes a cmake install
plus a Metal llama.cpp build from mini's critical path, and a comparator rebuild
from intel's, at the cost of two `UNMEASURED` Arm-B sub-cells that are honest.

### 3.3 A declaration/artifact drift worth a line

`scripts/llama_pin.toml` declares `build_flags_lambda = "... -DCMAKE_CUDA_ARCHITECTURES=89"`.
The artifact on lambda carries `sm_86 sm_89 sm_120`. `89` is present so the ratio is
sound, but the file no longer describes the binary. (`strings -a libggml-cuda.so | grep sm_`
gives a *misleading* `sm_120`-only answer; `cuobjdump --list-elf` is the authority.)

Also: two stale llama.cpp trees sit beside the pinned one on lambda
(`~/src/llama.cpp/llama-cli` = `4230 (0c39f44d)`, Nov 2024) and three on gx10
(`version: 1 (…)`). `llama_bin.sh` is safe here — it takes `$LLAMA_BENCH_PATH` only
and never PATH — but any hand-run must pass the `-pin`/`build/bin` path explicitly.

---

## 4. Blockers not on the original list

### 4.1 `apr` uses 3.08× the comparator's VRAM — this is what makes #2774 bite

`evidence/parity-http/findings.json`, lambda, same 7B model, measured 2026-08-24:

```
vram_resident_mib:  llamacpp 4554   apr 14030   ratio 3.081
```

14 GB of a 24 GB card at **c=1**, before `init_batched_kv_cache_gpu` preallocates
its hardcoded `M=32`. That is the mechanism behind #2774's OOM, and it is a
*second* reason the c>1 bands are hard on lambda even after the M=32 fix.

It also propagates to **mini**, which the original list did not cover: mini has
**16 GB unified** (`sysctl hw.memsize` = 17179869184). If the 3.08× footprint
holds there, apr alone wants ~14 GB of a pool shared with the OS. `mini/W1` at
c=8 and c=16 is likely unrepresentable on memory grounds alone — which is
exactly what §4.9 already records (`mini: c=1,4 ✅; c=8,16 UNMEASURED`) but
`perf-matrix.yaml`'s `bands: [1,4,8,16]` does not, and §4.5 makes a band missing
any field **schema-invalid**. The matrix and the spec disagree about mini.

### 4.2 The CUDA defects are genuinely CUDA-only — the CPU/Metal reasoning holds

`init_batched_kv_cache_gpu` is defined at
`crates/aprender-serve/src/cuda/executor/kv_cache_gpu_init.rs:122` and
`.../cache_from_cache_from_kv_cache.rs:67` — both under `src/cuda/`. #2767
(cuMemcpy/streams), #2770 (cuBLAS `m>=4`) and #2771 are in the same subtree.
**None of them is on the CPU or wgpu path.** intel and mini are unaffected.

Two riders. (a) `reject_unsupported_ignore_eos` is called *only* from
`api/cuda_chat_backend.rs`, so the quantized-GGUF chat backend that intel and
mini would use honours `ignore_eos` — W1 is representable there. (b) apr
serialises everywhere (PERF-000, `max_in_flight` expected 1). That is not a
blocker: Arm A is a ratchet and recording a bad honest number is the deliverable.

### 4.3 mini cannot report `compute_class: metal`

`perf-matrix.yaml` declares `mini: compute_class: metal`. The only runtime
producer of that field is `crates/apr-cli/src/commands/bench.rs:283`:

```rust
if cfg!(feature = "cuda") { ... }
if cfg!(feature = "wgpu") { return "wgpu"; }
"cpu"
```

It can never return `"metal"`. Worse, `crates/apr-cli/Cargo.toml` defines
`wgpu = ["inference"]` — a feature that enables **no GPU backend at all**
(the real Metal path is `aprender-serve`'s `gpu = ["trueno/gpu"]`). So
`--features wgpu` yields a build that *reports* `wgpu` while taking the CPU
path: an I-2 violation ("the dispatch path **taken**, never the hardware
present") shipped as a feature flag. `bench_receipt.py:73` only checks that the
class appears in the declared feature list, so it would validate.

**Recommendation:** measure mini as `cpu` and say so, or fix the class producer
first. Do not let a receipt claim `metal`.

### 4.4 Nothing asserts W1's `512 ± 8` prompt-token budget

`git grep -rn target_prompt_tokens origin/main -- crates/ scripts/` matches the
corpus file and nothing else. The corpus's own `_meta` says
`token_count_verified: false` and *"the 512 ± 8 of §4.3.1 is asserted by the
harness against the model's own tokenizer at measurement time"*. No such
assertion exists. `body_words: 496` of short code-ish tokens may or may not land
inside 512 ± 8 under the Qwen BPE, and there is no `--body-words` retune loop
because nothing measures the miss.

**This must be resolved before the first replicate**, not after: every receipt
taken against an out-of-band corpus is discarded, and §4.4's `N=3` / I-9 rule
forbids re-running a replicate to green.

### 4.5 The receipt validator is behind the spec

`scripts/lib/bench_receipt.py` enforces `provenance`, the four join keys and
`samples_ms`. It does **not** enforce §4.4.6 `tokenization` (I-13 calls its
absence schema-fatal), §4.4.9's scheduler block, or `drain_ms`. A receipt that
omits all three validates. `perf_gate.sh` catches `drain_ms` and
`admission_rejected`; the tokenization block is caught by nothing.

So even after PERF-054-A lands, a receipt can be *accepted* while missing the
fields the spec marks REQUIRED. Worth closing in the same PR.

### 4.6 intel is the clean-room release gate, and an 11-hour perf run blocks it

§4.9.2 and I-7 put the perf run in a concurrency group **shared with clean-room
on intel**. The estimate in §5 is ~11 h for `intel/W1`. That is ~11 h during
which nothing can be released. Combined with §3.2 — Arm A needs no comparator —
the right move is to drop intel's Arm B lane, which removes ~2.3 h, and to
schedule the remaining ~9 h deliberately rather than opportunistically. intel's
load was 11.7 and its disk 91% full at probe time.

---

## 5. Wall-clock arithmetic

§4.4.2 per band, per replicate:

```
T_band = 2c warmup requests  +  5 s quiesce  +  max(60 s, n × t_req)  +  drain
n      = max(30, 8c)
```

apr serialises (`max_in_flight` = 1), so aggregate throughput is independent of
`c` and the sampling phase for `n` requests costs `n × t_req` wall-clock, where
for W1 (512-token prompt, 128 generated, ignore-EOS):

```
t_req = 512 / prefill_tok_s  +  128 / decode_tok_s
```

Bands 1/4/8/16, N=3 replicates, apr lane + comparator lane.

| Host | prefill / decode tok/s | source | `t_req` | apr lane ×3 | comparator ×3 | **cell total** |
|---|---|---|---|---|---|---|
| lambda | 1373 / 100 | `M` `evidence/parity-http/bands/apr-c1.json` | 1.65 s | 28 min | 18 min | **~1 h** |
| gx10 | 900 / 90 | `E` from llama.cpp 185 tok/s c=1 in `llama_pin.toml` | 1.99 s | 32 min | 18 min | **~1 h** |
| intel | 40 / 6 | `E` — 6-ch DDR4 ≈ 90 GB/s ÷ 4.4 GB ⟹ ~20 tok/s ceiling; llama.cpp ~9, apr ~0.7× | 34.1 s | 8.9 h | 2.3 h | **~11 h** |
| mini | 60 / 8 | `E` — M4 ≈ 120 GB/s ⟹ ~27 tok/s ceiling; assumes apr falls back to CPU | 24.5 s | 6.4 h | 1.6 h | **~8 h** |

The two CUDA hosts are 60 s-wall-clock-bound at every band; the two others are
sample-count-bound at every band. Concretely for intel, the c=16 band alone is
`32 × 34.1` warmup + `128 × 34.1` sampling = **91 minutes for one replicate**.

**The intel and mini rows are the least trustworthy numbers in this document.**
They are derived from a bandwidth ceiling and a ratio, not measured — CLAUDE.md
rule 6, one input is an anecdote, applies doubly to zero inputs. The campaign's
first action on each of those hosts should be a **single c=1, 30-request
calibration** (≈18 min on intel, ≈13 min on mini) whose only purpose is to
replace `t_req` with a measurement before anyone commits to a 9-hour slot.

---

## 6. W2: four cells that should not be measured

§4.3.2 is explicit: W2 is **REPORTING, non-blocking** at v2.1/v2.2, because
*"a serialising server has no batch for a long prefill to interfere with, so W2
measures nothing today"*, and **"W2 becomes blocking in the same PR that lands
batching"**. Its stated expiry is *"PERF-001 merge + 30 days"*.

`scripts/perf-matrix.yaml` instead gives all four W2 cells `expires: '2026-09-25'`.
Under §4.7.3 those four go RED on that date, against a deadline §4.3.2 says does
not apply to them. Spending campaign compute on W2 now buys numbers the spec
says are meaningless and that must be re-taken the moment batching lands.

**Recommendation (matrix change, not made here):** re-tie the four W2 expiries to
PERF-001 + 30 days, per §4.3.2. That is a one-line-per-cell edit with the spec as
its authority, and it converts four cells from *"will go red on a date nobody
can meet"* to *"correctly pending"*.

Note the W2 corpus is 99 prompts (`prompts-w2.jsonl`), not 256, and §4.4.2
consumes up to 160 at c=16 — enough, but only just, and worth checking before W2
is ever run.

---

## 7. Receipt transport — the concrete answer

§4.9.1 specifies a forjar cron pushing signed receipts. `scripts/perf_receipt_sign.sh`
says so in its own header and is equally clear that it is **not deployed**:

> `WHAT THE FORJAR DEPLOYMENT WOULD BE (NOT IMPLEMENTED ON THIS BRANCH)` …
> *"Nothing below has ever run on a fleet host; that claim is not made."*

It lists what deployment needs: a 256-bit key per host at `/etc/apr/perf-receipt.key`
mode 0400, `key_id` = `<host>-<serial>`, a `machines/<host>/forjar.yaml` unit +
timer in `paiml/infra`, `forjar apply`, then `deploy-systemd-units` and
`verify-systemd-units` — and `launchd`, not systemd, on mini.

**For this campaign, do not wait for forjar.** The signer is a plain script and
its two refusals are the ones that matter: it will not sign a receipt with no
`commit` field, and it will not sign for a host its `key_id` is not scoped to.
Both properties survive being run by hand. So:

> **Run the signer ON the measuring host, then `scp` the signed file back.**

```bash
# on the measuring host, after the run, from the checkout at commit-under-test
scripts/perf_receipt_sign.sh \
    --receipt  receipt.json \
    --key-id   gx10-2026a \
    --keyring  /etc/apr/perf-receipt.key \
    --out      gx10-W1-<commit>.json

# from the coordinator
scp gx10:.../gx10-W1-<commit>.json \
    evidence/perf/gx10/W1/<commit>.json
```

Signing locally after `scp`-ing an unsigned receipt is the wrong order and
defeats the point: the signature would then attest that the *coordinator* saw
the file, not that the host produced it. The host-side keyring is the only thing
binding a document to the machine that measured it, and it is one
`openssl rand -hex 32` per host — cheaper than the forjar deployment and it
produces receipts the eventual cron will produce identically.

`--commit` on `perf_gate.sh --phase release` is what closes I-10; the signer
refuses a commitless receipt so the two ends agree by construction.

---

## 8. Runbook

### PERF-054-A — the receipt bridge (do this first, no host compute)

Wire `apr test llm bench` to `aprender_test_lib::perf_gate::ReceiptInput`. The
harness already emits `request_details[]` with `latency_ms`, `ttft_ms`,
`completion_tokens`, `prompt_tokens`, `itl_ms`, `finish_reason` — which is the
per-request terminal record `perf_gate::drain` is built to consume. Missing:

1. `drain_ms` + the §4.4.7 counters from those records (the producer exists; call it).
2. A §4.4.6 `tokenization` block — `client_tokenizer` is canonical, and it must
   declare which side of the chat-template boundary the count was taken on (the
   corpus `_meta` flags this as open).
3. The §4.4.9 scheduler block. `max_in_flight` must come from the **server**
   (I-16), not be inferred by the harness — expect `1`.
4. The §4.3.1 `512 ± 8` assertion (§4.4 above), which today does not exist.
5. Extend `bench_receipt.py` so a missing `tokenization` block is schema-fatal
   (I-13). Mutation to prove it: strip the block, receipt must be unparseable.

Land with the mutation that turns it RED, per the ratchet rule.

### Then, per host

**gx10 (do first — the cheapest proof the bridge works)**
```bash
ssh gx10
git -C ~/src/aprender fetch && git checkout <commit-under-test>
cargo build --release -p apr-cli --features cuda      # ~20 cores, 346 G free
export LLAMA_BENCH_PATH=~/src/llama.cpp-pin/build/bin/llama-bench   # 7746 (39173bcac), sm_121
# 4 bands x 3 replicates, apr lane + comparator lane  ->  ~1 h
scripts/perf_receipt_sign.sh --receipt … --key-id gx10-2026a …
```
Everything except the bridge is already in place. Load 0.02, GPU idle.

**lambda (this box) — c=1 only until #2774**
Use `. scripts/apr_bin.sh || exit 1` (never a bare `apr`; four coexisted here once).
Comparator `LLAMA_BENCH_PATH=/home/noah/src/llama.cpp/build/bin/llama-bench`.
Record c=1 and mark c=4/8/16 `UNMEASURED` blocked on **#2774** with that issue as
the owner-visible reason. Do not synthesise Arm A from a c=1-only cell —
`scaling_efficiency` has no numerator without `agg(c>1)`.

**intel — calibrate before committing the slot**
Needs an `apr` build (cargo 1.97.1 present). Drop Arm B (§3.2) rather than
rebuilding the comparator: its `llama-server` is `version: 1 (35bee03)`, not the
pin, and `llama-cli`/`llama-bench` are absent, so `llama_bin.sh` cannot resolve
an oracle there at all. Run the 30-request c=1 calibration first. Then schedule
the ~9 h explicitly against the clean-room concurrency group (I-7) — it blocks
releases for its whole duration. Disk is 91% full; check before starting.

**mini — last, and expect a partial cell**
Needs an `apr` build (Xcode CLT present, cargo 1.96.0). Drop Arm B: no llama.cpp,
no cmake, and a Metal llama.cpp build is a real install on a box with 10 cores.
Decide the `compute_class` question (§4.3) before measuring — `metal` is not a
value the code can emit. Expect c=1 and c=4 only; c=8/16 face a 16 GB ceiling
against a runtime that wants 3.08× the comparator's footprint. §4.9 already says
this; `perf-matrix.yaml` does not.

---

## 9. Defects surfaced by this scoping that deserve their own tickets

1. **The receipt emitter has no caller** — `perf_gate::{ReceiptInput,TokenizationBlock}`
   referenced only within its own module; `apr-cli` never mentions `perf_gate`.
2. **`apr test llm bench` ships in no installed binary** — added by #2705, after
   `v0.64.0`; all three remote hosts answer `unrecognized subcommand 'llm'`.
3. **`compute_class()` can never return `metal`**, and `apr-cli`'s
   `wgpu = ["inference"]` enables no GPU backend while making the build *report*
   `wgpu` — an I-2 violation that `bench_receipt.py` would validate.
4. **The `512 ± 8` assertion the corpus promises does not exist** anywhere in
   `crates/` or `scripts/`.
5. **`bench_receipt.py` does not enforce §4.4.6** — I-13 says its absence is
   schema-fatal; it is currently not even a warning.
6. **W2's expiry in `perf-matrix.yaml` contradicts §4.3.2** — four cells will go
   RED on 2026-09-25 under a deadline the spec ties to PERF-001 instead.
7. **`llama_pin.toml`'s `build_flags_lambda` no longer describes the artifact** —
   declares `CUDA_ARCHITECTURES=89`, binary carries `86;89;120`.
8. **Stale llama.cpp trees shadow the pinned one** — two on lambda (oldest
   `4230 (0c39f44d)`, Nov 2024), three on gx10. `llama_bin.sh` is safe by
   construction; hand-runs are not.
