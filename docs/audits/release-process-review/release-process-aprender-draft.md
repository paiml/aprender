# `release-process-aprender.md`

**Release process for aprender tagged releases**

| | |
|---|---|
| Status | **DRAFT v0.5.** Four decisions taken by the operator (RD-1, RD-5, RD-6, RD-7); four remain recommended-but-unconfirmed. |
| Author | Noah Gift (driver-assisted) |
| Date | 2026-09-08 (v0.1) · 2026-09-09 (v0.2–v0.4) |
| Applies from | v0.66.0 |
| Extends | `binary-release.yml` (tagged assets) · `cuda-nightly.yml` (live GPU) · `apr-dogfood` v3.0 |
| Related | PP-066 §5 R-5/R-6/R-7, §4 C4/C13/C14 · #2869 · #2971 · #2696 · #2982 · `paiml/infra#359` |
| Marks | `[V]` verified by a command here · `[C]` computed · `[A]` asserted, source named · `[U]` unverified, owner named |
| v0.4 | **Cut to about a third of v0.3**, after an `agy /grillme` grill and an `agy /teamwork` review of the plan. Nothing is done by hand: every check runs in CI or on the existing gx10 nightly lane. §7 (a PR-time GPU runner) is gone. §7 (rollback) is new. Changelog at the foot. |

Everything marked `[U]` is a question for the owner named, not a claim.

---

## §0 Why this exists

Four defects, each of which shipped:

| id | defect | what recurs without this |
|---|---|---|
| #2696 | published binary ran CPU while claiming GPU | hardware bound to the build; one artifact for every user |
| #2869 | no `apr-*` asset on any tagged release | "the published artifact" is whatever the user's `cargo` produced; no gate can bind to it |
| #2971 | GPU computes a different function than CPU on a supported model | correctness verified by users, not by gates |
| #2982 | a review receipt defined by the PR under review | producer is the gate |

**The one idea.** Verify **the artifact a user installs**, not the tree that produced it.
Everything below follows from that and nothing else does.

**What that does *not* mean.** v0.3 read it as "run the models by hand on owned
hardware". That does not follow: CI can download a release asset, install it and run it.
Only two things need silicon CI does not have, and they go to a lane that already exists.

---

## §1 Doctrine (inherited, not relitigated)

1. **Gates or theater.** Every gate carries a named mutation that must turn it RED.
2. **Build ≠ test.** A binary is verified when it is *installed from the release asset*
   and exercised on the hardware class it claims.
3. **Producer is never the gate.**
4. **Never enumerate from a written list.** Targets from the matrix, models from the
   manifest, commands from `--help` on the built binary.
5. **Exit 0 is not a pass.** Every probe excludes an outcome.
6. **Horizon, not a sample** (CF-4, #1864). Any gate over an autoregressive or cached
   path validates over **≥ 64 positions**.
7. **`basis=` or `[U]`.** No invented thresholds.
8. **Stop the line.** A red gate stops the release and becomes a ticket. Never `--skip`.
9. **Deterministic receipts.** Same tree, same inputs ⇒ byte-identical receipt body.

---

## §2 Hosts

| host | arch | compute | CUDA arch | role |
|---|---|---|---|---|
| hosted CI | x86_64 / aarch64 | GitHub-hosted, no GPU | — | builds every artifact; runs the no-GPU artifact checks |
| `gx10` | aarch64 | GB10, 120 GB unified | sm_121 | the **live-GPU lane** (`cuda-nightly`), aarch64 verification |
| `lambda-labs` | x86_64 | RTX 4090, 24 GB | sm_89 | x86 GPU verification. **Not a CI runner** — control host |
| `yoga` | x86_64, 22 cores, 30 GB `[V]` | RTX 4060 Laptop, 8188 MiB, driver 595.91.07 `[V]` | sm_89 `[V]` | build box. **No `~/models`** `[V]`. Not used by this process |
| `intel` | x86_64 | Xeon W-3245, 283 GB, CPU only | — | clean-room; the publish gate |

**Fleet shape** `[C]`: sm_89 × 2, sm_121 × 1. RD-6 names the aarch64 single-host gap; its
mirror is that the two x86 GPU hosts share a compute capability.

v0.3 carried a host-slot scheduler here. **Deleted** — with the work automated onto CI and
an existing nightly lane, there is no fleet to schedule.

---

## §3 Artifacts

### 3.1 What already ships

| workflow | fires on | builds | targets | runs on |
|---|---|---|---|---|
| `nightly.yml` | cron, dispatch | `apr-cli` | **5**: linux x86_64/aarch64, darwin x86_64/aarch64, windows | GitHub-hosted |
| `binary-release.yml` | `release: published` | **`pv` only** | **4**: x86_64/aarch64 × **musl/gnu** | `ubuntu-latest` + `cross` |

`binary-release.yml` is already tag-scoped, multi-target and uploads assets, and its header
says other CLIs *"can opt in"*. **`apr-cli` opts in. No new build workflow is written.**
§3.3 says no CUDA toolkit is needed to build, so hosted runners build every artifact;
house hardware only verifies.

### 3.2 The artifact set

`apr-<target-triple>-<backend>`, one signed `SHA256SUMS` manifest per tag.

**Decided (RD-1): four artifacts.** `x86_64-unknown-linux-gnu` and
`aarch64-unknown-linux-gnu`, each × `cpu` and `cuda`. **Darwin and windows are dropped** —
no verification host, and §9.1 forbids publishing what nothing can run. **musl is dropped**
— it doubles the matrix and interacts badly with §3.3's `dlopen` of `libcuda.so.1`, though
`binary-release.yml` keeps shipping it for `pv`.

**PP-066 C13 says five targets and must be amended to four in the implementing PR.**

Carry over from `nightly.yml`: default features on unix targets,
`--no-default-features --features inference` on windows only.

### 3.3 The cpu/cuda split must not re-create #2696

Splitting artifacts re-binds hardware to the build, which is why-2 of #2696. Three
fixtures make it safe; each is a registry catalogue entry:

| id | invariant | falsifier |
|---|---|---|
| **FX-16** | the `-cuda` artifact on a host with **no** driver prints `cuda unavailable reason=libcuda.so.1 not found`, serves CPU, exits 0 | a crash, a silent CPU run, or a `cuda ready` line is RED |
| **FX-17** | the `-cpu` artifact asked for `--gpu` refuses with `FeatureDisabled` **and prints the install command for the `-cuda` artifact of the same target** | a silent CPU fallback is RED — this is #2696 itself |
| **FX-18** | `install.sh` picks `-cuda` when a driver is present and `-cpu` otherwise, prints which and why; `--backend` overrides | driver present + `-cpu` chosen with no printed reason is RED |

**The user is never asked to know their hardware.** The installer detects, the binary
discovers, both print what they found.

**No CUDA toolkit is required to build.** The driver and cuBLAS are `dlopen`ed;
`ldd apr | grep -c cudart` must be **0** on every `-cuda` artifact. A linked `libcudart`
reproduces the llama.cpp version-mismatch class and is RED.

**PTX, not SASS.** `apr` emits PTX as text and the driver JITs it — cited: the driver-API
samples load PTX through `cuModuleLoadDataEx` / `CU_JIT_INPUT_PTX` after linking the driver
dynamically.

> **v0.1's acceptance — "`cuobjdump` lists no ELF section" — is withdrawn: it cannot
> fail.** `crates/aprender-gpu` generates PTX from Rust with *"no LLVM, no nvcc"*, so no
> `apr` binary ever contains device code and that probe answers the same for the `-cuda`
> artifact, the `-cpu` artifact and `/bin/true`. Measured: `cuobjdump --list-elf` on a real
> `apr` prints `does not contain device code` and exits **255**, so a guard written from it
> is also a false-RED generator; and `cuobjdump` is **absent on `intel`**. Two CUDA-docs
> queries for its exit status in that case returned **no authority**.
>
> **Acceptance instead:** a unit test over `PtxModule`'s emitted header
> (`crates/aprender-gpu/src/ptx/builder/ptx_module.rs:110`; floor validated at
> `ptx/mod.rs:43-67`). **Registered mutation:** raise the emitter's version constant above
> the floor → RED.

**The PTX floor's basis (RD-7).** Not "the fleet's oldest driver": the emitted `.version`
is a **source constant**, and a floor set by three machines in one house makes a published
binary's compatibility a function of who owns hardware. Use the published mapping — PTX ISA
**PTX ISA §11.1.1** (*".version … must be compiled with tools that support an equal or
greater version number"*) with Release-Notes Table 62 mapping ISA version → CUDA release → **driver**.
Derive from a **declared** minimum; the fleet is a cross-check that it is reachable.
**What that minimum should be is RD-7's.**

---

## §4 Release verification

**Every check below is automated.** Nothing in this section is run by hand.

### 4.1 What runs where, and why

The split is not a preference. It is what needs a GPU.

| check | GPU? | where |
|---|---|---|
| install from the release asset; binary runs; `apr devices --json` prints the registry | no | **hosted CI**, against the downloaded asset |
| FX-16, FX-17, FX-18 | no | hosted CI |
| artifact identity: installed sha256 == signed manifest | no | hosted CI |
| `--gpu` silently falls back on a host *with* a GPU (#2696) | **yes** | **`cuda-nightly` on `gx10`** |
| CPU/GPU parity over ≥ 64 positions (#2971, C14) | **yes** | **`cuda-nightly` on `gx10`**; `lambda-labs` for x86 |

The live-GPU lane already exists: `cuda-nightly.yml` runs on `gx10`, which is what
`paiml/infra#359` **Option A** left in place. This section adds a release-asset input to
it; it does not create a lane, and it does not add a PR-time GPU runner.

### 4.2 Gates

| id | gate | acceptance | registered mutation |
|---|---|---|---|
| **M1** | artifact identity | installed sha256 == manifest entry for this target | edit one byte of the tarball → RED |
| **M2** | registry readback | `apr devices --json` lists the expected backend `ready` on that host | force `selected: cuda` on a CPU cell → RED |
| **M3** | **CPU/GPU parity, horizon** | same host, model, prompt and seed: min cosine ≥ threshold over **≥ 64 positions**. **Threshold is measured, not `[U]`**: `evidence/parity/thresholds.yaml` sets 0.98 from n=5 on both hosts, both polarities (7B 0.9986/0.9985, 1.5B 0.9508/0.9506, stdev 0). The gate is `check_model_parity.sh --manifest` (#3026) | revert the #2971 kernel fix → 1.5B RED, 7B GREEN |
| **M4** | determinism | same cell, same seed, m=1 greedy, two runs → byte-identical token stream | inject an unordered reduction → RED |
| **M5** | refusal semantics | FX-16 / FX-17 / FX-18 | remove the refusal → silent CPU fallback → RED |
| **M6** | 7B service smoke | loads, > 0 tokens, stop reason recorded, no OOM; peak RSS/VRAM with `basis=`. **Per-host instrument**: `nvidia-smi` returns `[N/A]` on `gx10` (unified), so that cell records `UNMEASURED(vram, unified-memory)` — a blank is the `verified_hardware: UNKNOWN` §9 sets to zero | cap VRAM below need → refusal with a reason, not a crash |
| **M7** | performance | `pp512`/`tg128` **reported**, never gated in 0.66 | — (report-only, claims ratchet) |

**M3 is the reason this exists.** It is the gate #2971 escaped.

### 4.3 Models

`evidence/models/supported.yaml` **already exists** (#3026), derived by
`scripts/derive_model_manifest.sh` from README, `docs/BEATS.md`, `book/src/**`,
`evidence/dogfood/*` and `scripts/perf-matrix.yaml`; 18 models, every entry citing
`file:line`; `--check` refuses a hand-typed entry.

**It is a citation index, not a field pin.** It carries `name`, `family`, `size`,
`cited_by` — not the `sha256`, `hidden`, `heads`, `kv_heads` this process wants. Either
extend the deriver or use a second file. **Do not hand-write a manifest**; `--check`
exists to stop that.

Release models: **Qwen2.5-Coder-7B-Instruct Q4_K_M** (primary) and
**Qwen2.5-1.5B-Instruct Q4_K_M** (the #2971 sentinel — the release's own falsifier).

### 4.4 Receipt and verdict

One receipt per cell under `docs/audits/release/<tag>/`, deterministic, no timestamps,
recording the artifact sha256 from the signed manifest, host facts from
`apr devices --json`, and every gate result with each exclusion **named**.

| verdict | condition |
|---|---|
| **GO** | every cell green |
| **NO-GO** | any cell red, **or missing**, or a receipt whose `asset_sha256` is not in the signed manifest |
| ~~UNSERVICEABLE~~ | **Deleted by RD-5.** A model whose parity fails blocks the tag. M3 exists because of #2971; an escape hatch on it makes it theater |

A missing cell is NO-GO, not a skip.

---

## §5 Release sequence

Six steps. Each is idempotent; the DONE-IF runs first.

| # | step | status | DONE-IF |
|---|---|---|---|
| 1 | clean-room build on `intel` | **to build** (`run_clean_room.sh`) | — |
| 2 | `apr-dogfood --release` | exists | GO receipt for this build sha |
| 3 | tag `vX.Y.Z-rc1` as **prerelease** | exists (`gh`) | tag exists — **never move a tag** |
| 4 | `binary-release.yml` builds and attaches the artifacts + `SHA256SUMS` | **to build** (opt `apr-cli` in) | assets present, sha256 matching |
| 5 | `make sign-release TAG=<tag>` — **local, key-holding** | **to build** | `.minisig` verifies against the committed public key |
| 6 | §4 runs: hosted-CI cells, then the `gx10` lane. **Promotion flips `prerelease=false` from the receipts** | **to build** | `isPrerelease == false` |

Then: release notes generated from the manifest and the registry, **no number outside
those tables** (claims ratchet); and the crates.io cascade from a detached checkout of the
promoted tag.

**No workflow publishes to crates.io.** Signing (5) and publishing use local keys. Those
are the only two manual steps.

---

## §6 What enforces each rule

| guard | status | enforces | mutation that must go RED |
|---|---|---|---|
| `check_release_matrix.sh` | **absent** | §3.2 — the artifacts exist with sha256 + manifest | drop one artifact |
| `check_no_cudart_link.sh` | **absent** | §3.3 — no `libcudart` in a `-cuda` artifact | link cudart → RED |
| `check_ptx_version.sh` | **absent** | §3.3 — the emitter's `.version` ≤ the declared floor | raise the constant → RED |
| `check_model_parity.sh` | **exists** (#3026) | §4.2 M3 / **C14** | revert the parity fix → RED on the sentinel |
| `check_release_receipts.sh` | **absent** | §5 step 6 — promotion inputs | remove a cell receipt; tamper one byte; a receipt with `parity: skipped` |
| `check_readme_claims.sh` | **exists** | §4.3 — README model ∉ manifest | add an unmanifested name → RED |
| `check_rollback_drill.sh` | **absent** | §7.3 — the drill ran within its window | let the window lapse → RED |

**Contracts.** `contracts/apr-gpu-cpu-parity-v1.yaml` **exists** (#3026) — C14's contract.
`contracts/apr-cpu-vs-gpu-output-parity-v1.yaml` also exists, with 11 falsification tests.
**Two contracts now cover CPU/GPU parity and need an owner**: a falsifier added to one does
not constrain the other. Which is authoritative is open. The publish-cascade contract
extends `contracts/apr-cli-publish-v1.yaml`. Genuinely new:
`contracts/apr-release-assets-v1.yaml`.

*Stale citation to fix:* the older contract puts the `SKIP_PARITY_GATE` bypass at
`gguf/cuda/mod.rs:268-279` (YAML lines 18, 179). `grep` puts it at **`:333`** and
**`:349`**; `:268-279` is the `HGEMM_PREFILL` warm-up block.

**Of the artifacts this document names, 4 exist and 12 must be built** — re-tested with
`find` over the tree against `origin/main` @ `ebc9e9d81`. v0.3 named 21 with 17 absent;
deleting §7 and the slot scheduler removed five.

---

## §7 Rollback

**New in v0.4. The grill was right that its absence was the largest hole**: v0.3 had ten
refusals and no answer to "the release is out and it is bad".

### 7.1 Detection

A release is bad when any holds after promotion: a P0 filed by a non-maintainer against
the tag; a `cuda-nightly` cell that was green at promotion turning red on the same asset;
or `apr devices` on the published artifact disagreeing with its own receipt.

### 7.2 The path

| # | step | who |
|---|---|---|
| 1 | mark the GitHub release a prerelease again — reversible, immediate, stops new installers | automated |
| 2 | pin `install.sh` to the last GO tag | automated |
| 3 | `cargo yank` the affected crates | **local, key-holding** |
| 4 | edit the release notes to say what is wrong and what a user should do | human |
| 5 | file the defect with the receipt that passed, and the one that should not have | automated |

**A yank is not a fix and not a delete.** Yanked versions still resolve for existing
lockfiles; step 2 is what actually moves users.

### 7.3 The fire drill — rollback's registered mutation

A rollback nobody has executed is a rollback that does not work, and §1.1 does not exempt
it.

**`check_rollback_drill.sh`**: on a schedule, cut a throwaway prerelease, run §7.2 steps
1, 2 and 5 against it end to end, and record the wall-clock. **Registered mutation:** break
the un-promote step → the drill fails RED. A drill that has not run inside its window is
RED, exactly as a missing cell is NO-GO.

The drill measures the number that matters and nobody currently knows: **time from
detection to users no longer receiving the bad artifact.** `[U]` until the first drill.

---

## §8 Toyota Way targets

| principle | target | instrument |
|---|---|---|
| **jidoka** | GPU/CPU divergence found by a **user**: **0 per release** (baseline 1 — #2971). **Resolved by RD-5**: the `UNSERVICEABLE` hatch is deleted, so the target stands unqualified | P0 issues filed by non-maintainers |
| **poka-yoke** | every artifact carries asset + sha256 + signature; every cell green before promotion | §6 guards |
| **genchi genbutsu** | release claims verified on the hardware they name: **100%**; `verified_hardware: UNKNOWN` on a release-path feature: **0** | cell receipts |
| **jidoka (build)** | `-cuda` artifact crashes on a driverless host: **0** | FX-16 |
| **standardized work** | release steps executed by hand: **2** (sign, publish) — both key-holding | §5 |
| **andon** | releases promoted with a missing cell: **0** | promotion job |
| **kaizen** | time from detection to users no longer served a bad artifact: ratchet down from the first drill | §7.3 |

---

## §9 Refusals

1. **No artifact without a verification host.** If nothing can run it, it is not published
   as verified.
2. **No promotion from a build log.** Only cell receipts naming the manifest sha256.
3. **No `SKIP_PARITY_GATE` in any release path.** It exists in shipped code
   (`gguf/cuda/mod.rs:333,349`) as an override that prints and refuses; its presence in a
   release job's environment is RED and any receipt under it is `INVALID-CORRECTNESS`.
4. **No performance number in the release notes** outside the generated tables. 0.66
   delivers correctness; speed ships in 0.67 with its instruments.
5. **No hand-set `prerelease=false`.** The promotion job or nothing.
6. **No CI job on `lambda-labs`.** It is the control host.
7. **No `cargo publish` in any workflow.**
8. **No PR-time GPU runner.** `paiml/infra#359` decided this (Option A); live-GPU work is
   the `gx10` nightly lane. v0.3's §7 asked to reverse it and is deleted.

---

## §10 Decisions for review

**None are decided here — but each now has a recommended answer, an argument, and a
falsifier.** An `agy /grillme` lane was asked for positions rather than objections, after
four revisions and thirteen review rounds moved none of these. Its answers are below,
assessed. Two were rejected and one was rewritten; where that happened it says so and why.

**Status: the four blocking decisions are TAKEN** (Noah, 2026-09-09) — RD-1, RD-5, RD-6
and RD-7 below. **RD-2, RD-4, RD-8 and RD-9 remain recommended and unconfirmed**; each
defaults safely and none blocks 0.66.


| id | question | recommended answer, and the argument | falsifier — what makes it wrong | owner |
|---|---|---|---|---|
| **RD-1** | which targets does a tagged release carry? | **DECIDED — drop darwin, windows and musl.** The release carries **four artifacts**: `x86_64` and `aarch64`, linux-gnu, × `cpu` and `cuda`. Darwin and windows have no verification host and §9.1 forbids publishing what nothing can run. musl doubles the matrix and interacts badly with §3.3's `dlopen`. **`apr-cli` opts into `binary-release.yml`** rather than a new workflow. **PP-066 C13 must be amended from five targets to four in the PR that implements this** | A musl `-cuda` build runs cleanly on Alpine with a real driver — then musl is cheap and the drop was over-cautious | **taken** |
| **RD-2** | installer default when a driver is present | *Recommended, unconfirmed.* `-cuda`, reason printed, `--backend` overrides. FX-16 covers a present-but-unusable driver; FX-18 makes the choice visible. No objection in thirteen rounds | A stub or corrupt `libcuda.so.1` slips past FX-16 and segfaults before `--backend cpu` applies | team |
| ~~**RD-3**~~ | ~~`cuda-test` on `yoga`?~~ | **RETIRED, not answered.** The section it governed is deleted (§9.8) — there is no PR-time GPU check to schedule | A live-GPU PR check is proposed again; it needs a new row, and must answer `infra#359`'s intermittency cost | — |
| **RD-4** | `gx10` is the live-GPU lane **and** a perf host | *Recommended, unconfirmed.* One GitHub Actions `concurrency:` group shared by the nightly lane and any perf job. Three lines, no new machinery | A perf run started by hand over SSH never enters the group and overlaps anyway; then the lock has to move off Actions | team |
| **RD-5** | if #2971 is not fixed by the tag | **DECIDED — block the tag. The `UNSERVICEABLE` hatch is deleted** (§4.4). M3 exists because of #2971, so an escape hatch on that one gate makes it theater by §1.1, and it contradicted §8's jidoka target outright | #2971 proves architecturally unfixable, in which case blocking halts every release and this has to be revisited as an explicit, dated exception — not a standing hatch | **taken** |
| **RD-6** | `arm-gpu-cuda` has one verification host | **DECIDED — ship it, and state the single-host limit in the release notes.** Not shipping punishes users who can use it, and the x86 pipeline is **also** single-capability (`yoga` and `lambda-labs` are both sm_89), so requiring a cross-check for aarch64 alone was inconsistent | An sm_121-specific path silently returns garbage on Jetson Orin (sm_87). That is what a second aarch64 host would catch, and it is the 0.67 fleet ask | **taken** |
| **RD-7** | the declared PTX floor | **DECIDED — whatever is idiomatic for Hugging Face, which turns out to be what the code already does.** The HF/PyTorch wheel convention is a single global floor at **sm_70 (Volta)** with forward compatibility by driver JIT. `crates/aprender-gpu` already declares exactly that: `MIN_PTX_VERSION (7,0)`, `validate_target` rejects `sm_<70`, and `as_module()` says *"Uses sm_70 (Volta) as minimum baseline for broad compatibility"*. **The floor is GLOBAL, not per-artifact.** The `.version` is **derived per module, not declared**: `ptx_version_for_target()` (`kernels/mod.rs:139`) emits **8.8** for sm_100+ and **8.0** below (trueno#188) — cited: *"PTX ISA version 8.8 … Adds support for `sm_121` target architecture"*. **Consequence to write down: `.version 8.0` implies a driver supporting PTX ISA 8.0 (CUDA 12.0, ≈ r525), so the effective driver floor is r525, not sm_70's own r384.** `check_ptx_version.sh` asserts the mapping; its mutation is to raise a constant above the floor | A kernel needs an instruction above ISA 8.0 on a pre-Blackwell target, which would raise the driver floor again without anything noticing. That is exactly what the guard is for | **taken** |
| **RD-8** | bit-for-bit reproducibility across machines | *Recommended, unconfirmed.* Close permanently out of scope: it needs a hermetic toolchain this project is not building, and same-machine determinism is what M1 checks | A supply-chain compromise of `intel`'s toolchain stays undetectable precisely because the binary cannot be reproduced elsewhere. Close with eyes open | team |
| **RD-9** | `yoga`'s specs | *Recommended, unconfirmed.* Close — measured in §2, and `yoga` is not used by this process | `yoga` is a dynamically provisioned VM whose specs change between invocations | Noah |

---

## §11 Adoption

| release | contents |
|---|---|
| **0.66.0** | §3 artifacts via `binary-release.yml`; §4 verification, automated; §5 sequence; §6 surviving guards; §7 rollback with its first drill |
| **0.66.1** | RD-1's target decision implemented; a second aarch64 GPU host if RD-6 says so |
| **0.67** | performance promoted from reported to gated with its own ratchets; `metal` artifact |

**Blocker:** §4 installs from the release asset, which needs `install.sh` — PP-066 **R-6 /
PMAT-994, open, due 2026-10-16** (R-5 / PMAT-993 likewise). Either they come forward or
this row re-scopes. **Nothing here is gated on a number this document invents.**

---

## Appendix A — S0 ledger

| id | status |
|---|---|
| S0-Y1, S0-Y2 (`yoga` identity) | **measured** — §2 |
| S0-G2 (driver ISA floor) | **measured**; cross-checks the declared floor, no longer sets it |
| S0-M1 (7B on the dogfood hosts) | **measured**: `lambda-labs` and `gx10`, 4 683 073 536 B. Extend to cover *presence* |
| S0-R1, S0-R2, S0-N1 | **measured** — §3.1, §9 |
| S0-G1 (`gx10` build wall clock) | `[U]`, Noah. Less load-bearing now hosted CI builds |
| S0-Y3/Y4/Y5 (`yoga` cuda build and tests) | **moot** — `yoga` is not used by this process |

---

## Appendix B — Files

```
.github/workflows/binary-release.yml        # apr-cli opts in — NOT a new workflow
.github/workflows/cuda-nightly.yml          # the live-GPU lane; gains a release-asset input
scripts/run_clean_room.sh                   # the one build recipe
scripts/check_release_matrix.sh             # §6 guards …
scripts/check_rollback_drill.sh             # §7.3
scripts/publish_cascade.sh                  # local, fail-closed
evidence/models/supported.yaml              # EXISTS (#3026) — extend, do not replace
keys/apr-release-minisign.pub               # committed public key; private key local only
contracts/apr-release-assets-v1.yaml        # new
docs/audits/release/<tag>/                  # every receipt for a release
.claude/skills/apr-dogfood-models/SKILL.md  # REPO scope — ~/.claude/… is #2361
```

---

## Changelog

**v0.5 — 2026-09-09.** Four decisions taken by the operator; the spec records them.

- **RD-1** — four artifacts: x86_64 and aarch64, linux-gnu, × cpu/cuda. Darwin, windows
  and musl dropped. `apr-cli` opts into `binary-release.yml`. **PP-066 C13 must be amended
  from five targets to four.**
- **RD-5** — the tag blocks on #2971. `UNSERVICEABLE` is **deleted** from §4.4, which
  resolves the §8 jidoka contradiction the document has carried since v0.1.
- **RD-6** — `arm-gpu-cuda` ships with its single-host limit stated in the notes.
- **RD-7** — *"whatever is idiomatic for Hugging Face"*, which on inspection is what the
  code already does. sm_70 global floor (PyTorch's, and `as_module()` says so in those
  words); `.version` derived per target by `ptx_version_for_target()`, 8.8 for sm_100+ —
  cited: *"PTX ISA version 8.8 … Adds support for `sm_121`"*. The decision ratifies the
  implementation and writes down its consequence: **ISA 8.0 implies a driver ≈ r525**, so
  that, not sm_70's r384, is the effective driver floor.

RD-2, RD-4, RD-8 and RD-9 remain recommended and unconfirmed. None blocks 0.66.


**v0.4 — 2026-09-09.** Cut to about a third. An `agy /grillme` lane returned
`do-not-implement-as-written` — over-engineered for a project shipping via `cargo install`
— and an `agy /teamwork` lane then broke the plan's rebuttal.

The rebuttal was that in-tree testing cannot see a tarball defect, therefore verify by
hand. **Wrong**: CI can download the asset, install it and run it. "Test the artifact, not
the tree" does not imply "test it by hand". The residue is only what needs a GPU, and a
GPU lane already exists.

| change | |
|---|---|
| **§7 (PR-time GPU runner) deleted** | `infra#359` Option A stands; live-GPU work is the `gx10` nightly lane. Now a refusal (§9.8) |
| **§4 automated** | hosted CI runs the no-GPU checks on the downloaded asset; `cuda-nightly` runs the two that need silicon. No manual step |
| **`gx10` kept** | an earlier plan cut it and would have closed RD-6 by omission |
| **§7 Rollback added** | with a **fire drill** (§7.3) as its registered mutation, and the un-measured number it exists to measure |
| **host slots deleted** | no fleet left to schedule |
| **`check_release_receipts.sh` kept** | deleting enforcement while keeping the receipt is the pendulum swinging |
| **RD-8 recommended closed** by argument | 9 decisions → 8 |
| absent artifacts | 17 → **12** |
| **§10 gains recommended answers** | an `agy /grillme` lane asked for positions rather than objections; each row now carries an answer, its argument and its falsifier. Two rejected, one rewritten. Still not decided — the owner column is unchanged |

**v0.3 — 2026-09-09.** Rewritten against what the tree already does: #3026 had shipped the
manifest, C14, the parity contract and PR-time sentinels; `binary-release.yml` already
builds tagged assets.

**v0.2 — 2026-09-09.** Facts and mechanisms corrected from the PMAT-1092 review (#3056):
the `cuobjdump` gate that could not fail, the PTX floor's basis, the skill's scope, and
`status` columns throughout.

---

*Comments against section numbers; decisions against RD-ids.*
