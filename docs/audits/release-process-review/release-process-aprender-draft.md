# `release-process-aprender.md`

**Release process for aprender tagged releases — build matrix and physical model dogfood**

| | |
|---|---|
| Status | **DRAFT v0.2 — for team review.** Not yet normative: nine decisions (§11) are open, and §6 depends on artifacts that do not exist. |
| Author | Noah Gift (driver-assisted) |
| Date | 2026-09-08 (v0.1) · 2026-09-09 (v0.2) |
| v0.2 changes | Facts and mechanisms only, from the PMAT-1092 review (#3056) and eleven quorum rounds. **No RD is decided here.** §2's `[U]` rows measured; §3.4's `cuobjdump` acceptance replaced (it could not fail); §3.4's PTX-floor *basis* reframed with a published citation; §5.1 skill moved to repo scope; §6 and §8 gain a `status` column and §6 is retitled; §7 cites `infra#359`; §8's contracts corrected to ones that exist; §12 names its blocker. Every RD row now carries the finding that bears on it. Changelog at the foot. |
| Applies from | v0.66.0 |
| Supersedes | `nightly.yml` as the artifact matrix of record; extends `apr-dogfood` v3.0 |
| Related | PP-066 spec §5 R-5/R-6/R-7, §4 C4/C13/C14 · #2869 · #2971 · #2696 · #2982 |
| Marks | `[V]` verified by a command in this document · `[C]` computed · `[A]` asserted, source named · `[U]` unverified, owner named |

Review asks are collected in **§11**. Everything marked `[U]` is a question for the
reviewer who owns it, not a claim.

---

## §0 What changes and why now

Two machines joined the fleet: **gx10** (mostly idle) and **yoga** (a dedicated build
runner that can run CUDA unit tests). That converts three standing `[U]`s into
buildable, testable facts:

1. **A CUDA binary can be built by CI.** Until now no registered runner could compile
   or exercise a CUDA target, so the `cuda` path was verified only by hand on
   `lambda-labs`, which is not a CI runner.
2. **CUDA unit tests can run before merge.** `#2971` — a 1.5B model computing a
   different function on GPU than on CPU (cosine 0.9418, |Δlogit|max 5.38) — was found
   by a teammate running a released binary, not by a gate. `yoga` is what makes that
   class RED at PR time instead of RED in a Slack message.
3. **A four-cell physical dogfood is affordable.** `gx10` idle capacity plus
   `lambda-labs` gives two hosts × two backends for a real 7B model.

This document makes two processes normative:

- **A · Build matrix.** Every tagged release carries four Linux artifacts, each built
  by a named machine under one reproducible recipe (§3).
- **B · Physical model dogfood.** Before promotion, the 7B Qwen2.5-Coder model is
  actually run — CPU and GPU — on `lambda-labs` **and** `gx10`, from the released
  artifact, by an extended skill (§5).

### Standing defects this process is designed to prevent

| id | defect | what recurs without this process |
|---|---|---|
| #2696 | published binary ran CPU while claiming GPU | hardware bound to the build; one artifact for every user |
| #2869 | no `apr-*` asset on any tagged release | "the published artifact" is whatever the user's `cargo` produced; no gate can bind to it |
| #2971 | GPU computes a different function than CPU on a supported model | correctness verified by users, not by gates |
| #2982 | a review receipt defined by the PR under review | producer is the gate |

---

## §1 Doctrine (inherited, non-negotiable, not relitigated here)

1. **Gates or theater.** Every gate in this document carries a named mutation that
   must turn it RED. A gate with no registered mutation is inadmissible.
2. **Build ≠ test.** A build log is not verification. A binary is verified when it is
   *installed from the release asset* and exercised on the hardware class it claims.
3. **Producer is never the gate.** Promotion is decided by a base-owned job reading
   receipts, never by the job that produced the artifact.
4. **Never enumerate from a written list.** Targets from the matrix file, models from
   the manifest, backends from the registry, commands from `--help` on the built
   binary.
5. **Exit 0 is not a pass.** Every probe excludes an outcome.
6. **Horizon, not a sample (CF-4, #1864).** Any gate over an autoregressive, cached,
   or streaming path validates over **≥ 64 cached positions**.
7. **`basis=` or `[U]`.** No invented thresholds. A measured number carries the
   command that produced it.
8. **Stop the line.** A red gate stops the release and becomes a ticket with a
   five-whys root cause. Never `--skip`, never rerun-to-green.
9. **Deterministic receipts.** Same tree, same inputs ⇒ byte-identical receipt body.

---

## §2 Host and runner ledger

The build matrix is meaningless without knowing what each machine is. **The `[U]` rows
in v0.1 are now measured** — `ssh <host>` with `uname -m`, `nproc`, `free -g`, `df -h /`,
`nvidia-smi --query-gpu=name,memory.total,driver_version,compute_cap --format=csv`, run
2026-09-08 and reproduced independently by three review lanes. Appendix A records which
rows remain open.

| host | arch | compute | CUDA arch | role in this process | CI runner? |
|---|---|---|---|---|---|
| `intel` | x86_64 | Xeon W-3245, 283 GB, CPU only | — | clean-room CI runner; builds `x86-cpu`; **the publish gate** | yes (clean-room) |
| `yoga` | x86_64 `[V]`, 22 cores, 30 GB, 859 G free | RTX 4060 **Laptop**, **8188 MiB** (7807 free), driver 595.91.07 `[V]` | **sm_89** `[V]` | **new** — builds `x86-gpu-cuda`; §7 would also run CUDA unit tests at PR time, which `infra#359` bears on | **new**, dedicated build runner. **No `~/models` directory: no parity model is provisioned** `[V]` |
| `gx10` | aarch64 | GB10 Blackwell, 120 GB unified | sm_121 | builds `arm-cpu` and `arm-gpu-cuda`; model dogfood host | yes, JIT-scoped (§7) |
| `lambda-labs` | x86_64 | RTX 4090, 24 GB | sm_89 | model dogfood host; release-asset build for `x86-gpu-cuda` **only** if RD-3 rejects `yoga` | **no** — not a general CI runner |
| `mini` | aarch64 | Apple M4, 16 GB unified | — (Metal) | darwin artifact verification only, if RD-1 keeps darwin | no |

**The fleet is sm_89 × 2 and sm_121 × 1** `[C]`, now that `yoga` is measured. RD-6 names
the `arm-gpu-cuda` single-host gap; its mirror is that `x86-gpu-cuda`'s two verification
hosts (`yoga`, `lambda-labs`) are the **same compute capability**, so §4's two-host row
for it is redundancy, not cross-architecture coverage. Both asymmetries are facts the
decision now has in front of it; neither is decided here.

**Contention is real and must be scheduled.** `gx10` is simultaneously a builder, a
dogfood host, and a performance host. A build running during a performance or parity
measurement invalidates that measurement. Rule: **a host holds one release role at a
time**, claimed as a queue slot; the release workflow claims and releases it, and no
perf receipt may be taken while a build slot is held. Guard: `check_host_slot.sh`
(§8).

---

## §3 A — Build matrix

### 3.1 Artifact set

Every tagged release carries these four Linux artifacts. Names are
`apr-<target-triple>-<backend>`:

| artifact | target triple | backend | features | **built by** | rationale |
|---|---|---|---|---|---|
| `apr-x86_64-unknown-linux-gnu-cpu` | x86_64-unknown-linux-gnu | cpu | `cli` | `intel` | the clean-room machine builds the artifact the clean-room gate certifies |
| `apr-x86_64-unknown-linux-gnu-cuda` | x86_64-unknown-linux-gnu | cuda | `cli,cuda` | **`yoga`** | the reporter's configuration (#2971: x86 + RTX 4090) |
| `apr-aarch64-unknown-linux-gnu-cpu` | aarch64-unknown-linux-gnu | cpu | `cli` | `gx10` | native aarch64 build; no cross-compilation |
| `apr-aarch64-unknown-linux-gnu-cuda` | aarch64-unknown-linux-gnu | cuda | `cli,cuda` | `gx10` | GB10 / Jetson class |

Each artifact ships with `<name>.tar.gz`, `<name>.tar.gz.sha256`, and is listed in
**one** signed manifest per tag (`SHA256SUMS` + `SHA256SUMS.minisig`).

**Two changes this table makes silently, neither of them decided here.** `nightly.yml`
today builds its 5 targets on **GitHub-hosted** runners (`ubuntu-latest`,
`ubuntu-24.04-arm`, `macos-latest` ×2, `windows-latest`) with **default features** on the
four unix targets (`:78`); Windows alone (`:82`, guarded `if: runner.os == 'Windows'`)
adds `--no-default-features --features inference`, and `grep -c cuda` over that file is
**0** — there is no `-cuda` artifact anywhere in the fleet today. §3.1 moves every build
onto house hardware *and* adds `--features cli,cuda` to two of them. PP-066 **R-5** says
the assets *"build on the nightly.yml runners"*. Whatever RD-1 decides about the target
count, the runner-class and feature-set change survives every option and needs saying in
the text.

> **RD-1 (§11):** darwin and windows targets currently in `nightly.yml` are **not**
> in the four above. Reviewers decide: (a) carried forward unchanged as best-effort
> unverified artifacts, (b) dropped from tagged releases in 0.66 and restored when a
> verification host exists, or (c) `mini` verifies `aarch64-apple-darwin` and windows
> is dropped. **The spec's C13 currently says five targets — it must be amended to
> whatever this review decides.** Do not leave two numbers in two documents.

### 3.2 The split must not re-create #2696

Shipping a `-cpu` and a `-cuda` artifact re-binds hardware to the build — which is
**why-2** of the #2696 five-whys. It is safe only under the runtime registry (R-0).
Three invariants make it safe; each is a fixture in the registry catalogue:

| id | invariant | falsifier |
|---|---|---|
| **FX-16** | the `-cuda` artifact on a host with **no** driver prints `cuda unavailable reason=libcuda.so.1 not found`, serves CPU, and exits 0 — it never crashes and never claims GPU | run `-cuda` artifact on `intel`; a crash, a silent CPU run, or a `cuda ready` line is RED |
| **FX-17** | the `-cpu` artifact asked for `--gpu` refuses with `FeatureDisabled` **and prints the exact install command for the `-cuda` artifact of the same target** | run `-cpu --gpu` on `lambda-labs`; a silent CPU fallback is RED (this is #2696 itself) |
| **FX-18** | `install.sh` selects `-cuda` when a driver is present and `-cpu` otherwise, prints which it chose and why, and `--backend cpu\|cuda` overrides the choice | driver present + `-cpu` chosen (or the reverse) with no printed reason is RED |

**The user is never asked to know their hardware.** The installer detects; the binary
discovers; both print what they found.

### 3.3 One recipe, four machines

The build machine is an implementation detail; **the recipe is the invariant.**

- Every artifact is built by `scripts/run_clean_room.sh`'s recipe: pinned toolchain
  from `forjar`, `--locked`, fresh target directory, no host `~/.cargo` reuse.
- Every artifact's receipt records: builder host, toolchain fingerprint, `rustc -vV`,
  target triple, feature list, source SHA, artifact sha256.
- Two builds of the same tag on the same machine produce identical sha256, or the
  receipt records the diff and the release is NO-GO. *(Bit-for-bit reproducibility
  across different machines is **not** required in 0.66 — `[U]`, RD-8.)*

### 3.4 CUDA specifics

- **No CUDA toolkit is required to build.** The driver and cuBLAS are `dlopen`ed
  (S0-14, S0-20); `ldd apr | grep -c 'cudart'` must be **0** on every `-cuda`
  artifact. A linked `libcudart` reproduces the llama.cpp version-mismatch class and
  is RED.
- **PTX, not SASS.** `apr` emits PTX as text and the driver JITs it. Cited: the CUDA
  driver-API samples load PTX at runtime through `cuModuleLoadDataEx` /
  `CU_JIT_INPUT_PTX` after linking the driver dynamically, which is this model exactly.

  **v0.1's acceptance was `cuobjdump` lists no ELF section. That probe cannot fail and
  is withdrawn.** `crates/aprender-gpu` generates PTX from Rust — *"no LLVM, no nvcc"* —
  so no `apr` binary ever contains device code, and the probe returns the same answer
  for the `-cuda` artifact, the `-cpu` artifact and `/bin/true`. Measured:
  `cuobjdump --list-elf` on a real `apr` prints `does not contain device code` and exits
  **255**, so a guard written literally from v0.1 is also a false-RED generator; and
  `cuobjdump` is **absent on `intel`**, the host that gates the release. Two queries to
  the CUDA documentation for `cuobjdump`'s exit status on a binary with no device code
  returned **no authority**, so even the measured behaviour is one host's, not a
  documented contract.

  **Replacement acceptance.** Assert the emitted header where the invariant lives — a
  unit test over `PtxModule`'s output (`crates/aprender-gpu/src/ptx/builder/ptx_module.rs:110`
  writes `.version {}.{}`; the floor is validated at `ptx/mod.rs:43-67`). **Registered
  mutation:** raise the emitter's version constant above the floor → RED. If an
  artifact-level probe is also wanted, state it as
  `readelf -S <artifact> | grep -c '\.nv_fatbin'` **== 0**, which at least names the
  outcome it excludes.
- **The PTX floor's *basis*, not its value (RD-7).** v0.1 derived the floor from "the
  minimum ISA the fleet's oldest driver accepts". Two problems, and neither is a
  decision: the emitted `.version` is a **source constant**, not a property of the
  fleet; and the artifact ships to users, so a floor set by three machines in one house
  makes a published binary's compatibility a function of who owns hardware. There is a
  published mapping to use instead — the PTX ISA guide, §11.1.1: *".version … Indicates
  that this module must be compiled with tools that support an equal or greater version
  number."*, with Release-Notes Table 62 mapping every PTX ISA version to its CUDA
  release **and driver** (e.g. PTX ISA 6.0 → CUDA 9.0, driver r384). So the floor can be
  derived from a **declared** minimum compute capability and ISA, with the fleet as a
  cross-check that the declared floor is reachable. **What that declared minimum should
  be is RD-7's to settle.**
- **sm asymmetry is a first-class risk.** `lambda-labs` is sm_89, `gx10` is sm_121, and
  `yoga` — measured since v0.1 — is **also sm_89**. `.target`'s enumerated architecture
  list contains both `sm_89` and `sm_121` `[cited]`, so neither is a transcription
  error. The `arm-gpu-cuda` artifact has exactly **one** verification host (`gx10`);
  `x86-gpu-cuda` has two of the *same* capability. Both stated, neither hidden (RD-6).

---

## §4 Verification matrix — who proves what

**Build ≠ test.** Each artifact is verified by installing it from the release and
exercising it. Cells are the unit of the promotion gate.

| artifact | verified on | gates run |
|---|---|---|
| `x86_64…-cpu` | `intel`, `lambda-labs` | C4 (install + `apr devices`), FX-17, model dogfood CPU cell |
| `x86_64…-cuda` | `lambda-labs`, `yoga` | C4, C14 (parity), FX-16, model dogfood GPU cell |
| `aarch64…-cpu` | `gx10` | C4, FX-17, model dogfood CPU cell |
| `aarch64…-cuda` | `gx10` | C4, C14, FX-16, model dogfood GPU cell |

Receipts land at `docs/audits/release/<tag>/<host>-<artifact>-receipt.md` and each
records `asset_sha256` matching the signed manifest. **A receipt that does not name
the manifest's sha256 is not a receipt for this release.**

---

## §5 B — The extended model dogfood

### 5.1 The skill

New skill, invoked at release only:

```yaml
name: apr-dogfood-models          # EXPLICIT (#2332) — a directory-derived name
                                  # collides with user-scope skills and silently
                                  # never appears in the session listing
# LOCATION (v0.2): .claude/skills/apr-dogfood-models/SKILL.md — REPO scope.
# v0.1 placed it at ~/.claude/skills/…, which is where #2361 happened: a user-scope
# skill shadowed this repo's release-certifying skill, so hardening it edited a file
# that never ran. Both ~/.claude/skills/dogfood/ and .claude/skills/dogfood/ exist on
# the dev box today. A release gate outside the tree also cannot be reviewed in the PR
# that changes it, cannot be versioned with the artifact it certifies, and makes the
# producer the gate (§1.3).
description: Physically run the pinned release models from the released artifact on
  every GPU host, both backends, and emit one deterministic receipt per cell
```

It is a **sibling** of `apr-dogfood` v3.0, not a replacement: `apr-dogfood --release`
remains the surface/coverage gate; `apr-dogfood-models` is the *model* gate. Both
must be GO before promotion.

### 5.2 The model manifest

Models are **derived, never typed** — from `evidence/models/supported.yaml`, which is
itself derived from what the README, docs, cookbook, and perf matrix name. Required
entries for the release cells:

| model | quant | role | why |
|---|---|---|---|
| Qwen2.5-Coder-7B-Instruct | Q4_K_M | **primary release model** | the team's ask; the largest model the fleet can serve on both hosts |
| Qwen2.5-1.5B-Instruct | Q4_K_M | **parity sentinel** | the #2971 model — the one known to diverge; its cell is the release's own falsifier |

Each entry pins `sha256`, `hidden`, `heads`, `kv_heads`, `intermediate`,
`tie_word_embeddings`, `format`. A model named in the README that is absent from the
manifest is RED (`check_readme_claims.sh`).

### 5.3 The cell matrix

Four cells per model, sixteen gate results per release for two models:

| | `lambda-labs` (x86, sm_89) | `gx10` (aarch64, sm_121) |
|---|---|---|
| **CPU** | `x86_64…-cpu` artifact | `aarch64…-cpu` artifact |
| **GPU** | `x86_64…-cuda` artifact | `aarch64…-cuda` artifact |

Every cell installs via `install.sh` from the tag — never `cargo install`, never a
locally built binary. G0.1's provenance guard applies: the binary under test is
resolved through `scripts/apr_bin.sh` and hard-fails if it was not the downloaded
asset.

### 5.4 Gates

| id | gate | acceptance | registered mutation (must go RED) |
|---|---|---|---|
| **M1** | artifact identity | installed binary's sha256 == manifest entry for this target | edit one byte of the tarball → RED |
| **M2** | registry readback | `apr devices --json` lists the expected backend `ready` on that host; `--model <m>` shows the model serviceable | force `selected: cuda` on a CPU cell → RED |
| **M3** | **CPU/GPU parity, horizon** | same host, same model, same prompt/seed: cosine ≥ threshold over **≥ 64 cached positions**; per-op table retained | revert the #2971 kernel fix → 1.5B cell RED, 7B cell GREEN |
| **M4** | determinism | same cell, same seed, m=1 greedy, two runs → byte-identical token stream | inject nondeterminism (unordered reduction) → RED |
| **M5** | refusal semantics | FX-16 / FX-17 / FX-18 on that cell | remove the refusal → silent CPU fallback → RED |
| **M6** | 7B service smoke | 7B loads, produces > 0 tokens, records stop reason, no OOM, peak RSS/VRAM recorded with `basis=`. **Per-host instrument required**: `nvidia-smi` returns `[N/A]` for memory on `gx10` (unified), so that cell records an explicit `UNMEASURED(vram, unified-memory)` — a blank is the `verified_hardware: UNKNOWN` row §9 sets to zero | cap VRAM below need → refusal with reason, not a crash |
| **M7** | transport parity | `apr serve` + `GET /v1/effective-config`: `backend` == the resolved backend; first N tokens over HTTP == CLI for the same seed | make effective-config echo the request → RED |
| **M8** | performance | `pp512`/`tg128` **REPORTED separately** with `basis=`; never gated in 0.66 | — (report-only by the claims ratchet) |

**M3 is the release's reason to exist.** It is the gate that #2971 escaped.

### 5.5 Receipt and verdict

One receipt per cell: `docs/audits/release/<tag>/models/<host>-<backend>-<model>.md`,
deterministic, no timestamps in the body, containing: artifact sha256, host facts
from `apr devices --json`, model sha256, gate results M1–M8, the per-op parity table
for M3, and every exclusion **named**.

| verdict | condition |
|---|---|
| **GO** | every cell green for every manifest model |
| **NO-GO** | any cell red, **or** any cell missing, **or** any receipt whose `asset_sha256` is absent from the signed manifest |
| **UNSERVICEABLE** | a (model, backend) pair whose parity legitimately fails **and** whose refusal is honest (M5 green). Permitted only if that model's row in the release notes reads `UNSERVICEABLE` with the open issue number — it may not be silent, and it may not apply to a model the README advertises without the notes saying so |

A missing cell is NO-GO, not a skip. This is the whole design: the fleet may be
incomplete, and it may not silently become *more* incomplete.

---

## §6 Target release sequence — **not yet executable**

v0.1 titled this "normative, end to end". It is not: of the 21 scripts, workflows,
contracts, keys and manifests this document depends on, **1 exists**
(`scripts/check_readme_claims.sh`), **1 is correctly marked new**
(`contracts/apr-dogfood-models-v1.yaml`), and **19 of the remaining 20 are absent** —
re-tested path by path, with `find` over the whole tree, by three independent review
lanes. Step 1 and step 2 abort at `command not found`. The `status` column below says
which is which; the ordering and the content of the steps are unchanged.

Each step is idempotent — the DONE-IF check runs first and the step is skipped if it
holds.

| # | step | status | gate | DONE-IF |
|---|---|---|---|---|
| 1 | `scripts/release_criteria.sh --all` | **to build** | exit 0 | — |
| 2 | `scripts/run_clean_room.sh` on `intel` | **to build** | **exit 0 — the hard gate, first** | — |
| 3 | `apr-dogfood --release` | exists (`apr-dogfood` v3.0) | GO receipt for this build sha | receipt exists for this sha |
| 4 | claim host slots (`yoga`, `gx10`) | **to build** | `check_host_slot.sh` | slots held |
| 5 | tag `vX.Y.Z-rc1` as **prerelease** | exists (`gh`) | — | tag exists (**never move a tag**) |
| 6 | build the four artifacts on their machines | **to build** | one recipe; per-artifact build receipt | four artifacts + `SHA256SUMS` present, sha256 matching |
| 7 | `make sign-release TAG=<tag>` — **local, on Noah's workstation** | **to build** (key absent) | `SHA256SUMS.minisig` verifies against the committed public key | `.minisig` present and verifying |
| 8 | `apr-dogfood-models` on `lambda-labs` and `gx10` | **to build** (§5.1) | GO on all cells (§5) | cell receipts exist for this manifest sha256 |
| 9 | C4 verification cells (§4) | blocked on 6 | receipts on every listed host | receipts present |
| 10 | **promotion** — base-owned dispatch job reads receipts, flips `prerelease=false` | **to build** | four artifact receipts ∧ all model cells GO ∧ parity never `skipped` | `isPrerelease == false` |
| 11 | release notes | blocked on 6 | model table generated from the manifest with per-host parity; backend table from the registry; **no number outside those tables** (claims ratchet) | both tables present |
| 12 | crates.io cascade — `scripts/publish_cascade.sh` from a detached checkout of the promoted tag | **to build** | dry-run receipt, then one crate per call, stop on first non-zero | `cargo search` shows the version for every cascade crate |
| 13 | post-publish | blocked on 3+8 | C4 + model cells re-run against the **crates.io** install on one host | receipt present |

**No workflow publishes to crates.io.** Signing (7) and publishing (12) use local
keys. Everything else is machine-decided.

---

## §7 PR-time CUDA testing on `yoga`

> **This section proposes re-opening a decision that was made.** `paiml/infra#359`
> ("x86_64 GPU CI has no sanctioned runner", **OPEN**) put exactly this question and was
> answered by Noah: *"Decision made: cuda-nightly is gx10-only. **Option A**."*,
> implemented in #2740 by removing the `ada-4090` leg rather than relabelling it.
> `.github/workflows/cuda-nightly.yml:52` still reads *"ada retired, see infra#359"* and
> `:100` pins a single literal `[self-hosted, gpu, Linux, ARM64, cuda, blackwell]`.
>
> **Option B was "make yoga the x86_64 GPU runner", and it was declined with a named
> cost this section must answer:** *"yoga is off most of the time, so the lane becomes
> intermittent — and this fleet's dead-man's switch exists precisely because absent runs
> look like healthy ones. Would need `FORJAR_UNREACHABLE_DAYS`-style handling so 'yoga
> was off' is distinguishable from 'the lane died'."*
>
> What §7 asks for is **stronger** than what was declined: not a nightly lane on `yoga`,
> but a *required PR-time check*. Adopting it means adopting Option B plus the
> reachability handling that was its stated precondition. Whether to do that is the
> team's; that it reverses a recorded decision is not in doubt, and v0.1 cited neither
> the issue nor the cost.

This is the largest quality change in the document and the one with the largest
security surface. It is scoped deliberately.

### 7.1 What runs

A new required check `cuda-test` on `yoga`, PR-time:

- `cargo test -p aprender-gpu -p aprender-serve --features cuda` — the CUDA unit
  tests that no runner could execute before.
- **Two parity sentinels only** (1.5B RED-first, 7B witness) over ≥ 64 positions.
  The *full* manifest runs at release, nightly, and post-publish — a PR must not pay
  for sixteen cells.
- **`yoga` has no models.** `ssh yoga 'ls ~/models'` → no such directory. Capacity is
  fine — 7807 MiB free against 4.36 GiB of 7B Q4_K_M weights, ~3.3 GiB left for KV and
  activations — so the objection is provisioning and wall-clock, not VRAM. Appendix A's
  **S0-M1 must be extended to cover model *presence*, and to include `yoga`**, which §7.1
  assigns a cell and which S0-M1 never measures.
- Budget: `[U]`, owner Noah — and the defect is that **no population is named**, not the
  number. Three disagreeing samples are on record: `ci.yml` `conclusion=success`
  → min 38 / **median 65** / p90 117 / max 129 min (n=28, sampled 2026-09-08, and the
  window rolls); `workspace-test` on `--branch main` → 3 / 20 / 43 / 92; and v0.1's
  unsourced "≈ 34". None refutes another because each is a different workflow, ref and
  conclusion filter. A budget whose population is unspecified cannot be exceeded, so it
  cannot fail. **Name the workflow, ref, job and filter before naming a number, and carry
  the run list.** What is stable across every re-derivation: the median is in the sixties
  and the maximum above two hours.

### 7.2 Security constraints (non-negotiable) — **prerequisites, not descriptions**

Measured 2026-09-08: `paiml/aprender` is **PUBLIC**; all three org runner groups
(`Default`, `gpu-nodes`, `gpu-x86`) report `restricted_to_workflows: false` with an empty
`selected_workflows`; and `gpu-x86` — the group whose name suggests where an x86 GPU
runner belongs — has `allows_public_repositories: false`, so a public repo cannot use it.
`yoga-gpu` is online and carries `self-hosted,Linux,X64,gpu,cuda,yoga,ada`.

**Constraints 1, 3 and 5 below therefore describe configuration that does not exist.**
They are work with an owner and a ticket, not statements of the current state, and §7
must not land before they are real: on a public repo, PR-authored code would otherwise
execute on a machine in a house with no workflow restriction. Constraint 2 *is* buildable
today — `grep -n 'authorize' .github/workflows/pr-gate.yml` → `:10`. **A guard is owed
that reads the runner-group API and fails when the group holding `yoga-gpu` reports
`restricted_to_workflows != true`**; a constraint nobody can check is a comment, by §8's
own words.

A self-hosted runner executing PR code is arbitrary code execution on your hardware.

1. `yoga` lives in a **restricted runner group** whose workflow access is *selected
   workflows only*.
2. `cuda-test` runs **only after** the existing `PR Gate / authorize` job passes —
   no fork PR, no first-time contributor, reaches the GPU runner.
3. Ephemeral: JIT registration per job, deregistered after; no persistent state
   between PRs; `~/.cargo` and target dirs are per-job.
4. No secrets are available to the job. Signing and publishing never touch it.
5. Release-only GPU work (`x86-gpu-cuda` asset build) runs in a **separate** group
   restricted to `release-assets.yml@refs/tags/v*` — a tag-scoped path, unreachable
   from a pull request.

> **RD-3 (§11):** does `cuda-test` become a **required** context in 0.66, or land
> advisory for one release and get promoted in 0.66.1? Making it required the day it
> lands means one flaky GPU job blocks every merge. Recommendation: **advisory for 14
> days with its red/green record published, then required** — with the promotion
> criterion stated now (`0 false reds in 14 days, basis=` the run list), not decided
> later by feel.

---

## §8 What enforces each rule

Every row is a guard with a self-test and both polarities. A rule with no guard is a
comment — and **seven of these eight rows are currently comments**: only
`check_readme_claims.sh` exists. The `status` column is the build list, not an excuse;
the mutations are unchanged.

| guard | status | enforces | mutation that must go RED |
|---|---|---|---|
| `check_release_matrix.sh` | **absent** | §3.1 — the four artifacts exist with sha256 + manifest, built by the assigned host per the receipt | drop one artifact; swap a builder host |
| `check_no_cudart_link.sh` | **absent** | §3.4 — no `libcudart` in any `-cuda` artifact | link cudart → RED |
| `check_ptx_version.sh` | **absent** (and see §3.4 — its acceptance changed) | §3.4 — emitted PTX `.version` ≤ fleet minimum | bump the target above the fleet minimum → RED |
| `check_host_slot.sh` | **absent** | §2 — one release role per host | start a perf receipt while a build slot is held → RED |
| `check_model_parity.sh` | **absent** | §5 M3 / criterion **C14** | revert the parity fix → RED on the sentinel |
| `check_release_receipts.sh` | **absent** | §6 step 10 — promotion inputs | remove a cell receipt; tamper one byte; a receipt with `parity: skipped` |
| `check_readme_claims.sh` | **exists**, 533 lines | §5.2 — README model ∉ manifest | add an unmanifested model name → RED |
| `check_publish_cascade.sh` | **absent** | §6 step 12 — refuses on branch, dirty tree, prerelease, or a token in the environment | run from a branch → refuse |

**Contracts — two of v0.1's four already exist under other names, and minting a second
splits the ratchet.** M3/C14 must extend **`contracts/apr-cpu-vs-gpu-output-parity-v1.yaml`**,
which is in the tree, passes `pv validate` clean, and already carries **11 falsification
tests** (`FALSIFY-CPU-GPU-001..011`) plus 10 proof obligations — not a new
`apr-gpu-cpu-parity-v1.yaml`. The publish-cascade contract extends
**`contracts/apr-cli-publish-v1.yaml`**. Genuinely new:
`contracts/apr-release-assets-v1.yaml` and `contracts/apr-dogfood-models-v1.yaml`.

*A defect found inside that existing contract while checking this:* it cites the
`SKIP_PARITY_GATE` bypass at `crates/aprender-serve/src/gguf/cuda/mod.rs:268-279` (YAML
lines 18 and 179). `grep` puts it at **`:333`** and **`:349`**; `:268-279` is the
`HGEMM_PREFILL` warm-up block. A line-keyed citation inside a contract drifts silently.
Fixing it belongs in this work.

---

## §9 Toyota Way targets

Targets are zeros, ones, and ratchets from measured baselines — never invented
continuous thresholds.

| principle | target | instrument |
|---|---|---|
| **jidoka** | GPU/CPU divergence found by a **user**: **0 per release** (baseline: 1 — #2971). **This target and §5.5/RD-5 cannot both hold as worded** — §5.5 permits shipping #2971 (OPEN, P0) as `UNSERVICEABLE`, which guarantees the baseline is not improved. Note the instrument beside it is already narrower than the target's prose: it counts issues filed by **non-maintainers**, i.e. undeclared divergences. Whether that narrowing is the intent, or a looseness to tighten the other way, is **RD-5's to settle** | issues labelled P0 filed by non-maintainers |
| **poka-yoke** | artifacts with asset + sha256 + manifest signature: **4/4**; model cells green before promotion: **8/8** (2 models × 4 cells) | §8 guards |
| **genchi genbutsu** | release claims verified on the physical hardware they name: **100%**; `verified_hardware: UNKNOWN` rows for release-path features: **0** | cell receipts |
| **jidoka (build)** | `-cuda` artifact crashes on a driverless host: **0** | FX-16 |
| **heijunka** | host role conflicts during a release: **0** | `check_host_slot.sh` |
| **kaizen** | PR-time CUDA test false reds: **ratchet down from the first 14-day measurement** (`basis=` the run list) — never a typed target | RD-3's promotion criterion |
| **standardized work** | release steps executed by hand: **2** (sign, publish) — both key-holding, both deliberate | §6 |
| **andon** | releases promoted with a missing cell: **0** | promotion job |

---

## §10 Refusals — what this process forbids

1. **No artifact without a verification host.** If nothing in the fleet can run it, it
   is not published as a verified artifact (RD-1 decides whether it ships at all).
2. **No promotion from a build log.** Only cell receipts naming the manifest sha256.
3. **No `SKIP_PARITY_GATE` in any release path.** Its presence in the environment of
   any release job is RED, and any receipt produced under it is
   `INVALID-CORRECTNESS`.
4. **No performance number in the release notes** outside the generated tables
   (claims ratchet). 0.66 delivers correctness and honest discovery; speed ships in
   0.67 with its instruments.
5. **No hand-set `prerelease=false`.** The promotion job or nothing.
6. **No CI job on `lambda-labs`.** It stays the control host, not a runner (unless
   RD-3 is decided otherwise, in which case its perf role is re-homed first).
7. **No `cargo publish` in any workflow.**

---

## §11 Decisions for review

| id | question | recommendation | owner |
|---|---|---|---|
| **RD-1** | darwin/windows: carry, drop, or verify? C13 says five targets; this spec names four. | Unchanged from v0.1. Two findings bear on it, neither an answer: C13 (`PP-066-release-spec.md:159`) names five targets but does **not** itself name `nightly.yml` — **R-5** (`:218`) is the row that pins the five to that workflow — so the two documents will disagree until both are edited; and the runner-class plus feature-set change (§3.1) survives every option. | team |
| **RD-2** | installer default when a driver is present | `-cuda`, with the choice and reason printed and `--backend` overriding (FX-18) | team |
| **RD-3** | `cuda-test` on `yoga`: required in 0.66 or advisory→required in 0.66.1 | Unchanged from v0.1, with two findings against the *criterion*: (a) `0 false reds in 14 days` is satisfied by a box that was powered off for 14 days, so it needs a reachability term — which is `infra#359`'s declined-Option-B precondition arriving by another route; (b) the budget names no population (§7.1), so it cannot be exceeded and cannot fail. | team |
| **RD-4** | `gx10` builds two artifacts **and** hosts two dogfood cells **and** is a perf host | Unchanged from v0.1. One finding: the slot discipline it relies on is enforced by `check_host_slot.sh`, which does not exist (§8), so "one role at a time" is currently unenforced however this is decided. | team |
| **RD-5** | if #2971 is not root-caused by the tag | Cannot be settled while §9's jidoka target says the opposite — the document has to state which of the two readings it means before this question is answerable. #2971 is OPEN, labels `bug, P0, pp-066, inst:A` `[V]`. | Noah |
| **RD-6** | `arm-gpu-cuda` has exactly one verification host | Unchanged from v0.1. The measured ledger adds a second asymmetry beside it: `yoga` is **sm_89**, the same as `lambda-labs`, so `x86-gpu-cuda`'s two hosts are redundancy rather than cross-architecture coverage. Both facts are now in front of the decision. | team |
| **RD-7** | PTX `.version` floor | v0.1's *basis* is withdrawn in §3.4, not its question: the emitted `.version` is a source constant, and a floor derived from three machines in one house makes a shipped binary's compatibility a function of who owns hardware. PTX ISA §11.1.1 and Release-Notes Table 62 give a published ISA→driver mapping to derive it from instead. **What the declared minimum should be is still this row's to answer.** | team |
| **RD-8** | bit-for-bit reproducibility across machines | out of scope for 0.66; same-machine determinism required now | team |
| **RD-9** | `yoga` specs unknown | The measurement this row asks for is **taken** (§2, Appendix A): x86_64, 22 cores, 30 GB, RTX 4060 Laptop, 8188 MiB, driver 595.91.07, **sm_89**, and the proprietary driver is up so it executes kernels. S0-Y3/Y4/Y5 and S0-G1 remain unmeasured. Recording RD-9 as closed is Noah's act, not this document's — a measurement being taken is not a decision being closed. | Noah |

---

## §12 Adoption plan

| release | contents |
|---|---|
| **0.66.0** | §3 four artifacts; §4 verification matrix; §5 `apr-dogfood-models` with both models on both hosts; §6 sequence; §8 guards; §7 `cuda-test` **advisory**. **Blocker, added in v0.2:** §5.3 requires every dogfood cell to install via `install.sh`, which does not exist and is owned by PP-066 **R-6 / PMAT-994, `status: open`, due 2026-10-16** (R-5 / PMAT-993 likewise). §6 step 8 cannot run before those land, so either they are pulled forward or this row is re-scoped. v0.1 scheduled all of it here without naming the dependency. |
| **0.66.1** | `cuda-test` required per RD-3's criterion; RD-1's darwin decision implemented; second aarch64 GPU host if RD-6 says so |
| **0.67** | performance cells promoted from REPORTED to gated with their own ratchets; `metal` artifact; PR-time full-manifest parity if the budget allows |

**Nothing in 0.66 is gated on a number this document invents.** Every threshold that
does not yet have a measurement is `[U]` with a named owner, and the release ships
with the measurement or with an honest refusal.

---

## Appendix A — S0 ledger

Each row must be executed and its output recorded. **Executed 2026-09-08 and reproduced
independently by three review lanes; the results are in
`docs/audits/release-process-review/PMAT-1092-review.md` Appendix A.** Status per row:

| id | status |
|---|---|
| S0-Y1, S0-Y2 (`yoga` identity) | **measured** — folded into §2 |
| S0-G1 (`gx10` per-artifact build wall clock) | `[U]`, owner Noah. Feeds RD-4 |
| S0-G2 (`gx10` / `lambda-labs` driver ISA floor) | **measured**; but see §3.4 — the floor's basis is no longer "the fleet's oldest driver", so this row cross-checks a declared minimum rather than setting one |
| S0-M1 (7B fits both dogfood hosts) | **measured**: present on `lambda-labs` and `gx10`, 4 683 073 536 B. **Must be extended** to cover model *presence* and to include `yoga`, which §7.1 assigns a cell and which has no `~/models` |
| S0-Y3 (cuda feature builds on `yoga`, no toolkit) | `[U]`, owner Noah. Note `crates/aprender-gpu` advertises "no LLVM, no nvcc", so the expected `cudart` count of 0 is a property of the source, not of the box |
| S0-Y4, S0-Y5 (CUDA unit tests run there; wall clock) | `[U]`, owner Noah — and blocked behind the decision §7 reopens (RD-3), not behind the box |
| S0-R1, S0-R2, S0-N1 | **measured** — folded into §7.2 and §3.1 |

The commands remain below so the rows can be re-run; a measurement that is not
reproducible on demand is an assertion with a timestamp.

| id | question | command |
|---|---|---|
| S0-Y1 | `yoga` arch, CPU, RAM, disk | `uname -m; nproc; free -g; df -h /` |
| S0-Y2 | `yoga` GPU, driver, CUDA arch | `nvidia-smi --query-gpu=name,memory.total,driver_version,compute_cap --format=csv` |
| S0-Y3 | `cuda` feature builds on `yoga` with no toolkit | `cargo build -p apr-cli --release --features cuda 2>&1 \| tail -3; ldd target/release/apr \| grep -ci 'cudart'` → expect **0** |
| S0-Y4 | CUDA unit tests actually run there | `cargo test -p aprender-gpu --features cuda -- --nocapture 2>&1 \| tail -20` |
| S0-Y5 | wall-clock budget for `cuda-test` | `/usr/bin/time -v cargo test -p aprender-gpu -p aprender-serve --features cuda` |
| S0-G1 | `gx10` builds both aarch64 artifacts, wall clock each | `time cargo build -p apr-cli --release --features cli`; same with `cli,cuda` |
| S0-G2 | `gx10` and `lambda` driver ISA floor | `nvidia-smi --query-gpu=driver_version,compute_cap --format=csv` on both |
| S0-M1 | 7B Q4_K_M fits both dogfood hosts | model bytes vs `nvidia-smi` free VRAM on `lambda-labs`; unified memory on `gx10` |
| S0-R1 | runner groups and their workflow access | `gh api orgs/paiml/actions/runner-groups`; `gh api repos/paiml/aprender/actions/runners` |
| S0-R2 | current required contexts | `gh api repos/paiml/aprender/branches/main/protection --jq .required_status_checks.contexts` |
| S0-N1 | what `nightly.yml` builds today, per target and feature | `grep -nE 'target:\|features\|runs-on' .github/workflows/nightly.yml` |

---

## Appendix B — Naming and file locations

```
.github/workflows/release-assets.yml         # four artifacts, tag-scoped, per-host jobs
.github/workflows/promote-release.yml        # base-owned, manual dispatch, reads receipts
.github/workflows/cuda-test.yml              # PR-time, yoga, after PR Gate / authorize
scripts/run_clean_room.sh                    # the one build recipe
scripts/check_release_matrix.sh              # §8 guards …
scripts/check_model_parity.sh                # C14
scripts/publish_cascade.sh                   # local, fail-closed
evidence/models/supported.yaml               # the model manifest (derived)
keys/apr-release-minisign.pub                # committed public key; private key local only
contracts/apr-dogfood-models-v1.yaml         # new
docs/audits/release/<tag>/                   # every receipt for a release lives here
.claude/skills/apr-dogfood-models/SKILL.md   # the new skill, REPO scope (explicit `name:`)
                                             # v0.1 said ~/.claude/skills/… — that is #2361
```

---

## Changelog

**v0.2 — 2026-09-09.** Facts and mechanisms only. **No RD is decided**; every RD row now
carries the finding that bears on it and says what is still owed. Source: the PMAT-1092
review (#3056), one `agy /teamwork-preview` grill and eleven AD-04 quorum rounds, plus a
signed `pr-review` receipt whose §3.B arm supplied the PTX citations.

| § | change |
|---|---|
| header | status names *why* it is not normative: nine open RDs **and** absent dependencies |
| §2 | `yoga`'s four `[U]` cells measured; the sm_89 × 2 / sm_121 × 1 fleet shape stated |
| §3.1 | the unstated move off GitHub-hosted runners, and the per-target feature set, written down |
| §3.4 | **`cuobjdump` acceptance withdrawn** — it returns the same answer for every input, exits 255 on a correct artifact, and the tool is absent on the host that gates the release. Replaced with an emitter assertion carrying a registered mutation. PTX-floor *basis* reframed against a published ISA→driver mapping; RD-7's question left open |
| §5.1, App. B | skill moved to **repo scope** (#2332 / #2361) |
| §5.4 | M6 gains a per-host instrument and an explicit `UNMEASURED` for unified memory |
| §6 | retitled "target sequence"; `status` column; the 19-of-20 absence stated up front |
| §7 | cites `infra#359`, quotes the cost on which Option B was declined, and says plainly that §7 asks for something stronger than what was declined |
| §7.1 | `yoga` has no models; the budget's missing *population* named as the defect, with three disagreeing samples |
| §7.2 | constraints restated as prerequisites with a guard owed; the measured group configuration recorded |
| §8 | `status` column; contracts corrected to ones that exist, with the stale in-contract citation noted |
| §9 | the contradiction with §5.5/RD-5 stated where the target is, not left to be found |
| §12 | the R-5 / R-6 blocker named |
| App. A | per-row status; S0-M1 extension called for |

**Not changed, deliberately:** every RD's question, §1's doctrine, §5.2's manifest,
§10's refusals, and the content and ordering of §6's steps.

---

*End of draft. Comments in the review thread against section numbers; decisions
against RD-ids.*
