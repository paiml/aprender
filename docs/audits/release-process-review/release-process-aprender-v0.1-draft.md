# `release-process-aprender.md`

**Release process for aprender tagged releases — build matrix and physical model dogfood**

| | |
|---|---|
| Status | **DRAFT v0.1 — for team review.** Not yet normative. |
| Author | Noah Gift (driver-assisted) |
| Date | 2026-09-08 |
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

The build matrix is meaningless without knowing what each machine is. Rows marked
`[U]` must be filled by running the S0 commands in **Appendix A** before this spec
is adopted.

| host | arch | compute | CUDA arch | role in this process | CI runner? |
|---|---|---|---|---|---|
| `intel` | x86_64 | Xeon W-3245, 283 GB, CPU only | — | clean-room CI runner; builds `x86-cpu`; **the publish gate** | yes (clean-room) |
| `yoga` | `[U]` presumed x86_64 | `[U]` GPU model / VRAM | `[U]` (`sm_??`) | **new** — builds `x86-gpu-cuda`; runs CUDA unit tests at PR time | **new**, dedicated build runner |
| `gx10` | aarch64 | GB10 Blackwell, 120 GB unified | sm_121 | builds `arm-cpu` and `arm-gpu-cuda`; model dogfood host | yes, JIT-scoped (§7) |
| `lambda-labs` | x86_64 | RTX 4090, 24 GB | sm_89 | model dogfood host; release-asset build for `x86-gpu-cuda` **only** if RD-3 rejects `yoga` | **no** — not a general CI runner |
| `mini` | aarch64 | Apple M4, 16 GB unified | — (Metal) | darwin artifact verification only, if RD-1 keeps darwin | no |

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
- **PTX, not SASS.** `apr` emits PTX as text and the driver JITs it. Acceptance:
  `cuobjdump` lists **no** ELF section, and the emitted `.target`/`.version` is read
  on both GPU hosts. The `.version` must be **≤ the minimum ISA the fleet's oldest
  driver accepts**, `basis=nvidia-smi` on `lambda-labs` and `gx10`. A PTX version
  above a user's driver produces a JIT failure the registry cannot explain.
- **sm asymmetry is a first-class risk.** `lambda-labs` is sm_89, `gx10` is sm_121.
  The `arm-gpu-cuda` artifact has exactly **one** verification host in the fleet
  (`gx10`) — single-host verification with no cross-check. Stated, not hidden
  (RD-6).

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
| **M6** | 7B service smoke | 7B loads, produces > 0 tokens, records stop reason, no OOM, peak RSS/VRAM recorded with `basis=` | cap VRAM below need → refusal with reason, not a crash |
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

## §6 Release sequence (normative, end to end)

Each step is idempotent — the DONE-IF check runs first and the step is skipped if it
holds.

| # | step | gate | DONE-IF |
|---|---|---|---|
| 1 | `scripts/release_criteria.sh --all` | exit 0 | — |
| 2 | `scripts/run_clean_room.sh` on `intel` | **exit 0 — the hard gate, first** | — |
| 3 | `apr-dogfood --release` | GO receipt for this build sha | receipt exists for this sha |
| 4 | claim host slots (`yoga`, `gx10`) | `check_host_slot.sh` | slots held |
| 5 | tag `vX.Y.Z-rc1` as **prerelease** | — | tag exists (**never move a tag**) |
| 6 | build the four artifacts on their machines | one recipe; per-artifact build receipt | four artifacts + `SHA256SUMS` present, sha256 matching |
| 7 | `make sign-release TAG=<tag>` — **local, on Noah's workstation** | `SHA256SUMS.minisig` verifies against the committed public key | `.minisig` present and verifying |
| 8 | `apr-dogfood-models` on `lambda-labs` and `gx10` | GO on all cells (§5) | cell receipts exist for this manifest sha256 |
| 9 | C4 verification cells (§4) | receipts on every listed host | receipts present |
| 10 | **promotion** — base-owned dispatch job reads receipts, flips `prerelease=false` | four artifact receipts ∧ all model cells GO ∧ parity never `skipped` | `isPrerelease == false` |
| 11 | release notes | model table generated from the manifest with per-host parity; backend table from the registry; **no number outside those tables** (claims ratchet) | both tables present |
| 12 | crates.io cascade — `scripts/publish_cascade.sh` from a detached checkout of the promoted tag | dry-run receipt, then one crate per call, stop on first non-zero | `cargo search` shows the version for every cascade crate |
| 13 | post-publish | C4 + model cells re-run against the **crates.io** install on one host | receipt present |

**No workflow publishes to crates.io.** Signing (7) and publishing (12) use local
keys. Everything else is machine-decided.

---

## §7 PR-time CUDA testing on `yoga`

This is the largest quality change in the document and the one with the largest
security surface. It is scoped deliberately.

### 7.1 What runs

A new required check `cuda-test` on `yoga`, PR-time:

- `cargo test -p aprender-gpu -p aprender-serve --features cuda` — the CUDA unit
  tests that no runner could execute before.
- **Two parity sentinels only** (1.5B RED-first, 7B witness) over ≥ 64 positions.
  The *full* manifest runs at release, nightly, and post-publish — a PR must not pay
  for sixteen cells.
- Budget: the check must complete inside the merge-queue ceiling
  (`[U]` — measure and record `basis=`; current workspace-test observed ≈ 34 min).

### 7.2 Security constraints (non-negotiable)

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
comment.

| guard | enforces | mutation that must go RED |
|---|---|---|
| `check_release_matrix.sh` | §3.1 — the four artifacts exist with sha256 + manifest, built by the assigned host per the receipt | drop one artifact; swap a builder host |
| `check_no_cudart_link.sh` | §3.4 — no `libcudart` in any `-cuda` artifact | link cudart → RED |
| `check_ptx_version.sh` | §3.4 — emitted PTX `.version` ≤ fleet minimum | bump the target above the fleet minimum → RED |
| `check_host_slot.sh` | §2 — one release role per host | start a perf receipt while a build slot is held → RED |
| `check_model_parity.sh` | §5 M3 / criterion **C14** | revert the parity fix → RED on the sentinel |
| `check_release_receipts.sh` | §6 step 10 — promotion inputs | remove a cell receipt; tamper one byte; a receipt with `parity: skipped` |
| `check_readme_claims.sh` | §5.2 — README model ∉ manifest | add an unmanifested model name → RED |
| `check_publish_cascade.sh` | §6 step 12 — refuses on branch, dirty tree, prerelease, or a token in the environment | run from a branch → refuse |

Contracts: `contracts/apr-release-assets-v1.yaml`,
`contracts/apr-gpu-cpu-parity-v1.yaml`, `contracts/apr-publish-cascade-v1.yaml`,
and **new** `contracts/apr-dogfood-models-v1.yaml`.

---

## §9 Toyota Way targets

Targets are zeros, ones, and ratchets from measured baselines — never invented
continuous thresholds.

| principle | target | instrument |
|---|---|---|
| **jidoka** | GPU/CPU divergence found by a **user**: **0 per release** (baseline: 1 — #2971) | issues labelled P0 filed by non-maintainers |
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
| **RD-1** | darwin/windows: carry, drop, or verify? C13 says five targets; this spec names four. | Verify `aarch64-apple-darwin` on `mini`; drop windows from tagged releases until a verification host exists; amend C13 to match. | team |
| **RD-2** | installer default when a driver is present | `-cuda`, with the choice and reason printed and `--backend` overriding (FX-18) | team |
| **RD-3** | `cuda-test` on `yoga`: required in 0.66 or advisory→required in 0.66.1 | advisory 14 days with a published red/green record, then required at `0 false reds` | team |
| **RD-4** | `gx10` builds two artifacts **and** hosts two dogfood cells **and** is a perf host | one role at a time, slot-claimed; if the release wall-clock is unacceptable, move `arm-cpu` to a cross-build on `intel` and re-verify on `gx10` | team |
| **RD-5** | if #2971 is not root-caused by the tag | ship with honest refusal: the 1.5B row reads `UNSERVICEABLE(cuda, #2971)`; #2971 stays P0 for 0.66.1 | Noah |
| **RD-6** | `arm-gpu-cuda` has exactly one verification host | ship it, state the single-host limitation in the notes; a second aarch64+CUDA host is a 0.67 fleet ask | team |
| **RD-7** | PTX `.version` floor | the fleet minimum by `nvidia-smi`, re-derived each release, not pinned in a script | team |
| **RD-8** | bit-for-bit reproducibility across machines | out of scope for 0.66; same-machine determinism required now | team |
| **RD-9** | `yoga` specs unknown | fill the ledger (Appendix A) before adoption — a builder whose arch is `[U]` cannot be assigned a target | Noah |

---

## §12 Adoption plan

| release | contents |
|---|---|
| **0.66.0** | §3 four artifacts; §4 verification matrix; §5 `apr-dogfood-models` with both models on both hosts; §6 sequence; §8 guards; §7 `cuda-test` **advisory** |
| **0.66.1** | `cuda-test` required per RD-3's criterion; RD-1's darwin decision implemented; second aarch64 GPU host if RD-6 says so |
| **0.67** | performance cells promoted from REPORTED to gated with their own ratchets; `metal` artifact; PR-time full-manifest parity if the budget allows |

**Nothing in 0.66 is gated on a number this document invents.** Every threshold that
does not yet have a measurement is `[U]` with a named owner, and the release ships
with the measurement or with an honest refusal.

---

## Appendix A — S0 ledger to run before adoption

Each row must be executed and its output pasted into the review thread. `[U]` rows
in §2 close here.

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
~/.claude/skills/apr-dogfood-models/SKILL.md # the new skill (explicit `name:`)
```

---

*End of draft. Comments in the review thread against section numbers; decisions
against RD-ids.*
