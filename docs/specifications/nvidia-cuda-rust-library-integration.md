# NVIDIA CUDA Rust — Integration for GPU Quality Stabilization

**Status:** PROPOSED · target release **0.67** · authored 2026-09-09 · HEAD `1faf09015`
**Upstream trigger:** [Introducing CUDA Rust: Two Tracks for Writing GPU Kernels](https://developer.nvidia.com/blog/introducing-cuda-rust-two-tracks-for-writing-gpu-kernels/)
**Goal (operator's words):** *stabilize quality*. Not throughput. Every proposal is judged by the
defect class it makes impossible to ship, never by tok/s.
**Review chain:** draft → `agy --mode grillme` (NO-GO) → revision → `agy /teamwork-preview` (NO-GO,
rescope) → this document. §9 records every finding, which were accepted, and which were wrong.

> **Bottom line.** The blog's two-track choice is **not** the decision in front of aprender.
> 0.67 should ship **T0 + O2**, and **neither needs an NVIDIA dependency**. The NVIDIA crates are
> real and measured to work (§0 G7–G9), but every one of them is blocked behind a driver or toolkit
> prerequisite no fleet host meets today, and they are answering a question aprender has not yet
> earned the right to ask. See §7.

---

## §0. Measured ground truth

Executed 2026-09-09 against HEAD `1faf09015`. **The command is the fact; the value is a dated
sample.** Nothing here is quoted from the blog.

| # | Fact | Measured by |
|---|------|-------------|
| G1 | `aprender-gpu` is **109,365 LOC** of hand-rolled PTX generation — `kernels/` 57,655 · `ptx/` 18,454 · `driver/` 12,193 · `memory/` 8,534 · `monitor/` 7,705 | `find crates/aprender-gpu/src -name '*.rs' \| xargs wc -l` |
| G2 | It holds **2,620** `#[test]` fns and is excluded from **every required check** — `.github/workflows/ci.yml:417` passes `--exclude aprender-gpu`. The only other CI reference is `cuda-nightly.yml:242`, one filter, `perf053` | grep over `.github/workflows/*.yml` |
| G3 | **444 of those tests need no GPU**: `cargo test -p aprender-gpu --lib` → `ok. 444 passed; 0 failed`, run time **0.10s** | run, lambda-vector |
| G3b | Their **cold build is 31s** (263 dependency crates) in an empty target dir. The *marginal* cost inside `workspace-test` is lower — those deps are shared — and is **NOT measured** | `cargo test -p aprender-gpu --lib --no-run`, fresh `CARGO_TARGET_DIR` |
| G4 | Hand-PTX needs a **string-level regex rewriter** to survive Blackwell: `driver/module.rs:207` (uncached) and `:294` (cached) call `patch_backward_branches_sm121` on **all** PTX when `compute_capability().0 >= 12`, rewriting every unconditional backward branch to a predicated one (GH-480 JIT loop-drop) | file read |
| G5 | The hand FFI arch table stops at `CU_TARGET_COMPUTE_90` — **no sm_100/120/121 constants** | `driver/sys/mod.rs:162-179` |
| G6 | Cubins are cached to `~/.cache/trueno/ptx/{sha256}.cubin`, keyed `sha256(patched_ptx ‖ jit_target ‖ driver_version)` — patched **before** keying | `driver/ptx_cache.rs:201-213` |
| G7 | `cuda-core` 0.3.1 **builds on stable Rust** (1.98.0; no nightly, no LLVM) in 10.26s | scratch crate, lambda-vector |
| G8 | `cuda-core` **refuses at runtime** on driver 570.207: `DriverError(3, "CUDA driver library unavailable: CUDA driver too old: built against 13.0 but runtime is 12.8")` | same crate, `cargo run` |
| G9 | `cuda-core` **PASSES end-to-end on gx10** — GB10 sm_121, CUDA 13.0, driver 590.48.01, stable rustc 1.95: `load_module_from_ptx_src` → `load_function` → `launch_kernel_on_stream(&mut [*mut c_void])` → **0 mismatches**; reported `num_registers=10`, `static_shared_mem=0`, `max_threads_per_block=1024` | probe on gx10 |
| G10 | `cuda-bindings` build.rs **hard-fails** without a CUDA **13.0+** toolkit (`"no CUDA 13.0+ toolkit was found"`); **no docs.rs escape hatch** | `cuda-bindings-0.3.1/build.rs:201` |
| G11 | Fleet: **gx10** CUDA 13.0 / drv 590.48.01 / sm_121 · **lambda-vector** CUDA 12.6–12.8 (13.0 headers present) / drv 570.207 / sm_89 · **yoga** *no toolkit* / drv 595.91.07 / sm_89 · **jetson** unreachable | ssh probe |
| G12 | cutile needs CUDA **13.1+** (sm_100+), **13.2+** (sm_8x), 13.3 recommended. **No fleet host qualifies today** | `cutile-rs` README |
| G13 | crates.io, Apache-2.0, permitted by `deny.toml`: `cutile`, `cuda-core`, `cuda-async`, `cuda-bindings`, `cutile-compiler`, `cutile-ir`, `cutile-macro`, `oxide-artifacts`. **Git-only** (never a dependency of a published crate): `ptx-parse`, `cuda-device`, `cuda-host`, `cargo-oxide`, `rustc-codegen-cuda` | crates.io API |
| G14 | The blog is **already stale**: it cites `nightly-2026-04-03`; upstream `main` pins **`nightly-2026-08-28`** and prefers `llc-23/22/21` | `cuda-oxide` README |
| G15 | In-repo prior art: `experiments/cuda-oxide/` holds **5 parity-verified ports** (q4k-matvec, incremental-attention, rmsnorm, swiglu, rope), all cos = 1.0000000 vs CPU. Promotion to decode was **RETRACTED** — oxide is **~4× SLOWER** than the production `HwDp4a` | `experiments/cuda-oxide/README.md`, PR #2045 closed |
| G16 | `--all-features` is used by **five** Makefile targets — `112`, `114`, `659` (`make coverage`), `994`/`1010` (`make mutants`). A workspace member's non-default feature **is** activated there | `grep -n -- --all-features Makefile` |
| G17 | `cutile` 0.1.0 → 0.3.1 in under 4 months (0.3.1 published 2026-09-04, five days before this doc). Total downloads 5,873 | crates.io API |
| **G18** | **aprender already declares the occupancy check — and it is vacuous.** `contracts/trueno/ptx-codegen-safety-v1.yaml` (2026-04-06) equation `register_budget` has postcondition `cuOccupancyMaxActiveBlocksPerMultiprocessor > 0`. Its generated macro `contract_register_budget!` (defined `generated_contracts.rs:18732`) is **invoked nowhere in the tree** — not by real code, not by tests, not by the generated file itself; and `cuOccupancyMaxActiveBlocksPerMultiprocessor` is an unbound identifier that would not compile if it ever were. Its falsifier FALSIFY-PTX-003 is **prose** (`"Parse ptxas output for register count, assert <= 128"`), and prose never runs. It carries `registry: true` — the exemption class the operator killed 2026-08-21 (**535** contracts still carry it). Its `domain` stops at sm_90, and it cites pre-monorepo paths | contract + `grep -rn 'contract_register_budget!'` |
| G19 | `cuFuncGetAttribute` / `cuOccupancy*` are **absent from `crates/aprender-gpu/src/`** entirely. `driver/sys/mod.rs` already has **52** `load_sym!` bindings, so adding them is one line per symbol | grep |
| **G21** | **cutile-rs 0.3.1 BUILDS AND RUNS on gx10 (GB10, sm_121)** after the CUDA 13.3 install — stable rustc 1.95, 18.13s, JIT through CUDA Tile IR, `len=1024 mismatches=0`. The T3 blocker is **gone** | probe on gx10, 2026-09-09 |
| **G22** | `/usr/local/cuda` on gx10 is **not a plain symlink** — it is `-> /etc/alternatives/cuda`, a Debian alternatives link. `cuda-toolkit-13-3` registers priority **133** vs 13.0's **130**, so the default flipped to 13.3 at 11:32:57 (`/var/log/alternatives.log`). Rollback is one command: `update-alternatives --set cuda /usr/local/cuda-13.0` | `update-alternatives --query cuda` |
| G20 | `experiments/` is invisible to cargo: `cargo metadata --no-deps` lists **79 packages, none under `experiments/`**; all six `experiments/cuda-oxide/*/Cargo.toml` carry their own `[workspace]`; no `Cargo.lock` there. **But** `scripts/complexity_baseline.txt:688-692` still names five `experiments/cuda-oxide/**` entries | delegate-verified in a clean export |

---

## §1. The reframe — and the objection to it

The blog frames a **two-track choice**: SIMT (`cuda-oxide`) vs Tile (`cutile-rs`). Both sit on a
**third** component it barely names — `cuda-core` / `cuda-async` / `cuda-bindings`, the host runtime,
on crates.io, buildable on **stable** Rust (G7), a near 1:1 superset of aprender's hand-rolled
`driver/` (G9).

**Claim.** aprender does not lack ways to author kernels (G1). The one time it ported five kernels to
cuda-oxide, they were *correct but 4× slower* and were retracted (G15). cutile cannot be built on any
fleet host (G12). So no kernel track is schedulable into 0.67, and the question is what else the
announcement makes possible.

**The strongest objection, stated fairly.** The grill's RANK2: G4 and G5 are not incidental. A stack
that needs a *regex rewriter* over generated assembly to survive a GPU generation, and whose arch
table never learned Blackwell exists, is a stack whose **authoring layer is the defect source**.
Adding an oracle "freezes the fragile hand-PTX stack behind a differential gate" — work that changes
nothing.

**Response.** The objection is right about the debt and wrong about the remedy's availability: its fix
is refuted by G12 and G15, not by opinion. What it legitimately demands is that this document not
pretend the debt is fine — §8 gives falsifiable **exit criteria** with named owners.

**And G18 changes the argument.** The check this document proposes is not a new idea imported from
NVIDIA. aprender has *declared* it since 2026-04-06 and never implemented it. The honest framing of
0.67 is therefore not "adopt NVIDIA libraries" but **"make an existing safety contract fire, and put
a gate under the crate that carries it."** Neither needs a dependency.

---

## §2. Scope decision for 0.67

Both review lanes returned **do-not-implement-as-written** on a four-tier 0.67. The teamwork lane
argued for T0 alone; it reached that partly on a misreading (it wrote that T0 "unlocks the 2,620
existing tests" — T0 unlocks **444**, G3, which the spec states plainly). Its *direction* is right and
its stopping point is too early.

**Ship in 0.67:**

| | Item | Why it is the highest quality-per-risk |
|---|---|---|
| **T0** | Un-dark `aprender-gpu` in CI (§3) | Nothing else is measurable without it. 444 tests, 0.10s |
| **O2** | Arm the vacuous `register_budget` contract (§4) | dependency-free. G18: aprender has *claimed* this check for 5 months. G19: ~1 line per symbol against 52 existing `load_sym!` bindings. Works on **every** host, CUDA-12 included |

| **T3** | cutile A/B evaluation on gx10 (§7) | **unblocked 2026-09-09** — operator ordered the toolkit upgrade; cutile now runs on sm_121 (G21) |

**Defer to 0.68:** T1 loader-differential oracle (§5 — needs cuda-core, so gx10 only, driver R580+),
T2 oxide verification tooling (§6).

---

## §3. T0 — Un-dark the GPU crate  *(0.67, no dependency)*

`aprender-gpu` carries 2,620 tests behind zero required checks (G2).

**Change.** Add `aprender-gpu` (default features — `default = []`, no CUDA link, and the crate has no
`build.rs`) to the `workspace-test` job in `.github/workflows/ci.yml`.

**Honest cost.** 444 tests, **0.10s run** (G3), **31s cold build** (G3b). The first draft quoted only
run time; the grill was right to call that self-serving. 444 of 2,620 is **17%** — this gates the
CPU-reachable subset only; ~2,176 sit behind `--features cuda` and need hardware. It is **not**
comprehensive coverage and must not be described as such.

**Falsifier FALSIFY-GPU-DARK-001.** Seed a defect in the PTX emitter → new gate goes RED → revert the
ci.yml line → goes GREEN. Guard and its classification in the same commit.

> `.github/workflows/*` changes need a web-UI merge click per repo policy.

---

## §4. O2 — Arm the `register_budget` contract  *(0.67, no dependency)*

**This is the headline change, and it is a contract-integrity fix, not an integration.**

G18: `contracts/trueno/ptx-codegen-safety-v1.yaml` asserts every emitted kernel stays inside the
register and shared-memory budget, with postcondition
`cuOccupancyMaxActiveBlocksPerMultiprocessor > 0`. Nothing enforces it: the macro is never invoked,
the symbol is unbound, the falsifier is prose, and it is `registry: true`. Meanwhile aprender launches
hand-emitted PTX with hardcoded block sizes (128 / 256) and validates **nothing** about the kernel the
JIT actually produced.

**Implementation.** Bind `cuFuncGetAttribute` and `cuOccupancyMaxActiveBlocksPerMultiprocessor`
through the existing `load_sym!` pattern in `driver/sys/mod.rs` (G19), and assert per kernel at module
load:

- `CU_FUNC_ATTRIBUTE_NUM_REGS` ≤ the target's per-thread ceiling
- `CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES` ≤ the device's per-block limit
- `CU_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK` ≥ the hardcoded block size at that launch site
- `cuOccupancyMaxActiveBlocksPerMultiprocessor(...) > 0` — the contract's own postcondition, now real

**Contract work, same PR** (coverage + contracts co-evolve, Rule 7):
1. Replace FALSIFY-PTX-003's prose `test:` with an executable command.
2. Drop `registry: true` (operator ruling 2026-08-21).
3. Extend `domain` past sm_90 to sm_100/120/121 — G5's blindness exists at the contract level too.
4. Fix the pre-monorepo paths in `references:` (`trueno/src/backends/gpu/`, `realizar/src/cuda/…`).
5. `pv validate contracts/trueno/ptx-codegen-safety-v1.yaml` — never bash.

**Mutation proof (mandatory).** Raise a kernel's block size above
`CU_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK`, confirm RED, revert, confirm GREEN. A contract that has been
vacuous for five months does not get to be re-armed on assertion.

**Why this is not a route-around of the NVIDIA work.** It is the same check `cuda-core` would give
(G9 measured it reporting `num_registers`, `static_shared_mem`, `max_threads_per_block`) — obtained
without a dependency, on every host, including the CUDA-12 boxes `cuda-core` refuses to run on (G8).

---

## §5. T1 — `cuda-core` as loader differential  *(0.68)*

### §5.1 What the grill killed

The draft's O1 said *"load the same PTX through aprender's hand FFI and through cuda-core, assert
bit-identical output."* The grill called that a **tautology**. **Accepted as written.**

### §5.2 What survives — the paths are not the same

```
aprender:  PTX ─► patch_backward_branches_sm121 ─► JIT ─► cubin ─► disk cache
                  (rewrites EVERY unconditional       keyed sha256(patched ‖ target ‖ driver)
                   backward branch, ALL PTX, major>=12)
cuda-core: PTX ────────────────────────────────► JIT ─► fresh, no patch, no cache
```

- **O1a — is the GH-480 patch still necessary?** It fires on every kernel on any `major >= 12` device.
  GH-480 was diagnosed on CUDA 13.0. If unpatched-cuda-core now agrees with patched-aprender *and*
  with the CPU reference, the rewriter is dead code mutating every Blackwell kernel we ship. **Kept.**
- **O1b — is the patch correct?** The teamwork lane was right that a two-way patched-vs-unpatched
  comparison is confounded: on an affected driver the unpatched kernel is *known* miscompiled, so a
  mismatch proves nothing. **Repaired to a three-way test** — `{unpatched-cuda-core, patched-aprender,
  CPU reference}`. Patched≡CPU and unpatched≢CPU ⟹ patch working. Patched≢CPU ⟹ patch broken.
  All three equal ⟹ O1a fires.
- **O1c — is the cubin cache key sound?** The lane was right that one run cannot falsify a cache
  collision. **Repaired from a differential to a case table**: cold-load vs warm-load must agree; and
  perturbing each key input (`patched_ptx`, `jit_target`, `driver_version`) must change the key, with
  a must-match / must-not-match table re-run rather than re-read.

### §5.3 Packaging — and the residual reach-in

The draft proposed a workspace member with a non-default feature. **Wrong**, and the grill's strongest
hit: G16 — `--all-features` at Makefile:659/994/1010 activates non-default features on members, so
`cuda-bindings`' build.rs (G10) would hard-fail `make coverage` and `make mutants` for every
contributor without a CUDA 13.0+ toolkit. `publish = false` does not shield it.

**Correct home:** `experiments/cuda-core-oracle/` with its **own `[workspace]`** — the pattern already
proven here. G20 verifies cargo cannot see `experiments/` (79 packages, none under it).

**Residual, from the teamwork pass — both must be closed in the same PR:**
1. `scripts/complexity_baseline.txt:688-692` still names five `experiments/cuda-oxide/**` entries.
   The isolation is real for cargo and leaky for the repo's own tooling.
2. Root `cargo deny check` does **not** traverse an isolated workspace. R4 needs an explicit
   `cargo deny check --manifest-path experiments/cuda-core-oracle/Cargo.toml`.

**Gate host:** gx10 only — lambda-vector cannot run cuda-core (G8).
**Contract:** `contracts/nvidia-cuda-rs-loader-differential-v1.yaml`, obligations O1a/O1b/O1c.

### §5.4 Deletion criteria for the oracle *(raised by teamwork; the draft had none)*

Delete `experiments/cuda-core-oracle/` when **any** holds: O1a proves the GH-480 rewriter unnecessary
and it is removed (the oracle's main subject is gone); or it produces no finding across two
consecutive releases; or upstream API churn (G17) costs more than one fix per release.

---

## §6. T2 — cuda-oxide as a *verification tool*  *(0.68)*

The retraction stands (G15): **do not promote oxide kernels onto the decode path.** Adopt the two
pieces aprender has never used, both aimed at classes it has been bitten by:

- `cargo oxide sanitize --tool memcheck|racecheck|synccheck` over the five existing ports.
- **`ptx-schedule`** — schedule-perturbation / nanosleep-injection fuzzing, aimed at the intra-warp
  ordering class behind PP-26 kernel-family divergence and the GH-480 family.

Out-of-workspace in `experiments/` (forced by G13 — git-only crates can never be a dependency of a
published crate). Nightly on gx10. Pin `nightly-2026-08-28` (G14), **not** the blog's stale date.

---

## §7. T3 — cutile-rs on gx10  *(0.67; operator decision)*

**Operator ruling, 2026-09-09, verbatim: "false YOU WILL UPGRADE".** This overrules the grill's
RANK3 objection and this document's own earlier recommendation, both of which argued against
touching gx10's toolkit. Per repo doctrine a review lane may not reopen an operator decision; the
recommendation below is withdrawn and the upgrade is the decision of record.

**Executed.** `cuda-toolkit-13-3` (13.3.1-1, sbsa/arm64, 66 packages) installed on gx10-a5b5.
Deliberately *not* installed: any kernel-driver package — the plan was asserted driver-free before
apt ran, and the driver is unchanged at **590.48.01**. CUDA 13.0 remains on disk and intact.

**Result — the blocker is gone (G21).** cutile-rs 0.3.1 builds and runs on GB10 **sm_121**: stable
rustc 1.95, 18.13s, JIT through CUDA Tile IR, `len=1024 mismatches=0`. Before the upgrade the fleet
maximum was 13.0, below cutile's 13.1 floor, and this was untestable.

**What the upgrade actually changed, and what this document got wrong (G22).** `/usr/local/cuda` is
a **Debian alternatives** link, not a plain symlink. `cuda-toolkit-13-3` registers priority 133
against 13.0's 130, so **the default toolkit flipped to 13.3** — `nvcc` and `ptxas` on the CI PATH
are now 13.3. The claim first reported here, that the install was purely side-by-side with the CI
lane untouched, was **wrong**; see §9.4.

**Rollback is one command and instant** (no reinstall, no download):

```bash
sudo update-alternatives --set cuda /usr/local/cuda-13.0    # and cuda-13 likewise
```

**Required before this lands:** the GH-480 rewriter (G4) now meets `ptxas` 13.3, which is exactly
the interaction the objection named. `cargo test -p aprender-gpu --features cuda --lib --release`
must pass on gx10 under 13.3 — see §9.5. A red result is a rollback, not a debate.

**Deliverable:** a **measurement, not an adoption** — one cutile kernel vs the production hand-PTX
equivalent, same-data A/B on gx10, plus build reproducibility. Precedent: HuggingFace **Grout**, a
Qwen3 inference engine on cutile — the direct analogue of `aprender-serve`.

## §8. Exit criteria for the hand-PTX stack

Answering the grill's RANK2 rather than dodging it. Teamwork's objection — that these are unowned
escape hatches — is accepted; each now names an owner and a measurement.

| # | Criterion | Measured by | When |
|---|---|---|---|
| 1 | A cutile or oxide port **matches or beats** the production kernel (`HwDp4a`, **not** `TiledQ4KGemv` — the comparand error behind the retracted 1.45× claim, G15) at every decode-hotpath shape | the T3 A/B harness on yoga | each release |
| 2 | O1a shows the GH-480 rewriter (G4) is unnecessary on the current driver | T1 oracle, gx10 | each driver bump |
| 3 | `aprender-gpu` required-check coverage plateaus **< 50%** of 2,620 tests | **needs a script that does not exist** — `scripts/gpu_test_coverage.sh`, emitting `covered/total`, wired into the T0 PR and ratcheted like the other baselines | two consecutive releases with no rise |
| 4 | A CUDA generation ships that the arch table (G5) + contract `domain` (G18) cannot be extended to without a second Blackwell-class rewrite | O2's arch-table extension work | on arch release |

Criterion 3 was **unfalsifiable as drafted** (teamwork, correct). It is only a criterion once the
script exists; that script ships with T0 or criterion 3 is struck.

---

## §9. Risks, gaps, and the review record

### §9.1 Risks

| # | Risk | Mitigation |
|---|------|-----------|
| R1 | **Alpha churn.** cutile 0.1.0→0.3.1 in <4 months (G17); cuda-oxide self-described alpha, re-pinned its nightly since the blog (G14) | Nothing NVIDIA ships **in the product** in 0.67. T3 is an out-of-workspace evaluation; never on the decode path, never in the shipped dependency graph |
| R2 | **`--all-features` breakage** (G16); the draft's mitigation was insufficient | Own-`[workspace]` crate under `experiments/` (§5.3), verified by G20 |
| R3 | **Driver floor R580.** lambda-vector (570.207) cannot run cuda-core (G8) | gx10 is the only T1 host; lambda-vector needs a driver bump or stays out |
| R4 | **Licence / advisories.** Apache-2.0 is permitted by `deny.toml`, but root `cargo deny` **does not traverse an isolated workspace** | Explicit `cargo deny check --manifest-path experiments/cuda-core-oracle/Cargo.toml` (advisories, bans, sources — not licences alone) |
| R5 | **Oracle disagreement ≠ hand-PTX bug.** cuda-core can be wrong | Every O1 verdict is three-way against the CPU reference (§5.2) before either side is blamed |
| R6 | **gx10 SPOF** — the only Blackwell host and the only `cuda-nightly` host now defaults to CUDA 13.3 | Operator-directed (§7). Driver untouched; 13.0 still on disk; rollback is one `update-alternatives --set` with no download. Gated on the §9.5 regression run |
| R7 | **Neither review was a quorum.** grillme: `children=1, label=single-lane`. teamwork: `children=unknown, method=none` — the count was *never recorded*, which is not the same as zero | Both verdicts are one model's opinion. Every claim either lane made was re-verified here before acceptance |
| R8 | **Re-arming a five-month-vacuous contract on assertion** | O2 ships only with the revert-to-RED mutation proof (§4) |

### §9.2 Gaps this document still does not close

Named rather than hidden. The teamwork lane credited the spec with handling MSRV and `Cargo.lock`
churn; the delegate corrected that — the document contains **zero** occurrences of `MSRV`,
`rust-version`, `Cargo.lock`, `transitive` or `supply`.

- **MSRV.** `cuda-core`/`cutile` declare `rust-version = 1.89`; the repo pins 1.93.0. Structurally moot
  while §5.3 holds (out-of-workspace), but unstated and unverified.
- **Lockfile / transitive surface.** An isolated workspace has its own lock, so root `Cargo.lock` churn
  should be nil — **asserted, not measured.** `cuda-core` pulls `bindgen`, `clang-sys`, `libloading`,
  `half`; that transitive set has had no supply-chain review.
- **Who operates the gate.** T0's gate is CI. O2's is CI. T1/T2 are nightly on gx10 with **no named
  owner** for a red nightly.
- **`cargo package` behaviour** for an isolated `experiments/` member is unverified against
  `scripts/check_include_files.sh`.

### §9.3 Operator decision

**2026-09-09, verbatim: "false YOU WILL UPGRADE".** The grill's RANK3 (gx10 SPOF) and this
document's own §7 recommendation are overruled. The gx10 toolkit upgrade was executed the same day
(§7) and unblocked T3 (G21). A review lane may not reopen an operator decision.

### §9.4 A false-green check in this document's own verification

The upgrade was first reported here as "side-by-side, CI lane untouched". **That was wrong** (G22).
The check that produced it was:

```bash
[ "$(readlink -f /usr/local/cuda)" = "$(cat $B/cuda-symlink.before)" ]; chk "still -> $(cat $B/cuda-symlink.before)" $?
```

The command substitution **inside the message argument** runs before `$?` is expanded, so `$?`
carries `cat`'s status (0), never the test's. Mismatched strings printed `OK`. Verified by
reproduction: with the substitution in the message the same failing test prints `OK`; without it,
`FAIL`. This is the CLAUDE.md rule *"never read `$?` through a pipe"* generalised — **capture the
status into a variable before any other command substitution appears on the line**:

```bash
[ "$a" = "$b" ]; rc=$?; chk "msg with $(cat file)" $rc
```

The error was caught only because a later, independent probe reported `ptxas release 13.3` on a
path this document claimed was 13.0. A guard whose own status can be laundered is theater; this one
was, for one turn.

### §9.5 Post-upgrade regression gate — RUN, with a control

`cargo test -p aprender-gpu --features cuda --lib --release` on gx10 under CUDA 13.3 (the GH-480
rewriter, G4, meeting `ptxas` 13.3):

> **`2568 passed; 2 failed; 12 ignored`** in 28.77s.

Two failures is not a verdict until the mechanism is proven, so the alternative was rolled back to
13.0 and the same two tests re-run — a control, then restored to 13.3:

| Test | on 13.0 | on 13.3 | reading |
|---|---|---|---|
| `driver::memory_fuzz_tests::adversarial::test_alloc_oversize_100gb` | **FAIL** | **FAIL** | pre-existing; **not** caused by the upgrade |
| `driver::cublas_tests::test_cublas_gemm_f16_training_shape` | **pass** (alone) | **pass** (alone) | passes alone on both, failed only inside the full suite ⇒ load-contended, **not** a toolkit regression |

**Verdict: CUDA 13.3 caused neither failure. The upgrade is exonerated by control, not by assertion.**

Both are pre-existing defects on this host and should be filed separately:

1. `adversarial.rs:36` asserts a 100 GB allocation is *"impossible on RTX 4090"* — a hard-coded
   host assumption. gx10 is a **GB10 with unified memory**, where the allocation legitimately
   succeeds. The test encodes the wrong machine.
2. `cublas_tests.rs:174` asserts `> 50 TFLOP/s` and measured 15.4 TFLOP/s under full-suite load,
   while passing in isolation. This is a **wall-clock assertion**, the class the repo has already
   been bitten by four times; it must not sit in a required check.

Neither had been visible, because `cuda-nightly.yml:242` runs only the `perf053` filter — so the
`--features cuda` suite has never been green on the Blackwell host, and nothing said so.

### §9.6 Review record

**Grill** — `agy --mode grillme`, agy 1.1.28, **1 lane**, 125.7s, `do-not-implement-as-written`,
conversation `11f250f2-d5d1-438f-89c2-96cbd9a362b0`.
*Accepted:* RANK1 (O1 as drafted is a tautology → §5.1/§5.2), RANK3 (gx10 SPOF → §7 withdrawal),
RANK5/delegate-upgraded (`--all-features` → §5.3), RANK4 in part (build time → G3b, 31s).
*Rejected with grounds:* RANK2's **remedy** (migrate kernels in 0.67) — refuted by G12 and G15. The
debt itself is conceded and answered by §8.
*Corrected:* the lane's "~3.6s compilation time (measured)" was **never measured** (delegate-verified);
real cold build is **31s** (G3b). Its cuBLAS-link hypothesis for T0 is **unsupported** —
`crates/aprender-gpu` has no `build.rs` and `default = []`.

**Teamwork** — `agy /teamwork-preview`, agy 1.1.28, 211.7s, `do-not-implement-as-written`,
conversation `49075dbf-ed16-47a1-b702-bf5bb56c8cf3`, fan-out **unmeasured**.
*Accepted:* O1b confounded and O1c single-run-weak (→ §5.2 repairs); §5.3 verified correct for cargo
but `complexity_baseline.txt` reach-in survives (G20); root `cargo deny` misses an isolated workspace
(R4); exit criterion 3 unfalsifiable without a script (§8); scope too large for one release (§2).
*Rejected with grounds:* "rescope to T0 alone" — the lane reached it partly on a misreading, writing
that T0 "unlocks the 2,620 existing tests" when T0 unlocks **444** (G3), which the reviewed text
stated. O2 is dependency-free, works on hosts cuda-core refuses (G8), and repairs a contract that has
been vacuous for five months (G18); it belongs in 0.67.
*Correction made during citation verification:* the patcher call sites were written as `module.rs:206,293` — taken from the delegate's report rather than from this session's own earlier grep, which said **207/294**. 206/293 is the `if major >= 12` guard. A subagent's line numbers are a claim; re-run the grep.

*Neither lane found G18* — the vacuous `register_budget` contract — which is the single strongest
finding in this document and the reason §2 does not reduce to T0.
