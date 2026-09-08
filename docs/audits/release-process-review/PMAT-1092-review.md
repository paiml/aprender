# Review — `release-process-aprender.md` v0.1

| | |
|---|---|
| Ticket | PMAT-1092 |
| Reviewed | `docs/specifications/release-process-aprender.md` v0.1 (463 lines) |
| Against | `origin/main` @ `c04eda87d`, and the physical hosts, 2026-09-08 |
| Method | one `agy /teamwork-preview` lane (agy 1.1.27, conversation `55b4205c`), every finding then re-run here |
| Verdict | **do-not-implement-as-written** — adopt §1/§2/§5.2/§9/§10 now, block §3.4/§5.1/§5.3/§6/§7 on the items below |
| Marks | `[V]` verified by a command printed here · `[C]` computed · `[A]` asserted, source named · `[U]` unverified, owner named |

The draft is right about *what* to gate and mostly wrong about *what already exists*. Its
doctrine (§1), its host ledger's shape (§2), its model manifest (§5.2), its Toyota targets
(§9) and its refusals (§10) are admissible today. Its normative sequence is not.

Evidence pack with every command: `docs/audits/release-process-review/ground-truth.md`.

---

## Verdict per section

| section | verdict | blocked on |
|---|---|---|
| §1 doctrine · §2 ledger shape · §5.2 manifest · §9 targets · §10 refusals | **admissible** | — |
| §3.1 artifact set | **blocked** | RD-1, and the unstated runner-class + feature-set change (F11) |
| §3.2 FX-16/17/18 | admissible **as fixtures**; the artifacts they test do not exist yet | F1 |
| §3.3 one recipe | **blocked** | `run_clean_room.sh` does not exist (F1) |
| §3.4 CUDA specifics | **reject as written** | the `cuobjdump` probe cannot fail (F2); the PTX floor is derived from the wrong population (F7) |
| §4 verification matrix | admissible once §3.1 resolves | F1, RD-1 |
| §5.1 skill location | **reject as written** | user scope (F6) |
| §5.3 cell matrix | **blocked** | `install.sh` is unwritten and owned by open R-6/PMAT-994 (F1); yoga has no models (F9) |
| §5.4 M1–M8 | admissible; M6 needs a per-host instrument (F9) | — |
| §5.5 verdict table | **blocked** | contradicts §9 (F8) |
| §6 sequence | **reject as "normative"** | 19 of 21 named paths absent (F1) |
| §7 PR-time CUDA on yoga | **reject as written** | reopens a settled decision without answering it (F3); security config does not exist (F4) |
| §8 guards | admissible as a *build list*, not as an enforcement table | F1, F5 |
| §11 RD-1..RD-9 | escalated below — none decided here | Noah / team |

---

## Findings, ranked by consequence

### F1 · S1 · §6 is labelled "normative, end to end" and 19 of the 21 paths it depends on do not exist `[V]`

`§6` step 1 runs `scripts/release_criteria.sh --all`; step 2 runs `scripts/run_clean_room.sh`
and calls it "**the hard gate, first**"; step 12 runs `scripts/publish_cascade.sh`. `§3.3`
says "Every artifact is built by `scripts/run_clean_room.sh`'s recipe". `§8` is titled "What
enforces each rule" and tables eight `check_*.sh`. `§5.3` says every cell "installs via
`install.sh` from the tag".

Measured on the worktree, and then again with `find . -name '*<basename>*'` across the whole
tree so a moved file could not hide:

> 21 paths named. **1 exists** (`scripts/check_readme_claims.sh`, 533 lines). **1 is
> correctly marked "new"** (`contracts/apr-dogfood-models-v1.yaml`). **19 are absent** and are
> written in the present tense as the process of record.

Absent: `run_clean_room.sh`, `release_criteria.sh`, `publish_cascade.sh`,
`check_release_matrix.sh`, `check_no_cudart_link.sh`, `check_ptx_version.sh`,
`check_host_slot.sh`, `check_model_parity.sh`, `check_release_receipts.sh`,
`check_publish_cascade.sh`, `release-assets.yml`, `promote-release.yml`, `cuda-test.yml`,
`install.sh`, `evidence/models/supported.yaml`, `keys/apr-release-minisign.pub`,
`contracts/apr-release-assets-v1.yaml`, `contracts/apr-gpu-cpu-parity-v1.yaml`,
`contracts/apr-publish-cascade-v1.yaml`.

`install.sh` is not merely missing — it is **owned by PP-066 R-6 / PMAT-994, `status: open`,
due 2026-10-16**, and R-5/PMAT-993 (the assets themselves) is open with the same shape. So
`§6` step 8 cannot run before two open tickets land, and the draft's `§12` adoption plan puts
all of it in **0.66.0** without naming that dependency.

**Consequence.** A reader following `§6` aborts at step 1 with `command not found`. `§8`
enforces nothing. Worse for a document whose §1.1 is "gates or theater": a table titled
"What enforces each rule" whose rows do not exist is itself the theater it forbids.

**Smallest fix.** Add a `status` column (`exists | to-build (ticket)`) to `§6` and `§8`;
retitle `§6` "target sequence" and add a build order; make R-5/R-6 explicit blockers in `§12`.
Nothing about the *content* of `§6` changes — only its tense.

---

### F2 · S1 · §3.4's `cuobjdump` acceptance cannot exclude any outcome, and read literally it goes RED on a correct artifact `[V]`

`§3.4` says: "**PTX, not SASS.** … Acceptance: `cuobjdump` lists **no** ELF section".

aprender does not embed device code at all. `crates/aprender-gpu/Cargo.toml:7` —
`description = "Pure Rust PTX generation for NVIDIA CUDA - no LLVM, no nvcc"`. The PTX header
is written by Rust at `crates/aprender-gpu/src/ptx/builder/ptx_module.rs:110`:

```rust
let _ = writeln!(ptx, ".version {}.{}", self.version.0, self.version.1);
```

with `target: "sm_70".to_string()` as the default at `:35`, and
`crates/aprender-gpu/src/ptx/mod.rs:43-67` carrying `/// Minimum supported PTX version (7.0
for SM 7.0+)` plus `validate_version` / `validate_target` (rejects `sm_<70`).

Run against a real `apr`:

```console
$ readelf -S /home/noah/.cargo/bin/apr | grep -icE 'nv_fatbin|\.nv\.'
0
$ cuobjdump --list-elf /home/noah/.cargo/bin/apr
cuobjdump info    : File '/home/noah/.cargo/bin/apr' does not contain device code
$ echo $?
255
```

**Both polarities are broken.** "Lists no ELF section" is true of the `-cuda` artifact, of the
`-cpu` artifact, of a build with CUDA compiled out entirely, and of `/bin/true` — the probe
returns the same answer for every input, so it excludes nothing (§1.5). And a guard written
literally from that sentence — `cuobjdump --list-elf apr` — exits **255** on a *correct*
artifact, so the naive implementation is a false-RED generator.

Note the lane's version of this finding was wrong on the mechanism: it claimed the gate fails
because `cuobjdump` is not installed on the CPU-only clean room. It **is** installed —
`/usr/bin/cuobjdump` on `intel`, `yoga` and `lambda-labs`, `/usr/local/cuda/bin/cuobjdump` on
`gx10` `[V]`. The defect is vacuity, not absence.

**Smallest fix.** Assert on the emitter, where the invariant actually lives, not on the
shipped ELF: a unit test over `PtxModule`'s emitted header asserting `.version` ≤ floor and
`.target` ≥ the declared minimum, with the registered mutation "bump the constant above the
floor → RED". Keep a *separate* ELF probe if you want one, but state it as
`readelf -S | grep -c '\.nv_fatbin'` **== 0** — which at least names the outcome it excludes.

---

### F3 · S1 · §7 reopens a decision that was made, without answering the reason it was made `[V]`

`§0` claim 2 and all of `§7` rest on yoga running CUDA unit tests at PR time. That exact
question was adjudicated in **`paiml/infra#359` (OPEN)** — "x86_64 GPU CI has no sanctioned
runner". Its three options were A (accept aarch64-only GPU CI), **B (make yoga the x86_64 GPU
runner)**, C (keep the leg on lambda-labs — rejected). Noah's own comment on that issue:

> **Decision made: cuda-nightly is gx10-only. Option A.** Implemented in paiml/aprender#2740 —
> the `ada-4090` leg is removed rather than relabelled…

And the recorded cost of Option B, which the draft does not cite anywhere:

> yoga is off most of the time, so the lane becomes intermittent — and this fleet's dead-man's
> switch exists precisely because **absent runs look like healthy ones**. Would need
> `FORJAR_UNREACHABLE_DAYS`-style handling so "yoga was off" is distinguishable from "the lane
> died".

Corroborated in the tree: `.github/workflows/cuda-nightly.yml:52` —
`description: "Which silicon to run (all | blackwell) — ada retired, see infra#359"`, and
`:100` — `runs-on: [self-hosted, gpu, Linux, ARM64, cuda, blackwell]`, one literal leg.

The draft asks for something **strictly stronger** than what was rejected: not a nightly lane
on yoga, but a *required PR-time check* (RD-3). It cites neither the issue nor the cost.

**Smallest fix.** `§7` cites infra#359, states that it is proposing Option B, and answers the
intermittency cost with the dead-man's switch. RD-3's promotion criterion becomes
`0 false reds in 14 days` **and** `yoga reachable ≥ N of the last M days, basis=` the run
list — otherwise "no red in 14 days" is satisfied by a box that was off for 14 days.

---

### F4 · S1 · §7.2's security constraints are written as constraints but describe configuration that does not exist, on a PUBLIC repo `[V]`

`gh repo view paiml/aprender --json visibility` → **PUBLIC**.

`gh api orgs/paiml/actions/runner-groups`:

| group | visibility | `allows_public_repositories` | `restricted_to_workflows` | `selected_workflows` |
|---|---|---|---|---|
| `Default` | all | true | **false** | `[]` |
| `gpu-nodes` | selected | true | **false** | `[]` |
| `gpu-x86` | selected | **false** | **false** | `[]` |

`gh api orgs/paiml/actions/runners` → `yoga-gpu` **online**, labels
`self-hosted,Linux,X64,gpu,cuda,yoga,ada`; `gx10-blackwell` online; 17 `intel-clean-room*`;
**no `lambda-4090`**. `gh api repos/paiml/aprender/actions/runners` → `total_count: 0`.

So: §7.2 constraint 1 ("restricted runner group whose workflow access is *selected workflows
only*") and constraint 5 (a separate group scoped to `release-assets.yml@refs/tags/v*`) are
**not configured on any group**. And `gpu-x86`, the group whose name suggests it is where a
yoga runner belongs, has `allows_public_repositories: false` — a public repo cannot use it.

Constraint 2 is buildable today: `.github/workflows/pr-gate.yml` has an `authorize` job `[V]`.

**Consequence.** §7 landing before the group configuration means PR-authored code executes on
a machine in a house, with no workflow restriction. §7.2 calls these constraints
"non-negotiable" — as written they read as descriptions of the current state.

**Smallest fix.** Restate constraints 1/3/5 as *prerequisites with an owner and a ticket*, and
add a guard that reads the runner-group API and fails when `restricted_to_workflows != true`
for the group holding `yoga-gpu`. A constraint no one can check is a comment (§8's own words).

---

### F11 · S2 · §3.1 silently changes both the runner class and the feature set `[V]`

`.github/workflows/nightly.yml` today: 5 targets on **GitHub-hosted** runners
(`ubuntu-latest`, `ubuntu-24.04-arm`, `macos-latest` ×2, `windows-latest`), built with
`cargo build --release -p apr-cli` — **default features**, and `grep -c cuda
.github/workflows/nightly.yml` = **0**. There is no `-cuda` artifact anywhere in the fleet
today.

`§3.1` moves all four builds to self-hosted machines and adds `--features cli,cuda` to two of
them. PP-066 **R-5's own text** says the assets are "built on the nightly.yml runners", and
**C13 says five targets**. The draft's RD-1 spots the 4-vs-5 conflict and says "Do not leave
two numbers in two documents" — while being the second document.

**Smallest fix.** `§3.1` gains a sentence naming the runner-class change and its consequence
(hosted minutes → house hardware; macOS/Windows lose their builder unless RD-1 keeps them),
and RD-1 is resolved *in the same PR* that amends C13.

---

### F5 · S2 · §8 mints a second contract for an invariant that already has one `[V]`

`§8` names `contracts/apr-gpu-cpu-parity-v1.yaml` as the contract behind M3/C14. Already in
the tree:

```console
$ pv validate contracts/apr-cpu-vs-gpu-output-parity-v1.yaml
0 error(s), 0 warning(s)
Contract is valid.
```

with **10 proof obligations and 11 falsification tests** (`FALSIFY-CPU-GPU-001..011`), and it
already documents the bypass the draft's §10.3 forbids — `SKIP_PARITY_GATE=1`, implemented at
`crates/aprender-serve/src/gguf/cuda/mod.rs:268-279`, plus `bypass: SKIP_PARITY_GATE=1` as a
declared field in `contracts/layer-parity-v1.yaml:121`. `contracts/apr-gpu-parity-consistency-v1.yaml`
and `contracts/apr-cli-publish-v1.yaml` also exist.

**Consequence.** Two contracts for one invariant splits the ratchet: a falsifier added to one
does not constrain the other, and `pv coverage` counts an obligation twice.

**Smallest fix.** M3/C14 extend `apr-cpu-vs-gpu-output-parity-v1.yaml`; the publish-cascade
contract extends `apr-cli-publish-v1.yaml`. Only `apr-release-assets-v1.yaml` and
`apr-dogfood-models-v1.yaml` are genuinely new. §10.3's refusal is *correct and buildable* —
and should say that the bypass exists in shipped code, so the guard greps the env **and**
asserts the contract's declared bypass was not exercised.

---

### F6 · S2 · §5.1 and Appendix B put a release-gating skill outside the repository `[V]`

Appendix B: `~/.claude/skills/apr-dogfood-models/SKILL.md` — user scope.

This repository already paid for that. `.claude/skills/apr-dogfood/SKILL.md` (v3.0, **repo**
scope) opens with:

> Without this, the skill takes its name from the directory, and `dogfood` collides with a
> personal user-scope skill at `~/.claude/skills/dogfood/`. On any machine where both exist
> the USER one wins and this file NEVER APPEARS in the session's skill listing — it cannot be
> invoked and nothing warns. **Edits look effective and change nothing that runs.**

Both `~/.claude/skills/dogfood/` and `.claude/skills/dogfood/` exist on this machine today
`[V]`. The draft cites #2332 for the explicit `name:` and then places the file in the
shadowing location.

**Consequence.** A gate outside the tree cannot be reviewed in the PR that changes it, cannot
be versioned with the artifact it certifies, and cannot be read by
`check_release_receipts.sh`. It also fails §1.3 — a release decided by a file that exists only
on the releaser's laptop makes the producer the gate.

**Smallest fix.** `.claude/skills/apr-dogfood-models/SKILL.md`, explicit `name:`, sibling of
`apr-dogfood` v3.0 exactly as §5.1 intends.

---

### F7 · S2 · §3.4's PTX floor is derived from the wrong population `[V]`

`§3.4`: "The `.version` must be **≤ the minimum ISA the fleet's oldest driver accepts**,
`basis=nvidia-smi` on `lambda-labs` and `gx10`." RD-7 makes that re-derived each release.

Measured drivers: `yoga` 595.91.07, `lambda-labs` **570.207**, `gx10` 590.48.01.

Two problems. **The population is wrong**: the artifact ships to users, and a floor derived
from three machines in one house makes a published binary's compatibility a function of who
owns hardware — buy a newer GPU, and the release silently stops supporting older drivers.
**The artifact is wrong**: per F2 the emitted `.version` is a *source constant* in
`PtxModule`, already validated against a declared minimum in
`crates/aprender-gpu/src/ptx/mod.rs:43`. Re-deriving it from `nvidia-smi` each release makes a
source constant float on hardware inventory.

**Smallest fix.** RD-7 becomes: the floor is the project's **declared** minimum compute
capability and PTX ISA, asserted against the emitter (F2's unit test). The fleet's drivers are
a *cross-check that the declared floor is reachable*, never its source.

---

### F8 · S2 · §9's jidoka target and §5.5/RD-5 contradict each other `[V]`

`§9`: "GPU/CPU divergence found by a **user**: **0 per release** (baseline: 1 — #2971)".
`§5.5` UNSERVICEABLE + RD-5: if #2971 is not root-caused by the tag, ship with the 1.5B row
reading `UNSERVICEABLE(cuda, #2971)`.

`gh issue view 2971` → **OPEN**, labels `bug,P0,pp-066` `[V]`.

Shipping a release whose declared sentinel model diverges on GPU guarantees the baseline is
not improved — it just moves who writes it down. The target and the escape hatch cannot both
be true as worded.

**Smallest fix.** Pick one, in the document: either the metric counts *undeclared* divergences
only (and says so, which is defensible — an honestly-refused model is not a user discovery),
or #2971 blocks the tag. `§9`'s instrument ("issues labelled P0 filed by non-maintainers")
already implies the former; make it explicit.

---

### F9 · S3 · cells are assigned to hosts that do not have the model, and M6's instrument does not exist on gx10 `[V]`

`ssh <host> 'find ~/models …'` and `nvidia-smi --query-gpu=memory.total,memory.used,memory.free`:

| host | 7B Q4_K_M present | VRAM total | used | free |
|---|---|---|---|---|
| `lambda-labs` | **yes** — `~/models/qwen2.5-coder-7b-instruct-q4_k_m.gguf`, 4 683 073 536 B | 24564 MiB | 756 | 23291 |
| `gx10` | **yes** — same name, same size | `[N/A]` | `[N/A]` | `[N/A]` |
| `yoga` | **no `~/models` directory at all** | 8188 MiB | 2 | **7807** |

So `§5.3`'s four cells are provisioned; `§7.1`'s PR-time 7B witness on yoga is **not**. On
capacity yoga is fine — 7807 MiB free against 4.36 GiB of weights leaves ~3.3 GiB for KV and
activations, and the dGPU is not driving a display (2 MiB used) — so the objection is
provisioning and wall-clock, not VRAM.

`gx10` reports **no VRAM number at all** (unified memory). `§5.4` M6 requires "peak RSS/VRAM
recorded with `basis=`". On gx10 that number does not exist, so M6 must record an explicit
`UNMEASURED(vram, unified-memory)` rather than a blank — a blank in a receipt is the
`verified_hardware: UNKNOWN` row §9 sets to zero.

**Smallest fix.** Extend Appendix A **S0-M1** to cover *presence* as well as fit, and to
include `yoga` since §7.1 assigns it a cell. Give M6 a per-host instrument and an explicit
UNMEASURED verdict for unified memory.

---

### F10 · S3 · RD-3's wall-clock `[U]` uses a single number where the distribution is what binds `[V]`

`§7.1`: "Budget: the check must complete inside the merge-queue ceiling (`[U]` — measure and
record `basis=`; current workspace-test observed ≈ 34 min)."

Measured, 25 completed `ci.yml` runs on pull requests
(`gh run list --workflow ci.yml --limit 40 --json …`), minutes:

```
3  5  13  17  17  18  32  34  35  38  40  41  47  53  53  54  57  66  71  98
```

34 min is roughly the median of the non-trivial runs; **p90 ≈ 70 min, max 98 min**. Nearly
every run is `cancelled` rather than completed — consistent with a queue that evicts on a
newer push.

**Smallest fix.** RD-3's `basis=` is the distribution and the run list, not one number, and
the budget is stated against the tail: adding a required check to a pipeline whose existing
required check already spans 34–98 min is a throughput decision, not a rounding one.

---

## Already solved on `origin/main`, uncited by the draft

1. **The skill-shadowing problem the draft cites (#2332)** — solved by the explicit `name:`
   field in the **repo-scope** `.claude/skills/apr-dogfood/SKILL.md` v3.0. The draft repeats
   the defect by choosing user scope (F6).
2. **The CPU/GPU parity contract** — `contracts/apr-cpu-vs-gpu-output-parity-v1.yaml`, 11
   falsifiers, `pv validate` clean (F5).
3. **The PR authorization gate §7.2 constraint 2 needs** — `.github/workflows/pr-gate.yml`,
   job `authorize`.
4. **The x86 GPU runner question** — adjudicated in `paiml/infra#359` (F3).
5. **PTX floor validation** — `crates/aprender-gpu/src/ptx/mod.rs:43-67` already declares a
   minimum and validates it (F2, F7).

## Highest-value item that can land this week, using only what exists

**Run the physical model dogfood by hand and publish the receipts.** Both dogfood hosts have
the 7B and the 1.5B; both GPUs execute; `apr qa`, `apr devices`, `apr trace` and
`contracts/apr-cpu-vs-gpu-output-parity-v1.yaml` all exist. That produces §5's cell receipts
for a real tag without `install.sh`, without `release-assets.yml`, and without the promotion
job — and it is the artifact that would have caught #2971. Everything else in §5 is
automation *around* that measurement.

Second: **F2's emitter unit test**. It is small, it has a real registered mutation, and it
replaces a probe that cannot fail.

---

## Appendix A — the S0 ledger, executed

The draft asks that each row "be executed and its output pasted into the review thread". Done.
Every row is `[V]`. This closes **RD-9** and the four `[U]` rows in `§2`.

Commands: `ssh <host> 'uname -m; nproc; free -g; df -h /'`,
`nvidia-smi --query-gpu=name,memory.total,memory.used,memory.free,driver_version,compute_cap --format=csv,noheader`,
`nvcc --version`, `lsmod | grep -E '^(nvidia|nouveau)'`, `command -v cuobjdump`.

| id | host | measured |
|---|---|---|
| **S0-Y1** | `yoga` | x86_64 · **22 cores** · 30 GB RAM · 859 G free on `/` |
| **S0-Y2** | `yoga` | **RTX 4060 Laptop GPU** · 8188 MiB · driver **595.91.07** · compute cap **8.9 (sm_89)** · CUDA 13.2 (smi) · nvcc 12.4 · `libcuda.so.1` present |
| — | `yoga` | `nvidia`, `nvidia_uvm`, `nvidia_drm`, `nvidia_modeset` loaded; **no `nouveau`**; `nvidia-smi -L` names the AD107M. **yoga can execute kernels** — this reverses the 2026-09-08 morning probe that found nouveau bound and `nvidia-smi` failing. Any note claiming yoga is build-only is stale. |
| — | `yoga` | free VRAM **7807 MiB** of 8188 (2 MiB used — the dGPU is not driving a display) |
| — | `yoga` | **no `~/models` directory** — no parity model is provisioned (F9) |
| **S0-G2** | `lambda-labs` | x86_64 · 48 cores · 125 GB · 376 G free · **RTX 4090** 24564 MiB · driver **570.207** · cap **8.9** · CUDA 12.8 · nvcc 12.8 · 23291 MiB free |
| **S0-G2** | `gx10` | aarch64 · 20 cores · 119 GB · 324 G free · **GB10** VRAM `[N/A]` unified · driver **590.48.01** · cap **12.1 (sm_121)** · CUDA 13.1 · nvcc 13.0 |
| **S0-M1** | both dogfood hosts | `qwen2.5-coder-7b-instruct-q4_k_m.gguf` **4 683 073 536 B** present on `lambda-labs` **and** `gx10`. Fits lambda (23291 MiB free) with ~18.6 GiB spare; gx10 unified, no VRAM figure exists |
| **S0-R1** | org | groups `Default` / `gpu-nodes` / `gpu-x86`; **all three** `restricted_to_workflows: false`, `selected_workflows: []`; `gpu-x86` `allows_public_repositories: false`. Runners: `yoga-gpu` (online, `self-hosted,Linux,X64,gpu,cuda,yoga,ada`), `gx10-blackwell` (online), 17 × `intel-clean-room*`. **No `lambda-4090`.** Repo-scoped runners: `total_count: 0` |
| **S0-R2** | repo | required contexts `["ci / gate","workspace-test"]`; rulesets *Green Main*, *workspace-test*, *Merge Queue (main)* all `active` |
| **S0-N1** | `nightly.yml` | 5 targets, **GitHub-hosted** runners, `cargo build --release -p apr-cli`, **default features**, zero occurrences of `cuda` |
| **S0-Y3** | `yoga` | `[U]` — not run here; needs a `cargo build -p apr-cli --release --features cuda` on yoga plus `ldd \| grep -c cudart` (owner: Noah). Note `crates/aprender-gpu` advertises "no LLVM, no nvcc", so the expected count of 0 is a property of the source, not of the box |
| **S0-Y4/Y5** | `yoga` | `[U]` — CUDA unit-test execution and its wall-clock budget, unmeasured (owner: Noah). Blocked behind F3's decision, not behind the box |
| **S0-G1** | `gx10` | `[U]` — per-artifact build wall clock, unmeasured (owner: Noah). Feeds RD-4 |

**§2's `[U]` rows now close as:** `yoga` = x86_64, RTX 4060 Laptop, 8 GB, sm_89, driver
595.91.07. Note this makes the fleet **sm_89 ×2 (yoga, lambda) and sm_121 ×1 (gx10)** — the
`x86-gpu-cuda` artifact's two verification hosts are the *same* compute capability, so §4's
two-host row for it is redundancy, not cross-architecture coverage. RD-6 states the aarch64
single-host limitation; the x86 double-sm_89 coverage should be stated beside it.

---

## RD-1 … RD-9 — escalated, not decided

None of these were decided in this review. Each is a design decision reserved to its owner.
Recommendations carry the finding that motivates them.

| id | recommendation from this review | motivated by |
|---|---|---|
| **RD-1** | Decide before §3.1 can be adopted; whatever is chosen, amend PP-066 C13 **in the same PR**. Note the runner-class change is a separate consequence that survives every option (F11) | F11, F1 |
| **RD-2** | No objection — FX-18 is a fixture with both polarities, admissible as written | — |
| **RD-3** | Advisory→required is sound, but the criterion must add a reachability term, else "0 false reds in 14 days" is satisfied by a box that was off; and the budget's `basis=` is the 3–98 min distribution, not "≈34 min" | F3, F10 |
| **RD-4** | Unchanged; add that `check_host_slot.sh` does not exist yet (F1) | F1 |
| **RD-5** | Cannot be decided while §9 says the opposite — resolve F8 first | F8 |
| **RD-6** | Accept, and state the *other* asymmetry beside it: the two x86 GPU hosts are both sm_89 | S0 ledger |
| **RD-7** | Reframe: the floor is a declared source constant asserted against the emitter; the fleet is a cross-check, not the source | F7, F2 |
| **RD-8** | No objection — same-machine determinism now, cross-machine out of scope | — |
| **RD-9** | **Closed by Appendix A above** for `yoga`'s identity; S0-Y3/Y4/Y5 and S0-G1 remain `[U]` with Noah named | S0 ledger |

---

## Method, and what was re-run

One `agy /teamwork-preview` lane (agy 1.1.27, `--sandbox`, `writes=false`, conversation
`55b4205c-e3d0-4f6f-899a-c01d4b535d27`, 194 s, exit 0, schema-valid) returned
`do-not-implement-as-written` with 10 findings. `repo_root` was byte-identical before and
after; no lane writes leaked.

Every finding above was then re-executed here. Two lane findings were **overturned**:

1. The lane said §3.4's gate "will fail due to `command not found`" because `cuobjdump` is
   absent on the clean room. `cuobjdump` is present on **all four** hosts. The real defect is
   vacuity plus a 255 exit on a correct artifact (F2). The lane's proposed fix — grep for PTX
   text — carries no RED-turning mutation and would not satisfy §1.1 either.
2. The lane implied the 7B may be unprovisioned. It is present on **both** dogfood hosts; the
   gap is `yoga`, which §7.1 assigns a 7B cell and which has no models directory (F9).

Three items the lane left uncovered are answered here: the merge-queue budget (F10), the
`nightly.yml` runner class and feature set (F11), and the §12 adoption-plan dependency on
R-5/R-6 (F1).

Findings the lane raised that survive re-execution unchanged: F1, F3, F4, F6, F8.
Findings originated here: F5, F7 (reframed), F10, F11, and the S0 ledger.
