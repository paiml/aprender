# Ground truth for the `release-process-aprender.md` v0.1 review

Every row was produced by the command beside it, on `origin/main` @ `c04eda87d`
(worktree `/mnt/nvme-raid0/agent-wt/rel-process`), 2026-09-08. Nothing here is
transcribed from the draft, from memory, or from another document.

## GT-1 — artifacts the draft names as the process of record

`for f in <path>; do [ -f "$f" ] && echo PRESENT || echo ABSENT; done` on the worktree:

| path | draft cites it as | measured |
|---|---|---|
| `scripts/run_clean_room.sh` | §3.3 "the one build recipe"; §6 step 2 "the hard gate, first" | **ABSENT** |
| `scripts/release_criteria.sh` | §6 step 1 | **ABSENT** |
| `scripts/publish_cascade.sh` | §6 step 12; §8 | **ABSENT** |
| `scripts/check_release_matrix.sh` | §8 | **ABSENT** |
| `scripts/check_no_cudart_link.sh` | §8 | **ABSENT** |
| `scripts/check_ptx_version.sh` | §8 | **ABSENT** |
| `scripts/check_host_slot.sh` | §2, §8 | **ABSENT** |
| `scripts/check_model_parity.sh` | §8 | **ABSENT** |
| `scripts/check_release_receipts.sh` | §8 | **ABSENT** |
| `scripts/check_publish_cascade.sh` | §8 | **ABSENT** |
| `scripts/check_readme_claims.sh` | §8 | PRESENT (533 lines) |
| `.github/workflows/release-assets.yml` | Appendix B | **ABSENT** |
| `.github/workflows/promote-release.yml` | §6 step 10, Appendix B | **ABSENT** |
| `.github/workflows/cuda-test.yml` | §7, Appendix B | **ABSENT** |
| `install.sh` | §3.2 FX-18; §5.3 "every cell installs via install.sh" | **ABSENT** |
| `evidence/models/supported.yaml` | §5.2 "the model manifest" | **ABSENT** |
| `keys/apr-release-minisign.pub` | §6 step 7 | **ABSENT** |
| `contracts/apr-release-assets-v1.yaml` | §8 | **ABSENT** |
| `contracts/apr-gpu-cpu-parity-v1.yaml` | §8 | **ABSENT** |
| `contracts/apr-publish-cascade-v1.yaml` | §8 | **ABSENT** |
| `contracts/apr-dogfood-models-v1.yaml` | §8 (marked "new") | ABSENT (correctly marked new) |

A `find . -name '*<basename>*'` over the whole tree (target/ excluded) finds none of
them under any other path. Of the **21** paths named: **1 exists**
(`scripts/check_readme_claims.sh`), **1 is correctly marked "new"**
(`contracts/apr-dogfood-models-v1.yaml`), and of the remaining **20 non-new** paths
**19 are absent**.

## GT-2 — the fleet, measured on the hosts themselves

`ssh <host> 'uname -m; nproc; free -g; df -h /; nvidia-smi --query-gpu=name,memory.total,driver_version,compute_cap --format=csv,noheader; nvcc --version'`

| host | arch | cores | RAM | free disk | GPU | VRAM | driver | compute cap | CUDA (smi) | nvcc |
|---|---|---|---|---|---|---|---|---|---|---|
| `yoga` | x86_64 | 22 | 30 GB | 859 G | RTX 4060 **Laptop** | **8188 MiB** | 595.91.07 | **8.9** | 13.2 | 12.4 |
| `lambda-labs` | x86_64 | 48 | 125 GB | 376 G | RTX 4090 | 24564 MiB | **570.207** | **8.9** | 12.8 | 12.8 |
| `gx10` | aarch64 | 20 | 119 GB | 324 G | GB10 | [N/A] unified | 590.48.01 | **12.1** | 13.1 | 13.0 |

`ssh yoga 'lsmod | grep -E "^(nvidia|nouveau)"'` → `nvidia`, `nvidia_uvm`, `nvidia_drm`,
`nvidia_modeset` loaded; **no `nouveau`**. `ssh yoga 'nvidia-smi -L'` → `GPU 0: NVIDIA
GeForce RTX 4060 Laptop GPU (UUID: GPU-a3a7c6c0-…)`; the part number **AD107M** comes from
`lspci -nn`, not from `nvidia-smi`. yoga can
execute kernels as of this measurement.

## GT-3 — runners

`gh api orgs/paiml/actions/runners`:

- `yoga-gpu` — online, **not busy**, labels `self-hosted,Linux,X64,gpu,cuda,yoga,ada`
- `gx10-blackwell` — online, labels `self-hosted,Linux,ARM64,gpu,cuda,blackwell,gb10`
- `intel-clean-room` .. `intel-clean-room-16` — 17 runners, labels `self-hosted,Linux,X64,clean-room,intel`; `-16` carries `perf-solo` instead of `clean-room`
- **no `lambda-4090`** on the org list

`gh api repos/paiml/aprender/actions/runners` → `{"total_count":0}` (no repo-scoped runners).

`gh api orgs/paiml/actions/runner-groups`: `Default` (visibility all, public repos allowed),
`gpu-nodes` (selected, public repos allowed), `gpu-x86` (selected, **`allows_public_repositories:
false`**). **Every group has `restricted_to_workflows: false` and an empty `selected_workflows`.**

`gh repo view paiml/aprender --json visibility` → **PUBLIC**.

## GT-4 — the CUDA lane and the x86 GPU decision

`.github/workflows/cuda-nightly.yml` line 52:
`description: "Which silicon to run (all | blackwell) — ada retired, see infra#359"`
line 100: `runs-on: [self-hosted, gpu, Linux, ARM64, cuda, blackwell]` — a single literal
selector, one leg, gx10 only.

`paiml/infra#359` (**OPEN**) — "x86_64 GPU CI has no sanctioned runner". Its options were
A: accept aarch64-only GPU CI; **B: make yoga the x86_64 GPU runner**; C: keep ada on
lambda-labs (rejected). Noah's comment on that issue: *"Decision made: cuda-nightly is
gx10-only. Option A."* Implemented in aprender#2740 by removing the `ada-4090` leg. The
cost of option B was named in the issue and remains unaddressed: *"yoga is off most of the
time, so the lane becomes intermittent — and this fleet's dead-man's switch exists precisely
because absent runs look like healthy ones. Would need `FORJAR_UNREACHABLE_DAYS`-style
handling so 'yoga was off' is distinguishable from 'the lane died'."*

`.github/workflows/pr-gate.yml` — an `authorize` job exists (§7.2 constraint 2 is buildable).

## GT-5 — issues the draft cites

`gh issue view <n> --repo paiml/aprender`:

| issue | state | title |
|---|---|---|
| #2696 | **CLOSED** | P0: published apr 0.64.0 SILENTLY IGNORES `--gpu` |
| #2869 | OPEN | No pre-built apr binary on stable releases |
| #2971 | OPEN (`bug, P0, pp-066, inst:A`) | GPU inference refuses Qwen2.5-1.5B-Instruct GGUF (hidden=1536/heads=12/kv_heads=2): parity gate fails at cosine 0.94 |
| #2982 | **CLOSED** | C0-5: PRQ-013 single base-owned quorum workflow |

## GT-6 — the PP-066 spec this draft extends

`docs/specifications/PP-066-release-spec.md` (826 lines) is on `origin/main`.

- **C13 says 5 targets**, and pins them to *the 5 `nightly.yml` targets*, with a signature
  over the manifest — not to a cpu/cuda split.
- **R-5 / R-6 / R-7 are `open`** (PMAT-993, PMAT-994, …), due 2026-10-09 / 2026-10-16.
  R-6 *is* the install script the draft's §5.3 requires every dogfood cell to use.
- R-5's own text says the assets are "**built on the nightly.yml runners**".

`.github/workflows/nightly.yml` builds 5 targets on **GitHub-hosted** runners
(`ubuntu-latest`, `ubuntu-24.04-arm`, `macos-latest`, `windows-latest`), `cargo build
--release -p apr-cli` with **default features on every target — no `--features cuda`
anywhere in the file**, and attaches them to the rolling `nightly` prerelease.

## GT-7 — how aprender produces PTX

`crates/aprender-gpu/Cargo.toml:7` — `description = "Pure Rust PTX generation for NVIDIA
CUDA - no LLVM, no nvcc"`. PTX is emitted by aprender's own Rust code, so the emitted
`.target`/`.version` is a property of the **source**, not of the builder's toolchain, and no
CUDA toolkit is needed to build (consistent with §3.4's first bullet).

Command: `grep -n 'description' crates/aprender-gpu/Cargo.toml | head -1`.

`cuobjdump`, which §3.4 names as the acceptance instrument, ships only with the CUDA toolkit.
Measured with `ssh <host> 'command -v cuobjdump || echo ABSENT'`:

| host | `cuobjdump` |
|---|---|
| `intel` (the clean-room host, the publish gate) | **ABSENT** |
| `yoga` | `/usr/bin/cuobjdump` |
| `lambda-labs` | `/usr/bin/cuobjdump` |
| `gx10` | `/usr/local/cuda/bin/cuobjdump` |

**Correction of record.** An earlier draft of this pack recorded `cuobjdump` as PRESENT on
`intel`. That row came from running `command -v cuobjdump` on the local workstation
(`hostname` → `noah-Lambda-Vector`) and labelling it `intel`. It was a mislabelled probe, not
a measurement of `intel`, and the AD-04 quorum caught it. The clean-room CI job additionally
runs **inside a container** (`.github/workflows/ci.yml:85`), so the binding question is what
the image contains — which is unmeasured here and is `[U]`, owner Noah.

## GT-8 — skill scope

Commands: `sed -n '1,30p' .claude/skills/apr-dogfood/SKILL.md`;
`ls -d .claude/skills/*dogfood* ~/.claude/skills/*dogfood*`.

`.claude/skills/apr-dogfood/SKILL.md` is **repo-scope** and opens with a comment block
explaining that it carries an explicit `name:` *because* a user-scope
`~/.claude/skills/dogfood/` silently shadows a directory-named repo skill (#2332/#2361),
and that "edits look effective and change nothing that runs". Both
`~/.claude/skills/dogfood/` and `.claude/skills/dogfood/` exist on this machine today.

The draft's Appendix B places the new skill at
`~/.claude/skills/apr-dogfood-models/SKILL.md` — **user scope**.
