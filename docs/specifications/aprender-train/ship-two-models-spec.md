# Specification: Ship Two Models — Sovereign AI Stack Proof (INDEX)

**Document ID:** SPEC-SHIP-TWO-001
**Version:** 4.0.0 (split-into-three index)
**Status:** Live — see per-model specs below for active content.

This document was split from a single 8,468-line spec at v3.28.0 into three companion files. The split preserves all original `§N` section markers verbatim so cross-references in git history, PR descriptions, memory files, and contracts remain valid.

## Model identifiers

| Stable ID | Family name | Role | Size | HF artifact slug |
|---|---|---|---|---|
| **MODEL-1** | `aprender/qwen2.5-coder-7b-apache-q4k` | Distilled Apache-licensed Q4_K_M Qwen2.5-Coder-7B-Instruct teacher | 7B → 1.5B Q4_K_M | `paiml/qwen2.5-coder-7b-apache-q4k-v1` |
| **MODEL-2** | `aprender/albor-370m` | Sovereign Python code completion student | 370M | (not yet published) |

**Naming convention.** Family name uses the redistributor pattern established by Unsloth, Bartowski, TheBloke: `{org}/{base-name}-{license-tag}-{quant-tag}` where `{base-name}` is the upstream model for derivatives or the project codename for sovereign work. `aprender/` is the framework org prefix. Examples in the wild:

- `unsloth/Qwen2.5-Coder-7B-Instruct-bnb-4bit` — keeps full Qwen base name, adds `-bnb-4bit` quant tag.
- `bartowski/Qwen2.5-Coder-7B-Instruct-GGUF` — same base, `-GGUF` format tag.
- `TheBloke/CodeLlama-7B-Instruct-GGUF` — keeps upstream CodeLlama identity.

**Family name vs. HF artifact slug.** The family name (`aprender/...`) is the canonical spec identity. The HF artifact slug (`paiml/...-v1`) is the published handle, which uses the GitHub org (`paiml`) and a version tag (`-v1`). Both refer to the same artifact; the family name is what cross-references in this repo use.

**Sovereign work (MODEL-2).** Since `albor-370m` is original (no upstream to redistribute), the `{base-name}` slot holds the project codename `albor` from its original repo `paiml/albor`, plus the size suffix `-370m`.

MODEL-1 / MODEL-2 are stable document IDs (numeric, preserved across renames) used for cross-references in AC-SHIP1-*/AC-SHIP2-*, FALSIFY-SHIP-*, contracts, PR titles, and git history.

## Spec layout

| File | Scope | Status |
|---|---|---|
| [ship-model-1-spec.md](./ship-model-1-spec.md) | **aprender/qwen2.5-coder-7b-apache-q4k** (MODEL-1) — distilled 7B coder teacher | **🎉 100% — shipped via v0.33.0** |
| [ship-model-2-spec.md](./ship-model-2-spec.md) | **aprender/albor-370m** (MODEL-2) — sovereign 370M Python student | **79% — best val_loss 4.71 (§82)** |
| [ship-shared-methodology.md](./ship-shared-methodology.md) | Foundation (§1-§3, §6-§11, §13) + cross-cutting falsifiers (§18, §36, §41, §44, §45) | Stable |

## Repository lineage

Both models originated as standalone GitHub projects before the APR-MONO consolidation absorbed their code, contracts, and ticket systems into this monorepo. The standalone repos remain as historical references; active development is in `paiml/aprender`.

| Model | Lineage repo (standalone, dormant) | Active code repo | Artifact repo |
|---|---|---|---|
| `aprender/qwen2.5-coder-7b-apache-q4k` (MODEL-1) | [paiml/apr-leaderboard](https://github.com/paiml/apr-leaderboard) — last commit 2026-04-05 | [paiml/aprender](https://github.com/paiml/aprender) | [paiml/qwen2.5-coder-7b-apache-q4k-v1](https://huggingface.co/paiml/qwen2.5-coder-7b-apache-q4k-v1) (HuggingFace) |
| `aprender/albor-370m` (MODEL-2) | [paiml/albor](https://github.com/paiml/albor) — last commit 2026-04-05 | [paiml/aprender](https://github.com/paiml/aprender) | (not yet published — pending val_loss < 4) |

## Latest atomic next action

**v3.28.0 (2026-05-15) — §82 P2-A 5000-step training EARLY-STOP at val_loss=4.7111.** P0-trio dispatched against epoch-020 checkpoint: AC-SHIP2-009 LIVE-DISCHARGED at 325.1 tok/s; AC-SHIP2-010 blocked on P0-G + P0-H. Both P0-G (PR #1706) and P0-H (PR #1709) merged 2026-05-16 — full §82 cascade closed. **MODEL-1**: 100%. **MODEL-2**: 77% → 79%. See [MODEL-2 spec §82](./ship-model-2-spec.md) for full evidence.

## Section ownership (jump table)

### `aprender/qwen2.5-coder-7b-apache-q4k` (MODEL-1) — see [ship-model-1-spec.md](./ship-model-1-spec.md)
§4 (base), §7.1 (falsification), §12 (expedited teacher-first), §15-§17 (SHIP-007), §23 (sub-FFN), §27 (P3), §30-§32 (SHIP-007 refutations), §40 (LOCALIZED), §46-§48 (layer-0 attention), §58 (v0.32.0 release), §61 (SHIP-002/006/008), §63 (empirical floor), §67-§71 (SHIP-005 chain), §72 (5-AC cascade), §73-§74 (LM head F32 GEMV), §75 (🎉 100%), §76 (v0.33.0 published).

### `aprender/albor-370m` (MODEL-2) — see [ship-model-2-spec.md](./ship-model-2-spec.md)
§5 (base), §7.2 (falsification), §14 (Task #132 CUDA training), §19-§20 (CUDA dispatch), §22 (first real training), §24-§25 (corpus diagnosis), §26 (three-priority plan), §33-§35 (retrain + distill stub), §42-§43 (distill-train), §49 (strategy pivot), §50-§57 (§50.4 cascade + step 5g preflight), §77 (5g.1 discovered), §78 (5g.2 converged), §79 (audit + Five-Whys), §80 (prioritized backlog), §81 (P0 metadata gaps), §82 (P2-A 5000-step val_loss=4.71).

### Shared (see [ship-shared-methodology.md](./ship-shared-methodology.md))
§1-§3 (abstract, audit findings, motivation, design principles), §6 (compound ship gates), §7 (falsification tests intro), §8 (execution plan + DAG), §9 (risk matrix), §10 (failure protocol / Hansei), §11 (references), §13 (retrospective), §18 (training status snapshot), §36 (plain-language status), §41 (apr-cpu-vs-gpu-output-parity chain), §44 (FALSIFY-CPU-GPU-005 part b), §45 (5/5 LIVE DISCHARGE milestone).

## Methodology lessons (lessons #1-29 from §60-§82)

All 29 methodology lessons accumulated during the SHIP-TWO-001 work live in the per-model specs at their introducing section (e.g. lesson #29 "Class 3 defects come in waves of 4" is in MODEL-2 §82.6). The shared methodology spec collects the cross-cutting ones (#7, #11, methodology #14-#22 around SHIP-007). Individual memory files in `~/.claude/projects/-home-noah-src-aprender/memory/` mirror the load-bearing ones.

## How to amend going forward

- **MODEL-1 amendment**: append `## §N` section to [ship-model-1-spec.md](./ship-model-1-spec.md), bump that file's version banner.
- **MODEL-2 amendment**: append `## §N` section to [ship-model-2-spec.md](./ship-model-2-spec.md), bump that file's version banner.
- **Cross-cutting** (touches both models OR introduces a methodology lesson applicable to both): append to [ship-shared-methodology.md](./ship-shared-methodology.md).
- **This index file**: only bumps when one of the three children flips a ship-% bucket OR a new spec file is added.

## Historical record

The unified v3.28.0 file is available in git history at commit `b3ab72f73^` (parent of the P0-G merge). Use `git show <commit>:docs/specifications/aprender-train/ship-two-models-spec.md` to access it directly. No content was lost in the split — every line was migrated to one of the three companion files.
