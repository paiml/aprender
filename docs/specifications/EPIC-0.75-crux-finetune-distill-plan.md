# EPIC 0.75.0 "CRUX declarative fine-tune & distill": plan (paiml/aprender#4002)

**Status:** plan for operator review. Nothing is applied: no child issues, milestone moves, or closes.
**Ticket:** PMAT-4002 · **kind:** docs · **Ratchet:** the 6th slice of DEBT-RATCHET-001 (#4003 §3.E, operator: "ALL releases in .7 have some rachet")

The operator's words, quoted on #4002: *".75 is crux decleartive fine-tuning/distill for qwen 3.5 with 3-5 crux
competitors"*.

## 1. Baselines (measured 2026-09-23, `origin/main` @ `49fe19c28`)

| Fact | Measured |
|---|---|
| Declarative entry points today | **three separate, unrelated config paths**: `apr distill --config <yaml> --stage precompute\|train` (`crates/apr-cli/src/commands/distill.rs:507`, ALB-011); `apr train plan/apply --config` (`crates/apr-cli/src/train_commands.rs:33,83`); `finetune`'s `config_path: Option<&Path>` (`crates/apr-cli/src/commands/finetune.rs:86`). **No single recipe schema** |
| Recipe as a `pv` contract | **none**. `contracts/` holds 20+ finetune/distill/LoRA contracts about behaviour (`apr-finetune-metrics-v1`, `distill-per-position-kd-v1`, `apr-qlora-composed-forward-equivalence-beat-v1`, …), but none defines a recipe schema |
| Competitor engines provisioned on the fleet | **HF TRL + PEFT on gx10 only**: `machines/gx10/forjar.yaml:292` pip-installs `transformers peft bitsandbytes datasets accelerate trl` **unpinned** (no versions). No forjar declaration for Unsloth, Axolotl, torchtune, LLaMA-Factory or MLX-LM on any host |
| Competitor comparisons in-tree | one beat: `crates/aprender-train/tests/beat_unsloth_coldstart_speed.rs` (**cold start only**, not quality), run by `beat-speed-nightly.yml`. **It records a concession:** "apr CONCEDES in-loop QLoRA fine-tune THROUGHPUT on GPU (Unsloth's Triton-fused kernels + bitsandbytes win the per-step decode/backward race — see docs/BEATS.md Pillar-3 CONCEDED)" (`:25-27`). The book has `ch27-switch-from-unsloth.md` |
| #3700 (multi-label classify fine-tune) | OPEN, in `backlog` |

## 2. Exit bar, made measurable

**Unit: the CRUX cell** = (Qwen3.5 size × method × backend × engine). Methods: LoRA, QLoRA, teacher→student distill.
The sizes are those 0.71 certifies. Each cell produces:
- **quality**: a held-out eval score on the same data, token budget and seed;
- **cost**: wall-clock, peak memory, tokens/s.

apr passes a cell when:
- its quality ≥ the best competitor's − the band, where the band comes from each engine's own seed-to-seed spread
  (3 seeds), never chosen by hand;
- its cost is ≤ the best competitor's (wall-clock and peak memory).

**Known RED at baseline (quorum fix):** the cost bar contradicts the recorded concession above. The GPU QLoRA
tokens/s cell against Unsloth is **expected RED today**. Under the no-defer doctrine it stays RED until closed; it is
never dropped from the matrix. Whether 0.75 must close it or report it RED is **operator decision O-1** (below).

**The positive control:** a known-good published recipe must reproduce its published score in **each** competitor.
A competitor that fails its own control is RED **as a harness** and cannot be the bar.

Every receipt carries each engine's version and sha, the recipe hash and the data hash.

## 3. Rows

| Row | Item | done_when | Baseline | First-green proof |
|---|---|---|---|---|
| **R-1** | Recipe schema as a `pv` contract (base, data, method, teacher→student, eval, seed, and the receipt fields: every engine's version and sha, the recipe hash, the data hash); `pv validate` runs before any run | `pv validate contracts/apr-recipe-v1.yaml` green; `apr finetune --recipe r.yaml` refuses an invalid recipe before loading a model | no schema | a recipe with a missing `eval` block is refused (RED) before any GPU allocation, **and the error text names the recipe field**. A clap parse error does not count (quorum fix: otherwise the proof passes on a flag rejection); a valid one passes |
| **R-2** | `apr finetune` / `apr distill` driven **only** by the recipe; flags become overrides that are recorded in the receipt | the three config paths collapse into one; a run's receipt reproduces the run: same recipe + seed + binary sha → same eval score within the band | 3 separate config paths | two runs from one receipt agree within the band; a changed seed changes the score (the positive control for determinism) |
| **R-3** | Competitor harness legs, **pinned** (versions in forjar, like llama.cpp in infra#911) | each chosen engine runs the same recipe via a translator, and its positive control reproduces its published score | TRL/PEFT on gx10, **unpinned**; the others absent | per engine: the control passes; an engine given a planted-wrong data hash is RED |
| **R-4** | The CRUX cells for LoRA/QLoRA/distill on certified Qwen3.5 sizes | every cell green under §2's rule | none | per cell |
| **R-5** | #3700 multi-label classify fine-tune | #3700's own acceptance | OPEN (backlog) | #3700's case |
| **R-6** | **Ratchet slice 6** (hold + continued paydown) | #4003 §3.E: A ≥ 9,473 bp, `P_cuda` ≥ `B_cuda + 4·s_cuda`, B-1 ≥ 499, B-2 0, B-3 ≥ 3,129 `[U]` (pending #4003 decision 3), C 27, D 0/0/0 | see #4003 | see #4003 |

## 4. Open question for the quorum to DECIDE: the 3–5 competitor engines

The operator: "quorum decide". Candidates from #4002: HF TRL/PEFT, Unsloth, Axolotl, torchtune, LLaMA-Factory,
MLX-LM. Selection criteria the lanes apply, **each checkable**:
1. **Qwen3.5 support** at a pinnable release, for LoRA **and** QLoRA, and for distillation (native, or a documented recipe);
2. **runs on a fleet host**: CUDA sm_89 (lambda), sm_121 (gx10, ARM64, where Unsloth/bitsandbytes support is historically weakest), or Apple Silicon (MLX-LM);
3. **independent implementation**: a wrapper around another candidate adds no information (Axolotl and LLaMA-Factory both build on TRL/PEFT);
4. **a published reference score** exists for the positive control.

**The plan's proposal, for the lanes to confirm or overturn: four engines.**

| Engine | Why | Hosts |
|---|---|---|
| **HF TRL + PEFT** | the reference implementation; already on gx10 | lambda, gx10 |
| **Unsloth** | the speed/memory leader the book already positions against (`ch27`) | lambda (sm_89); gx10 support to be verified by the lanes |
| **torchtune** | an independent PyTorch-native implementation (not TRL-based) with first-class distillation recipes | lambda, gx10 |
| **MLX-LM** | the only candidate for Apple Silicon, and 0.71 adds a Mac ladder host | Mac |

**Proposed out:** Axolotl and LLaMA-Factory. Both are configuration layers over TRL/PEFT, so they add little
independent signal (criterion 3). Either could replace torchtune if the lanes find torchtune's Qwen3.5 support
missing at a pinnable release.

Other open questions:
- **Q2. Which eval?** Recommendation: a held-out split of the fine-tune data plus one public benchmark subset per
  method, fixed in the recipe. The eval is part of the recipe hash.
- **Q3. Distill teacher/student pair.** Recommendation: the largest and smallest Qwen3.5 sizes 0.71 certifies on the
  same host (for example 9B → 0.8B), so every engine runs both.

## 5. Commands

```bash
grep -n 'config' crates/apr-cli/src/commands/distill.rs | sed -n 1,5p
grep -nE 'config: Option<PathBuf>' crates/apr-cli/src/train_commands.rs
ls contracts | grep -iE 'finetune|distill|lora|recipe'
git -C "$HOME/src/infra" show origin/main:machines/gx10/forjar.yaml | grep -n 'pip install'   # paiml/infra repo, not this tree
git ls-files | grep -iE 'beat_unsloth'
gh issue view 3700 -R paiml/aprender --json state,milestone
```


## Quorum record: decision quorum, 2026-09-23 (aprender-cb)

**Lanes (ADVISORY: single family, all gemini):** gemini-3.1-pro-high, gemini-3.8-flash-high, gemini-3.7-flash-high,
all returning PASS-with-changes. gpt-oss returned 429. 3/3 exited 3 on foreign ref motion. **Lane 2 wrote `Cargo.lock`
in its own review clone** (a `writes=false` lane ran a cargo command despite the brief). The clone was KEPT as evidence
and the shared checkout is untouched. Conversations: `21331daf`, `e71b87e0`, `c28f90ba`.

| Q | Decision (tally) | Applied as |
|---|---|---|
| **Q1: the competitor engines (the operator asked the quorum to decide this)** | **DECIDED 3/3: four engines, HF TRL + PEFT, Unsloth, torchtune, MLX-LM.** Axolotl and LLaMA-Factory are excluded on criterion 3 (wrappers over TRL/PEFT). Lanes 2 and 3 scored each engine against criteria 1–4, all PASS for the four chosen | R-3 builds pinned harness legs for exactly these four |
| Q2 | **a held-out split of the fine-tune data plus one public benchmark subset per method, fixed in the recipe hash**, 3/3 | §2 quality |
| Q3 | **the largest → smallest Qwen3.5 size 0.71 certifies, on one host** (lanes 2 and 3 name 9B → 0.8B), 3/3 | the distill cells |

**Must-fix items applied:**
- the Unsloth concession is made explicit, with its known-RED cell;
- R-1's vacuity (clap) closed, and its receipt fields named;
- R-6 carries the full §3.E floors;
- the infra path is fixed.

**Must-fix items carried to step 2 as child-issue acceptance:**
- exact `done_when` commands;
- negative controls for R-4 and R-5;
- engine versions and SHAs pinned in forjar (R-3's first step);
- a rollback checklist for the release manager.

**Operator decision O-1:** must 0.75 close the conceded GPU QLoRA throughput gap against Unsloth, or ship with that
cell reported RED?

