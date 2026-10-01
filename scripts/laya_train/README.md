# scripts/laya_train — the Laya back office (phase 8, D-01 / D-02)

A pinned, hash-locked `uv` project that runs **Laya's own code** for everything the Rust side
(`aprender-core` ModernBERT, `aprender-decide`) is measured against. Rust owns inference; this
project owns training and the reference numbers. There is no `apr` subcommand for any of it
(D-02: `contracts/apr-cli-commands-v1.yaml` is untouched) — it is driven through `just`.

## Recipes

| Recipe | What it does |
|--------|--------------|
| `just laya-fixtures` | `metrics.py --selftest`, then `fixtures.py`: regenerates the two tiny synthetic CI fixtures. Prints each file's sha256 and size and ends with `FIXTURES OK`. A re-run is byte-identical. |
| `just laya-prepare-stance [cell]` | The demo data. `cell` = **`s64`** (default, the contract's `demo_s64`, D-19 as amended by A2): `data/decide/tweet-stance-64/` with the 192 `s64-seed13` shots, `eval.jsonl` built **by rule** (`eval_set.demo_rule`, asserted 459 rows, class counts [111, 291, 57]) and `shift.jsonl` (the 280 TweetEval stance_abortion test rows, the shift probe). `s16`: `data/decide/tweet-stance-16/`, the 1.2.0 demo data (48 shots, the 280 test rows as `eval.jsonl`), byte-identical to every earlier run. Every shot is verified against the manifest's `exact_hash`; an existing dir is never overwritten with different bytes (a re-run is a no-op). Needs `apr data tweet-eval-stance --output data/tweet-eval-stance` first. Output is under the root-anchored, gitignored `/data/`. |
| `just laya-train <data> <out> [args]` | Fine-tune, calibrate and gate (below). Exit **0** = GATE PASS, **3** = GATE FAIL, **2** = input refused. Production trains the **three gate seeds** 13 / 17 / 23 and ships the median-ECE seed (`--seeds` defaults to 3 and any other value is refused), writes the float64 noise record, and scores an optional `shift.jsonl` as a reported probe. Args: `--epochs E` (only above 16 shots/class, in [4, 12]), `--stopping early_stopping\|fixed_epochs` (default: the contract's `early_stopping`), `--device mps\|cuda\|cpu`. |
| `just laya-train-lifecycle` | A real train -> F16 save -> complete dir -> reload -> calibrate -> gate on the committed tiny checkpoint, on CPU in seconds: both stopping rules on the legacy single seed, the seed refusals, a three-seed median run, the noise record on every run, and the shift probe's gate invariance. Prints `LIFECYCLE OK`. With `LAYA_LIFECYCLE_KEEP=<dir>` it copies the three-seed shift run to `<dir>/run` and its data dir to `<dir>/data` (plan 08-15's Rust reader test). |
| `just laya-train-selftest` | `metrics.py`, `data.py` and `gate.py --selftest` (numpy + pyyaml only), then the lifecycle. Prints `METRICS SELFTEST OK`, `DATA SELFTEST OK`, `GATE SELFTEST OK`, `LIFECYCLE OK`, `LAYA TRAIN SELFTEST OK`. The `test_harness` of FALSIFY-LAYA-GATE-003..007, 009, 011 and 013 and of FALSIFY-LAYA-PARITY-006 (the Python halves). |
| `just laya-verify-suite [model]` | The whole Python-parity / real-weights surface in one **local-only** command (see "What CI does and does not run"): the three torch-free self-tests, then the four env-gated `aprender-decide` targets ARMED — `laya_parity` (`LAYA_MODEL_DIR`), `fail_closed_vectors` (+ `LAYA_FAIL_CLOSED_VECTORS=1`), `demo_run` (+ `LAYA_DEMO_RUN=1`) — then the lifecycle kept to a temp dir (`LAYA_LIFECYCLE_KEEP`) and `python_records` against it (`LAYA_PY_RUN_DIR` + `LAYA_PY_DATA_DIR`). `model` defaults to `$LAYA_MODEL_DIR`, else the pinned snapshot `~/.cache/huggingface/hub/models--convaiinnovations--laya/snapshots/55cf4c4ebb4ebe31b2550e8bdf3bd21b99753851`; a missing snapshot is refused (exit 2). An armed leg that prints `SKIP:` fails the recipe, because there it measured nothing. Needs the gitignored run dirs (`models/decide/tweet-stance-16*`, `models/decide/laya-stance-64`) and data dirs. Prints `LAYA VERIFY SUITE OK`. |

All recipes use `uv run --project scripts/laya_train --frozen`, so the committed `uv.lock` is what
runs — never a fresh resolution. Every threshold, recipe value, seed, stopping rule and the base pin
is read at run time from `contracts/laya-finetune-gate-v1.yaml` through `contract.py`; no script holds
one as a literal (D-04, D-07). The noise multiplier, the re-score floor and the ceiling are read the same
way from `contracts/laya-parity-v1.yaml`.

## Pins

Every package was verified by a human (plan 08-02 Task 1, package-legitimacy checkpoint, approved
2026-09-25) **before** the first `uv lock`. Exact `==` pins, one why-comment each in
`pyproject.toml`; transitive dependencies are hash-pinned in `uv.lock`.

| Package | Pin |
|---------|-----|
| Python | 3.13.7 (`.python-version`; `requires-python = ">=3.13,<3.14"`) |
| torch | 2.14.0 |
| transformers | 5.17.0 |
| tokenizers | 0.23.1 (matches the Rust workspace pin) |
| safetensors | 0.8.0 |
| huggingface-hub | 1.32.0 |
| numpy | 2.5.3 |
| pyyaml | 6.0.3 |
| laya | `git+https://github.com/NandhaKishorM/laya@4066d5d5fbf08b66c6757ddeedbd797bd7655bc0` (not on PyPI) |
| base weights (plan 08-08) | `convaiinnovations/laya@55cf4c4ebb4ebe31b2550e8bdf3bd21b99753851` |

The lock is resolved for **macOS arm64 only** (`[tool.uv] environments`): this is the laptop back
office, and a Linux/x86 split would be a resolution nobody verified.

To change a pin: edit `pyproject.toml`, get the new package verified the same way, `uv lock`, and
re-run `just laya-fixtures` — every committed fixture is a numerical artifact of these versions.

## Files

| File | Purpose |
|------|---------|
| `metrics.py` | numpy-only (no torch import): `macro_f1`, `f_avg`, `ece_top_label` (the HOUSE top-label ECE — floor binning, `aprender::calibration::expected_calibration_error_top_label`), `nll`, and `--selftest` (hand cases plus a replay of every frozen case in `scripts/setfit_fixtures/claims_stats/ece_top_label_cases.json` to 1e-6). Shared by `fixtures.py` and the 08-08 gate. |
| `common.py` | Torch-free at import: the shared hashing (`sha256_bytes`, streaming `sha256_file`, `tree_sha256`), JSON / f32 serialization (`write_json`, `f32_list`, `f32_hex_list`) and the ONE F16 checkpoint policy `save_f16` (`temperature` kept F32) used by `fixtures.py`, `train.py` and `lifecycle.py`. |
| `fixtures.py` | Writes `crates/aprender-core/tests/fixtures/modernbert_tiny/` and `crates/aprender-decide/tests/fixtures/laya_tiny/` (see below). |
| `contract.py` | The ONE reader of `laya-finetune-gate-v1` (and the probe task of `decide-apr-v1`, and the A1 noise constants of `laya-parity-v1`, `noise_policy`): thresholds, recipe, base, seed policy (`resolve_seeds`, `seed_selection_decl`), epoch rule (`resolve_epochs`), stopping rule (`resolve_stopping`), the `recipe.json` object, and the run-dir file list / report key sets decided by the contract's own conditional wording (`run_dir_files`, `expected_keys`). No torch. |
| `data.py` | `task.json` / `*.jsonl` validation, normalized-text (`nfc-trim-ws-v1`) overlap refusal (eval and shift), conflicting-duplicate refusal, the seeded, stratified, text-GROUP-disjoint calibration split returning sorted `slice_ids` + their sha256, and `in_distribution_heldout` (the `eval_set.demo_rule` as a pure function). `--selftest`. No torch. |
| `prepare_stance.py` | Writes a demo data dir, `--cell s64\|s16 --out DIR` (never commits tweet text; logs counts and sha256s only). |
| `train.py` | The CLI behind `just laya-train` (steps below). |
| `gate.py` | The bounded NLL temperature fit, the early-stopping rule (`EarlyStopper`), `evaluate_gate` (thresholds from the contract, non-finite never passes), the median rule (`rank_key`, `select_median_seed`) and `verify_report` (re-decides a report under the contract's thresholds; refuses mismatched thresholds, a `pass` its metrics contradict, and under `median_ece` a `shipped` seed that is not the median of its own `per_seed` rows). `--selftest`. No torch. |
| `lifecycle.py` | `just laya-train-lifecycle`: runs `train.py` as a subprocess on the tiny fixture and asserts every ordering, hash, schema, seed and re-score rule listed in its docstring. |

## Training (`just laya-train`, plans 08-08 and 08-14)

In order, every value from the contract:

1. **Validate** the data dir. `eval.jsonl` is required; `shift.jsonl` is optional (the shift probe). An
   eval or shift text equal to a train text after normalization is refused.
2. **Write `recipe.json` first** (`sort_keys`, compact) and print `RECIPE WRITTEN <recipe_id>`, where
   `recipe_id` = sha256 of those bytes — before any model scores anything (D-04). A three-seed run
   (every production run) carries `seed_selection` {policy, seeds, rank_scale, tie_break}.
3. **Device**: request mps -> cuda -> cpu, load Laya's own `Agent` on the pinned base with
   `expected_sha256={"model.safetensors": <sha>}`, and record the device READ BACK from the parameters
   (Laya falls back to CPU silently; CPU is flagged). D-03.
4. **Train**, per seed, with the spike-024 loop on `fit` rows built by `Agent._encode_state`. Under
   `early_stopping` (the default) the calibration slice's NLL at the bounded fitted T is the monitor
   after every epoch, the best epoch is restored and `STOP` is logged. Eval rows never reach training.
5. **Write the COMPLETE checkpoint dir** (F16 weights with an F32 `temperature`, `rl_agent_config.json`,
   `encoder/`, `tokenizer/` with `tokenizer_config.json` already in Laya's fixed form) and print
   `CHECKPOINT COMPLETE`, before anything reloads it. Three-seed runs write it to
   `seeds/seed-<s>/checkpoint/`.
6. **Reload** in fp32 on CPU (checkpoint sha256s asserted unchanged), print `SCORING START`, fit T by
   NLL in [0.5, 5.0] on the calibration slice (`clamp_hit` recorded), write T, reload again, and score
   eval -> that seed's `eval-probs.json` (`seeds/seed-<s>/eval-probs.json` for three seeds). Each seed's
   checkpoint and stopping record exist before its eval probabilities (asserted).
7. **Zero-shot**: the declared base on eval (`zero-shot-probs.json`), once.
8. **Median selection** (three seeds, A3): only after every seed's checkpoint is fixed, each seed's gate is
   evaluated against the shared zero-shot baseline, the seeds are ranked by (floor(ece_post x 10000),
   seed) and the MEDIAN ships: `MEDIAN seeds=13,17,23 rank_keys=... shipped=<s>` is logged, its checkpoint
   becomes `checkpoint/`, the other two checkpoints are deleted (`DELETED seed=<s> checkpoint`), its
   eval probabilities are copied byte for byte to `eval-probs.json`, and `probes.json` comes from a
   reload of `checkpoint/`. (A single-seed synthetic run skips this: seed 13 ships.)
9. **Noise record** (A1, laya-parity-v1 `rescore_noise_reference`): for the shipped checkpoint and the
   base, a manual fp32 forward must reproduce the Scorer's logits EXACTLY on the first min(5, n) rows
   (else `REFUSED noise-control`, exit 2, no record, no gate report); then the same model is cast to
   float64 and every eval row is re-scored at the applied T. `rescore-noise.json` stores every row's
   float64 probabilities, `max_abs` over exactly the written float32 values, `bound = max(floor, k x
   max_abs)` and the argmax agreement, with k and the floor copied from the contract. `NOISE which=...`
   is logged, and a `WARN pack will refuse` line when the bound exceeds the ceiling or an argmax
   differs — the trainer records, the Rust verifier decides.
10. **Shift probe** (A2, only with `shift.jsonl`), scored only now — after the gate, the median and the
    noise record — with a fresh reload of the shipped checkpoint and the base: `shift-probs.json`,
    `shift-zero-shot-probs.json` and gate-report `shift_probe` {gate_clause: false, ...}. `SHIFT PROBE
    ... (reported, not a gate clause)` is logged. It never touches `pass`, the per-seed rows or the exit
    code (the lifecycle proves the gate identical with and without it).
11. **Gate**: pass iff the SHIPPED seed has `ft.macro_f1 - zs.macro_f1 >= 0.05` AND `ece_post <= 0.10`
    (contract constants) -> `gate-report.json` (`rescore_noise_sha256` binds the record), `GATE PASS`
    (0) / `GATE FAIL` (3).

**Run dir** (`run_dir_layout`): `checkpoint/{model.safetensors, rl_agent_config.json, encoder/,
tokenizer/}` (exactly ONE `model.safetensors` in the whole dir), `task.json`, `recipe.json`,
`gate-report.json`, `eval-probs.json`, `zero-shot-probs.json`, `probes.json`, `rescore-noise.json`;
with three seeds also `variance-report.json` and `seeds/seed-<s>/eval-probs.json` for 13, 17 and 23;
with `shift.jsonl` also `shift-probs.json` and `shift-zero-shot-probs.json`. A run dir is written
once: a non-empty `--out` is refused.

**Seeds (D-08 as amended by A3, laya-finetune-gate-v1 1.4.0 `seed_policy`).** Production trains exactly
`production_seeds_required` = 3 seeds, `variance_seeds` 13, 17, 23, on the SAME data, calibration split
and recipe; the seed varies only the training RNG (head init, batch shuffle). The shipped seed is the
**median** by `rank_rule`: rank_key = floor(ece_post x `rank_scale` 10000) as an integer, order by
(rank_key ascending, seed ascending) — `tie_break: smaller_seed` — and ship index (N - 1) / 2 = 1. The
gate passes only if that seed passes BOTH clauses; the other two are reported in gate-report
`seeds.per_seed` (each row hash-bound to its `seeds/seed-<s>/eval-probs.json` and its
`model_safetensors_sha256`) and in `variance-report.json`, and never enter `pass`. **Honesty
(`seed_policy.honesty`): the median is selected WITH eval labels.** Median-of-3 is not best-of-3 — the
shipped ECE is the middle draw, not the minimum — but it is a selection on the gate eval set and is
recorded as one; it was chosen because spike 027 measured MPS training as not bitwise reproducible
(one replicate turned a FAIL into a PASS). The gate-report seeds label is the contract's literal `median-ECE seed of 3
seeds` (one seed ships, never a mean); `variance-report.json` keeps `mean ± sd over 3 seeds`, because it
does report the mean and sd. The synthetic-fixture variant also runs the **legacy** rule with `--seeds 1` (its default): no
`seed_selection`, label `single seed`, the declared seed 13 ships — such a run is never deploy-eligible
under 1.4.0 (`seed_policy.legacy_rule`).

**Refusals** (exit 2, message `REFUSED <rule>: ...`):

| Rule | Refused input |
|------|---------------|
| `task-type` | `type` other than `"choice"` |
| `task-too-few-criteria` / `task-duplicate-criterion` | fewer than 2 criteria / a repeated criterion name |
| `task-unknown-key` / `task-schema` / `task-missing` | a key outside `{type, instructions, criteria}`, a malformed or absent task.json |
| `train-row-label` / `eval-row-label` | a label that is not a criterion NAME (an index is refused too) |
| `train-row-unknown-key` / `*-row-schema` | a row key outside `{text, label}`, a blank or non-object line |
| `eval-missing` | no `eval.jsonl` (D-06) |
| `eval-train-overlap` / `shift-train-overlap` | an eval (or shift) text equal to a train text after NFC / trim / whitespace collapse |
| `train-conflicting-labels` | two train rows with the same normalized text and different labels |
| `train-class-too-small` | a class too small to give a calibration slice of `calibration_slice_min_per_class` and keep a fit row |
| `epochs` | `--epochs` at <= 16 shots/class (fixed to 12), or missing / outside [4, 12] above 16 |
| `stopping` / `seeds` | an unknown stopping rule / production `--seeds` other than 3; synthetic `--seeds` other than 1 or 3; a non-finite `ece_post` (no rank key) |
| `noise-control` | the manual fp32 forward does not reproduce the Scorer's logits exactly (no noise record, no gate report) |
| `base` | `--base` without `--variant synthetic-fixture` (production always uses the contract base) |
| `out-dir` | a non-empty `--out` |

## What the gate certifies (1.4.0)

laya-finetune-gate-v1 `eval_set.claim`, quoted verbatim:

> The gate now certifies margin and calibration on held-out data drawn like the tenant's shots. It does NOT certify robustness to a shifted input population.

This eval set was chosen **after** the SemEval-2016 test split failed the gate twice (the s16 record
below). Spike 027 (`.planning/spikes/027-laya-calibration-slice-and-tcap`) measured the cause as the
SemEval train-to-test shift, not the recipe: the same s64 early-stopping checkpoints at their own
slice-fitted T scored ECE 0.051 to 0.069 on 459 in-distribution held-out rows and 0.096 to 0.139 on the
test split. The contract records the change as a claims change, not a tuning step (`eval_set.history`).

For the demo (`demo_s64`) the eval set is a rule, not a file someone picked: TweetEval stance_abortion
`validation` rows then `train` rows, minus every s64-seed13 shot, every excluded train id and every
exclusion-group member, shot text overlap refused, duplicates dropped keeping the first — 459 rows,
[111, 291, 57]. `just laya-prepare-stance` rebuilds it and refuses anything else, and `data.py
--selftest` rebuilds it again from the pinned splits when they are present.

**The shift probe.** The 1.x eval set — the 280 SemEval test rows — stays visible as `shift.jsonl`. It
is scored with the shipped checkpoint and the base only after the gate is decided and reported in
`shift_probe` with `gate_clause: false`: the number a reader needs to see how the model does on a
shifted population, never a number the gate reads. The Rust verifier recomputes its metrics from the
hash-bound probability files (plan 08-15).

## The s16 demo record (history, 1.2.0)

**The D-19 demo's recorded outcome is GATE FAIL** (contract 1.2.0 `demo.outcome`). Both declared
recipes pass the margin and fail `ece_post`: `fixed_epochs` (recipe_id `d0f4e40d…`, ECE 0.377, T
clamped at 5.0) and `early_stopping` (`3d4b91da…`, ECE 0.222, T 3.14). Their gitignored run dirs
(`models/decide/tweet-stance-16-fixed-epochs/`, `models/decide/tweet-stance-16/`) are FAIL-CLOSED
TEST VECTORS: `gate.py --selftest` decides FAIL on both and, when the dirs are present, recomputes it
from their probability files; pack/verify (plan 08-09) must refuse both. Their FAIL was measured on
the SemEval test split, the 1.x eval set (`demo.vectors_scope`).

The demo is superseded by `demo_s64` (1.4.0): 64 shots per class, the in-distribution eval set above,
three seeds with the median shipped, ONE declared run (plan 08-16), outcome pending.

## The two fixtures

Both are random-init, synthetic, deterministic (seed 20260925, deterministic torch algorithms, one
thread) and at most 1 MiB per directory. The `model.safetensors` files are committed through two
exact-path negations of the global `*.safetensors` rule in the root `.gitignore`; real checkpoints
stay under the root-anchored `/models/` and are never committed.

- **`modernbert_tiny/`** — a plain transformers ModernBERT (HF names, no prefix): vocab 512,
  hidden 32, 3 layers (full, sliding, sliding), 2 heads, `local_attention` 8, RoPE theta 160000 /
  10000 in the transformers 5.17 `rope_parameters` layout, every LayerNorm weight drawn in
  [0.5, 1.5]. Saved F16, reloaded to fp32, and the ladder recorded for two id rows (24 and 21
  tokens). `oracle.json` stores each block as a flat list of the exact f32 values with its shape,
  plus a `window_mutation` record: moving the half-window by one changes the first local layer by
  6-10 % rms while the first global layer is bit-identical — the rows exercise the window.
  `initializer_range` is 0.2 (10x ModernBERT's default) so attention is not near-uniform; the
  generator refuses if the window check stops discriminating.
- **`laya_tiny/`** — a tiny Laya checkpoint in the `run_dir_layout` of
  `contracts/laya-finetune-gate-v1.yaml`, plus its `data/` dir (D-05 `task.json`, synthetic
  `train.jsonl` / `eval.jsonl`). Built with Laya's `DecisionModel` constructor (hidden 32, so the
  head gets `nhead = max(1, 32 // 64) = 1`), saved F16 (`temperature` F32), then **reloaded through
  `laya.Agent(dir, device="cpu")`** and scored through `Agent.predict`. The oracle records, per row,
  the builder inputs, ids, markers, qtype, bucket, applied temperature, the ladder, marker states,
  logits and probabilities; plus the `many` question whose markers Laya drops past `max_len` (and
  which Laya refuses). `recipe.json` is `variant: "synthetic-fixture"`, so every pack-for-serving,
  verify and deploy path refuses this run (`synthetic_not_deployable`).

Encoding (for the Rust readers):

- `laya_tiny/oracle.json` ladder blocks and `m_opts` are **`f32le_base64`**: standard padded base64
  of the little-endian f32 bytes, row-major `[n_tokens, d]` / `[k, d]`. Exact values; hex would not
  fit the 1 MiB bound. Logits and probabilities are decimal lists plus `*_f32_hex` lists.
- `*_f32_hex` / `probabilities_f32_hex`: one value per string, 8 hex digits, the **big-endian f32 bit
  pattern** (the `decide-apr-v1` probe convention).
- Probabilities are exactly the unrounded `p` of `Agent._decode_answers` (captured at its
  `answer_confidence(p, k)` call); `predict`'s 4-decimal public answer is asserted equal to
  `round(p, 4)`.
- `gate-report.json` `calibration.slice_ids_sha256` = sha256 of the compact JSON array bytes
  `json.dumps(slice_ids, separators=(",", ":"))`, e.g. `[0,2,6,7,8,11]`.
- `recipe_id` = sha256 of the exact `recipe.json` bytes; `inputs_sha256.*` are file-byte hashes.
- `gate-report.json` is an honest record that no gate ran: the declared base IS the tiny model, so
  `zero-shot-probs.json` equals `eval-probs.json`, `margin` is 0.0 and `pass` is `false`.

The generator refuses (non-zero) if a probe row exceeds `decide-apr-v1` `probe_max_row_tokens`, if
Laya's loader rewrote any checkpoint file (sha256 before and after the reload), if the oracle does
not cover truncation / option shrink / all three qtypes, or if a fixture directory exceeds 1 MiB.

## What CI does and does not run

CI does **not** install this project's torch stack. The checks CI must enforce are re-derived in
Rust instead (plan 08-09, `aprender_decide::verify`): the gate metrics recomputed from the
probability files, split disjointness from `slice_ids` and the data dir, and the synthetic-variant
refusal.

**Decided at plan 08-12's CI checkpoint (user, 2026-09-27): the Python-parity surface stays
local-only.** CI's integration step runs the two pure-Rust Phase 8 targets,
`cargo test -p aprender-decide --test ui` (the private-mint compile-fail proof) and
`cargo test -p aprender-mcp-decide --test e2e_stdio` (its tiny-fixture leg; the real-model leg SKIPs
there). It does NOT run:

- the torch-free self-tests (`metrics.py`, `data.py`, `gate.py --selftest`);
- the torch lifecycle;
- the four env-gated targets `laya_parity`, `fail_closed_vectors`, `demo_run` and `python_records`.

Converting a Python algorithm and proving the Rust port gives the same results is a recent addition to a
Rust-first project, so it is kept out of CI behind its existing env flags. In CI those four could only
print SKIP: they need the 0.84 GB base snapshot and gitignored run dirs. Run all of it with
**`just laya-verify-suite`** before any change to `aprender_decide::{pack,verify,laya}`, to
`scripts/laya_train`, or to `contracts/laya-*.yaml`. `gate.py --selftest`'s run-dir recompute needs the
gitignored demo run dirs and prints an explicit `SKIP` where they are absent; the literal-number
fail-closed cases run everywhere. The open item is D-ITEM-08-12-B in the phase's `deferred-items.md`.
