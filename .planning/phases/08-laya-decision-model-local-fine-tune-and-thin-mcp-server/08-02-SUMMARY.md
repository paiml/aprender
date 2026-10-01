---
phase: 08-laya-decision-model-local-fine-tune-and-thin-mcp-server
plan: 02
subsystem: testing
tags: [laya, modernbert, uv, python, torch, transformers, fixtures, oracle, parity, safetensors, calibration]

requires:
  - phase: 08-laya-decision-model-local-fine-tune-and-thin-mcp-server
    provides: "08-01 contracts: laya-finetune-gate-v1 (run_dir_layout, recipe/gate/eval-probs/probes schemas, recipe_variants, ece_top_label), decide-apr-v1 (task_json_schema, probe_policy, probe_max_row_tokens 48), laya-parity-v1 (tolerances, window_mutation)"
provides:
  - "scripts/laya_train/: pinned, hash-locked uv project (human-approved pins, Laya @ 4066d5d as a git dependency, Python 3.13.7, macOS-arm64 lock)"
  - "scripts/laya_train/metrics.py: numpy-only macro_f1 / f_avg / house top-label ECE / nll + --selftest (frozen-case replay)"
  - "scripts/laya_train/fixtures.py + `just laya-fixtures`: deterministic, byte-identical generator for both tiny fixtures"
  - "crates/aprender-core/tests/fixtures/modernbert_tiny/: HF-named tiny ModernBERT (F16) + fp32 ladder oracle + window-mutation record (consumed by 08-03)"
  - "crates/aprender-decide/tests/fixtures/laya_tiny/: tiny Laya run dir + data dir in the laya-finetune-gate-v1 layout, oracle from Laya's own Agent.predict on the F16 reload, probes.json, zero-shot-probs.json, synthetic-fixture recipe, honest pass=false gate report (consumed by 08-04, 08-05, 08-09, 08-10)"
  - "Exact-path .gitignore negations for the two tiny model.safetensors"
affects: [08-03, 08-04, 08-05, 08-08, 08-09, 08-10, 08-12]

actuals:
  tokens: 227216   # chars/4 over the realized diff (908866 chars); dominated by generated fixtures and uv.lock. Authored code/docs alone: 56509 chars ~ 14127
  tasks: 2         # Task 1 checkpoint (approved, no files) + Task 2 tracer
  commits: 1
plan_head_before: c916e0879035c4db97bbb383bc6908a06b031212

tech-stack:
  added:
    - "Python back office (uv 0.9.5): torch 2.14.0, transformers 5.17.0, tokenizers 0.23.1, safetensors 0.8.0, huggingface-hub 1.32.0, numpy 2.5.3, pyyaml 6.0.3, laya @ git 4066d5d (0.3.20)"
  patterns:
    - "Oracle numbers captured from the library's own public path (hooks + a spy on Agent._decode_answers' answer_confidence call), cross-checked against the public rounded answer, never re-implemented"
    - "Fixture generator self-refuses: probe budget, loader rewrite (sha256 before/after reload), window discrimination, coverage of truncation/shrink/qtypes, 1 MiB bound"
    - "Exact f32 transport in JSON: decimal f64-repr lists (small), f32le_base64 blocks (large), 8-hex-digit big-endian bit patterns (probe convention)"

key-files:
  created:
    - scripts/laya_train/pyproject.toml
    - scripts/laya_train/uv.lock
    - scripts/laya_train/.python-version
    - scripts/laya_train/.gitignore
    - scripts/laya_train/README.md
    - scripts/laya_train/metrics.py
    - scripts/laya_train/fixtures.py
    - crates/aprender-core/tests/fixtures/modernbert_tiny/{config.json,model.safetensors,oracle.json}
    - crates/aprender-decide/tests/fixtures/laya_tiny/checkpoint/{model.safetensors,rl_agent_config.json,encoder/config.json,tokenizer/tokenizer.json,tokenizer/tokenizer_config.json}
    - crates/aprender-decide/tests/fixtures/laya_tiny/data/{task.json,train.jsonl,eval.jsonl}
    - crates/aprender-decide/tests/fixtures/laya_tiny/{task.json,recipe.json,gate-report.json,eval-probs.json,zero-shot-probs.json,probes.json,oracle.json}
  modified:
    - justfile
    - .gitignore

key-decisions:
  - "laya_tiny ladder blocks and marker states are f32le_base64 (exact) because 8-hex-digit bit patterns measured 942 KB and pushed the directory to 1.15 MB, over the 1 MiB bound; base64 0.22 is already a workspace dependency"
  - "Tiny encoders use initializer_range 0.2 (10x ModernBERT's default) so attention is not near-uniform; the generator proves the rows discriminate the local window (half-window +-1 moves layer1 by 6.5-10.2 % rms, layer0 bit-identical) and refuses otherwise"
  - "Oracle probabilities are Laya's own f32 `_decode_answers` p (spied at answer_confidence), asserted equal to predict's public round(p, 4); not the spike's float64 re-softmax"
  - "gate-report calibration.slice_ids_sha256 = sha256(json.dumps(slice_ids, separators=(',', ':'))); seeds.declared and recipe seed are the fixture seed 20260925 (the synthetic-fixture variant allows any seed)"
  - "Every non-encoder bias and every LayerNorm bias is re-drawn in [-0.1, 0.1] (nn.TransformerEncoderLayer zero-initialises its attention biases, which would hide a missing bias add); the temperature buffer is set to the config temperatures"

patterns-established:
  - "Back-office Python runs only through `uv run --project scripts/laya_train --frozen` from a just recipe"
  - "Fixture files hash-identical before/after the reference loader touches them (T-08-02-04)"

requirements-completed: [D-01, D-02, D-05, D-13, D-17]

coverage:
  - id: D1
    description: "Pinned, hash-locked uv back office with every human-approved pin at exactly the approved version and Laya as a full-SHA git dependency"
    requirement: D-02
    verification:
      - kind: other
        ref: "tomllib assertion over scripts/laya_train/uv.lock: 7 approved pins == locked versions, laya source rev 4066d5d5fbf08b66c6757ddeedbd797bd7655bc0, direct deps exactly the approved 8 (PIN CHECK OK)"
        status: pass
      - kind: other
        ref: "uv sync --frozen; import versions 3.13.7 / 2.14.0 / 5.17.0 / 0.23.1 / 0.8.0 / 2.5.3 / 6.0.3; installed laya/common.py and agent.py cmp-identical to the spike-024 vendor copy"
        status: pass
    human_judgment: false
  - id: D2
    description: "metrics.py house metrics with a self-test that replays the frozen ECE cases and goes red under a true-label-confidence mutation"
    requirement: D-01
    verification:
      - kind: unit
        ref: "uv run --project scripts/laya_train --frozen python scripts/laya_train/metrics.py --selftest -> METRICS SELFTEST OK (5 frozen cases within 1e-6)"
        status: pass
      - kind: unit
        ref: "mutant conf = P[arange, y] -> rc 1, reproduces the frozen red_ece_true_label_conf values"
        status: pass
    human_judgment: false
  - id: D3
    description: "Two tiny synthetic fixtures generated from Laya's / transformers' own code on F16-reloaded weights, deterministic (byte-identical re-run), each dir <= 1 MiB, nothing gitignored, apr CLI registry unchanged"
    requirement: D-17
    verification:
      - kind: other
        ref: "plan verify 1: just laya-fixtures -> METRICS SELFTEST OK + FIXTURES OK, rc 0"
        status: pass
      - kind: other
        ref: "plan verify 2: 18 fixture files sha256-identical across a re-run (cmp), du 240 KiB / 864 KiB"
        status: pass
      - kind: other
        ref: "plan verify 3: per-path git check-ignore -q --no-index rc 1 for all 9 paths; git diff --quiet contracts/apr-cli-commands-v1.yaml"
        status: pass
    human_judgment: false
  - id: D4
    description: "laya_tiny oracle coverage: 9 rows, qtypes 0/1/2, truncation rows (ids == max_len 64), option shrink with all markers kept, injection row, unicode row, the `many` marker-loss question (14 of 16 markers, Laya refuses it), probes <= 48 tokens, D-05 task.json, synthetic-fixture recipe, pass=false / margin 0.0 gate report with sorted slice_ids"
    requirement: D-05
    verification:
      - kind: other
        ref: "acceptance-criteria python checks AC3-AC7 over data/task.json, oracle.json, gate-report.json, recipe.json, probes.json (all PASS)"
        status: pass
      - kind: other
        ref: "reload-rewrite guard mutant (tokenizer_class TokenizersBackend) -> FIXTURES FAILED naming checkpoint/tokenizer/tokenizer_config.json; real run prints 5 files UNCHANGED"
        status: pass
    human_judgment: false
  - id: D5
    description: "modernbert_tiny: plain HF names (no prefix) for D-13 prefix-aware reuse, and rows that discriminate the local window"
    requirement: D-13
    verification:
      - kind: other
        ref: "generator window-mutation check: half-window 3/5 -> layer1 rel rms 0.0653-0.1020 (> 10 x 1e-3), layer0 0.0 (recorded in oracle.json window_mutation)"
        status: pass
    human_judgment: false

duration: 15min
completed: 2026-09-26
status: complete
---

# Phase 8 Plan 02: Laya Back Office and Tiny Fixtures Summary

**A human-approved, hash-locked uv project on Laya @ 4066d5d generates two deterministic tiny CI fixtures: a plain-HF ModernBERT with an fp32 ladder, and a tiny Laya run dir scored through Laya's own `Agent.predict` on the F16 reload. The Laya fixture covers truncation, option shrink, marker loss, injection and unicode rows, and adds probes, zero-shot probabilities and an honest `pass: false` gate report.**

## Performance

- **Duration:** 15 min (this continuation run; Task 1's read-only pin preparation happened in the earlier run that stopped at the checkpoint)
- **Started:** 2026-09-26T00:38:13Z
- **Completed:** 2026-09-26T00:53:26Z
- **Tasks:** 2 of 2 (Task 1 checkpoint approved; Task 2 tracer)
- **Files modified:** 27 (25 created, 2 modified)

## Package-Legitimacy Approval (Task 1)

**Who / what / when:** the user approved the pin table as written ("approved"), **2026-09-25**. That covers the
macOS-arm64 lock restriction, Python 3.13.7 (`uv python find` resolves `/usr/local/bin/python3.13`,
confirmed 3.13.7) and uv 0.9.5. No `uv.lock` existed under `scripts/laya_train/` before the approval
(the directory did not exist).

| Approved pin | In uv.lock |
|--------------|------------|
| torch 2.14.0 | 2.14.0 |
| transformers 5.17.0 | 5.17.0 |
| tokenizers 0.23.1 | 0.23.1 |
| safetensors 0.8.0 | 0.8.0 |
| huggingface-hub 1.32.0 | 1.32.0 |
| numpy 2.5.3 | 2.5.3 |
| pyyaml 6.0.3 | 6.0.3 |
| laya git+https://github.com/NandhaKishorM/laya@4066d5d5fbf08b66c6757ddeedbd797bd7655bc0 | git rev 4066d5d5fbf08b66c6757ddeedbd797bd7655bc0 (laya 0.3.20) |
| HF base weights convaiinnovations/laya@55cf4c4e... | not a lock entry. Recorded in README; plan 08-08 downloads it |
| `[tool.uv] environments = ["sys_platform == 'darwin' and platform_machine == 'arm64'"]` | `resolution-markers` / `supported-markers` = that marker only |

`uv lock` resolved 36 packages in 1.01 s. It forced no different version and added no unexpected
top-level package: the root's direct dependencies are exactly the approved 8.

**Exact resolved transitive set (hash-pinned in `uv.lock`, all from the PyPI registry except laya):**
annotated-doc 0.0.5, anyio 4.15.1, certifi 2026.7.22, click 8.5.0, filelock 4.0.3, fsspec 2026.9.0,
h11 0.16.0, hf-xet 1.6.0, httpcore 1.0.9, httpx 0.28.1, idna 3.20, jinja2 3.1.6, markdown-it-py 4.2.0,
markupsafe 3.0.3, mdurl 0.1.2, mpmath 1.3.0, networkx 3.7, packaging 26.3, pygments 2.21.0,
regex 2026.9.10, rich 15.0.0, setuptools 84.0.0, shellingham 1.5.4, sympy 1.14.0, tqdm 4.70.1,
typer 0.27.2, typing-extensions 4.16.0. Adding the 8 direct deps and the virtual root `laya-train` 0.1.0 gives 36.

## Accomplishments

- `scripts/laya_train/` is a pinned uv project following the `scripts/setfit_fixtures` conventions:
  upper-bounded `requires-python`, exact pins each with a why-comment, and `package = false`. Installed Laya
  is byte-identical to the spike-024 vendor copy (`common.py` and `agent.py` compared with `cmp`).
- `metrics.py` (numpy only, no torch import) implements the house floor-binned top-label ECE. It replays all 5
  frozen `ece_top_label_cases.json` cases within 1e-6. A mutation test proved it can fail: reading
  confidence from the true label turns it red and reproduces the frozen `red_ece_true_label_conf`
  values.
- `fixtures.py` + `just laya-fixtures` generate both fixtures deterministically. A second run is
  byte-identical over all 18 files. modernbert_tiny is 240 KiB and laya_tiny is 864 KiB (`du -sk`).
- **laya_tiny oracle** (9 rows, all from `Agent.predict` on the F16 reload, captured via the spike-025
  hooks):

  | # | qid | type | ids | markers | notes |
  |---|-----|------|-----|---------|-------|
  | 0-3 | team | choice (task) | 50-57 | [14,19,24] | option shrink, all 3 markers kept |
  | 4 | urgent | noul | 53 | [18,26] | T noul:2 = 0.8 |
  | 5 | mood | score | 61 | [17,22,27] | T score:3-5 = 1.25 |
  | 6 | safe | noul | 58 | [18,26] | injection row: literal `[SEP]`/`[CLS]` in the state tokenize to ids 2/1 (Pitfall 10 reproduced) |
  | 7 | lang | choice (4 langs) | 64 | [18,21,24,27] | mixed-script unicode; also truncated (byte-level CJK) |
  | 8 | team | choice (task) | 64 | [14,19,24] | the D-12 truncation row (untruncated 120+ tokens) |

  Also recorded: the `many` question (16 criteria), where Laya's own `build_sequence` keeps **14 of 16**
  markers `[10,...,62]` in a 64-id row and `Agent._encode_state` refuses it with `question 'many' options
  exceed head_max_len=32`.
- **Probes** (decide-apr-v1 synthetic probe task, Laya's predict): input 0 = 21 tokens, label `no`;
  input 1 = 26 tokens, label `yes`. Both are under the 48 cap, and the generator refuses above it.
- **Checkpoint sha256 before/after the `Agent` reload** (generator output, every run):
  `laya_tiny: checkpoint sha256 before/after Agent reload: 5 files, UNCHANGED`. The files are
  `checkpoint/encoder/config.json`, `checkpoint/model.safetensors`, `checkpoint/rl_agent_config.json`,
  `checkpoint/tokenizer/tokenizer.json` and `checkpoint/tokenizer/tokenizer_config.json`. The guard was
  mutation-tested: writing `tokenizer_class: TokenizersBackend` makes Laya's loader rewrite the file,
  and the generator exits 1 naming `checkpoint/tokenizer/tokenizer_config.json`.
- **Gate report:** `pass: false` and `margin: 0.0`, because zero-shot and fine-tuned are the same model and
  `zero-shot-probs.json` is byte-equal to `eval-probs.json`. Thresholds are read from the contract
  (0.05 / 0.10 / 15). `slice_ids` [0,2,6,7,8,11] is sorted and stratified 2 per class; `device_used` reads `cpu`
  from the parameters. `recipe.json` has `variant: synthetic-fixture`, shots 0, epochs 0 and
  base `tiny-synthetic` with its sha256.

## Task Commits

1. **Task 1: Package legitimacy checkpoint** - no commit (checkpoint, no files; approved by the user 2026-09-25)
2. **Task 2: Tracer - pinned uv project -> Laya's own code -> F16 save -> reload -> oracle -> committed tiny fixtures** - `d8030eab6` (feat)

**Tracer gate:** interactive run, `human_verify_mode` end-of-phase, `<verify>` automated only (row 3).
All three verify commands were re-run after implementation and passed, so the tracer is verified end to end.
There are no expansion tasks in this plan.

## CB-510 Guards (run after the .gitignore edit)

```
$ bash scripts/check_include_files.sh          -> rc 0
usage: grep [-abcdDEFGHhIiJLlMmnOopqRSsUVvwXxZz] [-A num] [-B num] [-C[num]] ...
OK: All 0 include!() files are tracked by git
$ bash scripts/check_package_includes.sh       -> rc 0
scanned 10 publishable crate(s) containing include!(); 1548 include target(s)
PASS: every include!() target survives `cargo package` in all 10 crate(s).
```

`check_include_files.sh` is **vacuous on this macOS box**: BSD grep rejects its flags, so it checks 0
files and still passes. This problem predates the plan (see the memory note on the two include guards) and is out of scope,
so its green is not evidence. `check_package_includes.sh` did real work (1548 targets). Independently,
`git status --ignored -uall` shows all 18 fixture files and all 7 project files as visible to git, and
only `.venv/` and `__pycache__/` as ignored. A sibling `checkpoint/other.safetensors` still reports ignored
(rc 0), so the negations are exact-path.

## Files Created/Modified

- `scripts/laya_train/pyproject.toml` - pinned project (approved pins, laya git SHA, macOS-arm64 environments)
- `scripts/laya_train/uv.lock` - hash-locked resolution, 36 packages
- `scripts/laya_train/.python-version` - 3.13.7
- `scripts/laya_train/.gitignore` - `.venv/`, `__pycache__/`
- `scripts/laya_train/README.md` - pins, recipes, fixture encodings, what CI does and does not run
- `scripts/laya_train/metrics.py` - house metrics + `--selftest`
- `scripts/laya_train/fixtures.py` - both fixture generators (720 lines)
- `justfile` - `laya-fixtures` recipe (bash shebang, `set -euo pipefail`, `uv run --frozen`)
- `.gitignore` - two exact-path negations under `*.safetensors`, with a comment
- `crates/aprender-core/tests/fixtures/modernbert_tiny/` - config.json, model.safetensors (F16), oracle.json
- `crates/aprender-decide/tests/fixtures/laya_tiny/` - checkpoint/ (5 files), data/ (3 files), task.json, recipe.json, gate-report.json, eval-probs.json, zero-shot-probs.json, probes.json, oracle.json

## Decisions Made

- **laya_tiny ladder encoding is `f32le_base64`.** Hex bit patterns measured 942 KB for oracle.json
  (1.15 MB directory, over the 1 MiB bound). Base64 of little-endian f32 bytes is 633 KB and still exact,
  and `base64 = "0.22"` is already a workspace dependency. Logits and probabilities stay as decimal lists
  plus 8-hex-digit bit patterns. modernbert_tiny keeps the plan's decimal lists.
- **Probabilities are Laya's own f32 `p`.** They are spied at `_decode_answers`' `answer_confidence(p, k)`
  call, and every row asserts that `predict`'s public 4-decimal answer equals `round(p, 4)` and that `p`
  matches `softmax(logits / T)` within 1e-6. The spike computed a float64 re-softmax instead; this records
  what the served path computes.
- **`many` ids and markers come from `laya.common.build_sequence`,** Laya's own builder, because
  `Agent.predict` refuses that question. The oracle records Laya's refusal message alongside.
- **`slice_ids_sha256` = sha256 of `json.dumps(slice_ids, separators=(",", ":"))`.** The contract names
  the field but not its preimage. This definition is documented in the README and in the fixtures.py
  docstring so plan 08-09 can re-derive it.
- **The gate report omits `f_avg_rule`.** It is a schema rule ("f_avg may be null"), not a value field.
  `f_avg` is `null` for this non-stance task.
- **Recipe base:** `repo: "synthetic"`, `revision: "fixtures.py-seed-20260925"`, `checkpoint: "tiny-synthetic"`
  and the tiny model's sha256. `seeds.declared` is 20260925 (the synthetic variant allows any seed).
- **config.json files write every ModernBertConfig field explicitly,** plus `model_type`,
  `transformers_version` and `global_attn_every_n_layers`, so no Rust reader has to know a transformers
  default. The generic PreTrainedConfig keys (id2label, return_dict, ...) are dropped.
- **tokenizer_config.json uses `model_max_length` 8192,** the base checkpoint's value. At 64,
  transformers printed a misleading "sequence longer than max" warning on the truncation row. Laya's
  `max_len` is what governs.
- **The tokenizer mirrors the base's tokenizer.json structure:** NFC normalizer, ByteLevel
  pre-tokenizer and decoder, `[CLS] $A [SEP]` template, `[MASK]` lstrip. The specials are
  `[UNK]=0 [CLS]=1 [SEP]=2 [PAD]=3 [MASK]=4`, and modernbert_tiny uses the same ids.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 3 - Blocking] The laya_tiny oracle overflowed the 1 MiB fixture bound**
- **Found during:** Task 2 (first generator run)
- **Issue:** The per-row ladder (7 blocks x ~58 tokens x 32 over 9 rows) was 942 KB as hex bit patterns, and the directory totalled 1,149,968 bytes. The generator's own 1 MiB check refused it.
- **Fix:** Ladder blocks and `m_opts` are now standard base64 of little-endian f32 bytes. The values are exact and the directory is 840,553 bytes.
- **Files modified:** scripts/laya_train/fixtures.py, crates/aprender-decide/tests/fixtures/laya_tiny/oracle.json
- **Verification:** FIXTURES OK; `du -sk` 864 ≤ 1024
- **Committed in:** d8030eab6

**2. [Rule 2 - Missing critical] The tiny encoders would not have exercised the local window**
- **Found during:** Task 2 (design)
- **Issue:** At ModernBERT's default `initializer_range` 0.02, a 32-wide model attends almost uniformly, so a window or RoPE defect barely moves a layer. The laya-parity-v1 `window_mutation` invariant says that if the mutation does not fail the local layer, the rung has stopped being evidence.
- **Fix:** Both tiny encoder configs use `initializer_range` 0.2. The generator rebuilds modernbert_tiny with the half-window at 3 and 5 and refuses unless the first local layer moves more than 10 x 1e-3 while the first global layer is bit-identical. Measured: 0.0653-0.1020 vs 0.0. The result is recorded in oracle.json `window_mutation`.
- **Files modified:** scripts/laya_train/fixtures.py
- **Verification:** the generator check passes on every run
- **Committed in:** d8030eab6

**3. [Rule 2 - Missing critical] Zero biases would hide a missing bias add in the Laya head**
- **Found during:** Task 2 (design)
- **Issue:** `nn.TransformerEncoderLayer` zero-initialises its attention in-proj and out-proj biases, so a Rust port that dropped them would still match.
- **Fix:** Every non-encoder bias (head, scorer, act_head) and every LayerNorm bias is re-drawn uniform in [-0.1, 0.1]. LayerNorm weights are drawn in [0.5, 1.5] as the plan specifies. The `temperature` buffer is set to the config temperatures so reading either source gives the same values.
- **Files modified:** scripts/laya_train/fixtures.py
- **Verification:** determinism re-run byte-identical
- **Committed in:** d8030eab6

---

**Total deviations:** 3 auto-fixed (1 blocking, 2 missing-critical fixture strength). The other entries above are recorded choices within Claude's discretion.
**Impact on plan:** None changes a contract or a consumer-facing schema. Deviation 1 changes the encoding
of the laya_tiny ladder blocks, and plans 08-04 and 08-05 read them as base64 (documented in oracle.json
`encoding`, the README and the fixtures.py docstring).

## Issues Encountered

- `check_include_files.sh` checks 0 files on macOS (BSD grep usage error) and still exits 0. The problem
  predates this plan and is out of scope; it is recorded here so nobody cites its green.
- The unicode row (7) is also truncated, because byte-level BPE with a 512-token vocab expands CJK.
  Coverage is unaffected: row 8 is the dedicated D-12 truncation row.
- The tiny model's probabilities are near-uniform (random init). The argmax margins are ≥ 7e-4, far above
  the 1e-5 `probs_abs` bar, so `argmax_exact` stays meaningful.

## Known Stubs

None. No placeholder text, empty data paths or TODO/FIXME markers in the created files (stub scan rc 1).

## User Setup Required

None. The Python environment is laptop-only and set up by `uv sync --frozen` / `just laya-fixtures`.

## Next Phase Readiness

- Plan 08-03 can read `modernbert_tiny/` (plain HF names, F16, decimal fp32 ladder, shapes).
- Plan 08-04 can read `laya_tiny/oracle.json`. Ladder blocks and `m_opts` are f32le_base64; ids, markers,
  logits and probabilities are exact; the `truncated` flag and the `many` marker-loss case are present.
- Plans 08-05, 08-09 and 08-10 have the run dir (recipe, gate report with `slice_ids`, eval/zero-shot probabilities,
  probes) and its synthetic-fixture refusal hook.
- Plan 08-08 extends the same uv project (`metrics.py` is shared).
- 08-12's CI checkpoint decides whether `metrics.py --selftest` runs in CI.

## Self-Check: PASSED

- 14/14 key files found on disk. All 25 plan files are tracked by git (18 fixture + 7 project).
- Commit `d8030eab6` exists (`git log --oneline --all`).
- Acceptance criteria AC1-AC10 were re-run and all pass. The three plan `<verify>` commands pass (rc 0).

---
*Phase: 08-laya-decision-model-local-fine-tune-and-thin-mcp-server*
*Completed: 2026-09-26*
