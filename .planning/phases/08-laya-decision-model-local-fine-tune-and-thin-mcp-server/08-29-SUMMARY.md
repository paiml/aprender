---
phase: 08-laya-decision-model-local-fine-tune-and-thin-mcp-server
plan: 29
subsystem: training-back-office
tags: [laya, python, contracts, class-e, refusals, early-stopping, jsonl, pv]
status: complete
gap_closure: true

requires:
  - phase: 08-27
    provides: "laya-finetune-gate-v1 4.0.0 numeric_agreement (makes the 'same subtraction the Rust verifier does' claim true)"
  - phase: 08-21
    provides: "Rust verify.rs parse_rows / DataInvalid row rule and the VerifyError variants mapped in python_refusals"
provides:
  - "contract.number(value, key, kind) + ContractValueError (REFUSED contract-value): the one typed reader for every contract number; noise k written as the float Rust compares bit for bit"
  - "data.jsonl_lines == Rust str::lines; <role>-row-encoding refusals (invalid UTF-8, lone surrogate)"
  - "EarlyStopper.record typed refusal; train.py REFUSED early-stopping, exit 2"
  - "laya-finetune-gate-v1 5.0.0: one early-stopping rule (best-anchored) in the formula, FALSIFY-LAYA-GATE-009 and tie_break_rule; python_refusals table (35 rows)"
  - "data.py --selftest python_refusals sweep (PYTHON REFUSALS swept=42 lifecycle=5) with a prepare_stance.py CLI-boundary case"
  - "prepare_stance.write_once ignores dotfiles; --src; DataError -> REFUSED, exit 2"
affects: [08-verify-work, laya-train, laya-prepare-stance, laya-finetune-gate-v1]

actuals:
  tokens: 10400      # chars/4 over the realized diff (git diff 6ea8d0cc6..HEAD | wc -c = 41605)
  tasks: 3
  commits: 3
plan_head_before: 6ea8d0cc6ca0d1fa4a473db28ed079a72c2a1406

tech-stack:
  added: []
  patterns:
    - "One typed reader per contract number: a str/bool is refused, an int kind refuses any float, a float kind widens an int exactly; the refusal names the key"
    - "A refusal table in the contract that a self-test sweeps in both directions: every row has a case, every case has a row, every rule id in the sources has a row"
    - "Contract text corrected TO THE CODE of record, pinned by a trace on which the candidate rules disagree"

key-files:
  created: []
  modified:
    - scripts/laya_train/contract.py
    - scripts/laya_train/data.py
    - scripts/laya_train/gate.py
    - scripts/laya_train/train.py
    - scripts/laya_train/prepare_stance.py
    - scripts/laya_train/lifecycle.py
    - scripts/laya_train/fixtures.py
    - contracts/laya-finetune-gate-v1.yaml
    - .planning/phases/08-laya-decision-model-local-fine-tune-and-thin-mcp-server/deferred-items.md

key-decisions:
  - "The early-stopping rule of record is best-anchored. It is re-derived from the run, not assumed: gate.py at 24cea2a63 (the last gate.py commit before the 08-16 run's recipe.json mtime, 2026-09-27 11:08) implements m_e < m_best - min_delta, and the shipped seed-17 trace replays to its recorded best_epoch 2 / epochs_run 5 / patience under it. That trace does not separate the rules, since all three candidates give epoch 2 on it; the code of record does."
  - "rescore-noise.json k is written as a float (4.0). Rust reads f64 and compares bits; the run of record's integer 4 is the same f64. rescore_noise_schema.k is re-described as number (float64), beyond the plan's allowed-diff list (see Deviations)."
  - "laya-finetune-gate-v1 bumped to 5.0.0, because pv diff classifies the formula and invariant edit MAJOR. This follows the contract's own 3.0.0/4.0.0 precedent and uses a TEXT ALIGNMENT note in metadata.description."
  - "write_once ignores dotfiles only. A foreign non-dot file (a stale shift.jsonl that train.py would score as a shift probe) is still REFUSED prepare-out-dir."
  - "prepare_stance.fail() (PREPARE FAILED, exit 1) stays. It is a consistency check of the pinned public dataset, not a tenant-input refusal, and python_refusals.not_refusals records that."

patterns-established:
  - "Rust-parity row reader: data.jsonl_lines, pinned by a rustc 1.98 str::lines case table in the self-test"
  - "Lifecycle rows of python_refusals name the lifecycle.py function that exercises them; the sweep checks it is defined and called"

requirements-completed: [D-01, D-05, D-06, D-07]

coverage:
  - id: D1
    description: "Typed contract-number reader; no contract number coerced; noise k is a float that armed Rust python_records verifies"
    requirement: D-07
    verification:
      - kind: unit
        ref: "scripts/laya_train/data.py --selftest#contract-value (11 cases incl. k=2.5 kept, \"1e-6\"/true/15.0 refused naming the key, run-of-record recipe_id 6a5489af rebuilt)"
        status: pass
      - kind: integration
        ref: "LAYA_LIFECYCLE_KEEP + cargo test -p aprender-decide --test python_records (NOISE x2, MEDIAN rust=23 python=23, no SKIP)"
        status: pass
    human_judgment: false
  - id: D2
    description: "JSONL rows split exactly as Rust str::lines; invalid UTF-8 / lone surrogate refused <role>-row-encoding"
    requirement: D-05
    verification:
      - kind: unit
        ref: "scripts/laya_train/data.py --selftest#row-split and row-encoding cases"
        status: pass
      - kind: integration
        ref: "lifecycle.py check_train_cli_refusals (train.py CLI: REFUSED train-row-encoding, exit 2)"
        status: pass
    human_judgment: false
  - id: D3
    description: "Early-stopping contract states one rule (best-anchored) with a distinguishing trace; typed early-stopping refusal at train.py"
    requirement: D-06
    verification:
      - kind: unit
        ref: "scripts/laya_train/gate.py --selftest#best-anchored (3 cases) and early-stopping"
        status: pass
      - kind: integration
        ref: "lifecycle.py check_early_stopping_refusal"
        status: pass
      - kind: other
        ref: "pv validate contracts/laya-finetune-gate-v1.yaml (0 errors); frozen-block comparison vs 6ea8d0cc6"
        status: pass
    human_judgment: false
  - id: D4
    description: "python_refusals table enumerates every back-office refusal and is swept, including a CLI-boundary case"
    requirement: D-01
    verification:
      - kind: unit
        ref: "scripts/laya_train/data.py --selftest#python_refusals (PYTHON REFUSALS swept=42 lifecycle=5)"
        status: pass
    human_judgment: false
  - id: D5
    description: "prepare_stance re-run beside a .DS_Store succeeds when byte-identical; a differing or foreign file is refused"
    requirement: D-05
    verification:
      - kind: unit
        ref: "scripts/laya_train/data.py --selftest#prepare_stance.write_once (3 cases)"
        status: pass
      - kind: e2e
        ref: "prepare_stance.py --cell s64 and --cell s16 on the real local dataset: unchanged (byte-identical)"
        status: pass
    human_judgment: false

duration: 25min
completed: 2026-09-28
---

# Phase 8 Plan 29: Python back office reads, splits and refuses like the Rust verifier

**The back office now reads every contract number through one typed reader, so the noise record carries a float k that armed Rust verifies. It splits JSONL rows exactly as `str::lines`, and it turns every malformed input into a `REFUSED <rule>` line with exit 2. A 35-row `python_refusals` table lists every such refusal and a self-test sweeps it. laya-finetune-gate-v1 5.0.0 states one early-stopping rule: the best-anchored rule that the 08-16 run of record trained under.**

## Performance

- **Duration:** 25 min
- **Started:** 2026-09-28T21:45:30Z
- **Completed:** 2026-09-28T22:10:48Z
- **Tasks:** 3 of 3
- **Files modified:** 9 (0 created)
- **BASE (plan_head_before):** `6ea8d0cc6ca0d1fa4a473db28ed079a72c2a1406`

## Accomplishments

- **WR-09 / V13-d, contract numbers.** `contract.number(value, key, kind)` is the one typed reader, and `ContractValueError` renders as `REFUSED contract-value: <key> is <value> (<type>); ...`.
  - `noise_policy` reads k as a float. It used to apply `int(...)`, so a 2.5 would have been truncated to 2 and every Rust verify would then refuse the record.
  - `thresholds`, `early_stopping_decl`, `seed_selection_decl`, `resolve_seeds`, `resolve_epochs`, `recipe_json`, `probe_policy` and the direct reads in train.py, gate.py, data.py and fixtures.py all read through it.
  - `train.py` calls `contract.check_numbers()` in its refusal block, so a mistyped contract refuses before any model loads.
  - The run of record's recipe.json still rebuilds to recipe_id `6a5489af…`, byte for byte.
- **WR-05, row boundaries.** `data.jsonl_lines` splits on `'\n'` only and strips one `'\r'` before a `'\n'`.
  - It matches Rust `str::lines` on a 9-case table that I measured with rustc 1.98, including an unterminated `"a\r"`, which keeps its `\r`.
  - A tweet holding U+0085, U+2028 or U+2029 is now one row in both languages. `load_rows`, `prepare_stance.read_jsonl`, the demo_rule check and lifecycle's row counts all use it.
- **V13-a, encoding.** An invalid UTF-8 byte and a lone-surrogate text or label are both refused with `REFUSED <role>-row-encoding`. `prepare_stance` uses `source-row-encoding`.
- **IN-08, early stopping.** `EarlyStopper.record` raises `DataError("early-stopping")` when no monitor was ever finite. `train.stopping_record_or_refuse` turns that into `REFUSED early-stopping: ...` with exit 2.
- **V13-c, one early-stopping rule.** The following now state the best-anchored rule:
  - the `early_stopping_train_side` formula and its tie invariant
  - FALSIFY-LAYA-GATE-009
  - the `early_stopping.tie_break_rule` description

  Every early_stopping VALUE key, `tie_break: earliest` included, is byte-frozen. gate.py has a distinguishing trace: `[1.0, 0.9993, 0.9988]` gives best-anchored 3, running-min 1 and earliest-within-min_delta-of-min 2.
- **D3-5, dotfiles.** `prepare_stance.write_once` compares only the files it writes and ignores dotfiles. On the real dataset, a re-run of both cells is `unchanged (byte-identical)`.
- **gate.py comment.** The "same subtraction the Rust verifier does" comment now cites `numeric_agreement.quantities.margin`.
- **Refusal table.** `python_refusals` has 35 rows and is swept in both directions (details below).

## Task Commits

1. **Task 1 (tracer): one typed contract-number reader, carried into the noise record Rust verifies** — `21716c494` (feat)
2. **Task 2: rows split like Rust, typed refusals not tracebacks, one early-stopping rule** — `438b6c1fe` (fix; TDD, RED observed first)
3. **Task 3: python_refusals table and its sweep; mutation proof** — `97ed53425` (test)

## The rule of record (re-derived, not assumed)

- **The shipped run's own record.** `models/decide/laya-stance-64/checkpoint/rl_agent_config.json` `training.stopping` for seed 17:
  - monitors 0.75667, 0.75215, 0.85396, 0.87062, 0.92542
  - best_epoch 2, epochs_run 5, reason patience
- **The code that ran.** The run's recipe.json mtime is 2026-09-27 11:08. The last gate.py commit before it is 24cea2a63 (10:22), and its `EarlyStopper.update` is `m < self.best - self.min_delta`, which is the best-anchored rule.
- **Replay.** Under that rule, the trace replays to the recorded 2 / 5 / patience. `gate.py --selftest` pins this as `RUN_OF_RECORD_STOPPING` and checks it against the local run dir when that dir is present.
- **Limit of the evidence.** This trace alone does NOT separate the three candidate rules: all give epoch 2. The record is therefore consistent with every candidate rule, not inconsistent with all of them, so there was no checkpoint to raise. The code of record decides the rule.
- **Seeds 13 and 23.** Their stopping records survive only as best_epoch and epochs_run in variance-report.json (1/4 and 2/5). Their checkpoints were deleted.

## python_refusals enumeration (class E, refusal half)

`<role>` expands over the listed roles. **Sweep** is either `in` (an in-process torch-free case) or `life` (the named lifecycle.py function).

| Rule | Entry | Hostile input | Rust counterpart | Sweep |
|------|-------|---------------|------------------|-------|
| task-missing | data.load_task | no task.json | Read (task.json) | in |
| task-schema | data.load_task | not UTF-8 JSON, a repeated or missing key, an empty field | Pack(PackError::Schema) | in |
| task-unknown-key | data.load_task | a key outside {type, instructions, criteria} | Pack(PackError::Schema) | in |
| task-type | data.load_task | type is not "choice" | Pack(PackError::Schema) | in |
| task-duplicate-criterion | data.load_task | a criterion name twice | Pack(PackError::Schema) | in |
| task-too-few-criteria | data.load_task | one criterion | Pack(PackError::Schema) | in |
| <role>-missing [train, eval] | data.load_rows | no <role>.jsonl | Read | in |
| <role>-row-encoding [train, eval, shift] | data.load_rows | an invalid UTF-8 byte, or a lone surrogate in text or label | DataInvalid | in |
| <role>-row-schema [train, eval, shift] | data.load_rows | a blank line, not an object, a missing key, an empty text | DataInvalid | in |
| <role>-row-unknown-key [train, eval, shift] | data.load_rows | a key outside {text, label} | DataInvalid (deny_unknown_fields) | in |
| <role>-row-label [train, eval, shift] | data.load_rows | a label that is not a criterion NAME | DataInvalid | in |
| <role>-empty [train, eval, shift] | data.load_rows | an empty file | DataInvalid (no rows) | in |
| eval-class-coverage | data.load_rows | a criterion with no eval row | DataInvalid (check_eval_coverage) | in |
| <role>-train-overlap [eval, shift] | data.refuse_overlap | a text equal to a train text after normalization | SplitOverlap for eval; none for shift (Rust does not re-check the shift overlap) | in |
| heldout-shot-overlap | data.in_distribution_heldout | a held-out row equal to a shot | none: back-office only | in |
| train-conflicting-labels | data.group_train | the same normalized text under two labels | ConflictingLabels | in |
| train-class-too-small | data.calibration_split | a class too small for the calibration slice | SliceInvalid | in |
| contract-value | contract.number | "1e-6" as a string, a bool, or 15.0 where an int is declared | none: back-office only (Rust deserializes into typed fields) | in |
| epochs | contract.resolve_epochs | --epochs at <= 16 shots/class, or out of range above 16 | RecipeMismatch | in |
| variant | contract.resolve_epochs | neither production nor synthetic-fixture | RecipeMismatch / SyntheticNotDeployable | in |
| stopping | contract.resolve_stopping | not a declared stopping rule | RecipeMismatch | in |
| seeds | contract.resolve_seeds | production --seeds other than 3 | SeedPolicyViolated / SeedPolicyMissing | in |
| seeds | gate.select_median_seed | an even N, an empty list, a repeated seed, a non-finite ECE | SeedPolicyViolated | in |
| seeds | gate.verify_report | ships a seed that is not the median | SeedPolicyMismatch | in |
| thresholds | gate.verify_report | thresholds differ from the contract | ThresholdMismatch | in |
| pass | gate.verify_report | pass disagrees with the metrics | PassDisagrees | in |
| early-stopping | gate.EarlyStopper.record | no finite calibration monitor | none: back-office only | in |
| early-stopping | train.stopping_record_or_refuse | the same, at the CLI | none: back-office only | life: check_early_stopping_refusal |
| source-row-encoding | prepare_stance.read_jsonl | an invalid UTF-8 byte or a lone surrogate in a source split | none: back-office only | in, plus the CLI-boundary subprocess |
| source-row-schema | prepare_stance.read_jsonl | a line that is not an object with an id | none: back-office only | in |
| prepare-out-dir | prepare_stance.write_once | a differing written file or a foreign non-dot file | none: back-office only | in |
| base | train.Base | production with --base, or synthetic without --base | BaseMismatch | life: check_train_cli_refusals |
| out-dir | train.main | a non-empty --out | none: back-office only | life: check_train_cli_refusals |
| noise-control | train.write_noise_record | the manual fp32 forward is not equal to the Scorer's logits | RescoreNoiseInvalid | life: check_noise_control_refusal |
| data-changed | train.refuse_if_data_changed | the data dir was edited during the run | InputHashMismatch | life: check_data_changed_refusal |

The sweep prints `PYTHON REFUSALS swept=42 lifecycle=5`. That is 35 rows, with the templated roles expanded into 42 in-process refusals plus 5 lifecycle rows.

A static scan found 32 distinct rule ids in the five back-office sources, and each one has a row. The scan matches `DataError("…")` and `"REFUSED <id>:` literals.

## Mutation table (each applied alone, then restored byte for byte, sha256 checked)

| # | Mutation | Turned RED |
|---|----------|------------|
| M1 | `noise_policy` k back to `int(...)` | data: `contract-value k = 4 … float 4.0` and `k = 2.5 stays 2.5` (both got 2 or 4 as an int); lifecycle: `rescore-noise.json k / floor_abs 4 … != laya-parity-v1` |
| M2 | `decode_jsonl` returns `text.splitlines()` | data: the three `row-split: a text holding U+0085/U+2028/U+2029 is ONE row` cases (REFUSED train-row-schema, line 13 not JSON) |
| M3 | the decode `try` removed | data: `train/eval.jsonl with an invalid UTF-8 byte` (UnicodeDecodeError) and the python_refusals rows `train-row-encoding` and `eval-row-encoding`; lifecycle: `a train.jsonl with an invalid UTF-8 byte was not REFUSED train-row-encoding … (exit 1)` |
| M4 | `EarlyStopper.record` back to `ValueError` | gate: the `early-stopping: … REFUSED early-stopping (typed)` case; data: the python_refusals row `early-stopping via gate.EarlyStopper.record`; lifecycle: `raised ValueError instead of REFUSED early-stopping, exit 2` |
| M5 | `write_once` compares the whole dir (dotfiles included) | data: `a byte-identical re-run beside a .DS_Store succeeds` (REFUSED prepare-out-dir) |

I also mutated the sweep's own guards and the train.py boundary:

- **M6.** An unlisted `REFUSED brand-new-rule:` added in train.py failed `python_refusals lists every rule id … no row for ['brand-new-rule']`.
- **M7.** Deleting the `pass` row failed both `sweep case pass via gate.verify_report has a table row` and the source scan.
- **Boundary.** Removing train.py's `stopping_record_or_refuse` try/except failed lifecycle with `raised DataError instead of REFUSED early-stopping, exit 2`.

## TDD RED evidence (Task 2, written before the implementation, run against HEAD's code)

- **data.py.** Eleven cases failed:
  - the U+0085/2028/2029 rows (REFUSED train-row-schema)
  - the jsonl_lines table (the function did not exist)
  - invalid UTF-8 in train and eval (a UnicodeDecodeError escaped)
  - the lone surrogate in a text (accepted)
  - the lone surrogate in a label (the wrong rule, shift-row-label)
  - the three write_once cases (exit 1, PREPARE FAILED)

  The CRLF case was already green, as expected: splitlines handles `\r\n`.
- **gate.py.** Two cases failed: the typed early-stopping refusal (a ValueError was raised) and the contract-text case (stale wording in all three places).
- **Already green.** The best-anchored trace case and the run-of-record replay were green at HEAD, because the code was always best-anchored. The defect was in the contract text, and the contract-text case is the one that was RED.

## Verification

- `just laya-train-selftest`: `LAYA TRAIN SELFTEST OK`. The output contains the contract-value, row-encoding, early-stopping, best-anchored and `PYTHON REFUSALS swept=42 lifecycle=5` cases, and 0 occurrences of `Traceback`.
- Task 1, Task 2 and Task 3 `<automated>` blocks, each run through the plan's own `r()` wrapper: all rc 0.
- **Armed Rust check of the Python-written record** (`LAYA_LIFECYCLE_KEEP` + `cargo test -p aprender-decide --test python_records`):
  - the written k is `4.0`
  - `NOISE which=fine_tuned rust=4.2993464954843574e-8 python=4.2993464954843574e-8`
  - `NOISE which=zero_shot rust=3.1062774519252656e-8 python=…`
  - `MEDIAN rust=23 python=23`
  - no SKIP
- `pv validate contracts/laya-finetune-gate-v1.yaml`: 0 errors, 0 warnings. `pv diff` against BASE: v4.0.0 → v5.0.0, suggested bump major (the early_stopping_train_side formula and invariants).
- **Frozen blocks against BASE (yaml.safe_load).**
  - `constants`, `recipe`, `seed_policy` and `base` are equal.
  - `early_stopping` is equal on every key except `tie_break_rule`.
  - The differing top-level keys are `early_stopping` (tie_break_rule only), `equations` (early_stopping_train_side only), `falsification_tests` (FALSIFY-LAYA-GATE-009 only), `python_refusals` (new), `rescore_noise_schema` (k description) and `metadata` (version and description).
- `grep -c "splitlines()"` gives 0 in data.py and in prepare_stance.py. lifecycle.py's remaining `splitlines()` splits train.py's stdout, not a JSONL file.
- `cargo test -p aprender-decide --lib --tests`: 167 passed. `make contract-audit-phase8`: rc 0.
- `LAYA_GATES_ONLY=train-selftest just laya-gates-selftest`: `ROW train-selftest: GREEN`, with both the must-fail uv shim and the must-pass run.
- `prepare_stance.py --cell s64` and `--cell s16` on the real local dataset print `unchanged (byte-identical)`, and the demo_rule case still rebuilds 459 rows as [111, 291, 57].

## Files Created/Modified

- `scripts/laya_train/contract.py`: `ContractValueError`, `number`, `constant`, `recipe_number`, `declared_seed`, `check_numbers`; every numeric read goes through them.
- `scripts/laya_train/data.py`:
  - `jsonl_lines`, `decode_jsonl`, `refuse_unencodable`
  - the contract-value, row-split, row-encoding and write_once cases
  - the `python_refusals` sweep and the CLI-boundary subprocess
  - one DataError class under `__main__`
- `scripts/laya_train/gate.py`:
  - typed `early-stopping` refusal, and the best-anchored docstring
  - `RUN_OF_RECORD_STOPPING`, `_running_min_rule`, `_within_min_delta_of_minimum_rule`, `_best_anchored_cases`
  - the numeric_agreement comment, and typed reads
- `scripts/laya_train/train.py`: `check_numbers` in the refusal block, `stopping_record_or_refuse`, `refuse_if_data_changed`, and typed reads.
- `scripts/laya_train/prepare_stance.py`: Rust row split, `source-row-*` refusals, dotfile-tolerant `write_once` (`prepare-out-dir`), `--src`, `main` returning exit 2 on a DataError.
- `scripts/laya_train/lifecycle.py`:
  - the written k must be a float
  - `jsonl_lines` for row counts
  - `check_early_stopping_refusal`, `check_train_cli_refusals`, `check_data_changed_refusal`, `check_noise_control_refusal`
- `scripts/laya_train/fixtures.py`: typed reads only. The values are identical.
- `contracts/laya-finetune-gate-v1.yaml`: 5.0.0; the early-stopping text alignment; `python_refusals`; the `rescore_noise_schema.k` description.
- `deferred-items.md`: the drift in the `laya-fixtures` byte-identity (pre-existing).

## Decisions Made

See `key-decisions` in the frontmatter. The main one is that the early-stopping rule of record is taken from the code that trained the 08-16 run, because the recorded trace cannot separate the candidate rules.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 3 - Blocking] lifecycle.py required an integer k**
- **Found during:** Task 1
- **Issue:** `check_noise_record` asserted `isinstance(rec["k"], int)`. The plan's float k would fail the lifecycle.
- **Fix:** It now requires a float. lifecycle.py was not in the plan's files list.
- **Commit:** 21716c494

**2. [Rule 2 - Missing critical] Lifecycle cases for base, out-dir, data-changed and noise-control**
- **Found during:** Task 3
- **Issue:** The plan says these rows are "checked to be exercised by lifecycle.py's existing cases (name them)". No existing lifecycle case exercised any of the four.
- **Fix:**
  - Added `check_train_cli_refusals`, `check_data_changed_refusal` and `check_noise_control_refusal`. The noise-control case injects the control mismatch by replacing `rescore_noise_set` for one call.
  - Added `check_early_stopping_refusal` for the train.py boundary.
  - Factored `train.refuse_if_data_changed` out of `main` so it can be exercised.
- **Commits:** 438b6c1fe, 97ed53425

**3. [Rule 1 - Bug] A self-test `DataError` class split**
- **Found during:** Task 2
- **Issue:** When data.py runs as `__main__`, `import data` elsewhere creates a second `DataError` class, and the self-test's `except DataError` did not catch prepare_stance's refusals.
- **Fix:** `sys.modules.setdefault("data", ...)` before the self-test.
- **Commit:** 438b6c1fe

**4. [Rule 2] prepare_stance refusals exit 2**
- **Found during:** Task 2
- **Issue:** The plan asks for a CLI-boundary REFUSED line with exit 2. `prepare_stance` had no way to point at a hostile source, and it rendered every refusal as `PREPARE FAILED`, exit 1.
- **Fix:** Added `--src`, and `main` now catches `DataError` (REFUSED, exit 2), which covers source-row-*, prepare-out-dir and heldout-shot-overlap. `fail()` stays for dataset and manifest consistency checks.
- **Commit:** 438b6c1fe

**5. [Rule 2] Contract edits beyond the Task 2 allowed-diff list**
- **Found during:** Tasks 2 and 3
- **Issue:** The Task 2 criterion says only early_stopping_train_side, FALSIFY-LAYA-GATE-009, tie_break_rule, python_refusals and comments differ. Three more changes were needed:
  - `metadata.version` 4.0.0 → 5.0.0, with a TEXT ALIGNMENT note, because pv diff classifies the edit MAJOR and the contract bumps on such edits (3.0.0, 4.0.0).
  - `rescore_noise_schema.k` said `'integer: …'` while the trainer now writes a float.
  - the tie invariant inside `early_stopping_train_side` ("ties go to the earliest epoch") was re-worded to the best-anchored statement.
- **Fix:** Made all three edits. Every frozen block (constants, recipe, seed_policy, base, and early_stopping minus tie_break_rule) is verified equal.
- **Commit:** 438b6c1fe, 97ed53425

**6. [Rule 2] More contract reads routed through the typed reader**
- **Found during:** Task 1
- **Issue:** The plan's Task 1 named contract.py only. The must-have truth says every contract number goes through one typed reader.
- **Fix:** Also routed train.py, gate.py, data.py and fixtures.py reads. The values are identical: recipe_id 6a5489af is rebuilt, and fixture regeneration shows no new drift from this change (see Issues).
- **Commit:** 21716c494

---

**Total deviations:** 6 auto-fixed (1 blocking, 1 bug, 4 missing-critical)
**Impact on plan:** All are needed for the plan's own truths ("never a traceback", "exercised by lifecycle", "one typed reader"). No threshold, constant, recipe or early_stopping value moved.

## Issues Encountered

- **The `just laya-fixtures` drift is pre-existing and logged in deferred-items.md.**
  - Regenerating the fixtures rewrites `laya_tiny/gate-report.json` `fine_tuned.ece_pre`, from …136 to …192, a gap of 5.6e-17.
  - I reproduced this on a pristine detached worktree at BASE, so this plan did not cause it. Plan 08-27's fsum ECE moved the value and the fixture was never regenerated.
  - I restored the file and did not commit the regenerated fixture.
- **The Task 2 verify's `grep -q Traceback` would have matched a case name.** The name was `… no Traceback`, so I renamed it to lowercase. The only remaining occurrences are in source files, never in the output.
- **The rtk hook's `grep -q` gave a false exit 1 when I ran the Task 3 verify by hand.** Re-run through the plan's own `r()` wrapper (`rtk run`), it passed with rc 0.

## Known Stubs

None.

## Threat Flags

None. The new surfaces are refusals only: `prepare_stance --src` reads a local directory the operator names, and a tampered file there is REFUSED.

## User Setup Required

None.

## Next Phase Readiness

Class E is now closed on the Python side for input, refusal and early stopping. The deployed artifact 24a44d7e is unaffected: no value moved, recipe_id 6a5489af rebuilds byte for byte, and the run of record's integer k=4 is the same f64 that Rust compares.

## Self-Check: PASSED

- The commits 21716c494, 438b6c1fe and 97ed53425 are present in `git log`.
- All 8 modified source and contract files exist, plus deferred-items.md.
- `commits: 3` is measured with `git rev-list --count 6ea8d0cc6..HEAD` (3), before this SUMMARY's own commit.
