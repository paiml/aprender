# apr-cli Silent Failure Enforcement Specification

Version: 1.0
Status: proposed
Date: 2026-04-09

**Version**: 1.1.0
**Status**: Active — Enforcement Required (Falsified 2026-04-04)
**Created**: 2026-04-04
**Parent**: [cli.md](cli.md) §6.3, [provable-contracts.md](provable-contracts.md)
**Crate**: `apr-cli`
**Contracts**: `apr-cli-v1.yaml`, `apr-cli-operations-v1.yaml`, `cli-dispatch-v1.yaml`

---

## 1. Problem Statement

Tarantula fault localization (suspicion scores 0.82–1.00) and cross-repo
contract audit reveal that `apr-cli` has **systematic silent failure modes**
where commands accept user input, appear to succeed, but silently discard
parameters or mask errors. These violate the exit code contract (cli.md §6.3),
the parameter propagation invariant (inference.md §5), and the Design by
Contract postcondition principle (design-by-contract.md).

### 1.1 Incident Classification

| ID | Severity | Tarantula | Description |
|----|----------|-----------|-------------|
| SF-001 | P0 | 1.00 | `lib_parse_rosetta.rs` struct initializers missing 7 fields — blocks entire lib test suite |
| SF-002 | P1 | 0.98 | `apr run` silently drops 7 sampling parameters (temperature, top_k, top_p, seed, repeat_penalty, repeat_last_n, split_prompt) |
| SF-003 | P1 | 0.90 | `apr inspect`/`apr validate` return exit code 0 on format errors |
| SF-004 | P1 | 0.88 | `dispatch_analysis.rs` catch-all `_ => return None` swallows unhandled commands |
| SF-005 | P2 | 0.82 | `apr finetune` accepts `--learning-rate`, `--plan` but never uses them |

### 1.2 Research Context

Silent failures in CLI dispatch are a known verification gap. Recent research
provides formal frameworks for detecting and preventing them:

- **Spectrum-Based Fault Localization** (Jones & Harrold, 2005; improved by
  SemLoc, arXiv:2603.29109): Tarantula suspicion ratio identifies components
  executed disproportionately by failing tests. Our analysis applies this to
  dispatch paths where tests compile-fail (SF-001) or silently pass (SF-002).

- **Agent Behavioral Contracts** (arXiv:2602.22302): Runtime enforcement
  detects 5.2–6.8 soft violations per session that uncontracted baselines miss.
  Directly applicable: our CLI parameters are "soft violations" — the system
  runs without error but produces wrong behavior.

- **Taming Silent Failures** (arXiv:2510.22224): Contract-based runtime
  envelope keeps behavior within provably safe bounds even under distribution
  shift. Our parameter-drop is analogous: the CLI accepts valid input but the
  execution envelope silently narrows.

- **Early Test Termination Impact on SBFL** (arXiv:2504.04557): Compilation
  failures in test suites (our SF-001) eliminate coverage data, making fault
  localization impossible. Slicing improves Tarantula by 10.7% MFR.

---

## 2. Five-Whys Root Cause Analysis

### 2.1 Why #1: Why do `apr run` sampling parameters have no effect?

**Observation**: User runs `apr run model.gguf --prompt "hello" --temperature 0.9 --top-p 0.95 --seed 42`. Output is identical to `--temperature 0.0` (greedy decode).

**Evidence**: `dispatch.rs:56-62` destructures `top_p`, `seed`, `repeat_penalty`, `repeat_last_n`, `split_prompt` from `Commands::Run` but `dispatch_run()` call on lines 88-110 omits all 5. Additionally, `temperature` and `top_k` are only used in the `batch_jsonl` branch (lines 73-84), not the main `dispatch_run()` path.

**Direct cause**: The `dispatch_run()` function signature (dispatch_run.rs:4-26) has 20 parameters but no sampling parameters. The `run::run()` entry point (run_entry.rs:8-27) has 18 parameters, also without any sampling parameters.

### 2.2 Why #2: Why does `dispatch_run()` lack sampling parameters?

**Observation**: The function was written when `apr run` was trace/format/profile-focused. Sampling parameters (top_p, seed, repeat_penalty, repeat_last_n, split_prompt) were added later (GH-569..572, PMAT-381..385) to the `Commands::Run` struct but never threaded through the dispatch chain.

**Evidence**: `commands_enum.rs:80-100` has comments referencing `F-CLIPARITY-01 / PMAT-381..385 / paiml/aprender#569..572`, indicating these fields were added as part of a CLI parity effort. The dispatch layer was not updated to propagate them.

**Direct cause**: **Struct evolution not synchronized with dispatch chain**. Clap derives the struct, so the CLI accepts the flags. But the manual dispatch code was not updated.

### 2.3 Why #3: Why wasn't the mismatch caught?

**Observation**: Rust emits `unused variable` warnings for all 5 dropped parameters (`dispatch.rs:57-62`). These warnings existed since the fields were added.

**Evidence**: `cargo clippy` output shows 5 warnings: `top_p`, `seed`, `repeat_penalty`, `repeat_last_n`, `split_prompt` — "help: try ignoring the field".

**Direct cause**: **Warnings treated as noise, not as contract violations**. No CI gate fails on unused-variable warnings in dispatch code. The pre-commit hook runs complexity/SATD checks but not unused-parameter detection.

### 2.4 Why #4: Why is there no contract enforcement for parameter propagation?

**Observation**: The provable contract `apr-cli-v1.yaml` specifies `command_parse_determinism` and `contract_gate_enforcement` but has **no equation for parameter propagation completeness**.

**Evidence**: The 26 contracts at Grade A (mean 0.97) verify structural properties (dispatch completeness, exit codes, output format) but not behavioral properties (every parsed parameter reaches the execution layer). This is the exact gap identified in GH-280: "Contracts enforce structural invariants but not behavioral invariants."

**Direct cause**: **No contract equation requires: ∀ param ∈ Commands::Run fields → param ∈ run::run() arguments**.

### 2.5 Why #5 (Root Cause): Why don't contracts cover parameter propagation?

**Root cause**: The provable contract system validates **existence** (function bound, signature matches) and **structure** (equations have invariants, falsification tests exist) but cannot verify **data flow** — that a value parsed at the CLI layer reaches the execution layer unchanged.

This is a category error in the contract taxonomy:
- **L1 (build.rs)**: Verifies function exists ✓
- **L2 (traits)**: Verifies function has right signature ✓
- **L3 (build.rs + traits)**: Both ✓
- **L4 (Kani)**: Bounded model checking on function body ✓
- **L5 (Lean)**: Unbounded proof ✓
- **MISSING**: **L-flow**: Data flow from CLI parse to execution — **no enforcement level covers this**

---

## 3. Falsification Protocol

### 3.1 SF-002: Parameter Propagation Falsification

**Contract**: `apr-cli-v1.yaml` / equation: `parameter_propagation_completeness`

```
∀ field f ∈ Commands::Run:
  ∃ param p ∈ run::run() arguments:
    dispatch_run(f) = run::run(p) ∧ value(f) == value(p)
```

**FALSIFY-PARAM-001**: Set `--temperature 0.9`, verify inference output differs from `--temperature 0.0`.
- **Prediction**: Output should be non-deterministic (temperature > 0 = sampling)
- **Actual**: Output is identical (greedy decode) — **FALSIFIED**
- **Root cause**: `temperature` never reaches `run::run()`

**FALSIFY-PARAM-002**: Set `--seed 1 --seed 2`, verify outputs differ.
- **Prediction**: Different seeds produce different token sequences
- **Actual**: Outputs identical — **FALSIFIED**
- **Root cause**: `seed` dropped at `dispatch.rs:58`

**FALSIFY-PARAM-003**: Set `--repeat-penalty 100.0`, verify repetition suppressed.
- **Prediction**: High penalty eliminates repeated tokens
- **Actual**: No effect — **FALSIFIED**
- **Root cause**: `repeat_penalty` dropped at `dispatch.rs:59`

### 3.2 SF-003: Exit Code Falsification

**Contract**: `cli-dispatch-v1.yaml` / equation: `exit_code_semantics`

```
∀ cmd ∈ {inspect, validate}:
  cmd(invalid_file) → exit_code ≠ 0
```

**FALSIFY-EXIT-001**: Run `apr inspect /dev/null`, check exit code.
- **Prediction**: Exit code 3 (format error)
- **Actual**: Exit code 0 with error printed to stderr — **FALSIFIED**

**FALSIFY-EXIT-002**: Run `apr validate /tmp/empty`, check exit code.
- **Prediction**: Exit code 1 (general error)
- **Actual**: Exit code 0 with error printed to stderr — **FALSIFIED**

### 3.3 SF-004: Dispatch Completeness Falsification

**Contract**: `cli-dispatch-v1.yaml` / equation: `dispatch_completeness`

```
∀ v ∈ ExtendedCommands::variants():
  dispatch_analysis(v) ≠ None
```

**FALSIFY-DISPATCH-001**: Add new `ExtendedCommands::Foo` variant, compile.
- **Prediction**: Compilation error (exhaustive match)
- **Actual**: Compiles silently, `_ => return None` catches it — **FALSIFIED**
- **Root cause**: `dispatch_analysis.rs` line 223 catch-all

---

## 4. Enforcement Solutions

### 4.1 Provable-Contracts Enforcement (pv)

**New equation for `apr-cli-v1.yaml`**:

```yaml
  parameter_propagation_completeness:
    formula: |
      ∀ field f ∈ Commands::Run struct:
        if f.is_user_facing():
          ∃ corresponding parameter in run::run() OR dispatch_run()
          value(f) is threaded through dispatch chain unchanged
    domain: Commands::Run struct fields (from clap derive)
    codomain: run::run() function arguments
    invariants:
    - Every #[arg] field in Commands::Run has a corresponding dispatch parameter
    - No #[arg] field is destructured but unused (Rust unused-variable = contract violation)
    - Batch mode and single-run mode propagate the same parameter set
    preconditions:
    - Commands::Run struct is well-formed (clap derives successfully)
    postconditions:
    - run::run() receives all user-specified sampling parameters
    - Default values match #[arg(default_value = ...)] annotations exactly
    lean_theorem: Theorems.Parameter_Propagation_Completeness
```

**New falsification test**:

```yaml
- id: FALSIFY-CLI-PARAM-001
  rule: Every Run field reaches run::run()
  prediction: Adding #[arg] field to Commands::Run without dispatch_run() param causes CI failure
  test: |
    #[test]
    fn falsify_param_propagation() {
        // Count #[arg] fields in Commands::Run
        let run_fields = count_struct_fields::<Commands>("Run");
        // Count parameters in dispatch_run() signature
        let dispatch_params = count_fn_params("dispatch_run");
        // dispatch_run has some extra (cli-level) and some missing (prompt merging)
        // but sampling parameters MUST be present
        let sampling_params = ["temperature", "top_k", "top_p", "seed",
                               "repeat_penalty", "repeat_last_n", "split_prompt"];
        for param in &sampling_params {
            assert!(dispatch_params.contains(param),
                "Sampling parameter '{}' missing from dispatch_run()", param);
        }
    }
  if_fails: dispatch_run() silently drops user-specified sampling parameters
```

### 4.2 PMAT Comply Enforcement

**New CB check: CB-1500 — Parameter Propagation Completeness**

```
CB-1500: For each Commands variant with >5 #[arg] fields:
  1. Parse the struct fields from AST
  2. Parse the dispatch function parameters from AST
  3. Verify: unused-variable warnings == 0 for destructured fields
  4. Severity: ERROR (blocks commit)
```

**Implementation**: Add to `pmat comply check` as a new provable-contracts
check that runs `cargo clippy -p apr-cli -- -W unused-variables 2>&1` and
fails if any dispatch.rs destructured field triggers unused-variable warning.

### 4.3 Pre-Commit Hook Enforcement

**Gate: `unused-dispatch-params`**

```bash
# In .pmat-metrics.toml:
[gates.unused-dispatch-params]
command = "cargo clippy -p apr-cli -- -W unused-variables 2>&1 | grep 'dispatch.rs' | grep -c 'unused variable'"
threshold = 0
direction = "at_most"
severity = "error"
```

This makes any unused variable in dispatch.rs a **commit-blocking error**.

### 4.4 Exhaustive Match Enforcement (SF-004)

Replace the catch-all in `dispatch_analysis.rs`:

```rust
// BEFORE (line 223):
_ => return None,

// AFTER:
// No catch-all — compiler enforces exhaustive match.
// Adding a new ExtendedCommands variant without a dispatch arm
// is a compilation error.
```

If a catch-all is needed for feature-gated commands:

```rust
#[cfg(not(feature = "training"))]
ExtendedCommands::Finetune { .. } |
ExtendedCommands::Train { .. } => {
    return Some(Err(CliError::FeatureDisabled("training")));
}
```

### 4.5 Exit Code Contract Enforcement (SF-003)

**Invariant**: No command that prints "error" or "Error" to stderr may return exit code 0.

**pv enforcement equation** for `cli-dispatch-v1.yaml`:

```yaml
  exit_code_error_consistency:
    formula: |
      ∀ cmd, input:
        if stderr(cmd(input)).contains("error") || stderr(cmd(input)).contains("Error"):
          exit_code(cmd(input)) ≠ 0
    invariants:
    - Error messages on stderr imply non-zero exit code
    - Success (exit 0) implies no error text on stderr
    - Warnings on stderr are permitted with exit code 0
```

---

## 5. Cross-Repo Verification Matrix

### 5.1 Contract Coverage by Interface

| Interface | Contract | Structural | Behavioral | Data Flow | Gap |
|-----------|----------|------------|------------|-----------|-----|
| CLI dispatch | cli-dispatch-v1 | ✓ dispatch completeness | ✗ param propagation | ✗ | SF-002, SF-004 |
| CLI exit codes | cli-dispatch-v1 | ✓ exit code mapping | ✗ error→exit consistency | ✗ | SF-003 |
| CLI operations | apr-cli-operations-v1 | ✓ side-effect class | ✗ resource cleanup | ✗ | — |
| HTTP API | http-api-v1 | ✓ schema validation | ✓ error envelope | ✗ | — |
| MCP tools | mcp-tool-schema-v1 | ✓ JSON schema | ✓ tool dispatch | ✗ | — |
| Model graph | apr-model-graph-v1 | ✓ tensor shapes | ✓ forward pass | ✗ | — |
| Kernels | 9 kernel contracts | ✓ shape invariants | ✓ numerical bounds | ✗ | GH-280 |
| Inference | inference.md spec | ✓ sampling equations | ✗ **not enforced** | ✗ | SF-002 |
| Format | apr-format-safety-v1 | ✓ magic bytes | ✓ roundtrip | ✗ | — |

### 5.2 Provable-Contracts ↔ apr-qa Integration

| Repo | Role | Contract Status |
|------|------|-----------------|
| `aprender/contracts/` | 26 YAML contracts, Grade A mean 0.97 | Structural ✓, Behavioral partial |
| `provable-contracts/` | `pv lint`/`pv score` tooling | Gates 1-6 pass, Gate 7 (reverse-coverage) skipped |
| `apr-model-qa-playbook/` | 5 QA contracts (gateway, format, MQS, oracle) | Falsification tests defined |
| `paiml-mcp-agent-toolkit/` | `pmat comply` CB-1200..1214 checks | Enforcement partial (CB-1213 TODO) |

### 5.3 Enforcement Level Gap Analysis

| Level | Mechanism | Covers | Misses |
|-------|-----------|--------|--------|
| L0 | Paper-only | Nothing | Everything |
| L1 | build.rs | Function exists | Signature, behavior |
| L2 | Traits | Function exists + signature | Behavior, data flow |
| L3 | build.rs + traits | Function + signature + build | Behavior, data flow |
| L4 | Kani | Function body bounded check | Cross-function flow |
| L5 | Lean | Unbounded proof | Implementation binding |
| **L-flow** | **MISSING** | — | **CLI parse → dispatch → execute data flow** |

---

## 6. Implementation Roadmap

### Phase 1: Stop the Bleeding (Week 1)

1. **SF-001**: Fix `lib_parse_rosetta.rs` — add missing struct fields
2. **SF-002**: Thread 7 sampling params through `dispatch_run()` → `run::run()`
3. **SF-003**: Fix `inspect`/`validate` to return non-zero exit on error
4. **SF-004**: Remove catch-all in `dispatch_analysis.rs`, use exhaustive match

### Phase 2: Contract Enforcement (Week 2)

5. Add `parameter_propagation_completeness` equation to `apr-cli-v1.yaml`
6. Add `exit_code_error_consistency` equation to `cli-dispatch-v1.yaml`
7. Add pre-commit gate for unused dispatch parameters
8. Add `CB-1500` check to `pmat comply`

### Phase 3: Cross-Repo Verification (Week 3)

9. Enable `pv lint --binding --crate-dir` (Gate 7) in CI
10. Add L-flow verification: AST-based parameter tracing from Commands struct to run()
11. Cross-validate against `apr-model-qa-playbook` falsification protocol
12. Run `pmat comply check` across all 5 stack repos

---

## 7. Toyota Way Principles

### Jidoka (Automation with a Human Touch)
The unused-variable warning IS the Andon cord — Rust already signals the
defect. We failed to wire the Andon cord to the production line (CI gate).
Fix: treat unused variables in dispatch code as hard errors.

### Poka-Yoke (Mistake-Proofing)
Make the wrong thing impossible: if `Commands::Run` gains a new `#[arg]` field,
the dispatch code MUST NOT compile until `dispatch_run()` accepts it. Enforce
via a compile-time assertion or a `#[deny(unused_variables)]` attribute on the
dispatch function.

### Genchi Genbutsu (Go and See)
The falsification protocol tests REAL commands with REAL parameters. We don't
mock the CLI — we run `apr run --temperature 0.9` and verify the output
distribution changes. If it doesn't, the contract is violated.

### Kaizen (Continuous Improvement)
Each silent failure found adds a new falsification test. The test suite grows
monotonically. The pre-commit gate prevents regression. The contract score
can only go up.

---

## 8. References

### Internal
- [cli.md](cli.md) §6.3 — Exit code contract
- [inference.md](inference.md) §5 — Sampling parameter specification
- [provable-contracts.md](provable-contracts.md) — Contract methodology
- [design-by-contract.md](../../design-by-contract.md) — Meyer + Popperian
- [unified-contract-by-design.md](../archive/unified-contract-by-design.md) §1 — GH-280 Five-Whys
- [GH-280-capability-gate.md](../archive/GH-280-capability-gate.md) — Behavioral contract gap

### External (arXiv)
- [SemLoc: Structured Grounding for Fault Localization](https://arxiv.org/html/2603.29109) — Semantic constraint violations as spectrum (2026)
- [Impact of Early Test Termination on SBFL](https://arxiv.org/html/2504.04557) — Slicing improves Tarantula by 10.7% MFR (2025)
- [Spectrum-Based FL Without Fault-Triggering Tests](https://arxiv.org/pdf/2405.00565) — SBEST: 32% MAP improvement (2025)
- [Agent Behavioral Contracts](https://arxiv.org/abs/2602.22302) — Runtime enforcement catches 5.2–6.8 soft violations/session (2026)
- [Taming Silent Failures](https://arxiv.org/html/2510.22224v1) — Contract-based runtime envelope for ML components (2025)
- [Formal Verification Pipeline](https://openaccess.city.ac.uk/id/eprint/34357/1/a%20piramid.pdf) — Compositional layered verification (2025)

### Provable-Contracts
- `apr-cli-v1.yaml` — CLI interface contract (Grade A, 0.98)
- `cli-dispatch-v1.yaml` — Dispatch completeness (Grade A, 0.95)
- `apr-cli-operations-v1.yaml` — Side-effect classification (Grade A, 0.97)
- `pmat comply` CB-1200..1214 — Provable-contracts compliance checks

---

## 9. Self-Falsification (Popperian Audit)

This specification was subjected to Popperian falsification on 2026-04-04.
Every claim was tested against the actual codebase using `pmat query` and
live `apr` binary execution. Results:

### 9.1 Claims Confirmed

| Claim | Evidence | Verdict |
|-------|----------|---------|
| **SF-001**: lib_parse_rosetta.rs blocks compilation | `cargo test -p apr-cli --lib` fails with E0063 (2 errors) | **CONFIRMED → FIXED** (5c46243e) |
| **SF-002**: 7 sampling params dropped in `apr run` | `RunOptions` struct (run.rs:126-157) has NO sampling fields. `dispatch_run()` signature (dispatch_run.rs:4-26) omits all 7. `dispatch.rs:57-62` destructures but never passes them. | **CONFIRMED → FIXED** (8c5078af, b14f2e06) |
| SF-002 batch path uses temp/top_k | `run_batch()` (run_entry.rs:276-284) takes temperature and top_k | **CONFIRMED** — batch path partial, main path total drop |
| Five-whys root cause (L-flow gap) | Provable contracts verify L1-L5 (existence, signature, body) but nothing verifies CLI param → execution data flow | **CONFIRMED** |

### 9.2 Claims Falsified (WRONG)

| Claim | Attempted Falsification | Actual Result | Verdict |
|-------|-------------------------|---------------|---------|
| **SF-003**: inspect/validate exit 0 on error | `apr inspect /tmp/fake.gguf` → exit 4; `apr validate /tmp/fake.gguf` → exit 5 | Both return correct non-zero exit codes | **FALSIFIED** — claim was wrong |
| **SF-004**: catch-all "silently swallows" | `dispatch_analysis.rs:1033` uses `unreachable!()` macro — panics at runtime, does NOT return None silently | Runtime panic, not silent failure | **PARTIALLY FALSIFIED** — still a bug (should be compile error not runtime panic), but characterization as "silent" was wrong |
| **SF-005**: finetune unused params | `finetune.rs:1022` passes `learning_rate` to `dispatch_finetune_mode()`. `plan_only` used 17 times in finetune.rs. | Both parameters ARE used | **FALSIFIED** — claim was wrong |

### 9.3 Corrected Incident Table

| ID | Severity | Status | Description |
|----|----------|--------|-------------|
| SF-001 | P0 | **CONFIRMED** | `lib_parse_rosetta.rs` missing 7 struct fields — blocks lib test suite |
| SF-002 | P1 | **CONFIRMED** | `apr run` drops 7 sampling params (temp, top_k, top_p, seed, repeat_penalty, repeat_last_n, split_prompt) in main path |
| SF-003 | ~~P1~~ | **FALSIFIED** | ~~inspect/validate exit 0 on error~~ — actual: exit 4/5 correctly |
| SF-004 | P2→P3 | **PARTIALLY FALSIFIED** | catch-all is `unreachable!()` (panic), not silent. Still wrong (should be compile-time exhaustive match) but not silent |
| SF-005 | ~~P2~~ | **FALSIFIED** | ~~finetune unused params~~ — both learning_rate and plan_only are used |

### 9.4 Tarantula Score Corrections

Original tarantula scores were computed from sub-agent analysis, not from
direct code execution. Three of five had inflated suspicion scores due to
incorrect premises:

| ID | Original Score | Corrected Score | Reason |
|----|---------------|-----------------|--------|
| SF-001 | 1.00 | 1.00 | Confirmed — blocks test compilation |
| SF-002 | 0.98 | **0.98** | Confirmed — 7 params dropped |
| SF-003 | 0.90 | **0.00** | False positive — exit codes are correct |
| SF-004 | 0.88 | **0.30** | Downgraded — panic not silent, but still bad |
| SF-005 | 0.82 | **0.00** | False positive — params are used |

**Lesson**: Sub-agent tarantula analysis trusted test failure messages
without verifying against running code. 3/5 claims were wrong. Only
direct `pmat query` + live binary execution produces reliable results.

---

## 10. Five-Whys: Why Did the Spec Contain False Claims?

### Why #1: Why were SF-003, SF-004, SF-005 wrong?

The initial investigation used sub-agent reports that analyzed code
structure (grep patterns, unused-variable warnings) without running
the actual binary or tracing the full call chain.

### Why #2: Why were sub-agents wrong?

Sub-agents read dispatch.rs and saw unused variables (correct) but then
inferred behavior from variable names without tracing downstream:
- SF-003: Saw error text in output → assumed exit 0 (didn't test)
- SF-005: Saw `learning_rate` as function param → assumed unused inside (didn't read body)

### Why #3: Why wasn't live verification done first?

The investigation prioritized breadth (test all 51 commands) over depth
(verify each claim). Tarantula scoring was applied to static analysis
results, not dynamic execution traces.

### Why #4: Why is static analysis insufficient for silent-failure detection?

Silent failures are definitionally undetectable from source code alone —
the code compiles, runs, and returns Ok(()). Only behavioral testing
(run command, check output) can distinguish "silently drops parameter"
from "parameter flows through correctly."

### Why #5 (Root Cause): No automated behavioral falsification in CI.

**Root cause**: The project has structural contracts (pv score: A) and
compilation gates (pre-commit) but no **behavioral regression tests**
that run `apr run --temperature 0.9` and verify the output distribution
actually changes. The falsification tests in YAML contracts are
*specified* but not *executed*.

---

## 11. Chain of Thought: What We Actually Know

After falsification, the verified ground truth is:

**CONFIRMED defects:**
1. `lib_parse_rosetta.rs` has 2 compilation errors (7 missing struct
   fields) blocking the entire lib test suite. Fix: add the fields.
   (Already fixed in commit 5c46243e.)

2. `apr run` accepts `--temperature`, `--top-p`, `--seed`,
   `--repeat-penalty`, `--repeat-last-n`, `--split-prompt`, `--top-k`
   from the CLI but **none of these reach the inference engine**:
   - `Commands::Run` struct has all 7 fields (commands_enum.rs:80-100)
   - `dispatch.rs:56-62` destructures all 7 but passes none
   - `dispatch_run()` (dispatch_run.rs:4-26) has 20 params, 0 sampling
   - `run::run()` (run_entry.rs:8-27) has 18 params, 0 sampling
   - `RunOptions` struct (run.rs:126-157) has 0 sampling fields
   - `run_model()` receives `RunOptions` → calls realizar with no sampling config
   - **User sees**: `apr run model.gguf --temperature 0.9 --seed 42` runs
     without error. Output is greedy (temperature=0). No warning printed.
   - **Batch path exception**: `run_batch()` does use temperature + top_k

3. `dispatch_extended_command()` has `_ => unreachable!()` on line 1033
   that will panic at runtime if a new ExtendedCommands variant is added
   without a dispatch arm. Not silent, but should be a compile error.

**FALSIFIED claims (removed from enforcement):**
- ~~SF-003~~: inspect/validate exit codes are correct (exit 4, exit 5)
- ~~SF-005~~: finetune parameters are properly propagated

**Enforcement priority (revised):**
1. **P0**: Fix SF-001 (done) — unblock test suite
2. **P1**: Fix SF-002 — add sampling params to RunOptions, dispatch_run, run::run
3. **P3**: Fix SF-004 — replace unreachable!() with exhaustive match
4. **New P1**: Add behavioral falsification tests that actually RUN commands
   and verify output changes with different parameter values

**Contract gap (confirmed):**
The L-flow gap is real. No enforcement level covers data flow from CLI
parse → struct → dispatch → options → execution. The fix requires either:
- (a) Compile-time: derive RunOptions from Commands::Run (share fields)
- (b) Runtime: behavioral regression tests in CI
- (c) Static analysis: pmat check for unused struct destructuring in dispatch
