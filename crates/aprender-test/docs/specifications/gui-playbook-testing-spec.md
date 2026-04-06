---
title: "GUI Playbook Testing: State Machine Verification"
version: "1.0.0"
status: "DRAFT"
ticket: "PROBAR-004"
issue_refs: ["#102", "#103", "#104"]
authors: ["Claude Code", "Noah Gift"]
created: "2024-12-14"
citations:
  - "Goldsmith2007"
  - "Fabbri1994"
  - "Mesbah2009"
  - "Leucker2009"
  - "Pesic2007"
---

# GUI Playbook Testing: State Machine Verification

**Version**: 1.0.0
**Status**: DRAFT
**Ticket**: PROBAR-004
**Target**: Declarative YAML + Formal State Machine Verification
**Research Basis**: SCXML (W3C), TLA+ (Lamport), Model Checking (Clarke)
**Falsification**: Popper Protocol for Implementation Validation

---

## 0. Acceptance Criteria

> **MUST** pass all criteria before implementation begins (Popperian threshold: 95/100)

**Acceptance Criteria**:
- [ ] **AC-01**: YAML schema validates all example playbooks (Falsify: Invalid YAML → parser error)
- [ ] **AC-02**: State machine rejects orphaned states (Falsify: Add unreachable state → error)
- [ ] **AC-03**: State machine rejects non-deterministic transitions (Falsify: Ambiguous trigger → error)
- [ ] **AC-04**: Forbidden transitions block at runtime (Falsify: Trigger forbidden → "Forbidden transition")
- [ ] **AC-05**: Performance budgets enforced per transition (Falsify: Exceed `max_time` → "exceeded budget")
- [ ] **AC-06**: O(n) complexity verified empirically (Falsify: Inject O(n²) → "complexity violation")
- [ ] **AC-07**: State invariants checked on entry/exit (Falsify: Violate invariant → "Invariant violation")
- [ ] **AC-08**: Playbook steps execute in declared order (Falsify: Log step order → matches YAML order)
- [ ] **AC-09**: Captured variables accessible in assertions (Falsify: Reference `${var}` → value substituted)
- [ ] **AC-10**: SVG state diagram exports valid SVG (Falsify: Export → renders in browser)
- [ ] **AC-11**: JSON execution trace is valid JSON (Falsify: Export → `jq .` succeeds)
- [ ] **AC-12**: CI exit code 0 on success, 1 on failure (Falsify: Fail test → exit 1)
- [ ] **AC-13**: Setup runs before steps, teardown after (Falsify: Mock actions → correct order)
- [ ] **AC-14**: Teardown runs even on failure (Falsify: Fail mid-test → teardown executes)
- [ ] **AC-15**: All 5 mutation classes (M1-M5) detected (Falsify: Inject each → corresponding error)

---

## 1. Overview

A **Playbook** is a declarative YAML specification that combines:
1. **State Machine Definition** — Allowed states and transitions
2. **Sequential Test Steps** — Actions that trigger transitions
3. **Performance Criteria** — Time/memory budgets per transition
4. **Complexity Assertions** — O(n) invariants on operations
5. **Falsification Hooks** — Entry points for validity testing

### 1.1 Design Principles

| Principle | Implementation |
|-----------|----------------|
| **Declarative** | YAML schema, no imperative code |
| **Verifiable** | State machine is formally checkable |
| **Falsifiable** | Every assertion can be proven wrong |
| **Observable** | All transitions emit metrics |
| **Deterministic** | Same input → same state path |

---

## 2. YAML Schema

### 2.1 Complete Example

```yaml
# File: transcription.playbook.yaml
version: "1.0"
name: "Realtime Transcription E2E"
description: "Verify transcription pipeline state transitions"

# ═══════════════════════════════════════════════════════════════════
# MACHINE DEFINITION
# ═══════════════════════════════════════════════════════════════════
machine:
  id: TranscriptionPipeline
  initial: uninitialized

  # Global performance budget (falsifiable)
  performance:
    total_time: 30s
    peak_memory: 200MB
    complexity: O(n)  # n = audio_samples / chunk_size

  # State definitions
  states:
    uninitialized:
      description: "WASM module not yet loaded"

    loading:
      description: "Model weights being loaded"
      invariant: "memory_used < 150MB"

    ready:
      description: "Model loaded, awaiting input"
      invariant: "is_model_loaded == true"

    processing:
      description: "Transcribing audio chunk"
      invariant: "chunks_pending >= 0"

    completed:
      description: "Transcription finished"
      invariant: "transcript.length > 0"

    error:
      description: "Unrecoverable failure"
      terminal: true

  # Transition definitions with performance criteria
  transitions:
    - id: t1
      from: uninitialized
      to: loading
      trigger:
        wasm: init_test_pipeline
        args: [16000]
      expect:
        returns: true
      performance:
        max_time: 2s

    - id: t2
      from: loading
      to: ready
      wait:
        wasm: is_model_loaded
        returns: true
        poll_interval: 100ms
      performance:
        max_time: 10s
        memory_delta: "<100MB"

    - id: t3
      from: ready
      to: processing
      trigger:
        wasm: process_test_chunk
      expect:
        returns: true
      performance:
        max_time: 100ms
        complexity: O(1)  # Dispatch is constant time

    - id: t4
      from: processing
      to: completed
      wait:
        wasm: get_transcript
        condition: not_empty
        poll_interval: 500ms
      performance:
        max_time: 20s
        complexity: O(n)  # Linear in tokens

    - id: t5
      from: [loading, processing]
      to: error
      on: exception

  # Forbidden transitions (must never occur)
  forbidden:
    - from: ready
      to: error
      reason: "Ready state must be stable"
    - from: completed
      to: processing
      reason: "No reprocessing after completion"

# ═══════════════════════════════════════════════════════════════════
# PLAYBOOK STEPS
# ═══════════════════════════════════════════════════════════════════
playbook:
  setup:
    - action: { wasm: inject_test_audio, args: ["${TEST_AUDIO_PATH}"] }
      description: "Load test audio samples"

  steps:
    - name: "Initialize Pipeline"
      transitions: [t1, t2]  # uninitialized -> loading -> ready
      timeout: 15s

    - name: "Process Audio"
      transitions: [t3, t4]  # ready -> processing -> completed
      timeout: 25s
      capture:
        - { var: transcript, from: "get_transcript()" }
        - { var: duration_ms, from: "get_processing_time()" }

  teardown:
    - action: { wasm: cleanup }
      ignore_errors: true

# ═══════════════════════════════════════════════════════════════════
# ASSERTIONS
# ═══════════════════════════════════════════════════════════════════
assertions:
  # Path assertion — exact state sequence
  path:
    expected: [uninitialized, loading, ready, processing, completed]

  # Output assertions
  output:
    - var: transcript
      not_empty: true
    - var: transcript
      matches: "\\w+"  # Contains at least one word
    - var: duration_ms
      less_than: 20000

  # Complexity assertion — O(n) verified by sampling
  complexity:
    operation: "transcribe"
    measure: duration_ms
    input_var: audio_samples
    expected: O(n)
    tolerance: 0.2  # 20% variance allowed
    sample_sizes: [1000, 5000, 10000, 50000]

# ═══════════════════════════════════════════════════════════════════
# FALSIFICATION PROTOCOL
# ═══════════════════════════════════════════════════════════════════
falsification:
  # Mutation targets — implementation MUST detect these
  mutations:
    - id: M1
      description: "Skip loading state"
      mutate: "remove transition t1"
      expected_failure: "Invalid transition: uninitialized -> ready"

    - id: M2
      description: "Exceed time budget"
      mutate: "set t2.performance.max_time = 1ms"
      expected_failure: "Transition t2 exceeded time budget"

    - id: M3
      description: "Violate state invariant"
      mutate: "set ready.invariant = 'is_model_loaded == false'"
      expected_failure: "Invariant violation in state: ready"

    - id: M4
      description: "Trigger forbidden transition"
      mutate: "inject error after ready state"
      expected_failure: "Forbidden transition: ready -> error"

    - id: M5
      description: "Violate O(n) complexity"
      mutate: "replace O(n) algorithm with O(n²)"
      expected_failure: "Complexity violation: expected O(n), measured O(n²)"

  # Property-based fuzzing
  properties:
    - id: P1
      description: "All paths terminate"
      property: "forall path: reaches(terminal_state) OR reaches(error)"

    - id: P2
      description: "No state is orphaned"
      property: "forall state: reachable(initial, state)"

    - id: P3
      description: "Deterministic transitions"
      property: "forall (state, input): |next_states| <= 1"

  # Chaos injection
  chaos:
    - id: C1
      description: "Random delay injection"
      inject: "delay(random(0, 5000ms)) before each transition"
      expect: "Still completes within total_time * 2"

    - id: C2
      description: "Memory pressure"
      inject: "allocate 100MB before processing"
      expect: "peak_memory assertion still holds OR graceful OOM"
```

---

## 3. Formal Semantics

### 3.1 State Machine Definition

A Playbook State Machine is a 7-tuple:

```
M = (S, Σ, δ, s₀, F, I, P)

Where:
  S = finite set of states
  Σ = alphabet of triggers (WASM calls, events)
  δ = transition function: S × Σ → S
  s₀ = initial state
  F = set of terminal states
  I = invariant function: S → Boolean expression
  P = performance function: δ → (max_time, max_memory, complexity)
```

### 3.2 Transition Semantics

```
TRANSITION(from, to, trigger, performance):
  PRE:
    - current_state == from
    - I(from) holds

  EXECUTE:
    - start_time = now()
    - start_memory = heap_size()
    - result = trigger()
    - end_time = now()
    - end_memory = heap_size()

  POST:
    - current_state = to
    - I(to) holds
    - (end_time - start_time) <= performance.max_time
    - (end_memory - start_memory) <= performance.memory_delta
    - complexity(trigger) == performance.complexity

  ON_FAILURE:
    - IF (from, to) in forbidden: FAIL("Forbidden transition")
    - IF timeout: FAIL("Transition timeout")
    - IF invariant_violation: FAIL("Invariant violation")
    - ELSE: transition to error state
```

### 3.3 Complexity Verification

Complexity is verified empirically using curve fitting:

```
VERIFY_COMPLEXITY(operation, expected_O, samples):
  measurements = []

  FOR n IN samples:
    input = generate_input(size=n)
    time = measure(operation(input))
    measurements.append((n, time))

  # Fit curves
  fit_constant = fit(measurements, λn.c)
  fit_linear = fit(measurements, λn.a*n + b)
  fit_quadratic = fit(measurements, λn.a*n² + b*n + c)
  fit_log = fit(measurements, λn.a*log(n) + b)

  # Select best fit
  best_fit = argmin_r²([fit_constant, fit_linear, fit_quadratic, fit_log])

  # Verify matches expected
  ASSERT best_fit.complexity == expected_O
```

---

## 4. Falsification Protocol

### 4.1 Theoretical Basis

Following Popper's falsificationism, a specification is valid only if:
1. It makes **testable predictions**
2. Those predictions can be **proven false**
3. Attempts to falsify it **fail**

### 4.2 Mutation-Based Falsification

Each mutation MUST cause a detectable failure:

| Mutation Class | Target | Detection Method |
|----------------|--------|------------------|
| **State Removal** | Remove a state from S | Unreachable state error |
| **Transition Removal** | Remove edge from δ | Invalid transition error |
| **Invariant Negation** | Negate I(s) | Invariant violation |
| **Performance Tightening** | Halve time budgets | Timeout error |
| **Complexity Degradation** | Replace O(n) with O(n²) | Complexity mismatch |
| **Forbidden Injection** | Force forbidden transition | Forbidden transition error |

### 4.3 Implementation Test Suite

```rust
#[cfg(test)]
mod falsification_tests {
    use super::*;

    /// M1: Specification MUST reject missing transitions
    #[test]
    fn falsify_missing_transition() {
        let mut spec = load_playbook("transcription.playbook.yaml");
        spec.machine.transitions.remove("t1");

        let result = run_playbook(&spec);

        assert!(result.is_err());
        assert!(result.unwrap_err().contains("Invalid transition"));
    }

    /// M2: Specification MUST enforce time budgets
    #[test]
    fn falsify_time_budget() {
        let mut spec = load_playbook("transcription.playbook.yaml");
        spec.machine.transitions["t2"].performance.max_time = Duration::from_millis(1);

        let result = run_playbook(&spec);

        assert!(result.is_err());
        assert!(result.unwrap_err().contains("exceeded time budget"));
    }

    /// M3: Specification MUST enforce invariants
    #[test]
    fn falsify_invariant() {
        let mut spec = load_playbook("transcription.playbook.yaml");
        spec.machine.states["ready"].invariant = "is_model_loaded == false";

        let result = run_playbook(&spec);

        assert!(result.is_err());
        assert!(result.unwrap_err().contains("Invariant violation"));
    }

    /// M4: Specification MUST block forbidden transitions
    #[test]
    fn falsify_forbidden_transition() {
        let mut spec = load_playbook("transcription.playbook.yaml");
        // Inject an error trigger after ready state
        spec.chaos.inject_error_after("ready");

        let result = run_playbook(&spec);

        assert!(result.is_err());
        assert!(result.unwrap_err().contains("Forbidden transition: ready -> error"));
    }

    /// M5: Specification MUST detect complexity violations
    #[test]
    fn falsify_complexity() {
        let mut spec = load_playbook("transcription.playbook.yaml");
        // Replace O(n) transcribe with O(n²) implementation
        spec.inject_on2_transcribe();

        let result = run_playbook(&spec);

        assert!(result.is_err());
        assert!(result.unwrap_err().contains("Complexity violation"));
    }

    /// P1: All paths must terminate
    #[test]
    fn property_all_paths_terminate() {
        let spec = load_playbook("transcription.playbook.yaml");
        let machine = spec.machine.to_formal();

        for path in machine.enumerate_paths(max_depth: 100) {
            assert!(
                path.ends_with_terminal() || path.ends_with_error(),
                "Non-terminating path: {:?}", path
            );
        }
    }

    /// P2: No orphaned states
    #[test]
    fn property_no_orphaned_states() {
        let spec = load_playbook("transcription.playbook.yaml");
        let machine = spec.machine.to_formal();

        for state in &machine.states {
            assert!(
                machine.is_reachable(machine.initial, state),
                "Orphaned state: {}", state
            );
        }
    }

    /// P3: Deterministic transitions
    #[test]
    fn property_deterministic() {
        let spec = load_playbook("transcription.playbook.yaml");
        let machine = spec.machine.to_formal();

        for state in &machine.states {
            for trigger in &machine.alphabet {
                let next_states = machine.next(state, trigger);
                assert!(
                    next_states.len() <= 1,
                    "Non-deterministic: {} --{}--> {:?}", state, trigger, next_states
                );
            }
        }
    }
}
```

### 4.4 Falsification Coverage Matrix

Implementation is valid when ALL mutations are caught:

| Mutation ID | Description | Caught? | Error Message |
|-------------|-------------|---------|---------------|
| M1 | Skip loading state | ☐ | |
| M2 | Exceed time budget | ☐ | |
| M3 | Violate state invariant | ☐ | |
| M4 | Trigger forbidden transition | ☐ | |
| M5 | Violate O(n) complexity | ☐ | |
| P1 | Non-terminating path | ☐ | |
| P2 | Orphaned state | ☐ | |
| P3 | Non-deterministic transition | ☐ | |

**Validity Threshold**: 100% of mutations must be caught.

---

## 5. Runtime Architecture

### 5.1 Components

```
┌─────────────────────────────────────────────────────────────────┐
│                      Playbook Runner                            │
├─────────────────────────────────────────────────────────────────┤
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────────┐  │
│  │ YAML Parser │→ │ State Machine│→ │ Transition Executor    │  │
│  └─────────────┘  │ Validator    │  │ (with metrics)         │  │
│                   └─────────────┘  └─────────────────────────┘  │
│                          ↓                    ↓                 │
│  ┌─────────────────────────────────────────────────────────────┐│
│  │                  Invariant Checker                          ││
│  │  - State invariants (per-state boolean expressions)         ││
│  │  - Transition performance (time, memory, complexity)        ││
│  │  - Forbidden transition detector                            ││
│  └─────────────────────────────────────────────────────────────┘│
│                          ↓                                      │
│  ┌─────────────────────────────────────────────────────────────┐│
│  │                  Reporter / Exporter                        ││
│  │  - State diagram (SVG/DOT)                                  ││
│  │  - Execution trace (JSON)                                   ││
│  │  - Performance metrics (Prometheus format)                  ││
│  │  - Falsification coverage matrix                            ││
│  └─────────────────────────────────────────────────────────────┘│
└─────────────────────────────────────────────────────────────────┘
```

### 5.2 Execution Flow

```
1. PARSE: Load YAML → PlaybookSpec
2. VALIDATE: Check state machine is well-formed (no orphans, deterministic)
3. SETUP: Run setup actions
4. EXECUTE: For each step:
   a. For each transition in step.transitions:
      - Check current state matches transition.from
      - Execute trigger OR wait for condition
      - Measure time, memory
      - Verify invariant on new state
      - Check not forbidden
      - Record metrics
5. ASSERT: Verify path, outputs, complexity
6. TEARDOWN: Run teardown actions
7. REPORT: Generate outputs
```

---

## 6. CLI Interface

```bash
# Run a playbook
probar playbook run transcription.playbook.yaml

# Validate a playbook (static checks only)
probar playbook validate transcription.playbook.yaml

# Run with falsification tests
probar playbook falsify transcription.playbook.yaml

# Generate state diagram
probar playbook diagram transcription.playbook.yaml -o states.svg

# Export execution trace
probar playbook run transcription.playbook.yaml --trace trace.json

# Run with chaos injection
probar playbook run transcription.playbook.yaml --chaos C1,C2
```

---

## 7. Integration with Existing Probar

### 7.1 Module Location

```
crates/probar/src/
├── playbook/
│   ├── mod.rs           # Public API
│   ├── schema.rs        # YAML schema types
│   ├── parser.rs        # YAML → PlaybookSpec
│   ├── validator.rs     # Static validation
│   ├── machine.rs       # State machine runtime
│   ├── executor.rs      # Transition executor
│   ├── complexity.rs    # O(n) verification
│   ├── falsify.rs       # Falsification protocol
│   └── export.rs        # SVG, JSON, Prometheus
```

### 7.2 Reused Components

| Existing Component | Usage in Playbook |
|--------------------|-------------------|
| `driver.rs` | WASM function calls |
| `assertion/retry.rs` | Wait conditions with polling |
| `perf/metrics.rs` | Time/memory measurement |
| `perf/trace.rs` | Execution trace export |
| `reporter.rs` | HTML/JSON report generation |

---

## 8. Example: whisper.apr Integration

```yaml
# demos/tests/playbooks/transcription.playbook.yaml
version: "1.0"
name: "Whisper.apr Realtime Transcription"

machine:
  id: WhisperTranscription
  initial: uninitialized

  performance:
    total_time: 60s
    peak_memory: 350MB  # Base model limit
    complexity: O(n)

  states:
    uninitialized:
    model_loading:
      invariant: "memory_used < 200MB"
    ready:
      invariant: "is_worker_ready == true && is_model_loaded == true"
    processing:
      invariant: "chunks_pending >= 0"
    transcribed:
      invariant: "transcript.length > 0"
    error:
      terminal: true

  transitions:
    - from: uninitialized
      to: model_loading
      trigger: { wasm: init_test_pipeline, args: [16000] }
      performance: { max_time: 2s }

    - from: model_loading
      to: ready
      wait: { wasm: is_worker_ready, returns: true, poll_interval: 100ms }
      performance: { max_time: 15s, memory_delta: "<150MB" }

    - from: ready
      to: processing
      trigger: { wasm: process_test_chunk }
      performance: { max_time: 100ms, complexity: O(1) }

    - from: processing
      to: transcribed
      wait: { wasm: get_transcript, condition: not_empty, poll_interval: 500ms }
      performance: { max_time: 30s, complexity: O(n) }

  forbidden:
    - { from: ready, to: error, reason: "Model must be stable after load" }

playbook:
  setup:
    - { wasm: inject_test_audio, args: ["${HELLO_WORLD_AUDIO}"] }

  steps:
    - name: "Load Model"
      transitions: [uninitialized->model_loading, model_loading->ready]

    - name: "Transcribe"
      transitions: [ready->processing, processing->transcribed]
      capture: [{ var: result, from: "get_transcript()" }]

assertions:
  path: [uninitialized, model_loading, ready, processing, transcribed]
  output:
    - { var: result, not_empty: true }
    - { var: result, not_contains: "Error" }
```

---

## 9. Success Criteria

| Criterion | Metric | Target |
|-----------|--------|--------|
| Falsification coverage | Mutations caught | 100% |
| Spec parsing | Valid YAML accepted | 100% |
| Invalid spec rejection | Malformed YAML rejected | 100% |
| Performance accuracy | Time measurement error | < 5% |
| Complexity detection | O(n²) detected when present | 100% |
| State diagram export | Valid SVG generated | 100% |

---

## 10. References

### 10.1 Standards & Textbooks

[1] W3C. **"State Chart XML (SCXML): State Machine Notation for Control Abstraction."** *W3C Recommendation*, 2015. https://www.w3.org/TR/scxml/

[2] Lamport, L. **"Specifying Systems: The TLA+ Language and Tools for Hardware and Software Engineers."** *Addison-Wesley*, 2002. ISBN: 978-0321143068

[3] Clarke, E. M., Grumberg, O., & Peled, D. A. **"Model Checking."** *MIT Press*, 1999. ISBN: 978-0262032704

[4] Popper, K. **"The Logic of Scientific Discovery."** *Routledge*, 1959. ISBN: 978-0415278447

### 10.2 Peer-Reviewed Research

[5] Goldsmith, S. F., Aiken, A. S., & Wilkerson, D. S. **"Measuring Empirical Computational Complexity."** *ESEC/FSE 2007*, pp. 395-404. https://doi.org/10.1145/1287624.1287681

[6] Fabbri, S. C. P. F., Maldonado, J. C., Sugeta, T., & Masiero, P. C. **"Mutation Testing Applied to Validate Specifications Based on Statecharts."** *ISSRE 1999*, pp. 210-219. https://doi.org/10.1109/ISSRE.1999.809326

[7] Mesbah, A., & van Deursen, A. **"Invariant-Based Automatic Testing of AJAX User Interfaces."** *ICSE 2009*, pp. 210-220. https://doi.org/10.1109/ICSE.2009.5070522

[8] Leucker, M., & Schallhart, C. **"A Brief Account of Runtime Verification."** *Journal of Logic and Algebraic Programming 78(5)*, 2009, pp. 293-303. https://doi.org/10.1016/j.jlap.2008.08.004

[9] Pesic, M., Schonenberg, H., & van der Aalst, W. M. P. **"DECLARE: Full Support for Loosely-Structured Processes."** *EDOC 2007*, pp. 287-298. https://doi.org/10.1109/EDOC.2007.14

---

## 11. Post-Completion QA Checklist (100 Points)

**Verification Method**: Each item is verified through **falsification** — inject the defect, confirm detection.

### 11.1 YAML Schema Validation (15 points)

| # | Check | Falsification Test | ☐ |
|---|-------|-------------------|---|
| 1 | Valid YAML parses without error | Inject syntax error → parser rejects | ☐ |
| 2 | Missing `version` field rejected | Remove `version` → schema error | ☐ |
| 3 | Missing `machine.id` rejected | Remove `machine.id` → schema error | ☐ |
| 4 | Missing `machine.initial` rejected | Remove `machine.initial` → schema error | ☐ |
| 5 | Empty `states` rejected | Set `states: {}` → validation error | ☐ |
| 6 | Empty `transitions` rejected | Set `transitions: []` → validation error | ☐ |
| 7 | Invalid state reference rejected | Reference non-existent state → error | ☐ |
| 8 | Duplicate state IDs rejected | Add duplicate state → error | ☐ |
| 9 | Duplicate transition IDs rejected | Add duplicate transition → error | ☐ |
| 10 | Invalid duration format rejected | Set `max_time: "foo"` → parse error | ☐ |
| 11 | Invalid memory format rejected | Set `max_memory: "bar"` → parse error | ☐ |
| 12 | Invalid complexity rejected | Set `complexity: O(n³)` → validation error | ☐ |
| 13 | Circular `initial` state rejected | Set `initial` to terminal → error | ☐ |
| 14 | Self-loop without explicit allow rejected | Add `A -> A` without flag → warning | ☐ |
| 15 | Unknown fields rejected (strict mode) | Add `foo: bar` → strict mode error | ☐ |

### 11.2 State Machine Semantics (20 points)

| # | Check | Falsification Test | ☐ |
|---|-------|-------------------|---|
| 16 | Initial state is reachable | Remove all transitions to initial → N/A (always reachable) | ☐ |
| 17 | All states reachable from initial | Add orphan state → "unreachable state" error | ☐ |
| 18 | Terminal states have no outgoing | Add transition from terminal → error | ☐ |
| 19 | Deterministic transitions enforced | Add two transitions same (from, trigger) → error | ☐ |
| 20 | Transition `from` must exist | Reference missing state in `from` → error | ☐ |
| 21 | Transition `to` must exist | Reference missing state in `to` → error | ☐ |
| 22 | Forbidden transitions block execution | Trigger forbidden path → "forbidden" error | ☐ |
| 23 | Multiple `from` states supported | Use `from: [A, B]` → both work | ☐ |
| 24 | Wildcard `from: "*"` supported | Use `from: "*"` → matches any | ☐ |
| 25 | State invariants checked on entry | Violate invariant on entry → error | ☐ |
| 26 | State invariants checked on exit | Violate invariant on exit → error | ☐ |
| 27 | Invariant syntax errors rejected | Invalid expression → parse error | ☐ |
| 28 | Empty invariant treated as `true` | Omit invariant → no error | ☐ |
| 29 | Nested state machines supported | Define child machine → works | ☐ |
| 30 | Parallel states supported | Define `parallel: true` → concurrent execution | ☐ |
| 31 | History states supported | Define `history: shallow` → restores | ☐ |
| 32 | Guard conditions evaluated | Add `guard: x > 0` → blocks when false | ☐ |
| 33 | Guard failure logged | Guard fails → log entry created | ☐ |
| 34 | Actions on entry executed | Define `on_entry` → action runs | ☐ |
| 35 | Actions on exit executed | Define `on_exit` → action runs | ☐ |

### 11.3 Transition Execution (15 points)

| # | Check | Falsification Test | ☐ |
|---|-------|-------------------|---|
| 36 | Trigger WASM function called | Mock WASM → verify call received | ☐ |
| 37 | Trigger arguments passed correctly | Check args in mock → match spec | ☐ |
| 38 | Trigger return value captured | Return value → available in context | ☐ |
| 39 | Wait condition polled at interval | Set `poll_interval: 100ms` → verify timing | ☐ |
| 40 | Wait timeout enforced | Set short timeout → timeout error | ☐ |
| 41 | Wait success stops polling | Condition true → polling stops | ☐ |
| 42 | Exception during trigger → error state | Throw in WASM → transitions to error | ☐ |
| 43 | Retry policy respected | Set `retries: 3` → retries 3 times | ☐ |
| 44 | Retry backoff respected | Set `backoff: exponential` → delays grow | ☐ |
| 45 | Transition timeout independent of wait | Transition timeout > wait timeout → works | ☐ |
| 46 | Captured variables stored | `capture: [x]` → `x` available later | ☐ |
| 47 | Captured variables typed correctly | Capture number → type is number | ☐ |
| 48 | Transition events emitted | Subscribe → receive transition event | ☐ |
| 49 | Transition metrics recorded | Query metrics → transition recorded | ☐ |
| 50 | Transition duration measured | Check duration → non-zero value | ☐ |

### 11.4 Performance Verification (15 points)

| # | Check | Falsification Test | ☐ |
|---|-------|-------------------|---|
| 51 | `max_time` enforced per transition | Exceed time → "exceeded time budget" | ☐ |
| 52 | `total_time` enforced globally | Exceed total → "total time exceeded" | ☐ |
| 53 | `memory_delta` enforced | Exceed delta → "memory budget exceeded" | ☐ |
| 54 | `peak_memory` enforced | Exceed peak → "peak memory exceeded" | ☐ |
| 55 | Time measurement accuracy < 5% | Compare to wall clock → within 5% | ☐ |
| 56 | Memory measurement accuracy < 10% | Compare to OS → within 10% | ☐ |
| 57 | O(1) complexity verified | Inject O(n) → "complexity violation" | ☐ |
| 58 | O(n) complexity verified | Inject O(n²) → "complexity violation" | ☐ |
| 59 | O(log n) complexity verified | Inject O(n) → "complexity violation" | ☐ |
| 60 | O(n log n) complexity verified | Inject O(n²) → "complexity violation" | ☐ |
| 61 | Complexity tolerance respected | Within tolerance → pass | ☐ |
| 62 | Complexity sample sizes configurable | Set `sample_sizes: [100, 1000]` → uses those | ☐ |
| 63 | JIT warmup excluded from timing | First run excluded → stable timing | ☐ |
| 64 | GC pauses handled | GC during test → timing adjusted or noted | ☐ |
| 65 | Performance metrics exportable | Export → valid Prometheus format | ☐ |

### 11.5 Playbook Execution (10 points)

| # | Check | Falsification Test | ☐ |
|---|-------|-------------------|---|
| 66 | Setup actions run before steps | Mock setup action → called first | ☐ |
| 67 | Teardown actions run after steps | Mock teardown → called last | ☐ |
| 68 | Teardown runs on failure | Fail mid-test → teardown still runs | ☐ |
| 69 | Steps execute in order | Log order → matches spec order | ☐ |
| 70 | Step timeout enforced | Exceed step timeout → error | ☐ |
| 71 | Step can span multiple transitions | Step with 3 transitions → all execute | ☐ |
| 72 | Variable interpolation works | Use `${VAR}` → replaced with value | ☐ |
| 73 | Environment variables accessible | Use `${env.HOME}` → replaced | ☐ |
| 74 | Missing variable → clear error | Use `${MISSING}` → "undefined variable" | ☐ |
| 75 | Conditional steps supported | `if: condition` → skips when false | ☐ |

### 11.6 Assertions (10 points)

| # | Check | Falsification Test | ☐ |
|---|-------|-------------------|---|
| 76 | Path assertion exact match | Wrong path → "path mismatch" | ☐ |
| 77 | Path assertion prefix match | Prefix mode + partial path → pass | ☐ |
| 78 | Output `not_empty` works | Empty output → assertion fails | ☐ |
| 79 | Output `contains` works | Missing substring → assertion fails | ☐ |
| 80 | Output `matches` regex works | Non-matching → assertion fails | ☐ |
| 81 | Output `equals` works | Different value → assertion fails | ☐ |
| 82 | Output `less_than` works | Greater value → assertion fails | ☐ |
| 83 | Output `greater_than` works | Lesser value → assertion fails | ☐ |
| 84 | Multiple assertions AND'd | One fails → overall fails | ☐ |
| 85 | Soft assertions collect all | Multiple fail → all reported | ☐ |

### 11.7 Falsification Protocol (10 points)

| # | Check | Falsification Test | ☐ |
|---|-------|-------------------|---|
| 86 | M1: Missing transition detected | Remove transition → error detected | ☐ |
| 87 | M2: Time budget violation detected | Tighten budget → error detected | ☐ |
| 88 | M3: Invariant violation detected | Negate invariant → error detected | ☐ |
| 89 | M4: Forbidden transition detected | Force forbidden → error detected | ☐ |
| 90 | M5: Complexity violation detected | Inject O(n²) → error detected | ☐ |
| 91 | P1: Non-terminating path detected | Add cycle → error or warning | ☐ |
| 92 | P2: Orphaned state detected | Add unreachable → error detected | ☐ |
| 93 | P3: Non-determinism detected | Add ambiguous → error detected | ☐ |
| 94 | Chaos delay injection works | Inject delay → still completes | ☐ |
| 95 | Chaos memory pressure works | Inject allocation → handled gracefully | ☐ |

### 11.8 Export & Reporting (5 points)

| # | Check | Falsification Test | ☐ |
|---|-------|-------------------|---|
| 96 | SVG state diagram valid | Export → valid SVG that renders | ☐ |
| 97 | DOT format valid | Export → valid DOT that graphviz accepts | ☐ |
| 98 | JSON trace valid | Export → valid JSON, parseable | ☐ |
| 99 | HTML report generated | Export → valid HTML with all data | ☐ |
| 100 | CI exit code correct | Failure → exit 1; Success → exit 0 | ☐ |

---

### 11.9 Scoring

| Score | Grade | Status |
|-------|-------|--------|
| 100/100 | A+ | Production Ready |
| 95-99 | A | Release Candidate |
| 90-94 | B | Beta |
| 80-89 | C | Alpha |
| < 80 | F | Not Ready |

**Minimum for release**: 95/100 with all items in sections 11.6 (Assertions) and 11.7 (Falsification) passing.

---

### 11.10 Falsification Verification Script

```bash
#!/bin/bash
# verify-falsification.sh — Run all 100 QA checks

set -e

PASS=0
FAIL=0

check() {
    local id=$1
    local desc=$2
    local cmd=$3

    echo -n "[$id] $desc... "
    if eval "$cmd" > /dev/null 2>&1; then
        echo "PASS"
        ((PASS++))
    else
        echo "FAIL"
        ((FAIL++))
    fi
}

# Section 11.1: Schema Validation
check "001" "Valid YAML parses" "probar playbook validate tests/valid.playbook.yaml"
check "002" "Missing version rejected" "! probar playbook validate tests/no-version.playbook.yaml"
check "003" "Missing machine.id rejected" "! probar playbook validate tests/no-machine-id.playbook.yaml"
# ... (remaining 97 checks)

echo ""
echo "═══════════════════════════════════════"
echo "RESULTS: $PASS passed, $FAIL failed"
echo "SCORE: $PASS / 100"
echo "═══════════════════════════════════════"

if [ $PASS -ge 95 ]; then
    echo "STATUS: RELEASE CANDIDATE"
    exit 0
else
    echo "STATUS: NOT READY"
    exit 1
fi
```

