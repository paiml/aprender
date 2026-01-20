# QA MASTER AUDIT: APR-TRACE-001 (The "Final Boss" Prompt)

**Scope**: Tracing, Inference Correctness, AWS Parity, Multi-Format, Multi-Backend.
**Policy**: ZERO TOLERANCE. A single failure in *any* section = **REJECTION** of the release.

---

## 🎯 AUDIT MISSION
Your goal is to **FALSIFY** the claim that `aprender` provides robust, production-grade observability for LLM inference. You are not testing if it "works"; you are testing if it **breaks** under pressure, variance, and scrutiny.

---

## 🏗️ PHASE 1: STRUCTURAL INTEGRITY (The State Machine)

**Objective**: Prove the traces are disconnected, illogical, or malformed.

### 1.1 The Orphan Hunt (F-AWS-04)
**Hypothesis**: The system leaves "TaskStateEntered" events open without a matching "TaskStateExited".
**Action**:
```bash
apr run models/qwen2.5-0.5b-instruct.gguf --prompt "def hello():" --trace --trace-output trace_struct.json --max-tokens 5
```
**Falsification Checks**:
- [ ] Count of `TaskStateEntered` != Count of `TaskStateExited` (ignoring `ExecutionFailed`).
- [ ] Any `TaskStateExited` has `previous_event_id: null`.
- [ ] Any `TaskStateExited` links to an Entry event of a *different* step name (e.g., `DECODE` exit linking to `EMBED` entry).
- [ ] Timestamp of Exit < Timestamp of Entry.

### 1.2 The Monotonicity Stress Test (F-AWS-01)
**Hypothesis**: IDs skip or reset during complex execution.
**Action**: Run a long generation (> 50 tokens).
**Falsification Checks**:
- [ ] IDs are not strictly `1, 2, 3, 4...`.
- [ ] `execution_arn` is missing or invalid.

---

## 🔌 PHASE 2: THE BACKEND GAUNTLET (Hardware Parity)

**Objective**: Prove that tracing behaves differently (or fails) on different hardware paths.

### 2.1 The CUDA Graph Trap (F-HW-04-B)
**Hypothesis**: CUDA Graphs bypass the CPU-side tracer, resulting in missing steps.
**Action**:
```bash
apr run models/qwen2.5-0.5b-instruct.gguf --prompt "test" --trace --device cuda --enable-graphs
```
**Falsification Checks**:
- [ ] Trace contains gaps (e.g., missing `TRANSFORMER_BLOCK` events).
- [ ] Trace is empty.
- [ ] `apr` crashes / panics.

### 2.2 The "Grand Parity" Drift (F-HW-05)
**Hypothesis**: CPU and GPU traces produce different numerical outputs.
**Action**:
1. Run on CPU: `--device cpu --trace --trace-verbose --trace-output cpu.json --temp 0`
2. Run on CUDA: `--device cuda --trace --trace-verbose --trace-output gpu.json --temp 0`
**Falsification Checks**:
- [ ] `output` tensor values for `EMBED` differ > 1e-4.
- [ ] `output` tensor values for `LM_HEAD` differ > 1e-3.
- [ ] Selected token IDs differ (Determinism failure).

---

## 💾 PHASE 3: THE FORMAT WAR (File Types)

**Objective**: Prove that `.safetensors` or `.apr` are second-class citizens compared to `.gguf`.

### 3.1 The Missing Config (F-PAR-02)
**Hypothesis**: `aprender` panics when `safetensors` lacks a config, instead of tracing the error.
**Action**: Rename `config.json` -> `config.json.bak` and run `apr run model.safetensors --trace`.
**Falsification Checks**:
- [ ] Rust Panic (thread 'main' panicked).
- [ ] Missing `ExecutionFailed` event in the trace.

### 3.2 The APR Bundle Check
**Hypothesis**: The custom `.apr` format tracing logic is desynchronized from the main codebase.
**Action**:
```bash
apr run models/bundle.apr --prompt "test" --trace --trace-output apr.json
```
**Falsification Checks**:
- [ ] Step names differ from `.gguf` (e.g., "Encoder" instead of "TOKENIZE").
- [ ] Metadata in `TaskStateEntered` is missing compared to `.gguf`.

---

## 🔎 PHASE 4: DATA FIDELITY (Genchi Genbutsu)

**Objective**: Prove that the captured data is wrong or misleading.

### 4.1 The Vocab Trap
**Hypothesis**: The tracer reports incorrect token strings.
**Action**: Trace a known prompt: `--prompt "Hello"`.
**Falsification Checks**:
- [ ] `TOKENIZE` output shows IDs that don't map back to "Hello".
- [ ] `DECODE` output shows "<unk>" for standard ASCII.

### 4.2 The Shape Shifter
**Hypothesis**: Recorded tensor shapes are wrong.
**Action**: Check `EMBED` output shape.
**Falsification Checks**:
- [ ] Shape != `[seq_len, hidden_dim]` (e.g., `[1, 1024]`).

---

## 🚦 PHASE 5: ERROR HANDLING (Jidoka)

**Objective**: Prove the system fails silently.

### 5.1 The OOV Injection
**Hypothesis**: The system swallows errors when forced to process invalid tokens.
**Action**: Manually construct a prompt with an ID > Vocab Size (via API or custom harness).
**Falsification Checks**:
- [ ] No `ExecutionFailed` event.
- [ ] "Partial Pass" (program finishes but prints garbage).

### 5.2 The Broken Model
**Hypothesis**: Loading a truncated file with tracing enabled causes a segfault.
**Action**: `head -c 1000 model.gguf > broken.gguf && apr run broken.gguf --trace`.
**Falsification Checks**:
- [ ] Segfault / Bus Error.
- [ ] Panic without a trace log.

---

## 🏁 FINAL VERDICT CRITERIA

| Result | Consequence |
| :--- | :--- |
| **All Checks Pass** | **RELEASE CANDIDATE APPROVED**. |
| **1 Failure** | **REJECT**. Issue ticket must be created. |
| **Panic / Crash** | **CRITICAL REJECT**. Immediate rollback. |

**Auditor Note**: Do not trust the logs. Verify the artifacts. Trust is good, Falsification is better.
