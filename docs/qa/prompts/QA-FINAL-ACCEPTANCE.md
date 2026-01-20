# QA FINAL ACCEPTANCE: Qwen2.5 Showcase Demo

**Target Spec**: `docs/specifications/qwen2.5-coder-showcase-demo.md`
**Mandate**: COMPLETE VERIFICATION (Correctness + Performance + UX)
**Policy**: ZERO TOLERANCE. A single failure = **REJECT**.

---

## 🎯 MISSION: DEMO FALSIFICATION
Prove that the `aprender` showcase demo is broken. If you cannot break it, it is ready to ship.

---

## 📦 PHASE 1: COLD START & UX (The First 5 Minutes)

**Objective**: Verify the "Out of Box" experience.

### 1.1 The Clean Install
**Action**:
```bash
# Clean slate
rm -rf ~/.cache/aprender/models/qwen-demo
cargo build --release -p apr-cli
```
**Falsification Checks**:
- [ ] Build fails.
- [ ] `apr --version` returns error or wrong version.

### 1.2 The "Just Works" Pull
**Action**:
```bash
apr pull qwen2.5-coder:1.5b --format apr
```
**Falsification Checks**:
- [ ] Download hangs or is < 10 MB/s.
- [ ] Checksum verification fails.
- [ ] `apr list` does not show the model.

### 1.3 The "Hello World" Chat
**Action**:
```bash
apr chat qwen2.5-coder:1.5b
# Interactive Input: "Write a function to add two numbers."
```
**Falsification Checks**:
- [ ] Panic on startup.
- [ ] Output is garbage/Chinese chars (Regression of CORRECTNESS-012).
- [ ] Model refuses to code ("I don't able").
- [ ] Latency to first token > 500ms.

---

## 🧪 PHASE 2: CORRECTNESS (The Math Check)

**Objective**: Verify the engine is mathematically sound (CPU == GPU).

### 2.1 The Determinism Test
**Action**:
```bash
# Run twice with same seed
apr run qwen2.5-coder:1.5b --prompt "2+2=" --seed 42 --max-tokens 5 > run1.txt
apr run qwen2.5-coder:1.5b --prompt "2+2=" --seed 42 --max-tokens 5 > run2.txt
diff run1.txt run2.txt
```
**Falsification Checks**:
- [ ] Output differs (Non-deterministic).
- [ ] Output does not contain "4".

### 2.2 The Cross-Format Check
**Action**:
```bash
# Pull GGUF variant
apr pull qwen2.5-coder:1.5b --format gguf
# Run comparison
apr run qwen2.5-coder:1.5b --format apr --prompt "test" > apr.txt
apr run qwen2.5-coder:1.5b --format gguf --prompt "test" > gguf.txt
```
**Falsification Checks**:
- [ ] Outputs are semantically different (e.g. "Hello" vs "Goodbye").
- [ ] One format panics.

---

## ⚡ PHASE 3: PERFORMANCE (The 2X Challenge)

**Objective**: Verify we beat Ollama (582 tok/s GPU target).

### 3.1 The GPU Speed Run
**Action**:
```bash
apr run qwen2.5-coder:1.5b --benchmark --device cuda --batch-size 16 --prompt "bench" --max-tokens 500
```
**Falsification Checks**:
- [ ] Throughput < 582 tok/s.
- [ ] GPU Utilization < 80%.

### 3.2 The CPU Fallback
**Action**:
```bash
apr run qwen2.5-coder:1.5b --benchmark --device cpu --prompt "bench" --max-tokens 50
```
**Falsification Checks**:
- [ ] Throughput < 10 tok/s.
- [ ] Segfault or Illegal Instruction (AVX512 issues).

---

## 🌐 PHASE 4: PRODUCTION SERVING

**Objective**: Verify the API server works under load.

### 4.1 The Server Stress Test
**Action**:
```bash
apr serve qwen2.5-coder:1.5b --port 8080 --batch-size 8 & 
PID=$!
sleep 5

# Send 4 concurrent requests
for i in {1..4}; do
  curl -s http://localhost:8080/v1/completions \
    -d '{"prompt": "Hello", "max_tokens": 10}' & 
done
wait
kill $PID
```
**Falsification Checks**:
- [ ] Server crashes.
- [ ] Any request returns 500 error.
- [ ] Throughput drops significantly (Sequential processing instead of batched).

---

## 🕵️ PHASE 5: OBSERVABILITY (The Tracing Check)

**Objective**: Verify structural tracing is active.

### 5.1 The Inspection
**Action**:
```bash
apr inspect qwen2.5-coder:1.5b --trace-structure
```
**Falsification Checks**:
- [ ] `rope_theta` not 1,000,000.0.
- [ ] `bias_loaded` is False.

### 5.2 The Self-Test
**Action**:
```bash
apr check qwen2.5-coder:1.5b
```
**Falsification Checks**:
- [ ] Any step marked "FAIL".
- [ ] Output is missing the 10-step table.

---

## 🏁 RELEASE VERDICT

| Result | Status | Action |
| :--- | :--- | :--- |
| **0 Failures** | 🟢 **APPROVED** | Tag release `v1.0-showcase`. |
| **1+ Failure** | 🔴 **REJECTED** | Fix bug, restart Phase 1. |

**Auditor Signature**: __________________________
**Date**: 2026-01-17
