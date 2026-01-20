# QA SHOWCASE AUDIT: Qwen2.5-Coder (The "Just Works" Gate)

**Target Spec**: `docs/specifications/qwen2.5-coder-showcase-demo.md`
**Mandate**: UNIFIED INFERENCE STRATEGY (Just Works + 2X Speed)
**Audience**: QA Engineers / Automated Agents

---

## 🎯 AUDIT MISSION
Prove that `aprender` delivers a **seamless, high-performance** experience for the Qwen2.5-Coder model across all formats and hardware. If *any* command fails, hangs, or produces garbage, the release is **REJECTED**.

---

## 🏗️ PHASE 1: "JUST WORKS" (Coherence & UX)

**Objective**: Verify the model speaks English/Code, not garbage.

### 1.1 The "Hello World" Test (GGUF)
**Action**:
```bash
apr run models/qwen2.5-coder-1.5b-instruct-q4_k_m.gguf --prompt "Write a Python function to add two numbers." --max-tokens 50
```
**Falsification Criteria**:
- [ ] Output contains non-code/non-English characters.
- [ ] Output is empty.
- [ ] Output does not contain `def` or `return`.

### 1.2 The "Native Format" Test (APR)
**Action**:
```bash
apr run models/qwen2.5-coder-1.5b.apr --prompt "What is 2+2?" --max-tokens 10
```
**Falsification Criteria**:
- [ ] Output does not contain "4".
- [ ] Panic or "Format Error".

### 1.3 The "Interop" Test (SafeTensors)
**Action**:
```bash
apr run models/qwen2.5-coder-1.5b-safetensors/model.safetensors --prompt "Hello" --max-tokens 10
```
**Falsification Criteria**:
- [ ] "Config not found" error (should be handled gracefully).
- [ ] Output is garbage.

### 1.4 The "Polyglot" Test (UTF-8)
**Action**:
```bash
apr chat qwen2.5-coder:1.5b
# Input: "Write 'Hello' in Chinese."
```
**Falsification Checks**:
- [ ] Output contains mojibake (e.g., "ä½łåı¯").
- [ ] Output is not "你好".
- [ ] Terminal encoding breaks.

---

## 🧪 PHASE 2: CORRECTNESS (The Math Check)

**Objective**: Verify we beat the Ollama baseline.

**Baselines (Approximate)**:
*   CPU: ~15 tok/s (Target: >30 tok/s)
*   GPU: ~290 tok/s (Target: >580 tok/s)

### 2.1 CPU Benchmark (The Floor)
**Action**:
```bash
apr run models/qwen2.5-coder-1.5b.apr --prompt "bench" --benchmark --device cpu --max-tokens 100
```
**Falsification Criteria**:
- [ ] Throughput < 25 tok/s.
- [ ] CPU usage < 80% (indicates poor threading).

### 2.2 GPU Benchmark (The Ceiling)
**Action**:
```bash
apr run models/qwen2.5-coder-1.5b.apr --prompt "bench" --benchmark --device cuda --max-tokens 500 --batch-size 8
```
**Falsification Criteria**:
- [ ] Throughput < 400 tok/s (Minimum Viable).
- [ ] Throughput < 580 tok/s (Target Miss).
- [ ] CUDA Out of Memory panic.

---

## 🔧 PHASE 3: UNIFIED TRACING AUDIT (Correctness)

**Objective**: Verify the engine is observable and structurally sound (Per §C.5).

### 3.1 The "No More Ghosts" Check
**Action**:
```bash
apr inspect models/qwen2.5-coder-1.5b.apr --trace-structure
```
**Falsification Criteria**:
- [ ] `bias_mean` is 0.0 for QKV layers.
- [ ] `rope_theta` != 1,000,000.0.

### 3.2 The Zero-Token Collapse Check
**Action**:
```bash
apr run models/qwen2.5-coder-1.5b.apr --prompt "test" --trace=attention --trace-output attn.json
```
**Falsification Criteria**:
- [ ] Attention entropy < 0.1 (Collapsed).
- [ ] Top-1 attention target is ALWAYS position 0.

---

## 🚀 PHASE 4: SHOWCASE DEMO (End-to-End)

**Objective**: Verify the "Showcase" flow works for a new user.

### 4.1 The "Pull & Run" (UX)
**Action**:
```bash
# Clean slate
rm -rf ~/.cache/aprender/models/qwen-demo
# Pull
apr pull qwen2.5-coder:1.5b --format apr
# Run
apr chat qwen2.5-coder:1.5b
```
**Falsification Criteria**:
- [ ] Pull fails or hangs.
- [ ] Chat session crashes on first message.
- [ ] Control-C does not exit cleanly.

### 4.2 The "Serve & Batch" (Production)
**Action**:
```bash
apr serve qwen2.5-coder:1.5b --port 8080 --batch-size 4 &
PID=$!
sleep 5
curl http://localhost:8080/v1/completions -d '{"prompt": "Hello", "max_tokens": 10}'
kill $PID
```
**Falsification Criteria**:
- [ ] Server does not start.
- [ ] Curl request times out (> 2s).
- [ ] Response is not valid JSON.

---

## 🏁 FINAL VERDICT CRITERIA

| Result | Consequence |
| :--- | :--- |
| **All Phases Pass** | **SHOWCASE APPROVED**. Ready for public demo. |
| **Garbage Output** | **CRITICAL REJECT**. Return to CORRECTNESS-002. |
| **Perf Miss (<1.5x)** | **WARNING**. Ship with "Optimization Pending" note. |
| **Perf Miss (<1.0x)** | **REJECT**. Cannot ship slower than Ollama. |

**Auditor Note**: Trust the trace. Verify the tokens. Speed is nothing without correctness.
