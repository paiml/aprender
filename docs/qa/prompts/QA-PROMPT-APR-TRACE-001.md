# QA Falsification Prompt: APR-TRACE-001

**Target Spec**: `docs/specifications/apr-inference-tracing.md`
**Methodology**: Popperian Falsification & Toyota Way (Genchi Genbutsu)
**Audience**: QA Engineers / Automated Agents

---

## 🚨 CRITICAL QUALITY GATE: ZERO TOLERANCE POLICY

**The Engineering Team is NOT allowed to claim a "Partial Pass".**
*   **CUDA Graph Parity**: If tracing fails when CUDA Graphs are enabled: **HARD FAIL**.
*   **Format Parity**: If `.gguf` works but `.apr` or `.safetensors` fails: **HARD FAIL**.
*   **Hardware Parity**: If CPU works but SIMD, WGPU, or CUDA fails (or requires workarounds): **HARD FAIL**.

**Anything less than 100% parity across ALL formats and ALL backends is a REJECTION of the entire feature.**

---

## 🛑 QA AGENT INSTRUCTIONS

**Role**: You are a **Ruthless Quality Auditor**. Your job is **NOT** to confirm the feature works. Your job is to **prove it is broken**. 

**Philosophical Mandate**:
- **Popperian Falsification**: One single failure point (e.g., CUDA Graphs not instrumented) falsifies the entire implementation.
- **Jidoka**: If you find a "Partial (GPU)" status, stop the line. The feature is defective.
- **Formal Verification (Lamport)**: Treat the trace as a formal proof of execution. Any missing link falsifies the proof.

---

## 🧪 PHASE 1: GENCHI GENBUTSU (Go & See)

Execute the following commands. If *any* command fails to produce the expected output, **FAIL** the audit immediately.

### 1. The "Happy Path" Trace
```bash
apr run models/qwen2.5-0.5b-instruct.gguf --prompt "def hello():" --trace --max-tokens 10
```

### 2. The "Deep Dive" (Verbose & JSON)
```bash
apr run models/qwen2.5-0.5b-instruct.gguf --prompt "x" --trace --trace-verbose --trace-output trace.json --max-tokens 1
```

---

## 🔨 PHASE 2: AGGRESSIVE FALSIFICATION (Try to Break It)

| ID | Test Scenario | Hypothesis to Falsify | Command / Action |
|----|---------------|-----------------------|------------------|
| **F-01** | **The Void** | Tracer panics on empty/whitespace prompt. | `apr run model.gguf --prompt "   " --trace` |
| **F-05** | **The Lock** | JSON output file locked/read-only. | `touch locked.json && chmod 400 locked.json && apr run ... --trace-output locked.json` |

---

## ⚔️ PHASE 2b: TRI-FORMAT BATTLE (The Parity Check)

**Objective**: Falsify the claim that "Tracing works identically on all formats."

| ID | Test Scenario | Hypothesis to Falsify | Command / Action |
|----|---------------|-----------------------|------------------|
| **F-PAR-01** | **The Clone War** | Traces diverge between formats. | Run prompt "test" on .gguf, .safetensors, and .apr. Diff JSON outputs. |
| **F-PAR-02** | **The Missing Config** | SafeTensors panics without config. | Rename `config.json` -> `config.json.bak`. Run `apr run model.safetensors --trace`. |

---

## ⚡ PHASE 2c: THE SILICON GAUNTLET (Hardcore Mode)

**Objective**: Falsify the claim that "Tracing works identically on all hardware."

| ID | Test Scenario | Hypothesis to Falsify | Command / Action |
|----|---------------|-----------------------|------------------|
| **F-HW-04-B** | **CUDA Graph** | Tracing fails/skips steps in Graph mode. | Run with `--device cuda`. Verify ALL 7 steps are present and correct. |
| **F-HW-05** | **Grand Parity**| Drift between CPU/GPU > tolerance. | Diff JSON traces for same prompt on CPU vs CUDA. |

---

## 🤖 PHASE 2d: THE STATE MACHINE AUDIT (Step Function Parity)

**Objective**: Falsify the claim that "Tracing is equivalent to AWS Step Functions."

| ID | Test Scenario | Hypothesis to Falsify | Command / Action |
|----|---------------|-----------------------|------------------|
| **F-AWS-01** | **The Broken Chain** | State transitions are disconnected. | Verify every `TaskStateExited` event links to a valid `previous_event_id`. |
| **F-AWS-03** | **The Black Hole** | Input/Output data is missing. | Run `--trace-verbose`. Verify `input` and `output` fields are populated for `EMBED` and `LM_HEAD`. |
| **F-AWS-05** | **The Silent Fail** | Errors lack structured data. | Trigger a crash (e.g., bad vocab). Verify `ExecutionFailed` event contains `error` and `cause` fields. |

---

## 📝 PHASE 3: THE 100-POINT CHECKLIST AUDIT

Review the implementation against the **100-Point Popperian Checklist** in Section 8 of the spec.

**Critical Sections:**
*   Section I: CLI & Interface
*   Section II: Correctness & Accuracy
*   Section III: Performance & Stability
*   Section IV: JSON Schema & Interop
*   Section V: Toyota Way Jidoka
*   Section VI: Multi-Format Parity
*   Section VII: Hardware Backend Parity
*   Section VIII: AWS Step Function Parity

## 🏁 AUDIT CONCLUSION CRITERIA
- **100/100 Points**: RELEASE
- **99/100 Points**: REJECT (Fix the 1 point)
- **"Partial" anything**: REJECT IMMEDIATELY.
