# Master Falsification Prompt: Qwen2.5-Coder Showcase (Round 37)

**Target:** `docs/specifications/qwen2.5-coder-showcase-demo.md` (v9.5.1)
**Context:** Post-Remediation (Round 36.3) - SATD & Lock Poisoning reported fixed. APR Q4_K reported broken.

**Mission:** You are the **Lead Falsification Auditor**. Your goal is not to verify that the software works, but to **prove that the specification is lying**.
Execute the following 6-Phase Protocol. **Stop the line** (FAIL) immediately if any P0 criteria are met.

---

## Phase 1: The "Toyota Way" Quality Audit (P0)
**Claim:** "Zero SATD in production code. 95%+ Test Coverage."
*   **F-SATD-002 (Regression Check):**
    *   *Action:* `grep -r "TODO" src/ crates/ | grep -v "test" | grep -v "rosetta.rs"`
    *   *Falsification Criteria:* Any output. (Note: `rosetta.rs` TODOs were replaced with warnings; verify this).
*   **F-COV-002 (Coverage Verification):**
    *   *Action:* Check most recent `tarpaulin` report or run coverage check.
    *   *Falsification Criteria:* Coverage < 95.0%.

## Phase 2: Ground Truth Verification (SafeTensors) (P0)
**Claim:** "SafeTensors 1.5B/0.5B is Production Ready and serves as Ground Truth."
*   **F-GT-002 (The "2+2" Test):**
    *   *Action:* `apr run hf://Qwen/Qwen2.5-Coder-1.5B-Instruct "What is 2+2?" --chat`
    *   *Falsification Criteria:*
        1.  Output is not "4".
        2.  Output contains garbage (e.g., repeated tokens, wrong language).
        3.  Process hangs (>60s).

## Phase 3: The Known Defect (APR Q4_K Parity) (P1)
**Claim:** "APR Q4_K conversion is currently IN PROGRESS/BROKEN."
*   **F-PAR-002 (Garbage Confirmation):**
    *   *Action:*
        1.  `apr import hf://Qwen/Qwen2.5-Coder-1.5B-Instruct -o model.apr` (F32 import)
        2.  `apr convert model.apr --quantize q4_k -o model_q4k.apr`
        3.  `apr run model_q4k.apr "What is 2+2?" --chat`
    *   *Falsification Criteria:*
        *   **IF IT WORKS:** The spec is FALSIFIED (Claiming "Broken" when it works is a documentation error).
        *   **IF IT FAILS:** The spec is CORROBORATED (Behavior matches "Known Issue").

## Phase 4: Reliability & Panic Resilience (P0)
**Claim:** "All panics (Empty File, Missing Tokenizer, Lock Poisoning) are fixed."
*   **F-CRIT-004 (Empty File):**
    *   *Action:* `touch empty.safetensors && apr run empty.safetensors "hi"`
    *   *Falsification Criteria:* Stack trace / Panic. (Must be a clean Error exit).
*   **F-CRIT-005 (Missing Tokenizer):**
    *   *Action:* Load a SafeTensors model where `tokenizer.json` has been renamed.
    *   *Falsification Criteria:* Stack trace or raw token ID output.
*   **F-CRIT-006 (Lock Poisoning Regression):**
    *   *Action:* Verify `realizar/cuda/executor/*.rs` uses `.lock().expect(...)` or handles `PoisonError`.
    *   *Falsification Criteria:* Found `.lock().unwrap()` in CUDA executor.

## Phase 5: Performance Claims (P1)
**Claim:** "GPU > 2x CPU Speedup. SafeTensors CPU ~19 tok/s (1.5B)."
*   **F-PERF-004 (Speedup):**
    *   *Action:*
        1.  `apr bench model.safetensors --device cpu --prompt "A" --max-tokens 50`
        2.  `apr bench model.safetensors --device gpu --prompt "A" --max-tokens 50`
    *   *Falsification Criteria:* GPU Tok/s <= 2.0 * CPU Tok/s.
*   **F-PERF-005 (Baseline):**
    *   *Falsification Criteria:* CPU Tok/s < 10.0 (Severe regression).

## Phase 6: Tooling & UX (P2)
**Claim:** "All 13 Tools exist. Verbose mode provides complete telemetry."
*   **F-TOOL-002 (Tool Check):**
    *   *Action:* `apr --help`
    *   *Falsification Criteria:* Any of the 13 tools missing from list.
*   **F-UX-002 (Verbose Telemetry):**
    *   *Action:* `apr run model.safetensors "hi" --verbose`
    *   *Falsification Criteria:* Missing "Backend:", "Architecture:", or "Quantization:" in logs.

---

**Execution Output:**
Report one of the following for the **Entire Spec**:
1.  ✅ **CORROBORATED:** All claims hold (including acknowledged defects).
2.  ❌ **FALSIFIED:** One or more claims are false. (List specific F-ID).
