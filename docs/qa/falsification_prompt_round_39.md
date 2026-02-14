# Master Falsification Prompt: Qwen2.5-Coder Showcase (Round 39)

**Target:** `docs/specifications/qwen2.5-coder-showcase-demo.md` (v9.5.1)
**Context:** Post-Streaming Implementation & Conversion Path Fix.
**Status Claim:** "Ready for QA. Expected 32/32 (100%) pass rate."

**Mission:** You are the **Lead Falsification Auditor**. The engineering team claims to have fixed the conversion path bugs and implemented CUDA streaming. Your goal is to **prove them wrong**.

Execute the following 6-Phase Protocol. **Stop the line** (FAIL) immediately if any P0 criteria are met.

---

## Phase 1: The "Fix" Audit (Conversion Logic) (P0)
**Claim:** "Path resolution fix eliminates all 13 F-CONV-* conversion failures."
*   **F-CONV-001 (Certification Run):**
    *   *Action:*
        ```bash
        cd /home/noah/src/apr-model-qa-playbook
        cargo run --release -- certify --tier quick qwen2.5-coder-0.5b-instruct --fail-fast
        ```
    *   *Falsification Criteria:*
        *   Final score is NOT **32/32**.
        *   Any `F-CONV-*` test fails (proving the path fix is insufficient).

## Phase 2: The New Feature (CUDA Streaming) (P1)
**Claim:** "Streaming works on GPU for SafeTensors and APR (new `realizar/cuda/streaming.rs`)."
*   **F-STREAM-001 (SafeTensors GPU Stream):**
    *   *Action:* `apr run hf://Qwen/Qwen2.5-Coder-1.5B-Instruct "Count from 1 to 5" --gpu --stream`
    *   *Falsification Criteria:*
        1.  **Buffering:** Output appears all at once at the end (not streamed token-by-token).
        2.  **Garbage:** Output contains nonsense tokens.
        3.  **Panic:** Crash in `realizar/cuda/streaming.rs`.
*   **F-STREAM-002 (APR GPU Stream):**
    *   *Action:* `apr run model.apr "Count from 1 to 5" --gpu --stream` (Ensure `model.apr` exists from Phase 1).
    *   *Falsification Criteria:* Same as above.

## Phase 3: The Persistent Defect (Ground Truth Correctness) (P0)
**Claim:** "SafeTensors is Ground Truth and produces correct output." (Previous Round: Failed).
*   **F-GT-004 (Token Correctness):**
    *   *Action:* `apr run hf://Qwen/Qwen2.5-Coder-1.5B-Instruct "What is 2+2?" --chat --gpu`
    *   *Falsification Criteria:*
        *   Output is NOT "4" (e.g., "5", empty, or random words).
        *   *Note:* If this fails, the "Ground Truth" status in Section 0 remains **FALSIFIED**.

## Phase 4: Static Quality Gates (P0)
**Claim:** "Zero SATD. 96%+ Coverage."
*   **F-SATD-004:** `grep -r "TODO" src/ crates/ | grep -v "test" | grep -v "rosetta.rs"`
    *   *Falsification:* Any match found.
*   **F-COV-004:** Check coverage report.
    *   *Falsification:* < 96.0%.

## Phase 5: Performance & Parity (P1)
**Claim:** "GPU > 2x CPU Speedup."
*   **F-PERF-007:**
    *   *Action:* Compare `apr bench ... --device cpu` vs `--device gpu`.
    *   *Falsification:* GPU speedup < 2.0x.

## Phase 6: Universal Tooling (P2)
**Claim:** "All tools functional."
*   **F-TOOL-004:** Verify `apr serve` with the new streaming backend.
    *   *Action:* Start `apr serve ... --gpu` and hit `/generate` endpoint with `stream: true`.
    *   *Falsification:* Server panics or returns non-streamed JSON response.

---

**Execution Output:**
Report the **Certification Score** (e.g., "19/32" or "32/32") and the status of **SafeTensors Correctness** (Pass/Fail).
