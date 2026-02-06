# Master Falsification Prompt: Qwen2.5-Coder Showcase (Round 38)

**Target:** `docs/specifications/qwen2.5-coder-showcase-demo.md` (v9.5.1)
**Context:** Post-Remediation (Round 37) - `tie_word_embeddings` P0 fixed (logits no longer zeros), but SafeTensors output still produces incorrect tokens (P1). GGUF remains the only 100% correct path.

**Mission:** Prove the specification's claims about "SafeTensors Ground Truth" and "Production Ready" status are currently **FALSIFIED** due to tokenization/decoding mismatches.

---

### Phase 1: Static Quality & Coverage (P0)
**Claim:** "Zero SATD. 96%+ Test Coverage."
*   **F-SATD-003:** `grep -r "TODO" src/ crates/ | grep -v "test" | grep -v "rosetta.rs"`
    *   *Falsification:* Any matches found.
*   **F-COV-003:** Verify `tarpaulin` report.
    *   *Falsification:* Coverage < 96.0% (Current: 96.31%).

### Phase 2: Ground Truth Deconstruction (The P1 Failure)
**Claim:** "SafeTensors is the Ground Truth (Section 0) and is 'PASS' (Release Criteria)."
*   **F-GT-003 (The Parity Gap):**
    *   *Action:* Compare SafeTensors output against GGUF output for the same prompt.
    *   *Command A (ST):* `apr run hf://Qwen/Qwen2.5-Coder-1.5B-Instruct "What is 2+2?" --chat`
    *   *Command B (GGUF):* `apr run hf://Qwen/Qwen2.5-Coder-1.5B-Instruct-GGUF "What is 2+2?" --chat`
    *   *Falsification Criteria:*
        1.  GGUF outputs "4" (Corroborated).
        2.  SafeTensors outputs anything OTHER than "4" (FALSIFIED - Claim "PASS" is false).
        3.  SafeTensors logits are non-zero but argmax results in invalid tokens (FALSIFIED - "Ground Truth" is broken).

### Phase 3: Weight Tying Audit
**Claim:** "tie_word_embeddings: true is correctly handled."
*   **F-TIE-001 (Pointer/Value Check):**
    *   *Action:* Inspect `SafeTensorsCudaModel::upload_weights()` in `realizar/src/safetensors_cuda.rs`.
    *   *Falsification Criteria:* If `config.tie_word_embeddings` is true but `lm_head.weight` is still being loaded from the SafeTensors file instead of reusing `model.embed_tokens.weight`.

### Phase 4: Reliability & Hardware Dispatch
**Claim:** "All panics (Empty File, Lock Poisoning) are fixed. GPU > 2x CPU."
*   **F-CRIT-007 (Empty File Regression):** `touch empty.safetensors && apr run empty.safetensors "hi"`
    *   *Falsification:* Panic/Stack trace.
*   **F-PERF-006 (Hardware Gap):**
    *   *Action:* `apr bench model.gguf --device cpu` vs `--device gpu`.
    *   *Falsification:* GPU speedup < 2.0x (GGUF is the only valid benchmark path currently).

### Phase 5: Tooling Universalism
**Claim:** "13/13 tools verified." (Latest audit found 20).
*   **F-TOOL-003:** `apr --help`
    *   *Falsification:* Any core tool (run, chat, serve, import, export, rosetta) missing.
    *   *Note:* The spec claiming 13 when 20 exist is a "Minor Falsification" (Outdated Documentation).

---

**Final Report Requirements:**
1.  **Status of SafeTensors:** Must be marked ❌ **FALSIFIED** (Broken tokens).
2.  **Status of GGUF:** Must be marked ✅ **CORROBORATED** (Produces "4").
3.  **Status of spec v9.5.1:** Must be marked ⚠️ **UNRELIABLE** until Section 0 (Ground Truth) and the Release Table are updated to reflect SafeTensors token failures.
