# QA-SERVE Protocol: HTTP Inference Falsification

**Version:** 1.0.0
**Status:** PROTOCOL ACTIVE
**Target Component:** `apr serve` (realizar)
**Verification Tool:** `crates/apr-cli/scripts/qa-serve.sh`

---

## 1. Objective

To provide a **mechanized falsification** of the `apr serve` HTTP endpoint. This protocol defines a series of "Attacks" (Tests) that attempt to prove the server is non-compliant with the OpenAI API specification or the internal reliability standards of the *aprender* ecosystem.

**Hypothesis ($H_server$):** `apr serve` provides a fully compliant, robust, and performant OpenAI-compatible API for GGUF models.

**Refutation Conditions:** Any single test failure falsifies $H_server$.

---

## 2. Falsification Matrix (The Attacks)

The QA script MUST implement the following 20-point checklist.

### Section I: Connectivity & Health (Critical)
*   [ ] **F-HTTP-001**: Server accepts connection on port 8080.
*   [ ] **F-HTTP-002**: `/health` endpoint returns 200 OK.
*   [ ] **F-HTTP-003**: `/health` response includes `compute_mode` ("cpu" or "gpu").
*   [ ] **F-HTTP-004**: Server shuts down cleanly on SIGINT/SIGTERM.

### Section II: Basic Inference (Functional)
*   [ ] **F-HTTP-005**: `/v1/chat/completions` accepts valid JSON payload.
*   [ ] **F-HTTP-006**: Response contains `choices[0].message.content`.
*   [ ] **F-HTTP-007**: Response is valid JSON (rfc8259).
*   [ ] **F-HTTP-008**: Content is not empty string.
*   [ ] **F-HTTP-009**: Content does not contain raw token IDs (e.g., "token123").

### Section III: Advanced Features (Parity)
*   [ ] **F-HTTP-010**: Streaming (`stream=true`) returns valid SSE (`data: {...}`).
*   [ ] **F-HTTP-011**: Streaming ends with `data: [DONE]`.
*   [ ] **F-HTTP-012**: System prompt is respected (e.g., "You are a pirate").
*   [ ] **F-HTTP-013**: Multi-turn context is preserved (User name recall).
*   [ ] **F-HTTP-014**: `usage` stats (prompt/completion tokens) are present and > 0.

### Section IV: Robustness (Safety)
*   [ ] **F-HTTP-015**: Empty `messages` array returns 400 Bad Request (or informative error).
*   [ ] **F-HTTP-016**: Temperature=0 produces deterministic output (Run 2x, verify equality).
*   [ ] **F-HTTP-017**: Malformed JSON returns 400 Bad Request.
*   [ ] **F-HTTP-018**: OOM protection (Large max_tokens doesn't crash server).
*   [ ] **F-HTTP-019**: Concurrent requests do not cause deadlock.
*   [ ] **F-HTTP-020**: **Coherency Check**: Output entropy > threshold (detects "garbled" 0.5B output).

---

## 3. Protocol for `qa-serve.sh`

The script acts as the **Falsification Agent**.

1.  **Strict Mode:** Must run with `set -euo pipefail`.
2.  **Validation:** Must use Python `json` module for exact field verification (no loose `grep`).
3.  **Color Coding:** RED for Fail, GREEN for Pass, YELLOW for Warning.
4.  **Exit Code:**
    *   `0`: All critical tests passed.
    *   `1`: Functional falsification (API broken).
    *   `2`: Connectivity failure (Server dead).

---

## 4. Known Blockers (PAR-300 Series)

The script must explicitly check for regressions related to current blockers:

*   **PAR-301 (SafeTensors):** If run with a `.safetensors` model, MUST fail immediately or report "Not Implemented".
*   **PAR-303 (0.5B Coherency):** Test F-HTTP-020 is specifically designed to detect this. If output contains >50% repetition or high frequency of replacement characters (``), it MUST fail.

---

## 5. Usage

```bash
./qa-serve.sh <PORT> <MODEL_PATH> [EXPECTED_MODE:cpu|gpu]
```
