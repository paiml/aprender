# QA Verification Prompt: AWS Step Functions Parity (APR-TRACE-001)

**Target Spec**: `docs/specifications/apr-inference-tracing.md` (v3.1.0)
**Context**: Verification of recent implementation of AWS Step Functions Parity.
**Audience**: QA Engineers / Automated Agents

---

## 🎯 OBJECTIVE
Verify that the Inference Tracing implementation strictly adheres to the **AWS Step Functions Parity** requirements defined in Spec v3.1.0, specifically **Section VIII (F-AWS-01 to F-AWS-05)**.

**Key Changes to Verify**:
1.  **Renaming**: `Encode` -> `Tokenize`, `Transformer` -> `TransformerBlock`.
2.  **Event Structure**: `TaskStateEntered`, `TaskStateExited` events with valid ISO 8601 timestamps and monotonic IDs.
3.  **Linking**: `TaskStateExited` events MUST have a `previous_event_id` pointing to the correct `TaskStateEntered` event.

---

## 🧪 PHASE 1: GENCHI GENBUTSU (Go & See)

Execute the following commands and **inspect the raw JSON output**. Do not rely on "it didn't crash".

### 1. Generate a Trace
```bash
apr run models/qwen2.5-0.5b-instruct.gguf \
  --prompt "def hello():" \
  --trace \
  --trace-output trace_aws.json \
  --max-tokens 5
```

### 2. Inspect Event Types & Naming (F-AWS-01)
*   **Action**: Open `trace_aws.json`.
*   **Check**: Are there events with `type: "TaskStateEntered"` and `type: "TaskStateExited"`?
*   **Check**: Are step names `TOKENIZE` (not Encode) and `TRANSFORMER_BLOCK` (not Transformer)?
*   **Falsification**: If you see "Encode" or missing event types, **FAIL**.

### 3. Verify Monotonic IDs (F-AWS-01)
*   **Action**: Check the `id` field of sequential events.
*   **Check**: `id` starts at 1 and increments by 1 for *every* event.
*   **Falsification**: If IDs skip (1, 3, 4) or repeat, **FAIL**.

### 4. Verify Event Linking (F-AWS-02)
*   **Action**: Find a `TaskStateExited` event.
*   **Check**: Does it have a `previous_event_id` field?
*   **Check**: Does that ID correspond to the `TaskStateEntered` event for the *same* step?
*   **Falsification**: If `previous_event_id` is null or points to a different step, **FAIL**.

### 5. Verify ISO 8601 Timestamps (F-AWS-01)
*   **Action**: Check the `timestamp` field.
*   **Check**: Format should be `YYYY-MM-DDTHH:MM:SS.mmmZ` (RFC3339 with millis).
*   **Falsification**: If format is unix epoch or missing timezone `Z`, **FAIL**.

### 6. Verify Input/Output Capture (F-AWS-03)
*   **Action**: Run with `--trace-verbose`.
*   **Check**: `TaskStateEntered` events should have an `input` field.
*   **Check**: `TaskStateExited` events should have an `output` field.

---

## 🤖 PHASE 2: AUTOMATED CHECK (Optional)
Run this `jq` script on the output to auto-verify:

```bash
# Verify strictly monotonic IDs
jq '([.events[].id] | map(select(. > 0))) as $ids | $ids == ($ids | sort) and ($ids | length) == ($ids[-1] - $ids[0] + 1)' trace_aws.json

# Verify every Exit links to an Entry
jq '.events[] | select(.type == "TaskStateExited") | .previous_event_id != null' trace_aws.json
```

---

## 📝 AUDIT CHECKLIST (Section VIII) 

- [ ] **F-AWS-01**: Trace contains `TaskStateEntered` and `TaskStateExited` events.
- [ ] **F-AWS-02**: Every `TaskStateExited` links back to entry via `previous_event_id`.
- [ ] **F-AWS-03**: `Input` and `Output` fields are captured (verbose mode).
- [ ] **F-AWS-04**: Trace output forms a valid DAG / State Machine.
- [ ] **F-AWS-05**: Error events use `ExecutionFailed` structure (trigger with bad vocab/model).

---

## 🏁 CONCLUSION
*   **PASS**: All checks pass, JSON structure is valid AWS Step Functions format.
*   **FAIL**: Any naming mismatch, ID gap, or missing link.
