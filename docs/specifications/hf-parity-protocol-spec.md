# Hugging Face Parity Verification Protocol (PMAT-HF-001)

**Version:** 1.0.0  
**Status:** ACTIVE  
**Focus:** Full E2E coverage (`oracle`, `pull`, `chat`, `code`, `serve`)

## Abstract

This protocol defines the standard for verifying `apr` compatibility with the Top 50 trending `text-generation` models on the Hugging Face Hub. Verification must extend beyond static metadata parsing (`oracle`) to cover end-to-end inference and serving capabilities across all supported hardware tiers.

## Hardware Profiles Matrix

To ensure universal compatibility, parity tests are targeted across our core hardware boundaries:
- **`lambda-labs`**: High-performance edge (RTX 4090, 24GB VRAM)
- **`gx10`**: High-capacity compute (NVIDIA GB10, 128GB RAM)
- **`mini`**: Local/edge inference (Apple M4, 16GB RAM)

## Verification Tiers

The protocol is divided into two verification tiers, implemented programmatically via the `hf_parity_protocol.py` automation script.

### Level 1: Oracle Validation
*Validates static metadata, constraints, and architecture mappings without downloading full model weights.*

- **Command**: `apr oracle --json hf://<model_id>`
- **Pass Criteria**:
  1. Identifies the correct base architecture (e.g., Llama 3, Qwen 2.5, DeepSeek).
  2. Parses hidden dimensions, layer counts, and head configs.
  3. Accurately determines the memory constraints and target hardware limit.

### Level 2: End-to-End (E2E) Integration
*Requires pulling the model and exercising the primary `apr` user-facing commands to guarantee functional generation.*

- **Pull Verification**:
  - `apr pull hf://<model_id>`
  - *Pass Criteria*: Model correctly caches the `.safetensors` or `.gguf` variants locally.

- **Chat Verification (Conversational Inference)**:
  - `apr chat hf://<model_id> --prompt "Hello" --max-tokens 10`
  - *Pass Criteria*: Generates text contextually successfully without panicking.

- **Code Verification (Non-Interactive Generation)**:
  - `apr code hf://<model_id> -p "def main():" --max-tokens 10`
  - *Pass Criteria*: Respects formatting constraints and code-agent prompts.

- **Serve Verification (Capacity Planning & REST API)**:
  - `apr serve plan hf://<model_id>`
  - *Pass Criteria*: Validates the hardware memory planner successfully allocates KV cache and tensor graph before serving.

## Execution

To run the parity protocol against the top 50 models:

```bash
# Level 1 (Fast)
python3 hf_parity_protocol.py --mode oracle

# Level 2 (Full Download & Execution)
python3 hf_parity_protocol.py --mode e2e --limit 10
```

## Falsification Checklist
As part of the PMAT Quality Gates, the following assertions must hold true:
- [ ] **F-HF-001**: A model failing the `oracle` check must gracefully abort and not attempt a `pull`.
- [ ] **F-HF-002**: A `401 Unauthorized` network error correctly identifies a gated model and prompts the user for `HF_TOKEN`.
- [ ] **F-HF-003**: `apr serve plan` correctly identifies when a model exceeds available system memory and warns the user before starting the HTTP server.
