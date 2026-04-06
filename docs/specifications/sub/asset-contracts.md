# 29. Asset Contracts

## The Gap

Contracts today verify **functions** — Rust code with preconditions
and postconditions. But the sovereign stack also produces and consumes
**data assets**: model weights, tensors, tokenizer vocabularies,
configuration files, media files, documents. These assets have
invariants just as real as softmax's, and nobody verifies them.

Examples of unverified asset invariants:

| Asset | Invariant | What breaks if violated |
|-------|-----------|------------------------|
| `.apr` model file | Tensor shapes match architecture config | Inference panics on shape mismatch |
| `.safetensors` | Header is valid JSON, all tensors finite | Silent NaN propagation |
| `.gguf` | Metadata matches `arch-constraints-v1` | Wrong normalization, wrong activation |
| Tokenizer `vocab.json` | `vocab_size` == embedding table rows | Index out of bounds at runtime |
| `.mp4` video | Valid moov atom, decodeable streams | Playback fails |
| `.svg` diagram | Well-formed XML, valid viewBox | Rendering broken |
| `.md` documentation | Parses without errors, no broken links | User confusion |
| `config.json` | All required fields present, valid types | Deserialization panic |

## What Already Exists (Almost)

The contract corpus already has **shape contracts** and **metadata
bounds** that describe asset invariants — they just aren't verified
against actual files:

- `qwen2-shapes-v1.yaml` defines `[3584, 3584]` for Q projection
- `model-metadata-bounds-v1.yaml` defines `hidden_dim ∈ [1, 65536]`
- `arch-constraints-v1.yaml` defines per-architecture norm/activation
- `special-tokens-registry-v1.yaml` defines EOS/BOS/PAD token IDs

These are **asset contracts in disguise**. They declare data invariants
but the verification tool (`pv validate`) only checks the YAML
structure, not the actual model files.

## Three Types of Asset Contracts

### Type 1: Schema Contracts (structure)

Verify file format is well-formed without examining content.

```yaml
# contracts/assets/safetensors-schema-v1.yaml
asset_type: safetensors
invariants:
  - header: valid JSON, size < 100MB
  - tensors: each has dtype, shape, data_offsets
  - data_offsets: monotonically increasing, within file size
  - no overlapping tensor regions
verification: parse header, validate offsets
```

### Type 2: Shape Contracts (dimensions)

Verify tensor dimensions match the declared architecture.

```yaml
# contracts/assets/qwen2-7b-shapes-v1.yaml
asset_type: model_weights
architecture: qwen2
config:
  hidden_dim: 3584
  num_heads: 28
  num_kv_heads: 4
  num_layers: 28
  vocab_size: 152064
invariants:
  - embedding.weight: [152064, 3584]
  - layers.*.self_attn.q_proj.weight: [3584, 3584]
  - layers.*.self_attn.k_proj.weight: [512, 3584]
  - layers.*.self_attn.v_proj.weight: [512, 3584]
  - layers.*.mlp.gate_proj.weight: [18944, 3584]
  - lm_head.weight: [152064, 3584]
  - total_params: ~7.6B (within 5% tolerance)
verification: load safetensors header, check each shape
```

### Type 3: Value Contracts (content)

Verify tensor values satisfy numeric invariants.

```yaml
# contracts/assets/weight-health-v1.yaml
asset_type: tensor_values
invariants:
  - all_finite: no NaN or Inf in any tensor
  - norm_bounded: ||w||_2 < 1000 for each weight matrix
  - embedding_normalized: each row of embedding.weight has ||r||_2 > 0
  - no_dead_neurons: no all-zero rows in linear projections
verification: scan tensor data, check per-element and per-row
```

## CLI: `pv verify-asset`

> **Falsification (2026-04-03):** `pv verify-asset` is spec'd but NOT implemented.
> Running `pv verify-asset` returns "unrecognized subcommand". The entire
> `contracts/assets/` directory also does not exist. This section describes
> the **proposed design** — implementation tracked as future work.

```bash
# Verify a model file against its shape contract:
pv verify-asset model.safetensors \
    --contract contracts/assets/qwen2-7b-shapes-v1.yaml

# Verify all assets in a directory:
pv verify-asset models/ --contract-dir contracts/assets/

# Quick health check (no contract needed, checks all_finite + format):
pv verify-asset model.safetensors --health-check

# Output:
#   model.safetensors (safetensors, 7.6B params)
#   Schema:  PASS (valid header, 291 tensors)
#   Shapes:  PASS (all 291 tensors match qwen2-7b config)
#   Values:  PASS (all finite, no dead neurons)
```

## Integration with Existing Contracts

Asset contracts extend the existing two-tier model:

```
Tier 1: Kernel contracts      (algorithm math)
Tier 2: Per-repo bindings     (code → contract mapping)
Tier 3: Asset contracts (NEW)  (data → contract mapping)
```

The binding.yaml gains an `assets` section:

```yaml
# contracts/aprender/binding.yaml
critical_path: [softmax, matmul, attention]
bindings: [...]
assets:                          # NEW
  - file_pattern: "models/*.safetensors"
    contract: assets/weight-health-v1.yaml
    verification: health-check
  - file_pattern: "tokenizers/*.json"
    contract: special-tokens-registry-v1.yaml
    verification: schema
```

## Scoring

Asset contract coverage becomes an optional dimension:

```
CD6: Asset coverage = verified_assets / declared_assets
```

Only scores when `assets:` section is present in binding.yaml.
Repos without assets get no penalty (same as critical_path fallback).

## Asset Type Registry

`pv verify-asset` detects the contract type from the file extension:

| Contract Type | Extensions | Invariants |
|--------------|------------|------------|
| `tensor_weights` | `.safetensors` `.gguf` `.apr` `.onnx` | shapes match config, all finite, no dead neurons |
| `tokenizer` | `tokenizer.json` `vocab.json` `merges.txt` | vocab_size == embedding rows, special tokens valid |
| `config` | `config.json` `*.toml` `*.yaml` | required fields present, values in declared bounds |
| `media_video` | `.mp4` `.webm` `.mkv` | valid container, decodeable streams, duration > 0 |
| `media_audio` | `.wav` `.flac` `.mp3` `.ogg` | valid headers, sample_rate > 0, channels in {1,2} |
| `media_image` | `.png` `.jpg` `.svg` `.webp` | valid format, dimensions > 0, finite pixel values |
| `document` | `.md` `.html` `.pdf` `.tex` | parses clean, no broken internal links |
| `binary_artifact` | `.wasm` `.so` `.dylib` `.ptx` `.spv` | valid format, expected exports/entry points |
| `structured_data` | `.json` `.jsonl` `.parquet` `.arrow` `.csv` | schema conformance, row count, no null in required cols |
| `proof` | `.lean` `.olean` | compiles, no sorry, hash matches source |

Each sovereign stack component maps to specific asset types:

```
aprender/realizar   → tensor_weights, tokenizer, config
whisper.apr         → tensor_weights, media_audio, tokenizer
trueno              → binary_artifact (PTX, SPIR-V, Metal)
presentar           → media_image, document (SVG, PDF, MD)
rmedia              → media_video, media_audio, media_image
forjar              → binary_artifact (WASM, .so)
trueno-db           → structured_data (Parquet, SQLite)
trueno-rag          → structured_data, tensor_weights (embeddings)
provable-contracts  → proof (.lean), config (.yaml)
```

## Asset Contract YAML Schema

```yaml
# contracts/assets/safetensors-schema-v1.yaml
metadata:
  version: "1.0.0"
  description: "Safetensors format schema contract"
  asset_type: tensor_weights       # ← from type registry
  extensions: [".safetensors"]

invariants:
  schema:                           # Type 1: format well-formedness
    - "header is valid JSON"
    - "header size < 100MB"
    - "each tensor has dtype, shape, data_offsets"
    - "data_offsets monotonically increasing"
    - "no overlapping tensor regions"
    - "total file size == header_size + sum(tensor_bytes)"

  shape:                            # Type 2: dimension matching
    - "tensor.shape matches architecture config when provided"
    - "embedding.weight[0] == vocab_size"
    - "all linear projections have 2 dimensions"

  value:                            # Type 3: numeric health
    - "all elements finite (no NaN, no Inf)"
    - "||weight||_2 < 10000 per tensor"
    - "no all-zero rows in linear projections"

falsification_tests:
  - id: FALSIFY-ST-001
    rule: "Truncated file detection"
    prediction: "File truncated at random offset → schema error, not panic"
    test: "Truncate valid safetensors at 100 random positions"
  - id: FALSIFY-ST-002
    rule: "NaN injection detection"
    prediction: "Single NaN in weight tensor → value check fails"
    test: "Inject NaN at random position in valid file"
```

## Implementation Path

| Phase | What | Complexity |
|-------|------|------------|
| P1 | `pv verify-asset --health-check` | Read safetensors header, check finite. Low. |
| P2 | Shape contract YAML schema | New `asset_type`, `invariants` fields. Medium. |
| P3 | `pv verify-asset --contract` | Parse shape contract, verify against file. Medium. |
| P4 | Value contracts | Scan tensor data for dead neurons, norms. High. |
| P5 | CD6 in codebase scoring | Wire into `pv score` when `assets:` present. Low. |

## Runtime Integration: trueno BrickLayer + apr-cli (PROPOSED)

> **Falsification (2026-03-28):** All runtime integration below is
> PROPOSED DESIGN. None of this code exists in trueno or aprender yet.
> Only `WeightHealth` (F7) exists. The spec describes what SHOULD be
> built, not what IS built. See implementation status table at end.

Asset contracts become useful only when the runtime **checks them**.
Two integration points exist in the sovereign stack today:

### trueno: BrickLayer contract-aware tracing

trueno already has a per-kernel profiling system:

```
ComputeBrick<Op>       — wraps a kernel operation
BrickLayer             — orchestrates bricks, manages execution graph
BrickSample / BrickStats — records timing, memory, launch counts
AsyncTaskProfiler      — profiles async kernel dispatch
PerfMetrics            — records load, prefill, decode timings
```

**What exists:** `record_kernel_launch()` captures timing and memory.
`record_prefill()` / `record_decode()` track phase performance.

**What's missing:** No contract check at the recording site. The
profiler observes but doesn't verify postconditions. The integration:

```rust
// trueno/src/brick/compute_brick.rs (proposed)
impl<Op: ComputeOp> ComputeBrick<Op> {
    pub fn execute_with_contract(&self, input: &[f32]) -> Vec<f32> {
        contract_pre_softmax!(input);        // from generated_contracts.rs
        let result = self.op.execute(input);
        contract_post_softmax!(result);      // postcondition check
        self.profiler.record_kernel_launch(  // existing profiling
            &self.name, elapsed, input.len()
        );
        result
    }
}
```

The `generated_contracts.rs` macros already exist in trueno
(Section 27). The integration is: call the precondition macro before
execution and the postcondition macro after — then record the result
in the existing `BrickStats`.

**Contract violation → BrickStats anomaly.** When a postcondition
fires (e.g., softmax output doesn't sum to 1.0), the profiler records
it as a `contract_violation` event. This connects runtime behavior
to the contract-derived invariant, making `BrickLayer` a
**contract-aware execution engine**.

### apr-cli: Contract-verified model loading

aprender's `load_model` currently:
1. Reads safetensors file
2. Deserializes weights into tensors
3. Returns `Module`

**What's missing:** No verification that tensor shapes match the
architecture contract or that values are finite.

```rust
// aprender/src/nn/serialize.rs (proposed)
pub fn load_model_verified<M: Module>(
    path: &Path,
    shape_contract: Option<&Path>,  // e.g. qwen2-7b-shapes-v1.yaml
) -> Result<M> {
    let model = load_model::<M>(path)?;

    if let Some(contract) = shape_contract {
        // pv verify-asset logic embedded:
        let shapes = extract_tensor_shapes(&model);
        let expected = parse_shape_contract(contract)?;
        verify_shapes(&shapes, &expected)?;  // errors on mismatch
    }

    // Quick health check: all finite
    for tensor in model.tensors() {
        assert!(tensor.data().iter().all(|v| v.is_finite()),
            "NaN/Inf detected in loaded model weights");
    }

    Ok(model)
}
```

aprender already has `WeightHealth` / `health_status()` in
`src/inspect/weight_stats.rs` — this is the hook point.

### Roofline-derived serving budget

apr-cli's `serve plan` should derive performance ceilings from
`roofline-model-v1.yaml` instead of hardcoded formulas. The `pv
roofline` module (Section 24, already implemented) provides:

```rust
let ceiling = roofline::compute_roofline(
    model.total_params(),
    model.bits_per_weight(),
    &HardwareProfile::detect(),  // auto-detect hardware
);
// ceiling.throughput_ceiling = max achievable tok/s
// Use as SLA: warn if observed TPOT > 1/ceiling
```

## Full Verification Chain

```
                    Asset Contracts (§29)
                          │
    ┌─────────────────────┼──────────────────────┐
    │                     │                      │
load_model_verified  BrickLayer.execute    pv roofline
    │                with_contract              │
    │                     │                      │
shape check           pre/post check        SLA budget
value health          profiler record       throughput gate
    │                     │                      │
    └─────────────────────┼──────────────────────┘
                          │
              Contract-verified inference
```

## Implementation Status (measured 2026-03-28)

| Component | Status | Evidence |
|-----------|--------|----------|
| `pv codegen` macros in repos | **18/18 DONE** | All sovereign stack repos have `generated_contracts.rs` + `#[macro_use]` |
| Contract macro call sites | **27 total across 18 repos** | trueno: 3, entrenar: 2, realizar: 2, forjar: 2, pacha: 2, pepita: 2, ruchy: 2, trueno-rag: 2, aprender: 1, bashrs: 1, depyler: 1, batuta: 1, renacer: 1, alimentar: 1, simular: 1, trueno-viz: 1, trueno-db: 1, trueno-graph: 1 |
| Test-verified (0 contract failures) | **17/18** | All repos pass `cargo test --lib` with 0 contract-caused failures. ruchy has pre-existing build issue (missing wasmparser). |
| Assertion placement rule | **ENFORCED** | Contract assertions placed AFTER early-return guards, not before. Fixes applied to forjar, realizar, trueno-rag (v2.3.0). |
| `contracts/assets/` directory | **NOT IMPLEMENTED** | Directory does not exist |
| `pv verify-asset` CLI | **NOT IMPLEMENTED** | No such subcommand |
| `execute_with_contract()` | **NOT IMPLEMENTED** | Proposed design only |
| `load_model_verified()` | **NOT IMPLEMENTED** | Proposed design (aprender recovered) |
| apr-cli roofline integration | **NOT IMPLEMENTED** | Only in generated_contracts.rs |
| BrickStats violation tracking | **NOT IMPLEMENTED** | No violation field |
| Shape contract vs real files | **NOT IMPLEMENTED** | No code reads shapes YAML |
| `WeightHealth` NaN/Inf check | **EXISTS** | aprender/src/inspect/weight_stats.rs:22 |

## Why This Matters

The inference pipeline is: **load model → run kernels → produce output**.
We verify the kernels (Section 2-28) but not the model load. A corrupt
weight file silently produces wrong outputs even with perfect kernels.
Asset contracts close the last gap in the verification chain.

The trueno BrickLayer and apr-cli load_model are the two insertion
points where asset + function contracts meet runtime execution.

## References (Section 29)

- Atlas (arXiv:2502.19567, 2025). ML lifecycle provenance and
  transparency — verifiable records of model artifact authenticity.
- Data Quality Survey (arXiv:2406.19614, 2024). Data quality
  dimensions for ML: accuracy, completeness, consistency, timeliness.
- DQuag (arXiv:2502.10667, 2025). Automated data quality validation
  in end-to-end GNN frameworks.
- safetensors specification. HuggingFace.
  github.com/huggingface/safetensors
- GGUF specification. ggerganov/ggml.
  github.com/ggerganov/ggml/blob/master/docs/gguf.md
