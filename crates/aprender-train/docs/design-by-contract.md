# Design by Contract in Entrenar

Entrenar enforces Meyer-style Design by Contract (DbC) at configuration
boundaries and evaluation outputs. Each contract has a code (C-xx, N-xx,
R-xx) that appears in error messages and comments, making violations
traceable to a single source of truth.

## C-10/C-11: Required Fields in HuggingFace Config Parsing

**Location:** `src/config/train/loader.rs`, `parse_hf_config()`

When loading a HuggingFace `config.json`, five fields are **required** and
will fail with an explicit error if missing:

| Field                  | Contract | Rationale                                            |
|------------------------|----------|------------------------------------------------------|
| `hidden_size`          | C-11     | Cannot construct attention or FFN layers without it  |
| `num_attention_heads`  | C-11     | Head dimension = hidden_size / num_attention_heads   |
| `num_hidden_layers`    | C-11     | Determines model depth                               |
| `vocab_size`           | C-10     | Wrong vocab corrupts embedding table on first step   |
| `intermediate_size`    | C-11     | FFN gate/up/down projections depend on this          |

**Why not default?** A silent default for `vocab_size` (e.g., 32000) will
produce a shape mismatch against a Qwen tokenizer (151936 tokens) and
corrupt the embedding gradient on the first backward pass. The error is
silent: training runs but produces garbage. C-10 makes this a hard failure.

## R-04: Optional Fields with Warned Defaults

**Location:** `src/config/train/loader.rs`, `parse_hf_config()`

Optional fields fall back to generic defaults **with `eprintln!` warnings**:

- `num_key_value_heads` defaults to `num_attention_heads` (MHA fallback)
- `max_position_embeddings` defaults to 2048
- `rope_theta` defaults to 10000.0 (warning: Qwen uses 1000000.0)
- `rms_norm_eps` defaults to 1e-6 (warning: some models use 1e-5)
- `attention_bias` defaults to `false`

The warnings are deliberate: a silent default for `rope_theta` causes
position encodings to wrap at the wrong frequency, producing degraded
perplexity that is difficult to diagnose.

## R-05: Explicit Demo Config Fallback

**Location:** `src/config/train/loader.rs`

When no `config.json` is found at all, the loader falls back to a Qwen2-0.5B
demo configuration and prints a `WARNING` to stderr. This path exists
exclusively for testing without a downloaded model. Production callers must
always provide a real `config.json`. The warning text includes "NOT suitable
for production training" to prevent silent misuse.

## N-06: Leaderboard Missing-Score Semantics

**Location:** `src/eval/evaluator/leaderboard.rs`, `Leaderboard::sort()`

When a model has no score for the ranking metric, `unwrap_or(missing)` maps
it to the worst possible value:

- **Higher-is-better** metrics (Accuracy, F1, BLEU): missing = `NEG_INFINITY`
- **Lower-is-better** metrics (MSE, Perplexity, WER): missing = `INFINITY`

This guarantees missing-score models sort **last**, never to an arbitrary
middle position. The same contract applies to `sort_by()` for secondary
metrics. Display methods render missing scores as an em-dash, never "0.0000",
because zero is a valid measurement.

## TransformerConfig Factory Methods

**Location:** `src/transformer/config.rs`

Factory methods (`llama2_7b()`, `mistral_7b()`, `qwen2_0_5b()`, `tiny()`)
produce validated configurations where all fields are internally consistent.
The `head_dim()` method computes `hidden_size / num_attention_heads`; callers
can assert divisibility as a postcondition.

These factories serve as the "known good" configurations for tests and
examples. The `tiny()` factory (hidden=64, heads=2, layers=2, vocab=1000)
is designed for unit tests that need a valid config without real model weights.

## Source of Truth

Entrenar's contracts are downstream of the aprender DbC spec
(`docs/specifications/design-by-contract.md` in the aprender repo). The
contract codes (C-10, C-11, N-06, R-04, R-05) are shared across the PAIML
stack so that error messages are grep-able across repositories.

## Running the Falsification Tests

```bash
# Run all falsification tests (names contain "falsify")
cargo test --lib -- falsify

# Run N-06 leaderboard contract tests specifically
cargo test --lib -- falsify_n06

# Run the Design by Contract example
cargo run --example design_by_contract
```

## Cross-References

- `src/config/train/loader.rs` -- C-10, C-11, R-04, R-05 implementation
- `src/eval/evaluator/leaderboard.rs` -- N-06 implementation and falsify tests
- `src/transformer/config.rs` -- Factory methods and `head_dim()`
- `src/eval/evaluator/metric.rs` -- `Metric::higher_is_better()` used by N-06
