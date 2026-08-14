<!-- PCU: lib-models | contract: contracts/apr-page-lib-models-v1.yaml -->

# Module: `aprender::models`

Public module of the `aprender-core` crate.

## Source

[`crates/aprender-core/src/models/mod.rs`](https://github.com/paiml/aprender/blob/main/crates/aprender-core/src/models/mod.rs) or directory.

## Example

<!-- example-cost: skip -->
```rust
use aprender::models::Qwen2Model;
// See `cargo doc -p aprender-core --open` for full API reference.
```

## Module summary

`aprender::models` is where pre-implemented transformer architectures live —
BERT (encoder, cross-encoder) for embeddings and reranking, Qwen2 for
decoder-only generation. These are reference implementations of the
architectures used by the trained checkpoints aprender ships, designed for
training and offline analysis. For production inference at scale, use the
realizar runtime (re-exported as `aprender-serve`); these in-crate models are
for training, fine-tuning, and architecture experiments.

## Key types

| Type | Description |
|------|-------------|
| `Qwen2Model` | Qwen2 decoder-only transformer with grouped-query attention and SwiGLU MLP. |
| `BertConfig` | Hyperparameters for BERT (hidden size, attention heads, layers, intermediate size, max positions). |
| `BertEncoder` | Pre-norm BERT encoder for sentence embeddings. |
| `CrossEncoder` | Cross-attention BERT used for pairwise scoring / reranking. |

Inside `models::qwen2`, the building blocks `Embedding`, `Qwen2MLP`,
`Qwen2DecoderLayer`, `GroupedQueryAttention`, and `Qwen2Config` are publicly
accessible so you can compose custom variants.

## Usage patterns

### Pattern 1: Construct a BERT encoder from a config

<!-- example-cost: skip -->
```rust
use aprender::models::{BertConfig, BertEncoder};

let config = BertConfig {
    hidden_size: 384,
    num_attention_heads: 12,
    num_hidden_layers: 6,
    intermediate_size: 1536,
    max_position_embeddings: 512,
    vocab_size: 30_522,
    ..BertConfig::default()
};
let encoder = BertEncoder::new(&config);
println!("encoder ready with {} layers", config.num_hidden_layers);
```

### Pattern 2: Inspect Qwen2 components

<!-- example-cost: skip -->
```rust
use aprender::models::qwen2::{Qwen2MLP, Embedding};

// MLP block — SwiGLU: gate_proj + up_proj into down_proj
let mlp = Qwen2MLP::placeholder(896, 4864);

// Token embedding table for a 151,936-token vocab
let emb = Embedding::placeholder(151_936, 896);
println!("hidden size = {}", emb.weight().shape()[1]);
```

## See also

- [`nn`](nn.md) — the layer primitives (`Linear`, `RMSNorm`, attention) these models build on
- [`format`](format.md) — load weights from GGUF / SafeTensors / APR
- [`text`](text.md) — tokenization (BPE, chat templates) for prompt preparation
- [`autograd`](autograd.md) — `Tensor` and backward used by these architectures during training

## Full API

Run `cargo doc -p aprender-core --open` for the rendered rustdoc, or browse
[docs.rs/aprender](https://docs.rs/aprender) for the published version.
