<!-- PCU: lib-nn | contract: contracts/apr-page-lib-nn-v1.yaml -->

# Module: `aprender::nn`

Public module of the `aprender-core` crate.

## Source

[`crates/aprender-core/src/nn/mod.rs`](https://github.com/paiml/aprender/blob/main/crates/aprender-core/src/nn/mod.rs) or directory.

## Example

<!-- example-cost: skip -->
```rust
use aprender::nn::{Sequential, Linear, ReLU};
// See `cargo doc -p aprender-core --open` for full API reference.
```

## Module summary

`aprender::nn` is the neural-network toolkit: layers, activations, containers,
initializers, normalization, RNNs, transformer building blocks, dropout
variants, and a `Module` trait that ties everything together. The design
mirrors `torch.nn` — you compose layers in a `Sequential` container, apply
forward passes, and let `autograd` handle backprop.

## Key types

| Type | Description |
|------|-------------|
| `Module` | Core trait. Every layer / container implements `forward` and `parameters`. |
| `Sequential`, `ModuleList`, `ModuleDict` | Containers for composing layers in order or by name. |
| `Linear` | Fully-connected layer. |
| `ReLU`, `GELU`, `Sigmoid`, `Softmax`, `Tanh`, `LeakyReLU` | Activations. |
| `LayerNorm`, `RMSNorm`, `BatchNorm1d`, `GroupNorm`, `InstanceNorm` | Normalization layers (RMSNorm is what Llama/Qwen use). |
| `Dropout`, `Dropout2d`, `AlphaDropout`, `DropBlock`, `DropConnect` | Regularization. |
| `LSTM`, `GRU`, `Bidirectional` | Recurrent building blocks. |

The submodule `nn::transformer` exposes attention + transformer-block types;
`nn::optim` mirrors PyTorch's optimizer API for module-level training loops;
`nn::scheduler` exposes learning-rate schedulers; `nn::quantization` and
`nn::ssm` are specialized.

## Usage patterns

### Pattern 1: An MLP via `Sequential`

<!-- example-cost: skip -->
```rust
use aprender::nn::{Sequential, Linear, ReLU};

// Compose: 4 inputs → 8 hidden (ReLU) → 2 outputs
let mut model = Sequential::new();
model.add(Linear::new(4, 8));
model.add(ReLU::new());
model.add(Linear::new(8, 2));

println!("layers: {}", model.len());
```

### Pattern 2: Use RMSNorm (transformer-style)

<!-- example-cost: skip -->
```rust
use aprender::nn::{Linear, RMSNorm};
use aprender::nn::module::Module;

let norm = RMSNorm::new(64, 1e-6);
let proj = Linear::new(64, 64);

// In a transformer block you'd run: norm -> attention -> residual.
println!("RMSNorm has {} params", norm.parameters().len());
```

## See also

- [`autograd`](autograd.md) — `Tensor` and backward-pass machinery used by `Module`
- [`loss`](loss.md) — losses to drive backprop
- [`optim`](optim.md) — top-level stochastic optimizers (also re-exported as `nn::optim`)
- [`models`](models.md) — full transformer models built from these primitives
- [`regularization`](regularization.md) — non-dropout regularizers (Mixup, CutMix, label smoothing)

## Full API

Run `cargo doc -p aprender-core --open` for the rendered rustdoc, or browse
[docs.rs/aprender](https://docs.rs/aprender) for the published version.
