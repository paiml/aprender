<!-- PCU: lib-autograd | contract: contracts/apr-page-lib-autograd-v1.yaml -->

# Module: `aprender::autograd`

Public module of the `aprender-core` crate.

## Source

[`crates/aprender-core/src/autograd/mod.rs`](https://github.com/paiml/aprender/blob/main/crates/aprender-core/src/autograd/mod.rs) or directory.

## Example

<!-- example-cost: skip -->
```rust
use aprender::autograd::{Tensor, no_grad};
// See `cargo doc -p aprender-core --open` for full API reference.
```

## Module summary

`aprender::autograd` implements reverse-mode automatic differentiation in the
style of PyTorch. The central type is `Tensor` — a multidimensional array
that, when constructed with `requires_grad`, records the operations applied
to it in a global `ComputationGraph`. Calling `backward()` on a scalar tensor
then walks that graph backwards and accumulates gradients on the leaves. The
module also exposes the `no_grad` context for inference, helpers for
inspecting the graph, and the `GradFn` trait that custom ops implement.

## Key types

| Type | Description |
|------|-------------|
| `Tensor` | Owning n-D array. Has `data` (the values), `shape`, optional `grad`, and an autograd flag. |
| `TensorId` | Unique identifier used as a graph node key. |
| `ComputationGraph` | Global DAG of operations recorded for backward pass. |
| `GradFn` | Trait implemented by per-op backward functions (matmul, add, mean, …). |
| `no_grad`, `is_grad_enabled`, `clear_graph` | Context helpers to scope gradient tracking. |
| `get_grad(id)`, `clear_grad(id)` | Read or zero a leaf gradient by id. |

## Usage patterns

### Pattern 1: A simple gradient computation

<!-- example-cost: skip -->
```rust
use aprender::autograd::Tensor;

// y = x^2; dy/dx = 2x
let x = Tensor::from_slice(&[3.0]).requires_grad();
let y = x.clone() * x.clone();   // element-wise multiply
y.backward();

// gradient is 2*3 = 6
let g = x.grad().expect("leaf gets a gradient after backward");
assert!((g.item() - 6.0).abs() < 1e-5);
```

### Pattern 2: Inference without graph overhead via `no_grad`

<!-- example-cost: skip -->
```rust
use aprender::autograd::{Tensor, no_grad, is_grad_enabled};

let x = Tensor::from_slice(&[1.0, 2.0, 3.0]).requires_grad();

let preds: Vec<f32> = no_grad(|| {
    assert!(!is_grad_enabled(), "grad tracking is off inside no_grad");
    // Whatever forward pass you'd normally do here will not allocate
    // graph nodes — important for memory at inference time.
    x.data().iter().map(|v| v * 2.0).collect()
});
println!("preds: {:?}", preds);
```

## See also

- [`primitives`](primitives.md) — `Matrix` and `Vector` for non-differentiable work
- [`nn`](nn.md) — `Module` containers that build on `Tensor`
- [`loss`](loss.md) — loss functions whose backward pass feeds gradients to `Tensor` leaves
- [`optim`](optim.md) — optimizers that consume those gradients
- [`compute`](compute.md) — SIMD kernels invoked under the hood by tensor ops

## Full API

Run `cargo doc -p aprender-core --open` for the rendered rustdoc, or browse
[docs.rs/aprender](https://docs.rs/aprender) for the published version.
