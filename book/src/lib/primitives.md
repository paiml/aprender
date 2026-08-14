<!-- PCU: lib-primitives | contract: contracts/apr-page-lib-primitives-v1.yaml -->

# Module: `aprender::primitives`

Public module of the `aprender-core` crate.

## Source

[`crates/aprender-core/src/primitives/mod.rs`](https://github.com/paiml/aprender/blob/main/crates/aprender-core/src/primitives/mod.rs) or directory.

## Example

```rust
use aprender::primitives::{Vector, Matrix};
// See `cargo doc -p aprender-core --open` for full API reference.
```

## Module summary

`aprender::primitives` exposes the two foundational tensor types that every
other aprender module is built on: `Vector<T>` (1-D, contiguous, generic over
element type) and `Matrix<T>` (2-D, row-major, generic). Both are also
re-exported at the crate root and via `aprender::prelude::*` because almost
every ML signature takes a `Matrix<f32>` of features and a `Vector<f32>` of
targets.

## Key types

| Type | Description |
|------|-------------|
| `Vector<T>` | Owning, contiguous 1-D buffer. Implements `Index`/`IndexMut` plus arithmetic helpers. |
| `Matrix<T>` | Owning 2-D buffer in row-major layout. `get` / `set` are O(1); `matmul` / `matvec` use Trueno SIMD where available. |

`Matrix<f32>` adds linear-algebra entry points: `transpose`, `matmul`,
`matvec`, `add`, `sub`, `mul_scalar`, and `cholesky_solve` for SPD systems.
`Vector<f32>` adds reductions (`sum`, `mean`, `variance`, `std`, `norm`),
inner-product (`dot`), and `argmin` / `argmax`.

## Usage patterns

### Pattern 1: Build a matrix and inspect shape

```rust
use aprender::primitives::Matrix;

let m = Matrix::from_vec(2, 3, vec![
    1.0, 2.0, 3.0,
    4.0, 5.0, 6.0_f32,
]).expect("2x3 matrix");
assert_eq!(m.shape(), (2, 3));
assert_eq!(m.get(1, 2), 6.0);

let t = m.transpose();
assert_eq!(t.shape(), (3, 2));
```

### Pattern 2: Vector reductions and Cholesky solve

```rust
use aprender::primitives::{Matrix, Vector};

let v = Vector::from_slice(&[1.0, 2.0, 3.0, 4.0]);
assert!((v.mean() - 2.5).abs() < 1e-6);
assert!((v.norm() - (30.0_f32).sqrt()).abs() < 1e-5);

// SPD system Ax = b
let a = Matrix::from_vec(2, 2, vec![
    4.0, 1.0,
    1.0, 3.0,
]).expect("2x2");
let b = Vector::from_slice(&[1.0, 2.0]);
let x = a.cholesky_solve(&b).expect("SPD solve");
println!("x = {:?}", x.as_slice());
```

## See also

- [`compute`](compute.md) — Trueno-backed SIMD kernels that accelerate primitive ops
- [`autograd`](autograd.md) — differentiable `Tensor` wrapper built on top of `Vector`/`Matrix`
- [`traits`](traits.md) — the core traits whose signatures consume these primitives
- [`format`](format.md) — APR / SafeTensors / GGUF serialization of `Matrix`-shaped weights

## Full API

Run `cargo doc -p aprender-core --open` for the rendered rustdoc, or browse
[docs.rs/aprender](https://docs.rs/aprender) for the published version.
