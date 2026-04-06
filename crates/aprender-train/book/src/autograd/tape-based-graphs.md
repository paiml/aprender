# Tape-Based Computation Graphs

Entrenar uses a **tape-based** approach to record computational graphs during the forward pass and replay them in
reverse during backpropagation. This chapter explains how the tape works and why it's efficient.

## The Tape Metaphor

Think of the tape like a cassette recorder:

- **Forward pass**: Record each operation onto the tape
- **Backward pass**: Rewind and play back in reverse
- **Gradient computation**: Each operation knows how to propagate gradients

```
Forward (Recording):
  x → [Op1] → a → [Op2] → b → [Op3] → output
  Tape: [Op1, Op2, Op3]

Backward (Playback):
  dx ← [Op1*] ← da ← [Op2*] ← db ← [Op3*] ← dout=1.0
  Process tape in reverse: Op3* → Op2* → Op1*
```

## Tape Structure

Entrenar's tape stores operation metadata, not full tensors:

```rust
struct TapeEntry {
    operation: OpType,      // What operation (Add, Mul, MatMul, etc.)
    inputs: Vec<TensorId>,  // Input tensor IDs
    output: TensorId,       // Output tensor ID
    metadata: OpMetadata,   // Operation-specific data
}

enum OpType {
    Add,
    Mul,
    MatMul { rows, cols, batch },
    ReLU,
    LayerNorm,
    // ... etc
}
```

**Key insight**: We don't store actual tensor data on the tape, only **references** (IDs) and operation metadata.

## Example: Recording Operations

Let's trace a simple computation:

```rust
use entrenar::{Tensor, backward};

let x = Tensor::from_vec(vec![2.0], true);  // ID: 0
let y = Tensor::from_vec(vec![3.0], true);  // ID: 1

let a = &x + &y;  // ID: 2, records Add(0, 1) -> 2
let b = &a * &x;  // ID: 3, records Mul(2, 0) -> 3

backward(&b);
```

**Tape after forward pass:**

```
Tape = [
  Entry {
    operation: Add,
    inputs: [tensor_0_id, tensor_1_id],  // x, y
    output: tensor_2_id,                  // a
    metadata: {},
  },
  Entry {
    operation: Mul,
    inputs: [tensor_2_id, tensor_0_id],  // a, x
    output: tensor_3_id,                  // b
    metadata: {},
  },
]
```

## Backward Pass: Replaying the Tape

During `backward(&b)`, Entrenar processes the tape in **reverse order**:

### Step 1: Initialize Output Gradient

```rust
// db/db = 1.0 (seed gradient)
b.set_grad(vec![1.0]);
```

### Step 2: Process Tape Entry 1 (Mul)

```
Entry: Mul(a, x) -> b
Current: db = 1.0

Backward rule for Mul:
  da = db * x = 1.0 * 2.0 = 2.0
  dx += db * a = 1.0 * 5.0 = 5.0  (accumulate)

Update gradients:
  a.grad = [2.0]
  x.grad = [5.0]
```

### Step 3: Process Tape Entry 0 (Add)

```
Entry: Add(x, y) -> a
Current: da = 2.0

Backward rule for Add:
  dx += da * 1 = 2.0
  dy = da * 1 = 2.0

Update gradients:
  x.grad = [5.0 + 2.0] = [7.0]  (accumulated!)
  y.grad = [2.0]
```

### Final Gradients

```rust
println!("db/dx = {}", x.grad()[0]);  // 7.0 ✅
println!("db/dy = {}", y.grad()[0]);  // 2.0 ✅
```

**Verification** (manual chain rule):
```
b = a * x = (x + y) * x = x² + xy
db/dx = 2x + y = 2(2) + 3 = 7 ✅
db/dy = x = 2 ✅
```

## Gradient Accumulation

Notice that `x` appears twice in the computation graph:

```
    y
    │
    ▼
x ─┬─> Add -> a ─┐
   │              │
   └──────────────┴─> Mul -> b
```

**Gradients must accumulate** when a tensor has multiple consumers:

```rust
// First use: x in Add
dx_from_add = da * 1 = 2.0

// Second use: x in Mul
dx_from_mul = db * a = 5.0

// Total gradient (sum of paths)
dx_total = dx_from_add + dx_from_mul = 7.0
```

Entrenar handles this automatically via `+=` in gradient updates:

```rust
x.grad_mut()[i] += gradient_contribution;  // Accumulation
```

## Operation Metadata

Some operations need extra context for backward passes:

### Matrix Multiplication

```rust
Entry {
    operation: MatMul,
    inputs: [a_id, b_id],
    output: c_id,
    metadata: MatMulMeta {
        rows: 128,
        cols: 64,
        batch: 32,
    },
}
```

During backward:
```rust
// Need dimensions to compute dA = dC @ B^T
let dA = matmul(dC, B_transpose, rows, cols, batch);
```

### Layer Normalization

```rust
Entry {
    operation: LayerNorm,
    inputs: [x_id],
    output: y_id,
    metadata: LayerNormMeta {
        mean: 0.5,      // Saved from forward pass
        variance: 0.25,
    },
}
```

During backward:
```rust
// Need mean/variance from forward pass to compute gradients
let dx = layernorm_backward(dy, x, saved_mean, saved_variance);
```

## Memory Efficiency

Tape-based autograd is memory efficient because:

### 1. Store Operations, Not Tensors

**Bad** (store full tensors):
```rust
// Memory: O(n_ops * tensor_size)
struct TapeEntry {
    input_data: Vec<f32>,  // ❌ Wasteful
    output_data: Vec<f32>, // ❌ Wasteful
}
```

**Good** (store IDs):
```rust
// Memory: O(n_ops)
struct TapeEntry {
    input_ids: Vec<TensorId>,  // ✅ Just integers
    output_id: TensorId,        // ✅ Just one integer
}
```

### 2. Tensors Managed Separately

Tensors are reference-counted (`Rc<RefCell<TensorData>>`):

```rust
let x = Tensor::from_vec(vec![1.0, 2.0], true);
let y = &x * &x;  // y shares data with x via Rc

// When y is computed, x's data is still available
// Tape only stores IDs, not copies of data
```

### 3. Tape is Cleared After Backward

```rust
backward(&loss);  // Processes tape

// Tape is consumed and cleared
// Memory freed for next forward pass
```

## Dynamic Graphs

Entrenar's tape enables **dynamic computational graphs** - the graph can change every forward pass:

```rust
for epoch in 0..100 {
    let output = if epoch < 50 {
        // First 50 epochs: simple model
        &w1 * &x + &b1
    } else {
        // Last 50 epochs: complex model
        let h = relu(&(&w1 * &x + &b1));
        &w2 * &h + &b2
    };

    backward(&output);  // Different tape each epoch!
}
```

**Contrast with static graphs** (TensorFlow 1.x):
- Static: Define graph once, compile, reuse
- Dynamic (Entrenar): Build new graph every forward pass

**Trade-offs**:
- ✅ Dynamic: Flexible (control flow, debugging)
- ✅ Static: Faster (compiled optimizations)
- Entrenar chooses flexibility (similar to PyTorch)

## Tape Implementation Details

### Tape Creation

When you create a tensor with `requires_grad=true`:

```rust
let x = Tensor::from_vec(vec![1.0], true);
```

Entrenar initializes:
1. Tensor data storage
2. Gradient storage (same size as data)
3. Registration for tape recording

### Operation Recording

Every operation checks if recording is needed:

```rust
fn add(lhs: &Tensor, rhs: &Tensor) -> Tensor {
    // Forward computation
    let result_data = lhs.data() + rhs.data();

    // Check if we need to record
    if lhs.requires_grad() || rhs.requires_grad() {
        let result = Tensor::new(result_data, true);

        // Record on tape
        TAPE.with(|tape| {
            tape.borrow_mut().push(TapeEntry {
                operation: OpType::Add,
                inputs: vec![lhs.id(), rhs.id()],
                output: result.id(),
                metadata: {},
            });
        });

        result
    } else {
        // No gradients needed, skip tape
        Tensor::new(result_data, false)
    }
}
```

### Backward Execution

```rust
pub fn backward(loss: &Tensor) {
    // Seed gradient: dloss/dloss = 1.0
    loss.set_grad(vec![1.0]);

    // Get tape entries
    TAPE.with(|tape| {
        let entries = tape.borrow_mut().drain(..).collect::<Vec<_>>();

        // Process in reverse
        for entry in entries.into_iter().rev() {
            match entry.operation {
                OpType::Add => {
                    // Get output gradient
                    let grad_out = get_tensor(entry.output).grad();

                    // Propagate to inputs
                    get_tensor(entry.inputs[0]).accumulate_grad(&grad_out);
                    get_tensor(entry.inputs[1]).accumulate_grad(&grad_out);
                }
                OpType::Mul => {
                    let lhs = get_tensor(entry.inputs[0]);
                    let rhs = get_tensor(entry.inputs[1]);
                    let grad_out = get_tensor(entry.output).grad();

                    // d_lhs = grad_out * rhs
                    lhs.accumulate_grad(&(grad_out * rhs.data()));

                    // d_rhs = grad_out * lhs
                    rhs.accumulate_grad(&(grad_out * lhs.data()));
                }
                // ... other operations
            }
        }
    });
}
```

## Debugging the Tape

You can inspect the tape for debugging:

```rust
#[cfg(debug_assertions)]
fn print_tape() {
    TAPE.with(|tape| {
        println!("Tape contents:");
        for (i, entry) in tape.borrow().iter().enumerate() {
            println!("  [{}] {:?}", i, entry);
        }
    });
}

let x = Tensor::from_vec(vec![2.0], true);
let y = &x * &x;

print_tape();
// Output:
//   [0] Mul { inputs: [tensor_0, tensor_0], output: tensor_1 }
```

## Performance Considerations

### Tape Overhead

| Aspect | Cost | Mitigation |
|--------|------|------------|
| **Recording** | O(1) per operation | Minimal (just push to Vec) |
| **Storage** | O(n_ops) metadata | Small (typically <1MB for large models) |
| **Playback** | O(n_ops) | Necessary for gradients |

### Optimization: No-Grad Mode

Disable tape for inference:

```rust
// Inference (no tape recording)
let output = model.forward(&input);  // All tensors have requires_grad=false

// No tape entries created, faster forward pass
```

## Comparison with Graph-Based Autograd

| Aspect | Tape-Based (Entrenar) | Graph-Based (TensorFlow 1.x) |
|--------|-----------------------|-------------------------------|
| **Flexibility** | Dynamic (builds each forward) | Static (compile once) |
| **Debugging** | Easy (step through code) | Hard (symbolic graph) |
| **Performance** | Good (minimal overhead) | Excellent (compiled) |
| **Memory** | O(n_ops) | O(n_tensors + n_ops) |
| **Use Case** | Research, prototyping | Production at scale |

## Key Takeaways

1. **Tape records operations** during forward pass as metadata
2. **Backward replays tape in reverse** to propagate gradients
3. **Gradients accumulate** when tensors have multiple consumers
4. **Metadata stored** for operations needing forward pass values
5. **Dynamic graphs** rebuild tape each forward pass (flexibility)
6. **Memory efficient** - stores IDs and metadata, not full tensors

## What's Next?

- **[Backward Pass](./backward-pass.md)** - Detailed gradient propagation rules
- **[Gradient Computation](./gradient-computation.md)** - Chain rule mechanics
- **[Finite Difference Validation](./finite-difference.md)** - Testing gradient correctness

---

**Ready to understand backward passes?** Continue to [Backward Pass](./backward-pass.md) →
