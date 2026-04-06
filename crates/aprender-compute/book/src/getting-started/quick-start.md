# Quick Start

Get up and running with Trueno in 5 minutes.

## Your First Trueno Program

Let's build a simple vector addition program that automatically uses the best available SIMD backend.

### Create a New Project

```bash
cargo new trueno-quickstart
cd trueno-quickstart
```

### Add Trueno Dependency

Edit `Cargo.toml`:

```toml
[dependencies]
trueno = "0.14"
```

### Write the Code

Replace `src/main.rs`:

```rust
use trueno::Vector;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Create vectors from slices
    let a = Vector::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0]);
    let b = Vector::from_slice(&[10.0, 20.0, 30.0, 40.0, 50.0]);

    // Element-wise addition (automatically uses AVX2/SSE2/NEON)
    let sum = a.add(&b)?;
    println!("a + b = {:?}", sum.as_slice());
    // Output: [11.0, 22.0, 33.0, 44.0, 55.0]

    // Element-wise multiplication
    let product = a.mul(&b)?;
    println!("a * b = {:?}", product.as_slice());
    // Output: [10.0, 40.0, 90.0, 160.0, 250.0]

    // Dot product (reduction operation)
    let dot = a.dot(&b)?;
    println!("a · b = {}", dot);
    // Output: 550.0

    // Check which backend was selected
    println!("Using backend: {:?}", a.backend());

    Ok(())
}
```

### Run It

```bash
cargo run --release
```

**Expected output:**

```
a + b = [11.0, 22.0, 33.0, 44.0, 55.0]
a * b = [10.0, 40.0, 90.0, 160.0, 250.0]
a · b = 550.0
Using backend: Avx2  # (varies by CPU)
```

## Understanding What Just Happened

Let's break down the magic:

### 1. Automatic Backend Selection

```rust
let a = Vector::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0]);
```

When you create a `Vector`, Trueno:
1. Detects your CPU features (AVX2, SSE2, NEON, etc.)
2. Selects the best available backend
3. Stores this choice with the vector (no repeated detection)

**Backend priority:**
- ✅ AVX2 (4-8x faster) if available
- ✅ SSE2 (2-4x faster) as x86_64 baseline
- ✅ NEON (2-4x faster) on ARM64
- ✅ Scalar fallback (always works)

### 2. Safe, High-Level API

```rust
let sum = a.add(&b)?;  // Returns Result<Vector>
```

Trueno's API is:
- **100% safe Rust** - No `unsafe` in user code
- **Bounds-checked** - Size mismatches caught at runtime
- **Ergonomic** - Uses `?` operator for error handling

### 3. Zero-Copy Performance

```rust
println!("{:?}", sum.as_slice());
```

`as_slice()` returns a reference to internal data - no allocation or copying.

## Common Operations

### Element-Wise Operations

```rust
use trueno::Vector;

let a = Vector::from_slice(&[1.0, 2.0, 3.0, 4.0]);
let b = Vector::from_slice(&[5.0, 6.0, 7.0, 8.0]);

// Arithmetic
let sum = a.add(&b)?;      // [6.0, 8.0, 10.0, 12.0]
let diff = a.sub(&b)?;     // [-4.0, -4.0, -4.0, -4.0]
let prod = a.mul(&b)?;     // [5.0, 12.0, 21.0, 32.0]
let quot = a.div(&b)?;     // [0.2, 0.33, 0.43, 0.5]

// Scalar operations
let scaled = a.mul_scalar(2.0)?;  // [2.0, 4.0, 6.0, 8.0]
let offset = a.add_scalar(10.0)?; // [11.0, 12.0, 13.0, 14.0]
```

### Reduction Operations

```rust
let v = Vector::from_slice(&[1.0, 2.0, 3.0, 4.0]);

let sum = v.sum();      // 10.0
let mean = v.mean();    // 2.5
let min = v.min();      // 1.0
let max = v.max();      // 4.0
```

### Transformation Operations

```rust
let v = Vector::from_slice(&[1.0, 2.0, 3.0, 4.0]);

// Map function over elements
let squared = v.map(|x| x * x)?;  // [1.0, 4.0, 9.0, 16.0]

// Filter elements
let filtered = v.filter(|x| x > 2.0)?;  // [3.0, 4.0]

// Apply activation functions (coming in Phase 3)
// let activated = v.relu()?;
// let normalized = v.softmax()?;
```

## Error Handling

Trueno uses Rust's `Result` type for robust error handling:

```rust
use trueno::{Vector, TruenoError};

fn safe_divide() -> Result<Vector, TruenoError> {
    let a = Vector::from_slice(&[10.0, 20.0, 30.0]);
    let b = Vector::from_slice(&[2.0, 4.0]);  // Wrong size!

    // This returns Err(TruenoError::SizeMismatch)
    a.div(&b)
}

fn main() {
    match safe_divide() {
        Ok(result) => println!("Result: {:?}", result),
        Err(TruenoError::SizeMismatch { expected, actual }) => {
            eprintln!("Size mismatch: expected {}, got {}", expected, actual);
        }
        Err(e) => eprintln!("Error: {}", e),
    }
}
```

## Performance Tips

### 1. Use Release Mode

**Always** benchmark in release mode:

```bash
# ❌ Debug mode (10-100x slower!)
cargo run

# ✅ Release mode (full optimizations)
cargo run --release
```

### 2. Large Workloads for GPU

GPU backend only activates for large vectors (100K+ elements):

```rust
// ❌ Too small for GPU (uses SIMD)
let small = Vector::from_slice(&vec![1.0; 1000]);

// ✅ Large enough for GPU
let large = Vector::from_slice(&vec![1.0; 200_000]);
```

### 3. Batch Operations

Chain operations to minimize allocations:

```rust
// ❌ Multiple allocations
let temp1 = a.add(&b)?;
let temp2 = temp1.mul(&c)?;
let result = temp2.sub(&d)?;

// ✅ Better: use `map` for complex expressions
let result = a.zip(&b, &c, |a_i, b_i, c_i| {
    (a_i + b_i) * c_i - d_i
})?;
```

### 4. Reuse Buffers

For hot loops, reuse output buffers:

```rust
let mut output = Vector::zeros(1000);

for i in 0..iterations {
    // Writes into existing buffer (no allocation)
    a.add_into(&b, &mut output)?;
}
```

## What's Next?

Now that you've run your first Trueno program:

- **[Core Concepts](./core-concepts.md)** - Understand backends, safety, and performance
- **[First Program](./first-program.md)** - Build a more complex example
- **[API Reference](../api-reference/vector-operations.md)** - Explore all available operations
- **[Examples](../examples/vector-math.md)** - Real-world use cases
