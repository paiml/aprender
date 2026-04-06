# Depyler: Python → Rust

> **"Depyler transpiles Python to Rust with automatic type inference, NumPy→Trueno conversion, and sklearn→Aprender migration."**

## Overview

**Depyler** is Batuta's Python-to-Rust transpiler that converts Python projects into idiomatic Rust code with:

- **Automatic type inference**: Infers Rust types from Python code
- **NumPy → Trueno**: Converts NumPy operations to SIMD/GPU-accelerated Trueno
- **sklearn → Aprender**: Migrates scikit-learn to first-principles Aprender
- **PyTorch → Realizar**: Transpiles PyTorch inference to optimized Realizar
- **Project structure generation**: Creates full Cargo projects with dependencies

## Installation

```bash
# Install from crates.io
cargo install depyler

# Verify installation
depyler --version
# Output: depyler 3.20.0
```

## Basic Usage

### **Single File Transpilation**

```bash
# Transpile Python file to Rust
depyler transpile --input script.py --output script.rs

# View generated Rust code
cat script.rs
```

**Example:**

```python
# script.py
import numpy as np

def add_arrays(a, b):
    return np.add(a, b)

x = np.array([1, 2, 3])
y = np.array([4, 5, 6])
result = add_arrays(x, y)
print(result)
```

**Generated Rust:**

```rust
// script.rs
use trueno::Array;

fn add_arrays(a: &Array<f64>, b: &Array<f64>) -> Array<f64> {
    trueno::add(a, b)
}

fn main() {
    let x = Array::from_vec(vec![1.0, 2.0, 3.0]);
    let y = Array::from_vec(vec![4.0, 5.0, 6.0]);
    let result = add_arrays(&x, &y);
    println!("{:?}", result);
}
```

### **Project Transpilation**

```bash
# Transpile entire Python project
depyler transpile \
    --input /path/to/python_project \
    --output /path/to/rust_project \
    --format project

# Generated structure:
# rust_project/
# ├── Cargo.toml
# ├── src/
# │   ├── main.rs
# │   ├── lib.rs
# │   └── modules/
# ├── tests/
# └── benches/
```

## Batuta Integration

Batuta automatically uses Depyler for Python transpilation:

```bash
# Batuta detects Depyler and uses it
batuta transpile --input my_python_app --output my_rust_app
```

**Internal call:**
```bash
depyler transpile \
    --input my_python_app \
    --output my_rust_app \
    --format project
```

## ML Library Conversion

### **NumPy → Trueno**

Depyler converts NumPy operations to Trueno for SIMD/GPU acceleration:

| NumPy | Trueno | Backend |
|-------|--------|---------|
| `np.add(a, b)` | `trueno::add(&a, &b)` | SIMD/GPU |
| `np.dot(a, b)` | `trueno::dot(&a, &b)` | SIMD/GPU |
| `np.matmul(a, b)` | `trueno::matmul(&a, &b)` | GPU |
| `np.sum(a)` | `trueno::sum(&a)` | SIMD |
| `np.mean(a)` | `trueno::mean(&a)` | SIMD |

### **sklearn → Aprender**

Converts scikit-learn to first-principles Aprender:

| sklearn | Aprender |
|---------|----------|
| `LinearRegression()` | `aprender::LinearRegression::new()` |
| `LogisticRegression()` | `aprender::LogisticRegression::new()` |
| `KMeans(n_clusters=3)` | `aprender::KMeans::new(3)` |
| `StandardScaler()` | `aprender::StandardScaler::new()` |

### **PyTorch → Realizar**

Transpiles PyTorch inference to Realizar:

| PyTorch | Realizar |
|---------|----------|
| `model.generate(prompt)` | `realizar::generate_text(&model, prompt, max_len)` |
| `model.forward(x)` | `realizar::forward(&model, &x)` |
| `torch.load(path)` | `realizar::load_model(path)` |

## Features

### **Type Inference**

Depyler infers Rust types from Python:

```python
# Python (dynamic typing)
def multiply(x, y):
    return x * y

result = multiply(5, 10)  # int
```

```rust
// Rust (inferred types)
fn multiply(x: i32, y: i32) -> i32 {
    x * y
}

let result: i32 = multiply(5, 10);
```

### **Ownership Inference**

Converts Python references to Rust ownership:

```python
# Python
def process_list(items):
    items.append(42)
    return items
```

```rust
// Rust (mutable reference)
fn process_list(items: &mut Vec<i32>) -> &Vec<i32> {
    items.push(42);
    items
}
```

### **Error Handling**

Converts Python exceptions to Rust `Result`:

```python
# Python
def divide(a, b):
    if b == 0:
        raise ValueError("Division by zero")
    return a / b
```

```rust
// Rust
fn divide(a: f64, b: f64) -> Result<f64, String> {
    if b == 0.0 {
        Err("Division by zero".to_string())
    } else {
        Ok(a / b)
    }
}
```

## Command-Line Options

```bash
depyler transpile [OPTIONS]

OPTIONS:
    --input <PATH>      Input Python file or directory
    --output <PATH>     Output Rust file or directory
    --format <FORMAT>   Output format: file, project [default: file]
    --optimize <LEVEL>  Optimization level: 0, 1, 2, 3 [default: 2]
    --backend <BACKEND> Trueno backend: cpu, simd, gpu, auto [default: auto]
    --strict            Strict mode (fail on warnings)
    --no-ml             Disable ML library conversion
    -h, --help          Print help
    -V, --version       Print version
```

**Examples:**

```bash
# Strict mode (fail on type inference warnings)
depyler transpile --input script.py --output script.rs --strict

# Disable ML conversions (keep NumPy as-is)
depyler transpile --input ml_app.py --output ml_app.rs --no-ml

# Force GPU backend
depyler transpile --input gpu_code.py --output gpu_code.rs --backend gpu
```

## Limitations

Depyler has some known limitations:

- **Dynamic typing**: Complex dynamic types may require manual annotations
- **Metaprogramming**: Decorators and metaclasses not fully supported
- **C extensions**: Python C extensions cannot be transpiled
- **Runtime reflection**: `eval()`, `exec()`, `getattr()` limited support

**Workarounds:**

- Use type hints in Python code for better inference
- Refactor metaprogramming to explicit code
- Replace C extensions with pure Rust equivalents
- Avoid runtime reflection in critical paths

## Version

Current version: **3.20.0**

Check installed version:
```bash
depyler --version
```

Update to latest:
```bash
cargo install depyler --force
```

## Next Steps

- **[Bashrs: Shell → Rust](./bashrs.md)**: Shell script transpilation
- **[Trueno: Multi-target Compute](./trueno.md)**: SIMD/GPU acceleration
- **[Aprender: First-Principles ML](./aprender.md)**: ML algorithms in Rust

---

**Navigate:** [Table of Contents](../SUMMARY.md)
