# cuda-kernel-safety-v1

**Version:** 1.0.0

CUDA kernel safety contract for Decy transpiler

## References

- HPCTransCompile CUDA dataset [2506.10401]
- CASS NVIDIA to AMD transpilation [2505.16968]

## Equations

### host_transpilation

$$
forall f without CUDA qualifier: transpile(f) = normal Rust function
$$

**Domain:** $Host-side functions in .cu files$

**Codomain:** $Standard safe Rust functions$

**Invariants:**

- $Host functions use normal transpilation pipeline$
- $No FFI wrappers for host code$
- $Ownership inference applies normally$

### kernel_ffi

```
forall f with __global__: transpile(f) = extern "C" { fn name(raw_params); }
```

**Domain:** `CUDA __global__ kernel functions`

**Codomain:** $Rust extern "C" FFI declarations with raw pointer types$

**Invariants:**

- $Function name preserved in FFI declaration$
- $Pointer parameters become *mut T (raw pointers)$
- $Return type preserved (typically void)$
- $FFI declaration is inside extern "C" block$

### qualifier_preservation

$$
cuda_qualifier(AST) = cuda_qualifier(HIR) = cuda_qualifier(codegen input)
$$

**Domain:** $Functions with CUDA qualifiers through transformation pipeline$

**Codomain:** $Preserved qualifier at codegen stage$

**Invariants:**

- $Qualifier survives borrow_gen transformation$
- $Qualifier survives array_slice transformation$
- $Qualifier survives optimize transformation$

## Proof Obligations

| # | Type | Property | Formal |
|---|------|----------|--------|
| 1 | invariant | Kernel name preservation in FFI declaration |  |
| 2 | invariant | CUDA qualifier preservation through borrow/array/optimize transforms |  |
| 3 | postcondition | Host functions transpile without FFI wrapper |  |

## Falsification Tests

| ID | Rule | Prediction | If Fails |
|----|------|------------|----------|
| FALSIFY-CUDA-001 | __global__ kernel generates extern C | __global__ void kernel(int* a) -> extern "C" { fn kernel(a: *mut i32); } | Qualifier detection or FFI codegen broken |
| FALSIFY-CUDA-002 | Host code transpiles normally | void host_func(int x) in .cu file -> fn host_func(mut x: i32) | Host function incorrectly treated as kernel |
| FALSIFY-CUDA-003 | Qualifier preserved through ownership transform | __global__ qualifier survives borrow_gen and optimize | set_cuda_qualifier not called in transformation stage |
| FALSIFY-CUDA-004 | CUDA keywords detected in inline source | Source with __global__ enables C++ mode and empty macro definitions | Keyword detection regex or macro definitions broken |

## Kani Harnesses

| ID | Obligation | Bound | Strategy |
|----|------------|-------|----------|
| KANI-CUDA-001 | Qualifier preservation | 4 | bounded_int |

## QA Gate

**cuda-kernel-safety-v1 Contract** (F-CUDA-001)

CUDA kernel safety quality gate for Decy transpiler

**Checks:** validation, falsification

