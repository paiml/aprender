# PyTorch Kernel Extraction Pipeline

Version: 1.0
Status: proposed
Date: 2026-04-09

> Sub-spec of [pv-spec.md](../pv-spec.md) | Section 14

## Overview

`pv extract-pytorch` reads PyTorch reference implementations and generates:
1. YAML contract skeleton with equations extracted from docstrings/source
2. Lean 4 theorem stubs with the right types
3. Rust trait + failing test scaffold

PyTorch is the REFERENCE ORACLE — the ground truth for what a kernel SHOULD do.

## Pipeline

```
pytorch/torch/nn/functional.py::softmax
  ↓ pv extract-pytorch
contracts/softmax-kernel-v1.yaml (equations, pre/post, test refs)
  ↓ pv lean-gen
lean/ProvableContracts/Theorems/Softmax/*.lean (theorem stubs)
  ↓ human + LLM proves theorems
lean/ProvableContracts/Theorems/Softmax/*.lean (proven, no sorry)
  ↓ pv scaffold
src/kernels/softmax.rs (Rust trait + tests)
  ↓ implement + pv lint + pmat comply
DONE: equation → proof → contract → code → tests
```

## Extraction Rules

### From Python Docstrings

```python
def softmax(input, dim=None):
    r"""
    :math:`\text{Softmax}(x_{i}) = \frac{\exp(x_i)}{\sum_j \exp(x_j)}`
    elements lie in range [0, 1] and sum to 1
    """
```

Extracts:
- **formula**: `σ(x)_i = exp(x_i) / Σ_j exp(x_j)`
- **postcondition**: `ret ∈ [0, 1]`, `Σ ret = 1`
- **precondition**: `dim < input.ndim`

### From C++ Implementations

```cpp
// aten/src/ATen/native/SoftMax.cpp
// Uses log-sum-exp trick: subtract max for numerical stability
```

Extracts:
- **numerical variant**: `σ(x)_i = exp(x_i - max(x)) / Σ exp(x_j - max(x))`
- **edge case**: `max(x)` prevents overflow

### From Test Files

```python
# test/test_nn.py
def test_softmax():
    assert softmax(tensor([1,2,3])).sum() ≈ 1.0
```

Extracts:
- **falsification test reference**: `test_softmax_partition`

## CLI

```bash
# Extract from Python function
pv extract-pytorch pytorch/torch/nn/functional.py::softmax

# Extract from C++ kernel
pv extract-pytorch pytorch/aten/src/ATen/native/SoftMax.cpp

# Extract all known kernels
pv extract-pytorch --all pytorch/

# Generate Lean stubs from extracted contract
pv lean-gen contracts/softmax-kernel-v1.yaml
```

## Integration with apr-model-qa-playbook

```bash
# 1. Extract kernel from PyTorch
pv extract-pytorch pytorch/torch/nn/functional.py::layer_norm

# 2. QA playbook tests real models against the contract
apr-qa run --model meta-llama/Llama-3-8B --contract layer-norm-kernel-v1

# 3. Certification: model output matches contract postconditions
apr-qa certify --contract layer-norm-kernel-v1 --results qa-results.json
```

## Target Kernels

| Kernel | PyTorch Source | Lean Domain | Priority |
|--------|---------------|-------------|----------|
| softmax | functional.py | Softmax | Done |
| layer_norm | functional.py | LayerNorm | Done |
| rms_norm | functional.py | RMSNorm | Done |
| cross_entropy | functional.py | CrossEntropy | Done |
| gelu | Activation.cpp | Elementwise | High |
| silu | Activation.cpp | Sigmoid | Done |
| relu | functional.py | Elementwise | Done |
| linear | functional.py | MatMul | High |
| conv2d | Convolution.cpp | Conv2D | Medium |
| attention | functional.py | Attention | Critical |
| embedding | Embedding.cpp | Embedding | Medium |
| dropout | functional.py | Dropout | Low |
