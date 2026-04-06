# PyTorch to Realizar Integration

Batuta's `PyTorchConverter` maps PyTorch inference patterns to the `realizar`
inference engine. This conversion is **inference-only** -- training loops are
out of scope. Models must first be exported to GGUF or SafeTensors format.

## Model Loading

**Python (PyTorch / Transformers)**

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("model_name")
tokenizer = AutoTokenizer.from_pretrained("model_name")
```

**Rust (Realizar)**

```rust
use realizar::gguf::GGUFModel;
use realizar::tokenizer::Tokenizer;

let model = GGUFModel::from_file("model.gguf")?;
let tokenizer = Tokenizer::from_file("tokenizer.json")?;
```

Realizar loads GGUF and SafeTensors formats natively. GGUF column-major data is
automatically transposed to row-major at import time (see LAYOUT-002 in the
architecture docs). SafeTensors data is already row-major and loads directly.

## Text Generation

**Python (PyTorch)**

```python
inputs = tokenizer("Hello, world!", return_tensors="pt")
with torch.no_grad():
    outputs = model.generate(**inputs, max_length=50)
text = tokenizer.decode(outputs[0])
```

**Rust (Realizar)**

```rust
use realizar::generate::generate_text;

let tokens = tokenizer.encode("Hello, world!")?;
let output = generate_text(&model, &tokens, 50)?;
let text = tokenizer.decode(&output)?;
```

The `torch.no_grad()` context manager is unnecessary in Realizar because the
engine is inference-only by design. There is no autograd graph to disable.

## Forward Pass

**Python (PyTorch)**

```python
model.eval()
with torch.no_grad():
    logits = model(input_tensor)
```

**Rust (Realizar)**

```rust
let logits = model.forward(&input_tensor)?;
```

The `model.eval()` and `torch.no_grad()` guards map to nothing in Realizar.
The model is always in inference mode.

## Layer-Level Conversion

For custom architectures, individual layers have direct equivalents:

| PyTorch (`torch.nn`)       | Realizar                            |
|----------------------------|-------------------------------------|
| `nn.Linear(768, 512)`     | `LinearLayer::new(768, 512)`        |
| `nn.Embedding(50000, 512)`| `EmbeddingLayer::new(50000, 512)`   |
| `nn.LayerNorm(512)`       | `LayerNormLayer::new(512)`          |
| `nn.MultiheadAttention`   | `AttentionLayer::new(512, 8)`       |
| `nn.GELU()`               | `gelu(&input)`                      |
| `nn.Softmax(dim=-1)`      | `softmax(&input)`                   |

## Supported Model Formats

| Format       | Layout      | Loading                          |
|-------------|-------------|----------------------------------|
| GGUF        | Column-major| Transposed to row-major at load  |
| SafeTensors | Row-major   | Direct zero-copy loading         |
| APR v2      | Row-major   | Native format with LZ4/ZSTD     |

The APR v2 format (`.apr`) is the stack's native serialization. It supports
LZ4 and ZSTD tensor compression and full zero-copy loading. Models converted
through `aprender`'s import pipeline produce APR v2 files.

## Backend Selection

Inference operations are high-complexity by default. The MoE backend selector
routes based on model and batch size:

| Operation | Small Batch | Large Batch |
|-----------|-------------|-------------|
| Forward   | SIMD        | GPU         |
| Generate  | SIMD        | GPU         |
| Attention | SIMD        | GPU         |

For single-token generation (batch size 1), SIMD typically wins because the
PCIe transfer overhead dominates. Batch inference above the 5x threshold routes
to GPU automatically.

## Key Takeaways

- PyTorch conversion is inference-only. Export models to GGUF or SafeTensors
  before conversion.
- `torch.no_grad()` and `model.eval()` have no Realizar equivalent because
  the engine is always in inference mode.
- GGUF column-major data is transposed automatically at load time (LAYOUT-002).
- Individual `torch.nn` layers have direct Realizar equivalents for custom
  architectures.
- APR v2 is the recommended native format for production deployment.

---

**Navigate:** [Table of Contents](../SUMMARY.md)
