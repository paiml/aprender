# Shell Autocomplete Demo

Real-time shell command autocomplete powered by a trained N-gram Markov model.

**This is the Presentar showcase demo** - demonstrating Zero-Infrastructure AI deployment with WASM.

## Overview

The Shell Autocomplete demo loads a trained `.apr` model file and provides intelligent command suggestions as you type. No server required - everything runs in the browser via WebAssembly.

```
User Input → WASM Runtime → N-gram Model → Suggestions
     ↓
  "git c" → ["git commit", "git checkout", "git clone", ...]
```

## Key Features

- **Zero Infrastructure**: No Python, no server, no cloud - pure WASM
- **Real Trained Model**: Uses `aprender-shell-base.apr` (not random weights)
- **Sub-millisecond Inference**: <1ms suggestion latency
- **Dynamic Model Loading**: Fetch models at runtime via `fromBytes()`
- **574KB Total Size**: WASM binary with embedded model

## Running the Demo

```bash
cd /home/noah/src/presentar
make serve
# Open http://localhost:8080/shell-autocomplete.html
```

## Architecture

### Model Format (APR)

```
┌──────────────────────────────────────────────────────────────┐
│ 32-byte Header                                               │
├──────────────────────────────────────────────────────────────┤
│ Magic: "APRN" (4 bytes)                                      │
│ Version: 1.0 (2 bytes)                                       │
│ Model Type: 0x0010 (N-gram LM)                               │
│ Metadata Size, Payload Size, Compression Type                │
├──────────────────────────────────────────────────────────────┤
│ Payload (zstd compressed, bincode serialized)                │
│ - N-gram counts: HashMap<context, HashMap<token, count>>     │
│ - Command frequencies: HashMap<command, frequency>           │
│ - Total command count                                        │
└──────────────────────────────────────────────────────────────┘
```

### YAML Configuration

```yaml
presentar: "1.0"
name: "shell-autocomplete"
version: "1.0.0"

models:
  shell:
    source: "./assets/aprender-shell-base.apr"
    format: "apr"

layout:
  type: "app"
  sections:
    - id: "input-section"
      widgets:
        - type: "autocomplete"
          id: "shell-input"
          placeholder: "Type a command..."
          model: "{{ models.shell }}"
          suggestions: "{{ models.shell | suggest(state.input, 8) }}"
```

### Expression Language

The `suggest` transform enables model inference in expressions:

```
{{ models.shell | suggest(prefix, count) }}
```

Returns an array of suggestion objects:
```json
{
  "suggestions": [
    {"text": "git commit", "score": 0.101},
    {"text": "git checkout", "score": 0.056}
  ]
}
```

## Model Statistics

| Metric | Value |
|--------|-------|
| Model Type | N-gram Markov (n=3) |
| Vocabulary | 400 unique commands |
| N-grams | 712 transitions |
| Memory | ~19 KB |
| File Size | 9.4 KB (compressed) |

## WASM API

### Dynamic Loading (Recommended)

```javascript
import init, { ShellAutocompleteDemo } from './pkg/presentar.js';

await init();

// Fetch model dynamically
const response = await fetch('./assets/aprender-shell-base.apr');
const bytes = new Uint8Array(await response.arrayBuffer());

// Create autocomplete with fetched model
const autocomplete = ShellAutocompleteDemo.fromBytes(bytes);

// Get suggestions
const result = JSON.parse(autocomplete.suggest("git ", 5));
console.log(result.suggestions);
```

### Embedded Model (Testing)

```javascript
// Uses model compiled into WASM binary
const autocomplete = new ShellAutocompleteDemo();
```

## Files

| File | Description |
|------|-------------|
| `www/shell-autocomplete.html` | Browser demo UI |
| `www/assets/aprender-shell-base.apr` | Runtime model file |
| `crates/presentar/src/browser/shell_autocomplete.rs` | Rust implementation |
| `examples/apr/shell_autocomplete.yaml` | YAML manifest |
| `docs/specifications/showcase-demo-aprender-shell-apr.md` | Full specification |

## Academic References

The N-gram model implementation is based on:

1. Chen & Goodman (1999). "An Empirical Study of Smoothing Techniques for Language Modeling"
2. Stolcke (2002). "SRILM - An Extensible Language Modeling Toolkit"

See the [specification](../../../docs/specifications/showcase-demo-aprender-shell-apr.md) for complete references.

## 10X Competitive Advantage

| Metric | Presentar | Streamlit | Gradio |
|--------|-----------|-----------|--------|
| Server Required | No | Yes | Yes |
| Python Required | No | Yes | Yes |
| Cold Start | <100ms | 2-5s | 2-5s |
| Inference Latency | <1ms | 50-200ms | 50-200ms |
| Offline Support | Full | None | None |
| Bundle Size | 574KB | N/A | N/A |
