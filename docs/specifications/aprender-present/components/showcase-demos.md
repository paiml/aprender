# Showcase Demos

> Parent: [presentar-spec.md](../presentar-spec.md)

**Scope:** Shell command autocomplete demo, WASM integration, QA verification checklists.

---

## Shell Command Autocomplete Demo

A WASM-first showcase demonstrating real ML inference using the Sovereign AI Stack. Uses a genuinely trained N-gram Markov model (`aprender-shell-base.apr`) for intelligent shell command predictions.

### Key Properties

| Property | Value |
|----------|-------|
| Model | `aprender-shell-base.apr` |
| Model Type | N-gram Language Model (3-gram Markov) |
| Training Data | 404 developer commands |
| Model Size | ~10 KB |
| Inference | Client-side WASM only |
| External Dependencies | Zero (no API calls) |

### Model Architecture

3-gram Markov chain for sequential token prediction:
- `ngrams: HashMap<String, HashMap<String, u32>>` -- Context to NextToken to Count
- `command_freq: HashMap<String, u32>` -- Full command frequency ranking
- `trie: Trie` -- O(k) prefix lookup

### Inference Algorithm

```
suggest(prefix, count):
  1. Trie lookup: commands starting with prefix
  2. N-gram lookup: P(next_token | context)
  3. Score fusion: trie_score * 1.0 + ngram_score * 0.8
  4. Sort by score, truncate to top-K
  5. Return [(suggestion, score), ...]
```

### Training Corpus

| Category | Commands | % |
|----------|----------|---|
| Git | ~120 | 30% |
| Cargo | ~80 | 20% |
| Docker | ~60 | 15% |
| Kubectl | ~50 | 12% |
| npm/yarn | ~40 | 10% |
| System | ~51 | 13% |

**Validation (80/20 split):** Hit@1: 0.35, Hit@5: 0.62, Hit@10: 0.78, MRR: 0.48

### WASM Integration

```rust
#[wasm_bindgen]
pub fn showcase_suggest(prefix: &str, count: usize) -> JsValue;
#[wasm_bindgen]
pub fn showcase_model_info() -> JsValue;
#[wasm_bindgen]
pub fn showcase_init() -> bool;
```

Model embedded at compile time via `include_bytes!`. No `std::fs` or `std::net` (WASM compatible). Memory budget: < 5 MB.

### Performance Targets

| Metric | Target |
|--------|--------|
| Suggestion latency | < 1ms |
| Model load time | < 50ms |
| Memory footprint | < 10 MB |
| Bundle size (gzip) | < 150 KB |
| Typing at 60fps | < 16ms/frame |

## Implementation Issues

| # | Title | Priority | Status |
|---|-------|----------|--------|
| 1 | EPIC: Shell Command Autocomplete | - | OPEN |
| 2 | WASM Model Loader | P0 | OPEN |
| 3 | N-gram Inference Engine | P0 | OPEN |
| 4 | WASM Bindings + JS Interop | P0 | OPEN |
| 5 | Integration Tests + Quality Gates | P1 | OPEN |

## Quality Checklists

### Model Integrity (MI-001 to MI-010)

| # | Check | Status |
|---|-------|--------|
| MI-001 | Valid APRN magic bytes | PASS |
| MI-002 | Model type NgramLm (0x0010) | PASS |
| MI-003 | CRC32 checksum validates | PASS |
| MI-004 | Trained on documented corpus | PASS |
| MI-005 | Corpus committed to repository | PASS |
| MI-006 | Reproducible with fixed seed | PASS |
| MI-007 | Validation metrics documented | PASS |
| MI-008 | No data leakage | PASS |
| MI-009 | PII audit passed | PASS |
| MI-010 | Corpus uses synthetic placeholders | PASS |

### Inference Correctness (IC-001 to IC-010)

| # | Check | Status |
|---|-------|--------|
| IC-001 | `suggest("git ", 5)` returns git commands only | PASS |
| IC-002 | `suggest("cargo ", 5)` returns cargo commands | PASS |
| IC-003 | Empty prefix returns most frequent | PASS |
| IC-004 | Partial completion: "git c" -> "git commit" | PASS |
| IC-005 | Scores in [0, 1] | PASS |
| IC-006 | Results sorted by descending score | PASS |
| IC-007 | No corrupted suggestions | PASS |
| IC-008-010 | Empty/unicode/deterministic | PASS |

### WASM Build (WB-001 to WB-008)

| # | Check | Status |
|---|-------|--------|
| WB-001 | Target wasm32-unknown-unknown | PASS |
| WB-002 | No std::fs/std::net in WASM path | PASS |
| WB-003 | Model embedded at compile time | PASS |
| WB-005 | Bundle < 500KB gzipped | PENDING |
| WB-008 | Memory < 10MB at runtime | PASS |

## Demo QA Verification (100 Points)

### A. Performance Claims (20 pts)

| # | Check | Criteria |
|---|-------|---------|
| A1-A5 | Frame rate (Chrome/Firefox/throttled/burst/accuracy) | >= 55fps mean, P99 >= 45fps |
| A6-A10 | Latency (click, animation, data update, theme, inference) | < 16-100ms |
| A11-A15 | Throughput (candlesticks, particles, draw calls, memory, stress) | Exact counts, < 10% growth |
| A16-A20 | Rust/WASM (benchmarks, build, instantiation, native vs WASM) | 60fps, < 50ms init, within 2x |

### B. Size & Efficiency (15 pts)

| # | Check | Criteria |
|---|-------|---------|
| B1-B5 | Bundle (WASM, HTML, total, gzip, vs Gradio) | < 500KB WASM, < 600KB total |
| B6-B10 | Memory (initial, 1min, DOM nodes, canvas, particles) | < 20MB initial, < 50MB sustained |
| B11-B15 | Startup (FP, TTI, FCP, cold start, vs Streamlit) | < 200ms FP, < 500ms TTI |

### C. Data Format Integrity (15 pts)
- `.apr` magic bytes, layer count, param count, roundtrip (7 checks)
- `.ald` magic bytes, tensor count/shapes, OHLC validity, roundtrip (8 checks)

### D. Visualization Accuracy (15 pts)
- Candlestick chart (count, coloring, Y-axis, current price, wicks)
- Bar chart (count, heights, labels, month order, animation)
- Donut chart (segments, proportions, center total, rotation, color)

### E-H. Animation, Cross-Platform, Code Quality, Claims (25 pts)
- Smooth animations, particle physics, browser compatibility (Chrome/Firefox/Safari/Edge/mobile)
- All tests pass, no console errors, HTML validates, deterministic output
- Each marketing claim verified with measurement evidence

### Scoring

| Grade | Score |
|-------|-------|
| A+ | 95-100 (Production ready) |
| A | 90-94 (Safe to ship) |
| B | 80-89 (Significant issues) |
| C | 70-79 (Major issues, do not ship) |
| F | < 70 (Redesign required) |

## Security

- No telemetry (zero network calls)
- No persistence (suggestions not stored)
- No PII in corpus (synthetic placeholders only)
- SHA256 verification on model load

## File Hashes

```
corpus/developer-commands.txt:  SHA256=21ef9092...
models/aprender-shell-base.apr: SHA256=068ac67a...
```

## References

- Chen, S.F. & Goodman, J. (1999). Smoothing techniques for LMs. *Computer Speech & Language*.
- Davison, B.D. & Hirsh, H. (1998). Predicting sequences of user actions. *AAAI Workshop*.
- Fredkin, E. (1960). Trie memory. *CACM*.
- Haas, A. et al. (2017). WebAssembly. *PLDI '17*.
- Sculley, D. et al. (2015). Hidden Technical Debt in ML. *NeurIPS*.
