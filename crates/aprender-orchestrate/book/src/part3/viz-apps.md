# Visualization & Apps

The Sovereign AI Stack includes a complete visualization and application layer built on GPU-accelerated primitives. This eliminates the need for Python-based tools like Streamlit, Gradio, or Panel.

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│  Presentar (App Framework)                                      │
│  - YAML-driven configuration                                    │
│  - Auto-display for .apr/.ald files                             │
│  - Quality scoring (F-A grade)                                  │
├─────────────────────────────────────────────────────────────────┤
│  Trueno-Viz (GPU Rendering) v0.1.1                              │
│  - WGSL shaders for paths, fills, text                          │
│  - WebGPU + WASM targets                                        │
│  - 60fps rendering pipeline                                     │
├─────────────────────────────────────────────────────────────────┤
│  Trueno (Compute Foundation) v0.7.3                             │
│  - SIMD vectorization                                           │
│  - GPU compute dispatch                                         │
│  - Backend: CPU/WASM/WebGPU                                     │
└─────────────────────────────────────────────────────────────────┘
```

## Components

| Component | Version | Purpose |
|-----------|---------|---------|
| **Trueno-Viz** | 0.1.1 | GPU rendering primitives (paths, fills, text, charts) |
| **Presentar** | 0.1.0 | YAML-driven app framework with auto-display |

## Design Principles

Following the Toyota Way:

- **Muda (Waste Elimination):** No Python GIL, no runtime interpretation, no server round-trips
- **Jidoka (Built-in Quality):** Compile-time type safety, deterministic rendering
- **Poka-yoke (Mistake Proofing):** Schema validation at load time, not runtime

## 80/20 Rule

The visualization layer follows the stack's 80/20 principle:

- **80% Pure Stack:** All rendering via Trueno-Viz GPU primitives (WGSL shaders)
- **20% Minimal External:**
  - `winit` for cross-platform windowing (WASM lacks native window APIs)
  - `fontdue` for font rasterization (platform-specific font hinting)

## Use Cases

1. **Model Dashboards:** Display Aprender model performance metrics
2. **Data Exploration:** Interactive views of Alimentar datasets
3. **Inference UIs:** Real-time prediction interfaces
4. **Quality Reports:** TDG score visualization

## Further Reading

- [Trueno-Viz: GPU Rendering](./trueno-viz.md) - Low-level rendering primitives
- [Presentar: App Framework](./presentar.md) - High-level application framework

---

**Navigate:** [Table of Contents](../SUMMARY.md) | [Foundation Libraries](./foundation-libs.md)
