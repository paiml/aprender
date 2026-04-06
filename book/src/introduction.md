# Presentar - Sovereign AI Visualization Framework

**Presentar** is a **WASM-first visualization and rapid application framework** built entirely on the **Sovereign AI Stack**—a vertically integrated Rust ecosystem (Trueno, Aprender, Alimentar, Pacha) that eliminates Python/CUDA/cloud dependencies for fully self-hosted AI workloads.

## Why Presentar?

Unlike Streamlit, Gradio, or Panel which suffer from Python's GIL, poor testability, and runtime overhead, Presentar delivers:

- **60fps GPU-accelerated rendering** via WebGPU/WGSL shaders
- **Compile-time type safety** with zero runtime interpretation
- **Deterministic reproducibility** for every render
- **Zero external testing dependencies** - pure Rust test harness

## Core Principles

| Principle | Implementation |
|-----------|----------------|
| **80% Pure Stack** | All rendering via `trueno-viz` GPU primitives |
| **20% Minimal External** | Only `winit` (windowing) and `fontdue` (fonts) |
| **WASM-First** | Browser deployment without server dependencies |
| **YAML-Driven** | Declarative app configuration |
| **Graded Quality** | Every app receives F-A score via TDG metrics |

## Toyota Way Foundation

Presentar is built on Toyota Production System principles:

- **Muda (Waste Elimination)**: No Python GIL, no runtime interpretation
- **Jidoka (Built-in Quality)**: Compiler-enforced correctness
- **Kaizen (Continuous Improvement)**: Three-tier quality pipeline
- **Poka-yoke (Mistake Proofing)**: Strict schema validation

## Quick Example

```yaml
# app.yaml - A simple dashboard
presentar: "0.1"
name: "my-dashboard"

layout:
  type: "dashboard"
  sections:
    - id: "header"
      widgets:
        - type: "text"
          content: "Welcome to Presentar"
          style: "heading-1"

    - id: "metrics"
      widgets:
        - type: "metric"
          label: "Users"
          value: "{{ data.users | count }}"
```

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│  Layer 9: App Runtime                                           │
│  - YAML parser, .apr/.ald loaders, Pacha integration            │
├─────────────────────────────────────────────────────────────────┤
│  Layer 8: Presentar (Reactive UI Framework)                     │
│  - Widget tree, layout engine, event dispatch, state management │
├─────────────────────────────────────────────────────────────────┤
│  Layer 7: Trueno-Viz (GPU Rendering Primitives)                 │
│  - Paths, fills, strokes, text, charts, WGSL shaders            │
├─────────────────────────────────────────────────────────────────┤
│  Layer 6: Trueno (SIMD/GPU Compute)                             │
│  - Tensor ops, backend dispatch, memory management              │
└─────────────────────────────────────────────────────────────────┘
```

## What's in This Book?

This book covers:

1. **Getting Started** - Installation, quick start, first app
2. **Architecture** - Layer hierarchy, data flow, widget tree
3. **Widget System** - All built-in widgets and custom widget creation
4. **Layout** - Flexbox model, constraints, responsive design
5. **YAML Manifest** - Configuration schema, expressions, theming
6. **Testing** - Zero-dependency test harness, visual regression
7. **Quality** - Scoring system, grades, gates
8. **Examples** - Real-world applications

## Prerequisites

- Rust 1.75+ with `wasm32-unknown-unknown` target
- Basic familiarity with reactive UI concepts
- Understanding of YAML syntax

Let's get started!
