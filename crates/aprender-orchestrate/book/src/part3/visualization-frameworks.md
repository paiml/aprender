# Visualization Frameworks Integration

Batuta provides ecosystem visualization for Python data visualization and ML demo frameworks, showing how they map to sovereign Rust replacements. The `batuta viz` command displays framework hierarchies and PAIML replacement mappings.

## Core Principle

**Python visualization frameworks are replaced by sovereign Rust alternatives.** No Python runtime dependencies are permitted in the PAIML stack. Python code is transpiled to Rust via Depyler.

## Framework Replacement Matrix

| Python Framework | PAIML Replacement | Migration Path |
|------------------|-------------------|----------------|
| **Gradio** | Presentar | Depyler transpilation |
| **Streamlit** | Presentar | Depyler transpilation |
| **Panel** | Trueno-Viz | Depyler transpilation |
| **Dash** | Presentar + Trueno-Viz | Depyler transpilation |
| **Matplotlib** | Trueno-Viz | Direct API mapping |
| **Plotly** | Trueno-Viz | Direct API mapping |

## Toyota Way Principles

| Principle | Application |
|-----------|-------------|
| **Genchi Genbutsu** | Direct visualization enables first-hand observation |
| **Poka-Yoke** | Python interpreter eliminated from production |
| **Heijunka** | Frame-rate limiting prevents GPU saturation |
| **Jidoka** | Explicit component trees for predictable rendering |
| **Muda** | Signal-based rendering eliminates wasted computation |
| **Kanban** | Visual data flow with explicit signal graphs |

## CLI Usage

### View All Frameworks

```bash
batuta viz tree
```

Output:
```
VISUALIZATION FRAMEWORKS ECOSYSTEM
==================================

GRADIO (Python) → Presentar (Rust)
├── Interface
│   └── Interface → Presentar::QuickApp
├── Blocks
│   └── Blocks → Presentar::Layout
├── Components
│   ├── Image → Trueno-Viz::ImageView
│   ├── Audio → Presentar::AudioPlayer
│   ├── Chatbot → Realizar + Presentar
│   └── DataFrame → Trueno-Viz::DataGrid
└── Deployment
    └── HuggingFace Spaces → Batuta deploy

STREAMLIT (Python) → Presentar (Rust)
├── Widgets
│   ├── Input → Presentar::Widgets
│   └── Display → Presentar + Trueno-Viz
├── Caching
│   ├── @st.cache_data → Trueno::TensorCache
│   └── session_state → Presentar::State
└── Deployment
    └── Streamlit Cloud → Batuta deploy
...
```

### Filter by Framework

```bash
batuta viz tree --framework gradio
batuta viz tree --framework streamlit
batuta viz tree --framework panel
batuta viz tree --framework dash
```

### View PAIML Replacement Mappings

```bash
batuta viz tree --integration
```

Output:
```
PAIML REPLACEMENTS FOR PYTHON VIZ
=================================

UI FRAMEWORKS
├── [REP] Presentar::QuickApp ← gr.Interface
├── [REP] Presentar::Layout ← gr.Blocks
├── [REP] Presentar::App ← dash.Dash
├── [REP] Presentar::Layout ← st.columns/sidebar

VISUALIZATION
├── [REP] Trueno-Viz::Chart ← dcc.Graph
├── [REP] Trueno-Viz::Chart ← st.plotly_chart
├── [REP] Trueno-Viz::DataGrid ← st.dataframe
├── [REP] Trueno-Viz::GPURaster ← datashader

COMPONENTS
├── [REP] Presentar::TextInput ← st.text_input
├── [REP] Presentar::Slider ← st.slider
├── [REP] Trueno-Viz::ImageView ← gr.Image

STATE & CACHING
├── [REP] Presentar::State ← st.session_state
├── [REP] Trueno::TensorCache ← @st.cache_data
├── [REP] Presentar::on_event ← @callback

DEPLOYMENT
├── [REP] Batuta deploy ← HuggingFace Spaces
├── [REP] Batuta deploy ← Streamlit Cloud
├── [REP] Batuta deploy ← Dash Enterprise

Legend: [REP]=Replaces (Python eliminated)

Summary: 21 Python components replaced by sovereign Rust alternatives
         Zero Python dependencies in production
```

### JSON Output

```bash
batuta viz tree --format json
batuta viz tree --framework streamlit --format json
batuta viz tree --integration --format json
```

## Why Replace Python Frameworks?

### Gradio → Presentar

**Problems with Gradio:**
- Python server restarts on every interaction
- ~2s cold start time
- ~100ms interaction latency
- No offline capability

**Presentar Benefits:**
- Persistent state with sub-millisecond updates
- ~50ms cold start
- ~16ms interaction latency (60fps)
- WebAssembly deployment for edge/offline

### Streamlit → Presentar

**Problems with Streamlit:**
- Full script reruns on each interaction (Muda)
- ~3s cold start, ~200ms latency
- ~8MB bundle size
- ~200MB memory usage

**Presentar Benefits:**
- Signal-based reactivity (minimal DOM updates)
- Compile-time type checking
- ~500KB bundle size
- ~20MB memory usage

### Panel → Trueno-Viz

**Problems with Panel:**
- 6+ HoloViz dependencies (Panel, HoloViews, Datashader, Bokeh, Param, Colorcet)
- WebGL rendering (older API)
- Python GIL contention

**Trueno-Viz Benefits:**
- Single unified library
- Native WebGPU rendering
- Rust memory safety for big data
- Billion-point rendering capability

### Dash → Presentar + Trueno-Viz

**Problems with Dash:**
- Callback spaghetti (invisible data dependencies)
- Large Plotly.js bundle
- WebGL performance limits

**Presentar + Trueno-Viz Benefits:**
- Explicit signal graph (debuggable)
- Smaller WASM bundle
- WebGPU for maximum performance

## Performance Comparison

| Metric | Gradio | Streamlit | Dash | Presentar |
|--------|--------|-----------|------|-----------|
| Cold start | ~2s | ~3s | ~1s | ~50ms |
| Interaction | ~100ms | ~200ms | ~80ms | ~16ms |
| Bundle size | ~5MB | ~8MB | ~3MB | ~500KB |
| Memory | ~150MB | ~200MB | ~100MB | ~20MB |
| GPU | No | No | WebGL | WebGPU |
| Offline | No | No | No | Yes |
| WASM | No | No | No | Yes |

## Component Mapping Reference

### Gradio Components

| Gradio | Presentar/Trueno-Viz |
|--------|----------------------|
| `gr.Interface` | `Presentar::QuickApp` |
| `gr.Blocks` | `Presentar::Layout` |
| `gr.Image` | `Trueno-Viz::ImageView` |
| `gr.Audio` | `Presentar::AudioPlayer` |
| `gr.Chatbot` | `Realizar + Presentar` |
| `gr.DataFrame` | `Trueno-Viz::DataGrid` |

### Streamlit Components

| Streamlit | Presentar/Trueno-Viz |
|-----------|----------------------|
| `st.write` | `Presentar::Text` |
| `st.dataframe` | `Trueno-Viz::DataGrid` |
| `st.plotly_chart` | `Trueno-Viz::Chart` |
| `st.text_input` | `Presentar::TextInput` |
| `st.slider` | `Presentar::Slider` |
| `st.selectbox` | `Presentar::Select` |
| `st.session_state` | `Presentar::State` |
| `@st.cache_data` | `Trueno::TensorCache` |

### Dash Components

| Dash | Presentar/Trueno-Viz |
|------|----------------------|
| `dash.Dash` | `Presentar::App` |
| `dcc.Graph` | `Trueno-Viz::Chart` |
| `dcc.Input` | `Presentar::TextInput` |
| `dash_table` | `Trueno-Viz::DataGrid` |
| `@callback` | `Presentar::on_event` |

## See Also

- [Presentar: App Framework](./presentar.md) - Detailed Presentar documentation
- [Trueno-Viz: GPU Rendering](./trueno-viz.md) - Trueno-Viz capabilities
- [`batuta viz`](../part6/cli-viz.md) - CLI reference
