# `batuta viz`

Visualization frameworks ecosystem commands for viewing Python framework hierarchies and their PAIML Rust replacements.

## Synopsis

```bash
batuta viz <COMMAND> [OPTIONS]
```

## Commands

| Command | Description |
|---------|-------------|
| `tree` | Display visualization frameworks ecosystem tree |

## Global Options

| Option | Description |
|--------|-------------|
| `-v, --verbose` | Enable verbose output |
| `-d, --debug` | Enable debug output |
| `-h, --help` | Print help |

---

## `batuta viz tree`

Display hierarchical visualization of Python frameworks and their PAIML Rust replacements, or show component replacement mappings.

### Usage

```bash
batuta viz tree [OPTIONS]
```

### Options

| Option | Description | Default |
|--------|-------------|---------|
| `--framework <NAME>` | Filter by framework (gradio, streamlit, panel, dash) | All frameworks |
| `--integration` | Show PAIML replacement mappings | false |
| `--format <FORMAT>` | Output format (ascii, json) | ascii |

### Examples

#### View All Frameworks

```bash
$ batuta viz tree

VISUALIZATION FRAMEWORKS ECOSYSTEM
==================================

GRADIO (Python) → Presentar (Rust)
├── Interface
│   └── Interface → Presentar::QuickApp
│       ├── Inputs
│       ├── Outputs
│       └── Examples
├── Blocks
│   └── Blocks → Presentar::Layout
│       ├── Layout
│       ├── Events
│       └── State
├── Components
│   ├── Image → Trueno-Viz::ImageView
│   ├── Audio → Presentar::AudioPlayer
│   ├── Video → Presentar::VideoPlayer
│   ├── Chatbot → Realizar + Presentar
│   ├── DataFrame → Trueno-Viz::DataGrid
│   └── Plot → Trueno-Viz::Chart
└── Deployment
    └── Deployment → Batuta deploy

STREAMLIT (Python) → Presentar (Rust)
...

PANEL (Python) → Trueno-Viz (Rust)
...

DASH (Python) → Presentar + Trueno-Viz (Rust)
...

Summary: 4 Python frameworks replaced by 2 Rust libraries
```

#### Filter by Framework

```bash
$ batuta viz tree --framework gradio

GRADIO (Python) → Presentar (Rust)
├── Interface
│   └── Interface → Presentar::QuickApp
│       ├── Inputs
│       ├── Outputs
│       └── Examples
├── Blocks
│   └── Blocks → Presentar::Layout
├── Components
│   ├── Image → Trueno-Viz::ImageView
│   ├── Audio → Presentar::AudioPlayer
│   ├── Video → Presentar::VideoPlayer
│   ├── Chatbot → Realizar + Presentar
│   ├── DataFrame → Trueno-Viz::DataGrid
│   └── Plot → Trueno-Viz::Chart
└── Deployment
    └── Deployment → Batuta deploy
```

#### View Replacement Mappings

```bash
$ batuta viz tree --integration

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
├── [REP] Trueno-Viz::DataGrid ← dash_table
├── [REP] Trueno-Viz::GPURaster ← datashader
├── [REP] Trueno-Viz::Plot ← matplotlib/plotly/bokeh

COMPONENTS
├── [REP] Presentar::TextInput ← st.text_input
├── [REP] Presentar::Slider ← st.slider
├── [REP] Presentar::Select ← st.selectbox
├── [REP] Presentar::Button ← st.button
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

#### JSON Output

```bash
$ batuta viz tree --framework streamlit --format json

{
  "framework": "Streamlit",
  "replacement": "Presentar",
  "categories": [
    {
      "name": "Widgets",
      "components": [
        {
          "name": "Input",
          "description": "User input widgets",
          "replacement": "Presentar::Widgets",
          "sub_components": ["text_input", "number_input", "slider", "selectbox"]
        }
      ]
    }
  ]
}
```

### Integration Type Legend

| Code | Type | Meaning |
|------|------|---------|
| `REP` | Replaces | PAIML component fully replaces Python equivalent |

**Note:** All mappings are `REP` (Replaces) - Python is completely eliminated from production deployments.

### Supported Frameworks

| Framework | PAIML Replacement | Description |
|-----------|-------------------|-------------|
| `gradio` | Presentar | ML demo interfaces |
| `streamlit` | Presentar | Data apps and dashboards |
| `panel` | Trueno-Viz | HoloViz ecosystem visualizations |
| `dash` | Presentar + Trueno-Viz | Plotly enterprise dashboards |

## See Also

- [`batuta data`](./cli-data.md) - Data platforms integration
- [`batuta hf`](./cli-hf.md) - HuggingFace Hub operations
- [Visualization Frameworks](../part3/visualization-frameworks.md) - Detailed documentation
