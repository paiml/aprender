# Scene Format (`.prs`)

> Parent: [presentar-spec.md](../presentar-spec.md)

**Scope:** Presentar Scene Format v1.0 -- declarative YAML manifest for sharing visualization dashboards.

---

## Overview

The `.prs` format is a runtime-agnostic declarative manifest for sharing visualization dashboards, ML model interfaces, and interactive data applications. Unlike Gradio/Streamlit (Python-as-config), `.prs` files are parsed directly by WASM.

```
Extension: .prs
MIME Type: application/vnd.presentar.scene+yaml
Encoding: UTF-8, YAML 1.2
```

## Design Principles

- **Portability:** Share dashboards without embedding multi-GB models
- **Reproducibility:** Pin exact versions via BLAKE3 content-addressed hashes
- **Security:** Explicit permission grants enable sandboxing
- **Collaboration:** YAML diffs cleanly in version control

## Canonical Structure

```yaml
prs_version: "1.0"
metadata:
  name: "sentiment-analysis-demo"
  title: "Real-time Sentiment Analysis"
  author: "alice@example.com"
  license: "MIT"
  tags: ["nlp", "sentiment"]

resources:
  models:
    sentiment_model:
      type: apr
      source: "https://registry.paiml.com/models/sentiment-bert-q4.apr"
      hash: "blake3:a1b2c3d4e5f6..."
      size_bytes: 45_000_000
  datasets:
    examples:
      type: ald
      source: "./data/sentiment-examples.ald"

layout:
  type: grid
  columns: 2
  rows: 2
  gap: 16

widgets:
  - id: text_input
    type: textbox
    position: { row: 0, col: 0, colspan: 2 }
    config:
      label: "Enter text to analyze"
      max_length: 512
  - id: sentiment_chart
    type: bar_chart
    position: { row: 1, col: 0 }
    config:
      data: "{{ inference.sentiment_model | select('scores') }}"

bindings:
  - trigger: "text_input.change"
    debounce_ms: 300
    actions:
      - target: inference.sentiment_model
        input: "{{ text_input.value }}"

theme:
  preset: "dark"

permissions:
  network: ["https://registry.paiml.com/*"]
  filesystem: []
  clipboard: false
```

## Schema Reference

### Top-Level Fields

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `prs_version` | string | Yes | Format version (X.Y) |
| `metadata` | object | Yes | Scene metadata (name required, kebab-case) |
| `resources` | object | No | External models/datasets |
| `layout` | object | Yes | Widget arrangement |
| `widgets` | array | Yes | Widget definitions (unique IDs) |
| `bindings` | array | No | Event-action mappings |
| `theme` | object | No | Visual styling (presets: dark, light) |
| `permissions` | object | No | Security grants (default: all denied) |

### Resource Types

| Model Types | Dataset Types |
|-------------|---------------|
| `apr`, `gguf`, `safetensors` | `ald`, `parquet`, `csv` |

Remote sources (`https://`) require BLAKE3 hash. Local sources (`./`, `file://`) allow missing hash. Multiple sources enable offline-first fallback.

### Widget Types

| Widget | Purpose |
|--------|---------|
| `textbox` | Text input |
| `slider` | Numeric input (min/max/step) |
| `dropdown` | Selection (multi-select) |
| `button` | Action trigger |
| `image` | Display/upload |
| `bar_chart` | Bar visualization |
| `line_chart` | Time series |
| `gauge` | Single value with thresholds |
| `table` | Tabular data (sortable) |
| `markdown` | Rich text |
| `inference` | Model runner |

### Layout Types

- **Grid:** `columns`, `rows`, `gap`
- **Flex:** `direction` (row/column), `wrap`
- **Absolute:** `width`, `height`

### Expression Language

```
{{ source | transform | transform }}

Sources: widget.<id>.value, inference.<model>, dataset.<name>, state.<key>
Transforms: select, filter, sort, limit, count, sum, mean, percentage, format, join
```

## Security Model

Explicit permission grants. Runtime must: parse permissions before loading resources, reject URLs outside allowed domains, sandbox filesystem access, prompt for sensitive permissions.

## Comparison with Existing Formats

| Feature | `.prs` | Gradio | Streamlit | Grafana | Jupyter |
|---------|--------|--------|-----------|---------|---------|
| Declarative | Yes | No | No | Partial | No |
| WASM-native | Yes | No | No | No | No |
| External resources | Yes | Embedded | Embedded | Partial | Embedded |
| Content-addressed | Yes | No | No | No | No |
| Typical file size | < 10KB | N/A | N/A | 50-500KB | 1-100MB |

## QA Verification (100-Point Checklist)

| Category | Points | Scope |
|----------|--------|-------|
| Parsing & Schema | 20 | Required fields, version, YAML syntax, roundtrip |
| Metadata | 10 | Name validation (kebab-case), timestamps, tags |
| Widget Types | 11 | All 11 types parse correctly |
| Widget Config | 10 | Positions, expressions, duplicate ID rejection |
| Layout Types | 10 | Grid/flex/absolute, defaults, required fields |
| Resource Types | 12 | Model/dataset types, source arrays, size_bytes |
| Hash Validation | 8 | BLAKE3 format, remote requirement, local exemption |
| Bindings | 9 | Triggers, debounce, target validation |
| Theme & Permissions | 5 | Presets, custom colors, network arrays |
| Example Files | 5 | All example `.prs` files parse without error |

**Minimum passing score: 95/100.**

## References

- Satyanarayan, A. et al. (2017). Vega-Lite. *IEEE TVCG*, 23(1).
- Bostock, M. et al. (2011). D3: Data-Driven Documents. *IEEE TVCG*, 17(12).
- Kluyver, T. et al. (2016). Jupyter Notebooks. *ELPUB 2016*.
- Ohno, T. (1988). *Toyota Production System*. Productivity Press.
