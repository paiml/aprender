# Presentar Scene Format (.prs)

The **Presentar Scene Format** (`.prs`) is a declarative YAML-based manifest for sharing visualization dashboards, ML model interfaces, and interactive data applications.

## Overview

Unlike Gradio/Streamlit which use Python-as-config (requiring a Python runtime), `.prs` provides a **runtime-agnostic declarative format** that WASM binaries can parse directly. This eliminates the need for shipping interpreters with visualization applications.

A `.prs` file is a *bill of materials* for a visualization—it declares **what** to display and **where** data lives, not **how** to fetch or render it.

## Quick Start

### Minimal Example

```yaml
prs_version: "1.0"

metadata:
  name: "hello-world"

layout:
  type: flex
  direction: column

widgets:
  - id: greeting
    type: markdown
    config:
      content: "# Hello, Presentar!"
```

### Validate Your Scene

```bash
# Validate a .prs file
cargo run -p presentar-yaml --example validate_prs examples/prs/minimal.prs

# Validate with verbose output
cargo run -p presentar-yaml --example validate_prs -- -v examples/prs/sentiment-demo.prs

# Validate all .prs files
cargo run -p presentar-yaml --example validate_prs examples/prs/*.prs
```

## File Structure

### Required Fields

| Field | Type | Description |
|-------|------|-------------|
| `prs_version` | string | Format version (e.g., "1.0") |
| `metadata.name` | string | Unique identifier (kebab-case) |
| `layout` | object | Widget arrangement configuration |
| `widgets` | array | Widget definitions |

### Optional Fields

| Field | Type | Description |
|-------|------|-------------|
| `metadata.title` | string | Human-readable title |
| `metadata.description` | string | Description |
| `metadata.author` | string | Author email/identifier |
| `metadata.license` | string | License (MIT, Apache-2.0, etc.) |
| `metadata.tags` | array | Categorization tags |
| `resources` | object | External models/datasets |
| `bindings` | array | Event → action mappings |
| `theme` | object | Visual styling |
| `permissions` | object | Security grants |

## Layout Types

### Grid Layout

```yaml
layout:
  type: grid
  columns: 3
  rows: 2
  gap: 16

widgets:
  - id: header
    type: markdown
    position: { row: 0, col: 0, colspan: 3 }
    config:
      content: "# Dashboard"
```

### Flex Layout

```yaml
layout:
  type: flex
  direction: column  # or: row
  wrap: true
  gap: 8
```

### Absolute Layout

```yaml
layout:
  type: absolute
  width: 1200
  height: 800
```

## Widget Types

| Widget | Purpose | Key Config |
|--------|---------|------------|
| `textbox` | Text input | `label`, `placeholder`, `max_length` |
| `slider` | Numeric input | `min`, `max`, `step`, `default` |
| `dropdown` | Selection | `options`, `multi_select` |
| `button` | Action trigger | `label`, `action` |
| `image` | Display/upload | `source`, `mode`, `accept` |
| `bar_chart` | Bar visualization | `data`, `x_axis`, `y_axis` |
| `line_chart` | Time series | `data`, `x_axis`, `y_axis` |
| `gauge` | Single value | `value`, `min`, `max`, `thresholds` |
| `table` | Tabular data | `data`, `columns`, `sortable` |
| `markdown` | Rich text | `content` |
| `inference` | Model runner | `model`, `input`, `output` |

### Widget Example

```yaml
widgets:
  - id: temperature
    type: slider
    config:
      label: "Temperature"
      min: 0.0
      max: 2.0
      step: 0.1
      default: 0.7

  - id: confidence_gauge
    type: gauge
    position: { row: 1, col: 1 }
    config:
      value: "{{ inference.model | select('confidence') | percentage }}"
      min: 0
      max: 100
      thresholds:
        - { value: 50, color: "red" }
        - { value: 75, color: "yellow" }
        - { value: 100, color: "green" }
```

## Resources

### Model Resources

```yaml
resources:
  models:
    sentiment_model:
      type: apr          # apr | gguf | safetensors
      source: "https://registry.paiml.com/models/sentiment.apr"
      hash: "blake3:a1b2c3d4..."  # Required for remote sources
      size_bytes: 45000000        # Optional, for progress
```

### Dataset Resources

```yaml
resources:
  datasets:
    sales_data:
      type: ald          # ald | parquet | csv
      source: "./data/sales.ald"
      hash: "blake3:..."  # Optional for local sources
```

### Source Fallbacks

```yaml
resources:
  models:
    classifier:
      type: apr
      source:
        - "./local-cache/model.apr"           # Try local first
        - "https://cdn.example.com/model.apr" # Fallback to CDN
      hash: "blake3:abc123..."
```

## Bindings

Bindings connect widget events to actions:

```yaml
bindings:
  - trigger: "text_input.change"
    debounce_ms: 300
    actions:
      - target: inference.sentiment_model
        input: "{{ text_input.value }}"
      - target: sentiment_chart
        action: refresh
      - target: confidence_gauge
        action: refresh
```

### Trigger Format

- `widget_id.event` - Widget event (change, click, submit, hover)
- Events: `change`, `click`, `submit`, `hover`, `focus`, `blur`

### Action Targets

- Widget ID - Direct reference to a widget
- `inference.model_name` - Reference to a model in resources

## Theme

```yaml
theme:
  preset: "dark"  # or: light
  custom:
    primary_color: "#4A90D9"
    font_family: "Inter, sans-serif"
    border_radius: "8px"
```

## Permissions

Explicit security grants for sandboxing:

```yaml
permissions:
  network:
    - "https://registry.paiml.com/*"
    - "https://cdn.example.com/*"
  filesystem:
    - "./data/*"        # Read-only access
  clipboard: false      # No clipboard access
  camera: false         # No camera access
```

## Expression Language

Expressions use the `{{ source | transform }}` syntax:

```yaml
# Simple reference
data: "{{ dataset.sales }}"

# With transforms
data: "{{ dataset.sales | filter('region == \"West\"') | limit(100) }}"

# Chained transforms
value: "{{ inference.model | select('confidence') | percentage }}"
```

### Available Transforms

| Transform | Description |
|-----------|-------------|
| `select('field')` | Extract field |
| `filter(predicate)` | Filter rows |
| `sort('field', 'asc')` | Sort data |
| `limit(n)` | Take first n |
| `count()` | Count items |
| `sum('field')` | Sum numeric field |
| `mean('field')` | Average |
| `percentage()` | Convert to 0-100 |
| `format('%.2f')` | String formatting |
| `join(', ')` | Array to string |

## Validation

The parser performs these validations:

1. **Version format** - Must be "X.Y" (e.g., "1.0")
2. **Metadata name** - Must be kebab-case (lowercase, numbers, hyphens)
3. **Widget IDs** - Must be unique
4. **Binding targets** - Must reference existing widgets/models
5. **Remote hashes** - HTTPS sources require BLAKE3 hashes
6. **Layout requirements** - Grid needs `columns`, absolute needs `width`/`height`

## Rust API

```rust
use presentar_yaml::Scene;

// Parse from YAML string
let scene = Scene::from_yaml(yaml_content)?;

// Access scene data
println!("Name: {}", scene.metadata.name);
println!("Widgets: {}", scene.widgets.len());

// Get specific widget
if let Some(widget) = scene.get_widget("my_widget") {
    println!("Type: {:?}", widget.widget_type);
}

// Get model resource
if let Some(model) = scene.get_model("classifier") {
    println!("Source: {}", model.source.primary());
}

// Serialize back to YAML
let yaml_output = scene.to_yaml()?;
```

## Error Handling

```rust
use presentar_yaml::{Scene, SceneError};

match Scene::from_yaml(content) {
    Ok(scene) => println!("Valid scene: {}", scene.metadata.name),
    Err(SceneError::InvalidVersion(v)) => {
        eprintln!("Bad version: {v}");
    }
    Err(SceneError::DuplicateWidgetId(id)) => {
        eprintln!("Duplicate ID: {id}");
    }
    Err(SceneError::InvalidBindingTarget { trigger, target }) => {
        eprintln!("Binding {trigger} has invalid target: {target}");
    }
    Err(e) => eprintln!("Error: {e}"),
}
```

## Complete Example

```yaml
# sentiment-demo.prs
prs_version: "1.0"

metadata:
  name: "sentiment-analysis-demo"
  title: "Real-time Sentiment Analysis"
  description: "Interactive sentiment classifier"
  author: "alice@example.com"
  license: "MIT"
  tags: ["nlp", "sentiment", "demo"]

resources:
  models:
    sentiment_model:
      type: apr
      source: "https://registry.paiml.com/models/sentiment.apr"
      hash: "blake3:a1b2c3d4e5f6..."
      size_bytes: 45000000

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
      placeholder: "Type a sentence..."
      max_length: 512

  - id: sentiment_chart
    type: bar_chart
    position: { row: 1, col: 0 }
    config:
      title: "Sentiment Scores"
      data: "{{ inference.sentiment_model | select('scores') }}"

  - id: confidence_gauge
    type: gauge
    position: { row: 1, col: 1 }
    config:
      value: "{{ inference.sentiment_model | select('confidence') | percentage }}"
      min: 0
      max: 100
      thresholds:
        - { value: 50, color: "red" }
        - { value: 75, color: "yellow" }
        - { value: 100, color: "green" }

bindings:
  - trigger: "text_input.change"
    debounce_ms: 300
    actions:
      - target: inference.sentiment_model
        input: "{{ text_input.value }}"
      - target: sentiment_chart
        action: refresh
      - target: confidence_gauge
        action: refresh

theme:
  preset: "dark"
  custom:
    primary_color: "#4A90D9"

permissions:
  network:
    - "https://registry.paiml.com/*"
```

## See Also

- [Manifest Schema](./manifest-schema.md) - Original app.yaml format
- [Expression Language](./expression-language.md) - Transform syntax
- [Data Sources](./data-sources.md) - Loading data
- [Model References](./model-references.md) - Loading models
