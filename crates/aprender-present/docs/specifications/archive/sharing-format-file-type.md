# Presentar Scene Format (`.prs`): Shareable Visualization Manifests

**Version**: 1.0.0
**Status**: DRAFT
**Author**: Sovereign AI Stack Team
**Date**: 2025-12-06
**MSRV**: 1.75
**Repository**: [github.com/paiml/presentar](https://github.com/paiml/presentar)

---

## Executive Summary

This specification defines the **Presentar Scene Format** (`.prs`), a declarative YAML-based manifest for sharing visualization dashboards, ML model interfaces, and interactive data applications across the Sovereign AI Stack.

Unlike Gradio/Streamlit which use Python-as-config (requiring a Python runtime), `.prs` provides a **runtime-agnostic declarative format** that WASM binaries can parse directly [1, 2]. This eliminates the *Muda* (waste) of shipping interpreters with visualization applications and enables true edge deployment.

> **Theoretical Basis**: This specification draws from research on declarative visualization grammars [3, 4], reproducible computational documents [5, 6], and configuration management best practices [7]. Design follows Toyota Production System principles for lean, portable, and verifiable artifacts [8, 9].

**Design Philosophy**: A `.prs` file is a *bill of materials* for a visualization—it declares **what** to display and **where** data lives, not **how** to fetch or render it. The separation of concerns enables:
- **Portability**: Share dashboards without embedding multi-GB models
- **Reproducibility**: Pin exact versions of models/datasets
- **Security**: Explicit resource declarations enable sandboxing
- **Collaboration**: Version control friendly (YAML diffs cleanly)

---

## Table of Contents

1. [Design Rationale](#1-design-rationale)
2. [Format Specification](#2-format-specification)
3. [Schema Reference](#3-schema-reference)
4. [Use Cases](#4-use-cases)
5. [Comparison with Existing Formats](#5-comparison-with-existing-formats)
6. [Implementation Guidelines](#6-implementation-guidelines)
7. [Peer-Reviewed Citations](#7-peer-reviewed-citations)
8. [Appendices](#8-appendices)
9. [QA Verification Checklist (100 Points)](#9-qa-verification-checklist-100-points)

---

## 1. Design Rationale

### 1.1 Problem Statement

> **Rationale**: Satyanarayan et al. [3] demonstrate that declarative grammars dramatically reduce specification complexity while maintaining expressiveness. The `.prs` format applies this insight to visualization sharing.

Current visualization sharing approaches have fundamental limitations:

| Approach | Problem |
|----------|---------|
| **Gradio/Streamlit** | Requires Python runtime; cannot run in WASM |
| **Grafana JSON** | Tightly coupled to Grafana ecosystem [10] |
| **Jupyter Notebooks** | Mixes code/data/output; large file sizes [5] |
| **Embedded WASM** | Bundles models/data; inflexible, non-cacheable |

### 1.2 Solution: Declarative Scene Manifests

> **Rationale**: The ARTS framework [6] shows that separating "what" from "how" enables long-term reproducibility. A `.prs` file declares resources; the runtime resolves them.

The `.prs` format provides:

1. **Resource Declaration**: URLs/paths to `.apr` models and `.ald` datasets
2. **Layout Specification**: Widget arrangement using CSS-like positioning
3. **Interaction Bindings**: Event → state → widget data flow
4. **Version Pinning**: Content-addressed references via BLAKE3 hashes
5. **Security Boundaries**: Explicit permission grants for network/filesystem

### 1.3 Toyota Way Alignment

| TPS Principle | `.prs` Implementation |
|---------------|----------------------|
| **Jidoka** (Built-in Quality) | Schema validation before execution; type-safe widget configs [8] |
| **Muda** (Waste Elimination) | No embedded data; reference-only; minimal file size [9] |
| **Heijunka** (Level Loading) | Lazy resource loading; predictable memory footprint |
| **Poka-Yoke** (Error-Proofing) | Required fields enforced; invalid states unrepresentable |
| **Genchi Genbutsu** (Go and See) | Clear resource URLs; auditable dependencies |

---

## 2. Format Specification

### 2.1 File Extension and MIME Type

```
Extension: .prs
MIME Type: application/vnd.presentar.scene+yaml
Magic Bytes: N/A (text format)
```

### 2.2 Encoding

- **Character Set**: UTF-8 (required)
- **Line Endings**: LF (Unix-style preferred)
- **YAML Version**: 1.2

### 2.3 Canonical Structure

```yaml
# Presentar Scene Format v1.0
# https://github.com/paiml/presentar

prs_version: "1.0"

metadata:
  name: "sentiment-analysis-demo"
  title: "Real-time Sentiment Analysis"
  description: "Interactive sentiment classifier with confidence visualization"
  author: "alice@example.com"
  created: "2025-12-06T10:00:00Z"
  license: "MIT"
  tags: ["nlp", "sentiment", "demo"]

# External resources (models, datasets)
resources:
  models:
    sentiment_model:
      type: apr
      source: "https://registry.paiml.com/models/sentiment-bert-q4.apr"
      hash: "blake3:a1b2c3d4e5f6..."  # Content verification
      size_bytes: 45_000_000

  datasets:
    examples:
      type: ald
      source: "./data/sentiment-examples.ald"
      hash: "blake3:f6e5d4c3b2a1..."

# Widget layout
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
      x_axis: "{{ ['Positive', 'Negative', 'Neutral'] }}"

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

# Data flow bindings
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

# Theme and styling
theme:
  preset: "dark"
  custom:
    primary_color: "#4A90D9"
    font_family: "Inter, sans-serif"

# Security permissions (explicit grants)
permissions:
  network:
    - "https://registry.paiml.com/*"
  filesystem: []  # No local file access
  clipboard: false
```

---

## 3. Schema Reference

### 3.1 Top-Level Fields

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `prs_version` | string | Yes | Format version (semver) |
| `metadata` | object | Yes | Scene metadata |
| `resources` | object | No | External models/datasets |
| `layout` | object | Yes | Widget arrangement |
| `widgets` | array | Yes | Widget definitions |
| `bindings` | array | No | Event → action mappings |
| `theme` | object | No | Visual styling |
| `permissions` | object | No | Security grants |

### 3.2 Resource Types

```yaml
resources:
  models:
    <name>:
      type: apr | gguf | safetensors
      source: <url-or-path>
      hash: blake3:<hex-digest>  # Required for remote
      size_bytes: <integer>      # Optional, for progress

  datasets:
    <name>:
      type: ald | parquet | csv
      source: <url-or-path>
      hash: blake3:<hex-digest>
```

> **Rationale**: Content-addressed hashes ensure reproducibility [6] and enable caching. The batuta/pacha ecosystem uses BLAKE3 for performance [7].

### 3.3 Widget Types

| Widget | Purpose | Key Config |
|--------|---------|------------|
| `textbox` | Text input | `label`, `placeholder`, `max_length` |
| `slider` | Numeric input | `min`, `max`, `step`, `default` |
| `dropdown` | Selection | `options`, `multi_select` |
| `button` | Action trigger | `label`, `action` |
| `image` | Display image | `source`, `alt` |
| `bar_chart` | Bar visualization | `data`, `x_axis`, `y_axis` |
| `line_chart` | Time series | `data`, `x_axis`, `y_axis` |
| `gauge` | Single value | `value`, `min`, `max`, `thresholds` |
| `table` | Tabular data | `data`, `columns`, `sortable` |
| `markdown` | Rich text | `content` |
| `inference` | Model runner | `model`, `input`, `output` |

### 3.4 Expression Language

> **Rationale**: Inspired by Vega-Lite's declarative data flow [3], expressions enable reactive updates without imperative code.

```
{{ source | transform | transform }}

Sources:
  - widget.<id>.value      # Widget state
  - inference.<model>      # Model output
  - dataset.<name>         # Dataset reference
  - state.<key>            # Global state

Transforms:
  - select('field')        # Extract field
  - filter(predicate)      # Filter rows
  - sort('field', 'asc')   # Sort data
  - limit(n)               # Take first n
  - count()                # Count items
  - sum('field')           # Sum numeric field
  - mean('field')          # Average
  - percentage()           # Convert to 0-100
  - format('%.2f')         # String formatting
  - join(', ')             # Array to string
```

### 3.5 Layout Types

```yaml
# Grid layout
layout:
  type: grid
  columns: 3
  rows: 2
  gap: 16

# Flex layout
layout:
  type: flex
  direction: row | column
  wrap: true
  gap: 8

# Absolute positioning
layout:
  type: absolute
  width: 1200
  height: 800
```

---

## 4. Use Cases

### 4.1 Share ML Demo Without Bundling Model

**Problem**: You want to share a sentiment analysis demo, but the model is 500MB.

**Solution**: `.prs` references the model by URL; recipient's runtime fetches and caches it.

```yaml
resources:
  models:
    sentiment:
      source: "https://registry.paiml.com/models/sentiment-v2.apr"
      hash: "blake3:abc123..."
```

**Benefits**:
- `.prs` file is <5KB
- Model cached locally after first load
- Updates model by changing URL/hash

### 4.2 Embed Dashboard in Documentation

**Problem**: You want interactive visualizations in your mdBook documentation.

**Solution**: Reference `.prs` file in markdown; presentar-wasm renders inline.

```markdown
# Model Performance

<presentar-embed src="./dashboards/metrics.prs" height="400px" />
```

### 4.3 Version-Controlled Dashboards

**Problem**: Track dashboard changes in Git with meaningful diffs.

**Solution**: YAML diffs cleanly; JSON schemas for validation.

```diff
 widgets:
   - id: chart
     type: bar_chart
     config:
-      title: "Sales Q3"
+      title: "Sales Q4"
       data: "{{ dataset.sales | filter('quarter == 4') }}"
```

### 4.4 Offline-First with Fallbacks

**Problem**: Demo must work without internet.

**Solution**: Priority-ordered sources with local fallback.

```yaml
resources:
  models:
    classifier:
      source:
        - "./local-cache/model.apr"           # Try local first
        - "https://cdn.example.com/model.apr" # Fallback to CDN
      hash: "blake3:abc123..."
```

---

## 5. Comparison with Existing Formats

### 5.1 Feature Matrix

| Feature | `.prs` | Gradio | Streamlit | Grafana | Jupyter |
|---------|--------|--------|-----------|---------|---------|
| **Declarative** | Yes | No (Python) | No (Python) | Partial | No |
| **WASM-native** | Yes | No | No | No | No |
| **External resources** | Yes | Embedded | Embedded | Partial | Embedded |
| **Version control friendly** | Yes | No | No | Partial | No |
| **Content-addressed** | Yes | No | No | No | No |
| **Schema validation** | Yes | No | No | Partial | No |
| **File size (typical)** | <10KB | N/A | N/A | 50-500KB | 1-100MB |

### 5.2 Detailed Comparisons

#### vs. Gradio/Streamlit

> **Rationale**: Gradio and Streamlit use Python-as-config [1], which requires a Python runtime. This is incompatible with WASM edge deployment.

```python
# Gradio: Requires Python runtime
import gradio as gr
demo = gr.Interface(fn=predict, inputs="text", outputs="label")
demo.launch()
```

```yaml
# .prs: Declarative, parsed by WASM
widgets:
  - id: input
    type: textbox
  - id: output
    type: label
bindings:
  - trigger: input.submit
    actions:
      - target: inference.model
        input: "{{ input.value }}"
```

#### vs. Grafana Dashboard JSON

> **Rationale**: Grafana's JSON format [10] is powerful but tightly coupled to the Grafana ecosystem. `.prs` targets the Sovereign AI Stack.

- Grafana: 500+ lines of JSON for simple dashboard
- `.prs`: ~50 lines of YAML for equivalent
- Grafana: Requires Grafana server
- `.prs`: Runs in any presentar runtime (native, WASM)

#### vs. Jupyter Notebooks

> **Rationale**: Kluyver et al. [5] show notebooks mix code, data, and output, creating reproducibility challenges. `.prs` separates concerns.

- Jupyter: Code + data + output in one file (bloated)
- `.prs`: References only; runtime resolves
- Jupyter: Execution order matters
- `.prs`: Declarative; order irrelevant

#### vs. Vega-Lite

> **Rationale**: Vega-Lite [3] is the gold standard for declarative visualization. `.prs` extends this to full applications with ML inference.

- Vega-Lite: Visualization only
- `.prs`: Visualization + inference + interaction
- Vega-Lite: JSON only
- `.prs`: YAML (more readable, comments allowed)

---

## 6. Implementation Guidelines

### 6.1 Parser Requirements

```rust
use serde::{Deserialize, Serialize};

#[derive(Debug, Deserialize, Serialize)]
pub struct PresentarScene {
    pub prs_version: String,
    pub metadata: Metadata,
    #[serde(default)]
    pub resources: Resources,
    pub layout: Layout,
    pub widgets: Vec<Widget>,
    #[serde(default)]
    pub bindings: Vec<Binding>,
    #[serde(default)]
    pub theme: Option<Theme>,
    #[serde(default)]
    pub permissions: Permissions,
}

impl PresentarScene {
    pub fn from_yaml(yaml: &str) -> Result<Self, ParseError> {
        let scene: Self = serde_yaml::from_str(yaml)?;
        scene.validate()?;
        Ok(scene)
    }

    pub fn validate(&self) -> Result<(), ValidationError> {
        // 1. Check prs_version compatibility
        // 2. Validate resource hashes format
        // 3. Ensure widget IDs unique
        // 4. Verify binding targets exist
        // 5. Check expression syntax
        Ok(())
    }
}
```

### 6.2 Resource Resolution

```rust
pub trait ResourceResolver {
    async fn resolve_model(&self, resource: &ModelResource) -> Result<Model>;
    async fn resolve_dataset(&self, resource: &DatasetResource) -> Result<Dataset>;
}

pub struct CachingResolver {
    cache_dir: PathBuf,
    http_client: HttpClient,
}

impl ResourceResolver for CachingResolver {
    async fn resolve_model(&self, resource: &ModelResource) -> Result<Model> {
        // 1. Check local cache by hash
        // 2. If miss, fetch from source
        // 3. Verify hash matches
        // 4. Store in cache
        // 5. Return loaded model
    }
}
```

### 6.3 Security Model

> **Rationale**: Explicit permission grants follow the principle of least privilege, preventing supply-chain attacks [7].

```yaml
permissions:
  network:
    - "https://registry.paiml.com/*"  # Allowed
    - "https://cdn.example.com/*"     # Allowed
    # All other URLs blocked
  filesystem:
    - "./data/*"                       # Read-only access
  clipboard: false                     # No clipboard access
  camera: false                        # No camera access
```

Runtime MUST:
1. Parse permissions before loading resources
2. Reject resources outside allowed domains
3. Sandbox filesystem access
4. Prompt user for sensitive permissions

### 6.4 Expression Evaluation

```rust
pub struct ExpressionEngine {
    context: ExpressionContext,
}

impl ExpressionEngine {
    pub fn evaluate(&self, expr: &str) -> Result<Value> {
        let ast = parse_expression(expr)?;
        let result = self.eval_ast(&ast)?;
        Ok(result)
    }

    fn eval_ast(&self, ast: &Ast) -> Result<Value> {
        match ast {
            Ast::Source(name) => self.context.get(name),
            Ast::Pipe(left, transform) => {
                let value = self.eval_ast(left)?;
                self.apply_transform(value, transform)
            }
        }
    }
}
```

---

## 7. Peer-Reviewed Citations

### [1] Team, G. (2022). Gradio: Hassle-Free Sharing and Testing of ML Models
*GitHub repository. https://github.com/gradio-app/gradio*

Documents Python-as-config approach and `share=True` temporary URL mechanism.

### [2] Streamlit Inc. (2019). Streamlit: The fastest way to build data apps
*https://streamlit.io/*

Establishes Python script as application definition pattern.

### [3] Satyanarayan, A., Moritz, D., Wongsuphasawat, K., & Heer, J. (2017). Vega-Lite: A Grammar of Interactive Graphics
*IEEE TVCG, 23(1), 341-350. https://doi.org/10.1109/TVCG.2016.2599030*

Foundational work on declarative visualization grammars. Demonstrates that high-level specifications compile to low-level rendering instructions without loss of expressiveness.

### [4] Bostock, M., Ogievetsky, V., & Heer, J. (2011). D3: Data-Driven Documents
*IEEE TVCG, 17(12), 2301-2309. https://doi.org/10.1109/TVCG.2011.185*

Establishes data binding paradigm for visualization. Informs `.prs` expression language design.

### [5] Kluyver, T., et al. (2016). Jupyter Notebooks - a publishing format for reproducible computational workflows
*ELPUB 2016. https://doi.org/10.3233/978-1-61499-649-1-87*

Identifies reproducibility challenges with mixed code/data/output formats. `.prs` separates these concerns.

### [6] Krafczyk, M., et al. (2025). ARTS: An Open Framework for Archival, Reproducible, and Transparent Science
*arXiv:2504.08171. https://arxiv.org/abs/2504.08171*

Proposes container-based reproducibility. `.prs` content-addressing achieves similar guarantees.

### [7] Batuta Sovereign AI Stack Specification v2.0 (2025)
*https://github.com/paiml/batuta/docs/specifications/sovereign-ai-spec.md*

Defines BLAKE3 content addressing, pacha model registry, and privacy tiers that `.prs` integrates with.

### [8] Ohno, T. (1988). Toyota Production System: Beyond Large-Scale Production
*Productivity Press. ISBN 978-0915299140*

Jidoka (built-in quality) and Muda (waste elimination) principles guide format design.

### [9] Womack, J.P. & Jones, D.T. (1996). Lean Thinking: Banish Waste and Create Wealth
*Simon & Schuster. ISBN 978-0743249270*

Value stream mapping applied to visualization artifact creation.

### [10] Grafana Labs. (2024). Dashboard JSON Model
*https://grafana.com/docs/grafana/latest/dashboards/build-dashboards/view-dashboard-json-model/*

Reference for existing dashboard serialization. `.prs` simplifies structure for ML use cases.

---

## 8. Appendices

### A. JSON Schema

```json
{
  "$schema": "http://json-schema.org/draft-07/schema#",
  "$id": "https://presentar.dev/schemas/prs-v1.json",
  "title": "Presentar Scene Format",
  "type": "object",
  "required": ["prs_version", "metadata", "layout", "widgets"],
  "properties": {
    "prs_version": {
      "type": "string",
      "pattern": "^[0-9]+\\.[0-9]+$"
    },
    "metadata": {
      "type": "object",
      "required": ["name"],
      "properties": {
        "name": { "type": "string", "pattern": "^[a-z0-9-]+$" },
        "title": { "type": "string" },
        "description": { "type": "string" },
        "author": { "type": "string" },
        "license": { "type": "string" }
      }
    },
    "widgets": {
      "type": "array",
      "items": {
        "type": "object",
        "required": ["id", "type"],
        "properties": {
          "id": { "type": "string" },
          "type": { "enum": ["textbox", "slider", "dropdown", "button", "image", "bar_chart", "line_chart", "gauge", "table", "markdown", "inference"] }
        }
      }
    }
  }
}
```

### B. Example Gallery

#### B.1 Minimal Scene

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

#### B.2 Image Classifier

```yaml
prs_version: "1.0"
metadata:
  name: "image-classifier"
  title: "CIFAR-10 Classifier"

resources:
  models:
    classifier:
      type: apr
      source: "https://registry.paiml.com/models/cifar10-resnet.apr"
      hash: "blake3:abc123def456..."

layout:
  type: grid
  columns: 2
  rows: 1

widgets:
  - id: image_upload
    type: image
    position: { row: 0, col: 0 }
    config:
      mode: upload
      accept: ["image/png", "image/jpeg"]

  - id: predictions
    type: bar_chart
    position: { row: 0, col: 1 }
    config:
      title: "Predictions"
      data: "{{ inference.classifier | select('probabilities') }}"
      x_axis: "{{ ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck'] }}"

bindings:
  - trigger: image_upload.change
    actions:
      - target: inference.classifier
        input: "{{ image_upload.data }}"
```

#### B.3 Data Explorer

```yaml
prs_version: "1.0"
metadata:
  name: "data-explorer"

resources:
  datasets:
    sales:
      type: ald
      source: "./data/sales-2024.ald"
      hash: "blake3:789xyz..."

layout:
  type: flex
  direction: column

widgets:
  - id: filters
    type: dropdown
    config:
      label: "Region"
      options: "{{ dataset.sales | select('region') | unique }}"

  - id: chart
    type: line_chart
    config:
      title: "Sales Over Time"
      data: "{{ dataset.sales | filter('region == filters.value') }}"
      x_axis: date
      y_axis: revenue

  - id: table
    type: table
    config:
      data: "{{ dataset.sales | filter('region == filters.value') | limit(100) }}"
      columns: ["date", "region", "product", "revenue"]
      sortable: true
```

### C. Migration Guide

#### From Gradio

```python
# Gradio
import gradio as gr
def greet(name):
    return f"Hello {name}!"
gr.Interface(fn=greet, inputs="text", outputs="text").launch()
```

```yaml
# .prs equivalent
prs_version: "1.0"
metadata:
  name: "greeter"
layout:
  type: flex
  direction: column
widgets:
  - id: name_input
    type: textbox
    config:
      label: "Name"
  - id: greeting
    type: markdown
    config:
      content: "Hello {{ name_input.value }}!"
bindings:
  - trigger: name_input.change
    actions:
      - target: greeting
        action: refresh
```

### D. MIME Type Registration

```
Type name: application
Subtype name: vnd.presentar.scene+yaml
Required parameters: None
Optional parameters: version (e.g., version=1.0)
Encoding: UTF-8
Security considerations:
  - Parsers MUST validate against schema
  - Resource URLs MUST be sanitized
  - Expression evaluation MUST be sandboxed
Interoperability considerations: YAML 1.2 compliant
```

---

## 9. QA Verification Checklist (100 Points)

This checklist enables QA teams to verify `.prs` format implementation compliance. Each item is worth 1 point. **Minimum passing score: 95/100**.

### 9.1 Parsing & Schema Validation (20 points)

| # | Test Case | Expected Result | Pass |
|---|-----------|-----------------|------|
| 1 | Parse `minimal.prs` with only required fields | Scene object created successfully | ☐ |
| 2 | Parse `sentiment-demo.prs` with all optional fields | All fields populated correctly | ☐ |
| 3 | Reject YAML with missing `prs_version` | Error: missing required field | ☐ |
| 4 | Reject YAML with missing `metadata.name` | Error: missing required field | ☐ |
| 5 | Reject YAML with missing `layout` | Error: missing required field | ☐ |
| 6 | Reject YAML with missing `widgets` | Error: missing required field | ☐ |
| 7 | Accept `prs_version: "1.0"` | Valid version parsed | ☐ |
| 8 | Accept `prs_version: "2.1"` | Valid version parsed | ☐ |
| 9 | Reject `prs_version: "1.0.0"` (semver, not X.Y) | Error: InvalidVersion | ☐ |
| 10 | Reject `prs_version: "invalid"` | Error: InvalidVersion | ☐ |
| 11 | Reject malformed YAML syntax | Error: Yaml parse error | ☐ |
| 12 | Parse YAML with comments preserved | Comments ignored, data parsed | ☐ |
| 13 | Parse UTF-8 content in metadata.title | Unicode characters preserved | ☐ |
| 14 | Parse empty `widgets: []` array | Valid empty scene | ☐ |
| 15 | Parse empty `bindings: []` array | Valid scene with no bindings | ☐ |
| 16 | Parse scene with no `resources` section | Default empty resources | ☐ |
| 17 | Parse scene with no `theme` section | Theme is None | ☐ |
| 18 | Parse scene with no `permissions` section | Default permissions (all denied) | ☐ |
| 19 | Roundtrip: `from_yaml()` → `to_yaml()` → `from_yaml()` | Data preserved | ☐ |
| 20 | Parse large scene (50+ widgets) | No performance degradation | ☐ |

### 9.2 Metadata Validation (10 points)

| # | Test Case | Expected Result | Pass |
|---|-----------|-----------------|------|
| 21 | Accept `name: "valid-name"` (kebab-case) | Valid | ☐ |
| 22 | Accept `name: "my-app-v2"` (with numbers) | Valid | ☐ |
| 23 | Reject `name: "Invalid-Name"` (uppercase) | Error: InvalidMetadataName | ☐ |
| 24 | Reject `name: "-invalid"` (leading hyphen) | Error: InvalidMetadataName | ☐ |
| 25 | Reject `name: "invalid-"` (trailing hyphen) | Error: InvalidMetadataName | ☐ |
| 26 | Reject `name: "invalid--name"` (double hyphen) | Error: InvalidMetadataName | ☐ |
| 27 | Parse `created: "2025-12-06T10:00:00Z"` ISO 8601 | Timestamp preserved as string | ☐ |
| 28 | Parse `tags: ["ml", "demo", "nlp"]` | Array of 3 tags | ☐ |
| 29 | Parse `license: "MIT"` | License string preserved | ☐ |
| 30 | Parse `author: "user@example.com"` | Author string preserved | ☐ |

### 9.3 Widget Types (11 points)

| # | Test Case | Expected Result | Pass |
|---|-----------|-----------------|------|
| 31 | Parse `type: textbox` widget | WidgetType::Textbox | ☐ |
| 32 | Parse `type: slider` widget | WidgetType::Slider | ☐ |
| 33 | Parse `type: dropdown` widget | WidgetType::Dropdown | ☐ |
| 34 | Parse `type: button` widget | WidgetType::Button | ☐ |
| 35 | Parse `type: image` widget | WidgetType::Image | ☐ |
| 36 | Parse `type: bar_chart` widget | WidgetType::BarChart | ☐ |
| 37 | Parse `type: line_chart` widget | WidgetType::LineChart | ☐ |
| 38 | Parse `type: gauge` widget | WidgetType::Gauge | ☐ |
| 39 | Parse `type: table` widget | WidgetType::Table | ☐ |
| 40 | Parse `type: markdown` widget | WidgetType::Markdown | ☐ |
| 41 | Parse `type: inference` widget | WidgetType::Inference | ☐ |

### 9.4 Widget Configuration (10 points)

| # | Test Case | Expected Result | Pass |
|---|-----------|-----------------|------|
| 42 | Parse textbox with `max_length: 512` | Config field populated | ☐ |
| 43 | Parse slider with `min: 0.0, max: 1.0, step: 0.1` | All numeric fields correct | ☐ |
| 44 | Parse gauge with thresholds array | 3 threshold objects parsed | ☐ |
| 45 | Parse table with `columns: ["a", "b", "c"]` | String array preserved | ☐ |
| 46 | Parse image with `accept: ["image/png"]` | MIME type array | ☐ |
| 47 | Parse grid position `{ row: 0, col: 1, colspan: 2 }` | Position object complete | ☐ |
| 48 | Default `colspan: 1` when not specified | Default applied | ☐ |
| 49 | Default `rowspan: 1` when not specified | Default applied | ☐ |
| 50 | Parse widget with expression `data: "{{ source }}"` | Expression string preserved | ☐ |
| 51 | Reject duplicate widget IDs | Error: DuplicateWidgetId | ☐ |

### 9.5 Layout Types (10 points)

| # | Test Case | Expected Result | Pass |
|---|-----------|-----------------|------|
| 52 | Parse `type: grid` layout | LayoutType::Grid | ☐ |
| 53 | Parse `type: flex` layout | LayoutType::Flex | ☐ |
| 54 | Parse `type: absolute` layout | LayoutType::Absolute | ☐ |
| 55 | Grid layout: require `columns` field | Error if missing | ☐ |
| 56 | Grid layout: accept `rows` field | Optional field parsed | ☐ |
| 57 | Flex layout: parse `direction: row` | FlexDirection::Row | ☐ |
| 58 | Flex layout: parse `direction: column` | FlexDirection::Column | ☐ |
| 59 | Absolute layout: require `width` and `height` | Error if missing | ☐ |
| 60 | Default `gap: 16` when not specified | Default applied | ☐ |
| 61 | Custom `gap: 24` overrides default | Custom value used | ☐ |

### 9.6 Resource Types (12 points)

| # | Test Case | Expected Result | Pass |
|---|-----------|-----------------|------|
| 62 | Parse model `type: apr` | ModelType::Apr | ☐ |
| 63 | Parse model `type: gguf` | ModelType::Gguf | ☐ |
| 64 | Parse model `type: safetensors` | ModelType::Safetensors | ☐ |
| 65 | Parse dataset `type: ald` | DatasetType::Ald | ☐ |
| 66 | Parse dataset `type: parquet` | DatasetType::Parquet | ☐ |
| 67 | Parse dataset `type: csv` | DatasetType::Csv | ☐ |
| 68 | Parse single source string | ResourceSource::Single | ☐ |
| 69 | Parse multiple source array (fallback) | ResourceSource::Multiple | ☐ |
| 70 | `source.primary()` returns first source | Correct primary source | ☐ |
| 71 | `source.sources()` returns all sources | All sources listed | ☐ |
| 72 | Parse `size_bytes: 45000000` | Size preserved | ☐ |
| 73 | Parse resource without `size_bytes` | None value | ☐ |

### 9.7 Hash Validation (8 points)

| # | Test Case | Expected Result | Pass |
|---|-----------|-----------------|------|
| 74 | Accept `hash: "blake3:abc123..."` (valid hex) | Hash parsed | ☐ |
| 75 | Reject `hash: "sha256:abc123..."` (wrong algo) | Error: InvalidHashFormat | ☐ |
| 76 | Reject `hash: "blake3:xyz"` (invalid hex chars) | Error: InvalidHashFormat | ☐ |
| 77 | Reject `hash: "blake3:abc"` (too short) | Error: InvalidHashFormat | ☐ |
| 78 | Require hash for `https://` sources | Error: MissingRemoteHash | ☐ |
| 79 | Allow missing hash for `./local` sources | Valid (local trusted) | ☐ |
| 80 | Allow missing hash for `file://` sources | Valid (local trusted) | ☐ |
| 81 | Validate hash on fallback array with remote | Error if any remote lacks hash | ☐ |

### 9.8 Bindings & Actions (9 points)

| # | Test Case | Expected Result | Pass |
|---|-----------|-----------------|------|
| 82 | Parse binding with `trigger: "input.change"` | Trigger string preserved | ☐ |
| 83 | Parse binding with `debounce_ms: 300` | Debounce value set | ☐ |
| 84 | Parse binding with multiple actions | Actions array complete | ☐ |
| 85 | Parse action `target: widget_id` | Target to widget | ☐ |
| 86 | Parse action `target: inference.model_name` | Target to inference | ☐ |
| 87 | Validate binding target exists (widget) | Valid if widget exists | ☐ |
| 88 | Validate binding target exists (model) | Valid if model in resources | ☐ |
| 89 | Reject binding to non-existent widget | Error: InvalidBindingTarget | ☐ |
| 90 | Reject binding to non-existent model | Error: InvalidBindingTarget | ☐ |

### 9.9 Theme & Permissions (5 points)

| # | Test Case | Expected Result | Pass |
|---|-----------|-----------------|------|
| 91 | Parse `theme.preset: "dark"` | Preset string set | ☐ |
| 92 | Parse `theme.preset: "light"` | Preset string set | ☐ |
| 93 | Parse `theme.custom` with color values | HashMap populated | ☐ |
| 94 | Parse `permissions.network: ["https://*.com/*"]` | Network array set | ☐ |
| 95 | Parse `permissions.clipboard: false` | Boolean field set | ☐ |

### 9.10 Example File Validation (5 points)

| # | Test Case | Expected Result | Pass |
|---|-----------|-----------------|------|
| 96 | `minimal.prs` parses without error | Valid Scene object | ☐ |
| 97 | `sentiment-demo.prs` parses without error | Valid with resources & bindings | ☐ |
| 98 | `image-classifier.prs` parses without error | Valid with model & image widget | ☐ |
| 99 | `data-explorer.prs` parses without error | Valid with dataset & charts | ☐ |
| 100 | `parameter-tuner.prs` parses without error | Valid with 3 sliders | ☐ |

---

### 9.11 Scoring Guide

| Score | Grade | Status |
|-------|-------|--------|
| 100/100 | A+ | Production Ready |
| 95-99 | A | Production Ready |
| 90-94 | B+ | Minor Issues - Fix Before Release |
| 85-89 | B | Moderate Issues - Review Required |
| 80-84 | C | Significant Issues - Do Not Release |
| <80 | F | Critical Failures - Major Rework |

### 9.12 Test Execution Commands

```bash
# Run all unit tests (54 scene tests)
cargo test -p presentar-yaml --lib scene::

# Run integration tests (7 prs_examples tests)
cargo test -p presentar-yaml --test prs_examples

# Run full test suite (304 tests)
cargo test -p presentar-yaml

# Verify clippy compliance
cargo clippy -p presentar-yaml -- -D warnings

# Validate a single .prs file
cargo run -p presentar-yaml --example validate_prs examples/prs/minimal.prs

# Validate with verbose output (shows scene details)
cargo run -p presentar-yaml --example validate_prs -- -v examples/prs/sentiment-demo.prs

# Validate all example files at once
cargo run -p presentar-yaml --example validate_prs examples/prs/*.prs
```

### 9.13 Sign-Off

| Reviewer | Date | Score | Signature |
|----------|------|-------|-----------|
| QA Engineer | | /100 | |
| Dev Lead | | /100 | |
| Security | | /100 | |

---

## Approval

**Status**: DRAFT - AWAITING REVIEW

| Role | Name | Date | Signature |
|------|------|------|-----------|
| Author | Sovereign AI Stack Team | 2025-12-06 | Pending |
| Batuta Oracle | Consulted | 2025-12-06 | Pending |
| QA Lead | - | - | PENDING |
| Tech Lead | - | - | PENDING |

---

*This specification integrates with the Sovereign AI Stack (Trueno, Aprender, Alimentar, Realizar, Pacha, Batuta) and follows Toyota Production System principles for lean, reproducible, and shareable visualization artifacts.*
