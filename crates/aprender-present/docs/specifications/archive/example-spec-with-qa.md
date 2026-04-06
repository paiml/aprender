# Presentar Example Specification with QA Process

## Version: 1.0.0
## Status: Draft
## Last Updated: 2025-11-30

---

## 1. Overview

This specification defines **50 executable examples** for demonstrating Presentar's visualization capabilities with `.apr` (Aprender models), `.ald` (Alimentar datasets), and generic visualization features. Each example follows the **Toyota Way QA philosophy** with a **15-point quality checklist**.

### 1.1 Design Philosophy (Toyota Way 4P's)

| Principle | Application to Presentar Examples | Source |
|-----------|-----------------------------------|--------|
| **Philosophy** | Visualization as first-class citizen; every chart must tell a story | [1] Liker |
| **Process** | YAML-driven configuration eliminates code; reproducible renders (Standard Work) | [4] Poppendieck |
| **People** | WCAG AA accessibility by default; respect for end users | [5] W3C |
| **Problem Solving** | Visual regression tests catch defects before users do (Jidoka) | [10] Rother |

### 1.2 Quality Metrics

| Metric | Target | Rationale | Source |
|--------|--------|-----------|--------|
| Frame Rate | 60 FPS | Smooth interaction (<16ms/frame) | [6] Nielsen |
| Bundle Size | <500KB | Fast loading | [6] Nielsen |
| First Paint | <100ms | Responsive feel | [8] Card et al. |
| WCAG Level | AA | Accessibility | [5] W3C |
| Test Coverage | >90% | Reliability | [4] Poppendieck |
| Visual Regression | 0 diffs | Pixel-perfect | [2] Wilkinson |

---

## 2. 15-Point QA Checklist

Every example must pass this checklist before release:

### Category A: Rendering Quality (5 points)
- [ ] **A1**: Renders at 60 FPS (no frame drops) [6]
- [ ] **A2**: Anti-aliasing applied (no jagged edges) [2]
- [ ] **A3**: Text is crisp at all zoom levels [3]
- [ ] **A4**: Colors pass WCAG AA contrast (4.5:1 minimum) [5]
- [ ] **A5**: Visual regression test passes (0 pixel diff) [10]

### Category B: Data Integrity (4 points)
- [ ] **B1**: Data loads without corruption [4]
- [ ] **B2**: Axis labels match data range [2]
- [ ] **B3**: Legend matches series correctly [3]
- [ ] **B4**: Tooltips show accurate values [7]

### Category C: Interaction (3 points)
- [ ] **C1**: Hover states respond <16ms [8]
- [ ] **C2**: Click handlers fire correctly [7]
- [ ] **C3**: Keyboard navigation works (a11y) [9]

### Category D: Performance (3 points)
- [ ] **D1**: Initial render <100ms [6]
- [ ] **D2**: Memory stable (no leaks over 1000 frames) [1]
- [ ] **D3**: WASM bundle <500KB [4]

---

## 3. Example Categories

### 3.1 Section Overview

| Section | Description | Examples | YAML Examples |
|---------|-------------|----------|---------------|
| A | `.apr` Model Visualization | 10 | 5 |
| B | `.ald` Dataset Visualization | 10 | 5 |
| C | Basic Charts | 10 | 3 |
| D | Interactive Dashboards | 10 | 5 |
| E | Edge Cases & Stress Tests | 10 | 2 |
| **Total** | | **50** | **20 (40%)** |

---

## 4. Section A: `.apr` Model Visualization (APR-001 to APR-010)

### APR-001: Model Card Basic [YAML]
**QA Focus**: Model metadata displays correctly

```yaml
# examples/apr/model_card_basic.yaml
presentar: "1.0"
name: "model-card-basic"
title: "MLP Model Card"

data:
  model:
    source: "./models/mnist_mlp.apr"

layout:
  type: column
  children:
    - widget: model_card
      source: "{{ data.model }}"
      show_metrics: true
      show_architecture: true
```

**Acceptance Criteria**:
- Model name, type, and version displayed
- Architecture diagram renders
- Training metrics (loss, accuracy) shown
- 15-point checklist complete

---

### APR-002: Model Comparison
**QA Focus**: Side-by-side model comparison accurate

```yaml
# examples/apr/model_comparison.yaml
presentar: "1.0"
name: "model-comparison"

data:
  model_a:
    source: "./models/mlp_v1.apr"
  model_b:
    source: "./models/mlp_v2.apr"

layout:
  type: row
  gap: 16
  children:
    - widget: model_card
      source: "{{ data.model_a }}"
      flex: 1
    - widget: model_card
      source: "{{ data.model_b }}"
      flex: 1
```

---

### APR-003: Model Metrics Chart [YAML]
**QA Focus**: Training curves render correctly

```yaml
presentar: "1.0"
name: "model-metrics-chart"

data:
  model:
    source: "./models/trained.apr"

layout:
  type: column
  children:
    - widget: chart
      type: line
      data: "{{ data.model.metrics.loss_history }}"
      title: "Training Loss"
      x_label: "Epoch"
      y_label: "Loss"
```

**Acceptance Criteria**:
- Loss curve smooth with anti-aliasing
- Epoch labels correct
- Hover shows exact loss value
- 15-point checklist complete

---

### APR-004: Model Architecture Diagram
**QA Focus**: Layer visualization accurate

**Command**: `cargo run --example apr_architecture`

**Acceptance Criteria**:
- All layers displayed
- Parameter counts match
- Connection arrows render
- 15-point checklist complete

---

### APR-005: Model Inference Demo [YAML]
**QA Focus**: Real-time inference visualization

```yaml
presentar: "1.0"
name: "model-inference"

data:
  model:
    source: "./models/classifier.apr"
  input:
    source: "./data/sample_input.ald"

layout:
  type: column
  children:
    - widget: model_card
      source: "{{ data.model }}"
      mode: "inference"
    - widget: data_card
      source: "{{ data.input }}"
    - widget: chart
      type: bar
      data: "{{ inference(data.model, data.input) }}"
      title: "Prediction Probabilities"
```

---

### APR-006: Model Weight Histograms
**QA Focus**: Weight distribution visualization

**Command**: `cargo run --example apr_weight_histograms`

**Acceptance Criteria**:
- Histogram bins correct
- Bell curve visible for initialized weights
- Layer selector works
- 15-point checklist complete

---

### APR-007: Model Gradient Flow [YAML]
**QA Focus**: Gradient magnitude heatmap

```yaml
presentar: "1.0"
name: "gradient-flow"

data:
  model:
    source: "./models/deep_net.apr"

layout:
  type: column
  children:
    - widget: chart
      type: heatmap
      data: "{{ data.model.gradients }}"
      title: "Gradient Magnitudes by Layer"
      colormap: "viridis"
```

---

### APR-008: Model Size Breakdown
**QA Focus**: Parameter count pie chart

**Command**: `cargo run --example apr_size_breakdown`

**Acceptance Criteria**:
- Pie chart sums to total params
- Percentages labeled
- Legend matches colors
- 15-point checklist complete

---

### APR-009: Model Version History
**QA Focus**: Multi-version comparison timeline

**Command**: `cargo run --example apr_version_history`

---

### APR-010: Model Export Preview [YAML]
**QA Focus**: Export format preview

```yaml
presentar: "1.0"
name: "model-export-preview"

data:
  model:
    source: "./models/export_ready.apr"

layout:
  type: column
  children:
    - widget: model_card
      source: "{{ data.model }}"
    - widget: button
      text: "Export to ONNX"
      action: "export_onnx"
    - widget: button
      text: "Export to SafeTensors"
      action: "export_safetensors"
```

---

## 5. Section B: `.ald` Dataset Visualization (ALD-001 to ALD-010)

### ALD-001: Data Card Basic [YAML]
**QA Focus**: Dataset metadata displays correctly

```yaml
presentar: "1.0"
name: "data-card-basic"

data:
  dataset:
    source: "./data/mnist.ald"

layout:
  type: column
  children:
    - widget: data_card
      source: "{{ data.dataset }}"
      show_stats: true
      show_preview: true
```

**Acceptance Criteria**:
- Row count accurate
- Column types displayed
- Byte size formatted
- Preview renders first 10 rows
- 15-point checklist complete

---

### ALD-002: Data Table Virtualized [YAML]
**QA Focus**: Large dataset scrolling smooth

```yaml
presentar: "1.0"
name: "data-table-virtualized"

data:
  dataset:
    source: "./data/large_100k.ald"

layout:
  type: column
  children:
    - widget: data_table
      source: "{{ data.dataset }}"
      virtualized: true
      page_size: 50
      sortable: true
      filterable: true
```

**Acceptance Criteria**:
- 100k rows scrolls at 60 FPS
- Sorting responsive <100ms
- Filtering works correctly
- 15-point checklist complete

---

### ALD-003: Data Distribution Chart
**QA Focus**: Histogram renders accurately

```yaml
presentar: "1.0"
name: "data-distribution"

data:
  dataset:
    source: "./data/features.ald"

layout:
  type: column
  children:
    - widget: chart
      type: histogram
      data: "{{ data.dataset.columns.feature_1 }}"
      bins: 50
      title: "Feature Distribution"
```

---

### ALD-004: Data Scatter Plot [YAML]
**QA Focus**: Correlation visualization

```yaml
presentar: "1.0"
name: "data-scatter"

data:
  dataset:
    source: "./data/iris.ald"

layout:
  type: column
  children:
    - widget: chart
      type: scatter
      x: "{{ data.dataset.columns.sepal_length }}"
      y: "{{ data.dataset.columns.petal_length }}"
      color: "{{ data.dataset.columns.species }}"
      title: "Iris Scatter Plot"
```

---

### ALD-005: Data Heatmap Correlation
**QA Focus**: Correlation matrix visualization

**Command**: `cargo run --example ald_correlation_heatmap`

**Acceptance Criteria**:
- Correlation values [-1, 1] range
- Diagonal is 1.0
- Color scale correct
- 15-point checklist complete

---

### ALD-006: Data Time Series [YAML]
**QA Focus**: Temporal data visualization

```yaml
presentar: "1.0"
name: "data-timeseries"

data:
  dataset:
    source: "./data/stock_prices.ald"

layout:
  type: column
  children:
    - widget: chart
      type: line
      x: "{{ data.dataset.columns.timestamp }}"
      y: "{{ data.dataset.columns.price }}"
      title: "Stock Price Over Time"
      x_axis_type: "datetime"
```

---

### ALD-007: Data Missing Values Report
**QA Focus**: Data quality visualization

**Command**: `cargo run --example ald_missing_values`

---

### ALD-008: Data Class Balance [YAML]
**QA Focus**: Label distribution bar chart

```yaml
presentar: "1.0"
name: "class-balance"

data:
  dataset:
    source: "./data/classification.ald"

layout:
  type: column
  children:
    - widget: chart
      type: bar
      data: "{{ count_by(data.dataset.columns.label) }}"
      title: "Class Distribution"
      orientation: "horizontal"
```

---

### ALD-009: Data Schema Viewer
**QA Focus**: Column type inspection

**Command**: `cargo run --example ald_schema_viewer`

---

### ALD-010: Data Export Selection
**QA Focus**: Subset export workflow

**Command**: `cargo run --example ald_export_selection`

---

## 6. Section C: Basic Charts (CHT-001 to CHT-010)

### CHT-001: Line Chart Basic [YAML]
**QA Focus**: Simple line chart renders

```yaml
presentar: "1.0"
name: "line-chart-basic"

layout:
  type: column
  children:
    - widget: chart
      type: line
      data:
        - [0, 10]
        - [1, 15]
        - [2, 12]
        - [3, 18]
        - [4, 22]
      title: "Basic Line Chart"
      stroke_width: 2
      anti_alias: true
```

**Acceptance Criteria**:
- Line smooth with anti-aliasing
- Points at correct coordinates
- Title rendered
- 15-point checklist complete

---

### CHT-002: Bar Chart Grouped [YAML]
**QA Focus**: Grouped bar chart alignment

```yaml
presentar: "1.0"
name: "bar-chart-grouped"

layout:
  type: column
  children:
    - widget: chart
      type: bar
      data:
        labels: ["Q1", "Q2", "Q3", "Q4"]
        series:
          - name: "2023"
            values: [100, 120, 90, 150]
          - name: "2024"
            values: [110, 140, 100, 180]
      title: "Quarterly Revenue"
      grouped: true
```

---

### CHT-003: Pie Chart Basic [YAML]
**QA Focus**: Pie chart percentages correct

```yaml
presentar: "1.0"
name: "pie-chart-basic"

layout:
  type: column
  children:
    - widget: chart
      type: pie
      data:
        - { label: "A", value: 30 }
        - { label: "B", value: 50 }
        - { label: "C", value: 20 }
      title: "Category Breakdown"
      show_labels: true
      show_percentages: true
```

---

### CHT-004: Scatter Plot with Size
**QA Focus**: Bubble chart rendering

**Command**: `cargo run --example cht_scatter_bubble`

---

### CHT-005: Heatmap Basic
**QA Focus**: 2D tensor visualization

**Command**: `cargo run --example cht_heatmap_basic`

---

### CHT-006: Box Plot Distribution
**QA Focus**: Quartile visualization

**Command**: `cargo run --example cht_boxplot`

---

### CHT-007: Area Chart Stacked
**QA Focus**: Stacked area rendering

**Command**: `cargo run --example cht_area_stacked`

---

### CHT-008: Donut Chart
**QA Focus**: Ring chart with center label

**Command**: `cargo run --example cht_donut`

---

### CHT-009: Sparkline Inline
**QA Focus**: Compact chart in text

**Command**: `cargo run --example cht_sparkline`

---

### CHT-010: Multi-Axis Chart
**QA Focus**: Dual Y-axis alignment

**Command**: `cargo run --example cht_multi_axis`

---

## 7. Section D: Interactive Dashboards (DSH-001 to DSH-010)

### DSH-001: Model Training Dashboard [YAML]
**QA Focus**: Real-time training metrics

```yaml
presentar: "1.0"
name: "training-dashboard"

data:
  model:
    source: "./models/training.apr"
    watch: true  # Live updates

layout:
  type: grid
  columns: 12
  children:
    - widget: model_card
      source: "{{ data.model }}"
      col_span: 4
    - widget: chart
      type: line
      data: "{{ data.model.metrics.loss_history }}"
      title: "Loss"
      col_span: 4
    - widget: chart
      type: line
      data: "{{ data.model.metrics.accuracy_history }}"
      title: "Accuracy"
      col_span: 4
    - widget: text
      content: "Epoch: {{ data.model.current_epoch }}"
      col_span: 12
```

**Acceptance Criteria**:
- Updates at 60 FPS during training
- Charts animate smoothly
- Memory stable over 10 minutes
- 15-point checklist complete

---

### DSH-002: Dataset Explorer [YAML]
**QA Focus**: Interactive data exploration

```yaml
presentar: "1.0"
name: "dataset-explorer"

data:
  dataset:
    source: "./data/tabular.ald"

state:
  selected_column: null
  filter_query: ""

layout:
  type: row
  children:
    - widget: column
      flex: 1
      children:
        - widget: select
          options: "{{ data.dataset.column_names }}"
          bind: "state.selected_column"
        - widget: text_input
          placeholder: "Filter..."
          bind: "state.filter_query"
    - widget: column
      flex: 3
      children:
        - widget: data_table
          source: "{{ data.dataset }}"
          filter: "{{ state.filter_query }}"
        - widget: chart
          type: histogram
          data: "{{ data.dataset.columns[state.selected_column] }}"
          visible: "{{ state.selected_column != null }}"
```

---

### DSH-003: Model Comparison Dashboard [YAML]
**QA Focus**: A/B model comparison

```yaml
presentar: "1.0"
name: "model-comparison-dashboard"

data:
  models:
    - source: "./models/baseline.apr"
    - source: "./models/improved.apr"
    - source: "./models/experimental.apr"

layout:
  type: grid
  columns: 12
  children:
    - widget: chart
      type: bar
      data: "{{ map(data.models, m => m.metrics.accuracy) }}"
      labels: "{{ map(data.models, m => m.name) }}"
      title: "Accuracy Comparison"
      col_span: 6
    - widget: chart
      type: bar
      data: "{{ map(data.models, m => m.param_count) }}"
      labels: "{{ map(data.models, m => m.name) }}"
      title: "Model Size"
      col_span: 6
```

---

### DSH-004: System Metrics Dashboard
**QA Focus**: Performance monitoring

**Command**: `cargo run --example dsh_system_metrics`

---

### DSH-005: Experiment Tracker [YAML]
**QA Focus**: Multi-run comparison

```yaml
presentar: "1.0"
name: "experiment-tracker"

data:
  experiments:
    source: "trueno-db://experiments"
    query: "SELECT * FROM runs WHERE project = 'mnist'"

layout:
  type: column
  children:
    - widget: data_table
      source: "{{ data.experiments }}"
      columns: ["name", "accuracy", "loss", "duration"]
      sortable: true
    - widget: chart
      type: scatter
      x: "{{ data.experiments.columns.param_count }}"
      y: "{{ data.experiments.columns.accuracy }}"
      title: "Accuracy vs Model Size"
```

---

### DSH-006: Data Quality Dashboard
**QA Focus**: Data health metrics

**Command**: `cargo run --example dsh_data_quality`

---

### DSH-007: Feature Importance Dashboard
**QA Focus**: Explainability visualization

**Command**: `cargo run --example dsh_feature_importance`

---

### DSH-008: Confusion Matrix Dashboard [YAML]
**QA Focus**: Classification results

```yaml
presentar: "1.0"
name: "confusion-matrix"

data:
  predictions:
    source: "./data/predictions.ald"

layout:
  type: column
  children:
    - widget: chart
      type: heatmap
      data: "{{ confusion_matrix(data.predictions.columns.true, data.predictions.columns.pred) }}"
      title: "Confusion Matrix"
      colormap: "blues"
      annotate: true
```

---

### DSH-009: ROC Curve Dashboard
**QA Focus**: Threshold analysis

**Command**: `cargo run --example dsh_roc_curve`

---

### DSH-010: Model Registry Browser
**QA Focus**: Pacha registry integration

**Command**: `cargo run --example dsh_model_registry`

---

## 8. Section E: Edge Cases & Stress Tests (EDG-001 to EDG-010)

### EDG-001: Empty Dataset [YAML]
**QA Focus**: Graceful empty state

```yaml
presentar: "1.0"
name: "empty-dataset"

data:
  dataset:
    source: "./data/empty.ald"

layout:
  type: column
  children:
    - widget: data_card
      source: "{{ data.dataset }}"
      empty_message: "No data available"
```

**Acceptance Criteria**:
- No crash on empty data
- Empty state message shown
- Layout stable
- 15-point checklist complete

---

### EDG-002: Large Dataset (1M rows) [YAML]
**QA Focus**: Virtualization performance

```yaml
presentar: "1.0"
name: "large-dataset"

data:
  dataset:
    source: "./data/million_rows.ald"

layout:
  type: column
  children:
    - widget: data_table
      source: "{{ data.dataset }}"
      virtualized: true
      row_height: 32
```

**Acceptance Criteria**:
- Scrolling at 60 FPS
- Memory <200MB
- Initial render <500ms
- 15-point checklist complete

---

### EDG-003: Corrupt File Handling
**QA Focus**: Error recovery (Jidoka)

**Command**: `cargo run --example edg_corrupt_file`

**Acceptance Criteria**:
- Clear error message
- No panic/crash
- Graceful degradation
- 15-point checklist complete

---

### EDG-004: Network Timeout
**QA Focus**: Offline resilience

**Command**: `cargo run --example edg_network_timeout`

---

### EDG-005: High DPI Rendering
**QA Focus**: Retina display quality

**Command**: `cargo run --example edg_high_dpi`

---

### EDG-006: Rapid Updates (1000/sec)
**QA Focus**: Update throttling

**Command**: `cargo run --example edg_rapid_updates`

---

### EDG-007: Unicode/RTL Text
**QA Focus**: Internationalization

**Command**: `cargo run --example edg_unicode_rtl`

---

### EDG-008: Accessibility Audit
**QA Focus**: WCAG AA compliance [5]

**Command**: `cargo run --example edg_a11y_audit`

**Acceptance Criteria**:
- All contrast ratios pass [5]
- Keyboard navigation works [5]
- Screen reader labels present (Natural Language Descriptions) [9]
- Focus indicators visible
- 15-point checklist complete

---

### EDG-009: Memory Leak Soak Test
**QA Focus**: Long-running stability

**Command**: `cargo run --example edg_memory_soak -- --duration 3600`

**Acceptance Criteria**:
- Memory stable over 1 hour
- No frame rate degradation
- GC pauses <16ms
- 15-point checklist complete

---

### EDG-010: Bundle Size Verification
**QA Focus**: Production build size

**Command**: `wasm-pack build --release && ls -la pkg/*.wasm`

**Acceptance Criteria**:
- WASM <500KB gzipped
- No dead code
- Tree shaking effective
- 15-point checklist complete

---

## 9. Implementation Roadmap

### Phase 1: Core Examples (Week 1-2)
- [ ] APR-001 to APR-005 (Model basics)
- [ ] ALD-001 to ALD-005 (Data basics)
- [ ] CHT-001 to CHT-003 (Chart basics)

### Phase 2: Interactive Features (Week 3-4)
- [ ] DSH-001 to DSH-005 (Dashboards)
- [ ] APR-006 to APR-010 (Advanced models)
- [ ] ALD-006 to ALD-010 (Advanced data)

### Phase 3: Polish & Edge Cases (Week 5-6)
- [ ] CHT-004 to CHT-010 (All charts)
- [ ] DSH-006 to DSH-010 (All dashboards)
- [ ] EDG-001 to EDG-010 (Edge cases)

---

## 10. Quality Gates

### Tier 1: Fast (<5s)
```bash
make tier1  # Format, clippy, unit tests
```

### Tier 2: Integration (<30s)
```bash
make tier2  # Tier 1 + WASM build + visual regression
```

### Tier 3: Full (<5m)
```bash
make tier3  # Tier 2 + property tests + a11y audit + bundle size
```

---

## 11. Example File Structure

```
presentar/
├── examples/
│   ├── apr/                    # .apr model examples
│   │   ├── model_card_basic.yaml
│   │   ├── model_comparison.yaml
│   │   └── ...
│   ├── ald/                    # .ald dataset examples
│   │   ├── data_card_basic.yaml
│   │   ├── data_table_virtualized.yaml
│   │   └── ...
│   ├── charts/                 # Generic chart examples
│   │   ├── line_chart_basic.yaml
│   │   ├── bar_chart_grouped.yaml
│   │   └── ...
│   ├── dashboards/             # Interactive dashboards
│   │   ├── training_dashboard.yaml
│   │   └── ...
│   └── edge_cases/             # Stress tests
│       ├── empty_dataset.yaml
│       └── ...
├── tests/
│   ├── visual_regression/      # Snapshot tests
│   └── integration/            # Integration tests
└── docs/
    └── specifications/
        └── example-spec-with-qa.md  # This document
```

---

## 12. Appendix A: References

1. **Liker, J. K. (2004).** *The Toyota Way: 14 Management Principles from the World's Greatest Manufacturer*. McGraw-Hill.
2.  **Wilkinson, L. (2005).** *The Grammar of Graphics*. Springer.
3.  **Munzner, T. (2014).** *Visualization Analysis and Design*. CRC Press.
4.  **Poppendieck, M., & Poppendieck, T. (2003).** *Lean Software Development: An Agile Toolkit*. Addison-Wesley.
5.  **W3C. (2018).** *Web Content Accessibility Guidelines (WCAG) 2.1*. World Wide Web Consortium.
6.  **Nielsen, J. (1993).** *Usability Engineering*. Morgan Kaufmann.
7.  **Shneiderman, B. (1996).** "The Eyes Have It: A Task by Data Type Taxonomy for Information Visualizations". *Proceedings of the IEEE Symposium on Visual Languages*.
8.  **Card, S. K., Robertson, G. G., & Mackinlay, J. D. (1991).** "The Information Visualizer, an Information Workspace". *Proceedings of the SIGCHI Conference on Human Factors in Computing Systems*.
9.  **Kim, N. W., & Heer, J. (2021).** "Accessible Visualization via Natural Language Descriptions". *Computer Graphics Forum*.
10. **Rother, M. (2009).** *Toyota Kata: Managing People for Improvement, Adaptiveness and Superior Results*. McGraw-Hill.

---

## 13. Appendix B: Checklist Template

```markdown
## Example: [ID] - [Name]

### 15-Point QA Checklist

**Category A: Rendering Quality**
- [ ] A1: 60 FPS verified [6]
- [ ] A2: Anti-aliasing applied [2]
- [ ] A3: Text crisp [3]
- [ ] A4: WCAG AA contrast [5]
- [ ] A5: Visual regression pass [10]

**Category B: Data Integrity**
- [ ] B1: Data loads correctly [4]
- [ ] B2: Axis labels match [2]
- [ ] B3: Legend correct [3]
- [ ] B4: Tooltips accurate [7]

**Category C: Interaction**
- [ ] C1: Hover <16ms [8]
- [ ] C2: Click handlers work [7]
- [ ] C3: Keyboard nav works [9]

**Category D: Performance**
- [ ] D1: Initial render <100ms [6]
- [ ] D2: Memory stable [1]
- [ ] D3: Bundle <500KB [4]

**Sign-off**: _____________ Date: _______
```

---

## 14. Reviewer Notes

**Review in the Spirit of the Toyota Way:**

1.  **Genchi Genbutsu (Go and See):** This specification effectively moves away from abstract requirements to "executable examples." This aligns perfectly with the principle of making problems and standards visible. However, ensure that the "Quality Gates" [Section 10] are not just automated checks but are also manually verified for "Look and Feel" periodically, as automation cannot capture all nuances of user experience.
2.  **Jidoka (Built-in Quality):** The 15-point checklist acts as a "stop the line" signal. If an example doesn't pass, it shouldn't proceed. The heavy reliance on visual regression [A5] is a strong Jidoka implementation.
3.  **Standardized Work:** The YAML configuration [Section 4-8] standardizes how examples are created, reducing variability and waste (Muda) in setting up new visualizations.
4.  **Heijunka (Leveling):** The roadmap attempts to level the workload, but care should be taken to ensure "Edge Cases" [Section 8] are not left entirely for the end, as they often reveal structural flaws that require rework. Consider pulling some edge cases (like Empty Dataset EDG-001) into Phase 1.
5.  **Respect for People:** The explicit inclusion of Accessibility (WCAG) [5] and Performance metrics (<100ms) [6] demonstrates respect for the end user's time and ability.

**Action Items:**
- Verify that the *manual* checklist items (e.g., "Text is crisp") are clear enough to be objective.
- Ensure the "Visual Regression" tool is robust enough to avoid flaky tests, which would create "Muda" (waste) in investigating false positives.

---

**Document Control**
| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 1.1.0 | 2025-11-30 | Gemini | Added peer-reviewed annotations and Toyota Way review notes |
| 1.0.0 | 2025-11-30 | Claude | Initial specification |
