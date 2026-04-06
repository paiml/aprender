# Examples Catalog

> Parent: [presentar-spec.md](../presentar-spec.md)

**Scope:** 50 executable examples across 5 categories with 15-point QA checklist.

---

## Overview

50 executable examples demonstrating Presentar's visualization capabilities with `.apr` (Aprender models), `.ald` (Alimentar datasets), and generic visualization features. Each example follows a 15-point quality checklist.

## 15-Point QA Checklist

### A: Rendering Quality (5 points)
- **A1:** 60 FPS (no frame drops)
- **A2:** Anti-aliasing applied (no jagged edges)
- **A3:** Text crisp at all zoom levels
- **A4:** WCAG AA contrast (4.5:1 minimum)
- **A5:** Visual regression test passes (0 pixel diff)

### B: Data Integrity (4 points)
- **B1:** Data loads without corruption
- **B2:** Axis labels match data range
- **B3:** Legend matches series correctly
- **B4:** Tooltips show accurate values

### C: Interaction (3 points)
- **C1:** Hover states respond < 16ms
- **C2:** Click handlers fire correctly
- **C3:** Keyboard navigation works (a11y)

### D: Performance (3 points)
- **D1:** Initial render < 100ms
- **D2:** Memory stable (no leaks over 1000 frames)
- **D3:** WASM bundle < 500KB

## Example Categories

| Section | Description | Count | YAML |
|---------|-------------|-------|------|
| A | `.apr` Model Visualization | 10 | 5 |
| B | `.ald` Dataset Visualization | 10 | 5 |
| C | Basic Charts | 10 | 3 |
| D | Interactive Dashboards | 10 | 5 |
| E | Edge Cases & Stress Tests | 10 | 2 |
| **Total** | | **50** | **20** |

## Section A: `.apr` Model Visualization (APR-001 to APR-010)

| ID | Name | Focus | Format |
|----|------|-------|--------|
| APR-001 | Model Card Basic | Metadata display | YAML |
| APR-002 | Model Comparison | Side-by-side | YAML |
| APR-003 | Model Metrics Chart | Training curves | YAML |
| APR-004 | Architecture Diagram | Layer visualization | Code |
| APR-005 | Model Inference Demo | Real-time inference | YAML |
| APR-006 | Weight Histograms | Distribution viz | Code |
| APR-007 | Gradient Flow | Magnitude heatmap | YAML |
| APR-008 | Size Breakdown | Parameter pie chart | Code |
| APR-009 | Version History | Multi-version timeline | Code |
| APR-010 | Export Preview | Format preview | YAML |

**Example (APR-001):**
```yaml
presentar: "1.0"
name: "model-card-basic"
data:
  model: { source: "./models/mnist_mlp.apr" }
layout:
  type: column
  children:
    - widget: model_card
      source: "{{ data.model }}"
      show_metrics: true
```

## Section B: `.ald` Dataset Visualization (ALD-001 to ALD-010)

| ID | Name | Focus | Format |
|----|------|-------|--------|
| ALD-001 | Data Card Basic | Metadata display | YAML |
| ALD-002 | Data Table Virtualized | 100K row scrolling | YAML |
| ALD-003 | Data Distribution | Histogram rendering | YAML |
| ALD-004 | Scatter Plot | Correlation viz | YAML |
| ALD-005 | Heatmap Correlation | Correlation matrix | Code |
| ALD-006 | Time Series | Temporal data | YAML |
| ALD-007 | Missing Values Report | Data quality | Code |
| ALD-008 | Class Balance | Label distribution | YAML |
| ALD-009 | Schema Viewer | Column type inspection | Code |
| ALD-010 | Export Selection | Subset export | Code |

## Section C: Basic Charts (CHT-001 to CHT-010)

| ID | Name | Focus | Format |
|----|------|-------|--------|
| CHT-001 | Line Chart Basic | Simple line | YAML |
| CHT-002 | Bar Chart Grouped | Grouped bars | YAML |
| CHT-003 | Pie Chart Basic | Percentages | YAML |
| CHT-004 | Scatter with Size | Bubble chart | Code |
| CHT-005 | Heatmap Basic | 2D tensor viz | Code |
| CHT-006 | Box Plot | Quartile viz | Code |
| CHT-007 | Area Chart Stacked | Stacked area | Code |
| CHT-008 | Donut Chart | Ring chart | Code |
| CHT-009 | Sparkline Inline | Compact chart | Code |
| CHT-010 | Multi-Axis Chart | Dual Y-axis | Code |

## Section D: Interactive Dashboards (DSH-001 to DSH-010)

| ID | Name | Focus | Format |
|----|------|-------|--------|
| DSH-001 | Training Dashboard | Real-time metrics | YAML |
| DSH-002 | Dataset Explorer | Interactive exploration | YAML |
| DSH-003 | Model Comparison | A/B testing | YAML |
| DSH-004 | System Metrics | Performance monitoring | Code |
| DSH-005 | Experiment Tracker | Multi-run comparison | YAML |
| DSH-006 | Data Quality | Health metrics | Code |
| DSH-007 | Feature Importance | Explainability | Code |
| DSH-008 | Confusion Matrix | Classification results | YAML |
| DSH-009 | ROC Curve | Threshold analysis | Code |
| DSH-010 | Model Registry | Pacha integration | Code |

## Section E: Edge Cases & Stress Tests (EDG-001 to EDG-010)

| ID | Name | Focus | Acceptance |
|----|------|-------|------------|
| EDG-001 | Empty Dataset | Graceful empty state | No crash, message shown |
| EDG-002 | Large Dataset (1M) | Virtualization perf | 60fps, <200MB, <500ms |
| EDG-003 | Corrupt File | Error recovery (Jidoka) | Clear error, no panic |
| EDG-004 | Network Timeout | Offline resilience | Graceful degradation |
| EDG-005 | High DPI | Retina display | Crisp rendering |
| EDG-006 | Rapid Updates (1000/s) | Throttling | Stable frame rate |
| EDG-007 | Unicode/RTL | Internationalization | Correct layout |
| EDG-008 | Accessibility Audit | WCAG AA | All checks pass |
| EDG-009 | Memory Leak Soak (1hr) | Long-running stability | Stable memory/FPS |
| EDG-010 | Bundle Size | Production size | WASM < 500KB gzipped |

## Implementation Roadmap

| Phase | Weeks | Examples |
|-------|-------|---------|
| 1: Core | 1-2 | APR-001-005, ALD-001-005, CHT-001-003 |
| 2: Interactive | 3-4 | DSH-001-005, APR-006-010, ALD-006-010 |
| 3: Polish | 5-6 | CHT-004-010, DSH-006-010, EDG-001-010 |

## Quality Gates

| Tier | Timing | Scope |
|------|--------|-------|
| Tier 1 | < 5s | Format, clippy, unit tests |
| Tier 2 | < 30s | + WASM build, visual regression |
| Tier 3 | < 5m | + Property tests, a11y audit, bundle size |

## File Structure

```
examples/
  apr/         # .apr model examples (YAML configs)
  ald/         # .ald dataset examples
  charts/      # Generic chart examples
  dashboards/  # Interactive dashboards
  edge_cases/  # Stress tests
tests/
  visual_regression/  # Snapshot tests
  integration/       # Integration tests
```

## References

- Liker, J.K. (2004). *The Toyota Way*. McGraw-Hill.
- Wilkinson, L. (2005). *The Grammar of Graphics*. Springer.
- W3C (2018). *WCAG 2.1*. World Wide Web Consortium.
- Nielsen, J. (1993). *Usability Engineering*. Morgan Kaufmann.
