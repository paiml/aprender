# Presentar - Sovereign AI Visualization Framework

[Introduction](./introduction.md)

# Getting Started

- [Installation](./getting-started/installation.md)
- [Quick Start](./getting-started/quick-start.md)
- [First App](./getting-started/first-app.md)
- [Core Concepts](./getting-started/core-concepts.md)
- [YAML Configuration](./getting-started/yaml-configuration.md)

# Architecture

- [Overview](./architecture/overview.md)
- [Layer Hierarchy](./architecture/layer-hierarchy.md)
- [Unidirectional Data Flow](./architecture/data-flow.md)
- [Widget Tree](./architecture/widget-tree.md)
- [Layout Engine](./architecture/layout-engine.md)
- [Rendering Pipeline](./architecture/rendering-pipeline.md)
- [State Management](./architecture/state-management.md)
- [Event System](./architecture/event-system.md)

# Widget System

- [Brick Architecture](./widgets/brick-architecture.md)
- [Widget Trait](./widgets/widget-trait.md)
- [Core Widgets](./widgets/core-widgets.md)
  - [Container](./widgets/container.md)
  - [Row](./widgets/row.md)
  - [Column](./widgets/column.md)
  - [Stack](./widgets/stack.md)
  - [Text](./widgets/text.md)
  - [Button](./widgets/button.md)
  - [Slider](./widgets/slider.md)
  - [TextInput](./widgets/text-input.md)
  - [Select](./widgets/select.md)
  - [Checkbox](./widgets/checkbox.md)
- [Data Widgets](./widgets/data-widgets.md)
  - [DataTable](./widgets/data-table.md)
  - [Chart](./widgets/chart.md)
  - [ModelCard](./widgets/model-card.md)
  - [DataCard](./widgets/data-card.md)
- [Custom Widgets](./widgets/custom-widgets.md)

# Layout

- [Flexbox Model](./layout/flexbox-model.md)
- [Constraints](./layout/constraints.md)
- [Measure-Layout-Paint](./layout/measure-layout-paint.md)
- [Grid System](./layout/grid-system.md)
- [Responsive Design](./layout/responsive-design.md)
- [Layout Caching](./layout/layout-caching.md)

# YAML Manifest

- [Manifest Schema](./yaml/manifest-schema.md)
- [Scene Format (.prs)](./yaml/scene-format.md)
- [Data Sources](./yaml/data-sources.md)
- [Model References](./yaml/model-references.md)
- [Expression Language](./yaml/expression-language.md)
  - [Transforms](./yaml/transforms.md)
  - [Filters](./yaml/filters.md)
  - [Aggregations](./yaml/aggregations.md)
- [Interactions](./yaml/interactions.md)
- [Theming](./yaml/theming.md)

# Testing (presentar-test)

- [Philosophy: Zero External Dependencies](./testing/philosophy.md)
- [Test Harness](./testing/test-harness.md)
- [Selectors](./testing/selectors.md)
- [Event Simulation](./testing/event-simulation.md)
- [Assertions](./testing/assertions.md)
- [Visual Regression](./testing/visual-regression.md)
- [Accessibility Checking](./testing/accessibility-checking.md)
- [Performance Testing](./testing/performance-testing.md)

# Quality System

- [App Quality Score](./quality/app-quality-score.md)
- [Grade Thresholds (F-A)](./quality/grade-thresholds.md)
- [Quality Gates](./quality/quality-gates.md)
- [Structural Metrics](./quality/structural-metrics.md)
- [Performance Metrics](./quality/performance-metrics.md)
- [Accessibility Metrics](./quality/accessibility-metrics.md)
- [Data Quality Metrics](./quality/data-quality-metrics.md)

# Toyota Way Principles

- [Overview](./toyota-way/overview.md)
- [Muda (Waste Elimination)](./toyota-way/muda.md)
- [Jidoka (Built-in Quality)](./toyota-way/jidoka.md)
- [Kaizen (Continuous Improvement)](./toyota-way/kaizen.md)
- [Poka-yoke (Mistake Proofing)](./toyota-way/poka-yoke.md)
- [Visual Control](./toyota-way/visual-control.md)
- [Genchi Genbutsu (Go and See)](./toyota-way/genchi-genbutsu.md)

# Examples

- [Hello World](./examples/hello-world.md)
- [Counter App](./examples/counter-app.md)
- [Shell Autocomplete (Showcase)](./examples/shell-autocomplete.md)
- [Dashboard](./examples/dashboard.md)
- [Charts](./examples/charts.md)
- [Edge Cases](./examples/edge-cases.md)
- [Data Management](./examples/data-management.md)
- [Fraud Detection Dashboard](./examples/fraud-detection.md)
- [MNIST Explorer](./examples/mnist-explorer.md)
- [Data Table](./examples/data-table.md)
- [Model Card Display](./examples/model-card-display.md)

# Development Guide

- [Contributing](./development/contributing.md)
- [CLI Reference](./development/cli-reference.md)
- [EXTREME TDD](./development/extreme-tdd.md)
- [RED-GREEN-REFACTOR](./development/red-green-refactor.md)
- [Testing Strategy](./development/testing-strategy.md)
- [Property-Based Testing](./development/property-based-testing.md)
- [Mutation Testing](./development/mutation-testing.md)
- [Code Review Checklist](./development/code-review.md)
- [Makefile Targets](./development/makefile-targets.md)

# Terminal Backend

- [Direct Backend](./terminal/direct-backend.md)
- [Monitoring Examples](./terminal/monitoring-examples.md)
- [ptop - System Monitor](./terminal/ptop.md)

# Advanced Topics

- [GPU Rendering](./advanced/gpu-rendering.md)
- [WGSL Shaders](./advanced/wgsl-shaders.md)
- [Draw Commands](./advanced/draw-commands.md)
- [Anti-Aliasing](./advanced/anti-aliasing.md)
- [Virtualization](./advanced/virtualization.md)
- [Memory Management](./advanced/memory-management.md)
- [WASM Optimization](./advanced/wasm-optimization.md)
- [Bundle Size](./advanced/bundle-size.md)

# Ecosystem Integration

- [Trueno Integration](./ecosystem/trueno.md)
- [Trueno-Viz Primitives](./ecosystem/trueno-viz.md)
- [Aprender (.apr) Models](./ecosystem/aprender.md)
- [Alimentar (.ald) Datasets](./ecosystem/alimentar.md)
- [Pacha Registry](./ecosystem/pacha.md)
- [Ruchy Scripting](./ecosystem/ruchy.md)

# Specifications

- [Design Philosophy](./specifications/design-philosophy.md)
- [WASM-First Principles](./specifications/wasm-first.md)
- [Widget Specification](./specifications/widget-spec.md)
- [YAML Schema](./specifications/yaml-schema.md)
- [Quality Scoring](./specifications/quality-scoring.md)
- [Academic Foundations](./specifications/academic-foundations.md)

# Appendix

- [Glossary](./appendix/glossary.md)
- [References](./appendix/references.md)
- [FAQ](./appendix/faq.md)
- [Changelog](./appendix/changelog.md)
- [Migration Guide](./appendix/migration-guide.md)
- [WCAG 2.1 Checklist](./appendix/wcag-checklist.md)
