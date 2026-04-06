# Graph TUI Visualization Specification

**Version**: 1.0.0
**Status**: Draft
**Author**: PAIML Team
**Date**: 2025-12-08

---

## Executive Summary

This specification defines a lightweight, Toyota Way-aligned architecture for terminal-based graph visualization across the Sovereign AI Stack. The design leverages existing components (trueno-graph, trueno-viz, batuta) while introducing minimal new abstractions for TUI graph rendering.

---

## 1. Problem Statement (Genchi Genbutsu)

### 1.1 Current State

The PAIML stack lacks unified terminal-based graph visualization:

| Component | Graph Capability | TUI Support |
|-----------|------------------|-------------|
| trueno-graph | Algorithms (PageRank, Louvain, traversal) | None |
| trueno-viz | Visualization primitives, `graph` feature | PNG/SVG only |
| presentar | WASM application framework | Browser-based |
| batuta | ratatui/crossterm available | ASCII dashboards only |

### 1.2 Use Cases Requiring Graph TUI

1. **Dependency graphs** - `batuta stack publish-status` dependency visualization
2. **ML pipelines** - Visualization of DAG workflows
3. **Knowledge graphs** - trueno-rag retrieval paths
4. **Cluster analysis** - Louvain community detection results
5. **Performance profiling** - Call graphs and flame graphs

---

## 2. Architectural Decision (Heijunka - Level Loading)

### 2.1 Layer Placement Analysis

```
┌─────────────────────────────────────────────────────────────┐
│                    APPLICATION LAYER                         │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐          │
│  │   batuta    │  │  presentar  │  │   (CLI)     │          │
│  │  (TUI app)  │  │  (WASM app) │  │             │          │
│  └──────┬──────┘  └──────┬──────┘  └─────────────┘          │
│         │                │                                   │
├─────────┼────────────────┼───────────────────────────────────┤
│         │     VISUALIZATION LAYER                            │
│         │                │                                   │
│  ┌──────┴──────┐  ┌──────┴──────┐                           │
│  │ trueno-viz  │  │ trueno-viz  │                           │
│  │ (TUI mode)  │  │ (WASM mode) │                           │
│  └──────┬──────┘  └─────────────┘                           │
│         │                                                    │
├─────────┼────────────────────────────────────────────────────┤
│         │         DATA LAYER                                 │
│  ┌──────┴──────┐                                            │
│  │trueno-graph │  Graph algorithms, storage                  │
│  └─────────────┘                                            │
└─────────────────────────────────────────────────────────────┘
```

### 2.2 Decision: Extend trueno-viz with TUI Backend

**Rationale** (aligned with Toyota Way principles):

| Principle | Application |
|-----------|-------------|
| **Jidoka** | Layout algorithms in trueno-viz enable quality checks at render time |
| **Just-in-Time** | Layouts computed lazily, only when visualization requested |
| **Heijunka** | Single codebase for layout; multiple backends (TUI, SVG, PNG) |
| **Genchi Genbutsu** | Graph data stays in trueno-graph; viz logic in trueno-viz |
| **Kaizen** | Incremental addition of layout algorithms without breaking changes |

### 2.3 Component Responsibilities

| Component | New Responsibility |
|-----------|-------------------|
| **trueno-graph** | None (unchanged) - provides graph data structures |
| **trueno-viz** | Add `tui` feature with ratatui-based graph widgets |
| **batuta** | Consume trueno-viz TUI widgets for stack visualizations |
| **presentar** | Unchanged - continues WASM focus |

---

## 3. Layout Algorithms (Poka-Yoke - Error Prevention)

### 3.1 Algorithm Selection Matrix

| Algorithm | Complexity | Best For | Citation |
|-----------|------------|----------|----------|
| Fruchterman-Reingold | O(n²) per iteration | Small graphs (<500 nodes) | [1] |
| Kamada-Kawai | O(n² log n + nm) | Medium graphs, aesthetic | [2] |
| Multilevel FR | O(n log n) | Large graphs (>1000 nodes) | [3] |
| Sugiyama | O(nm log m) | DAGs, hierarchies | [4] |
| Radial | O(n + m) | Trees, star topologies | [5] |

### 3.2 Implementation Priority (Kaizen)

**Phase 1: MVP**
- Fruchterman-Reingold (force-directed)
- Grid layout (fallback)

**Phase 2: Hierarchical**
- Sugiyama (DAG layout)
- Radial (tree layout)

**Phase 3: Scalability**
- Multilevel force-directed
- Barnes-Hut approximation for O(n log n)

### 3.3 SIMD Acceleration Strategy

Force calculations are embarrassingly parallel:

```rust
// trueno-viz/src/layout/force_directed.rs
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

/// SIMD-accelerated repulsive force calculation
/// Processes 4 vertex pairs simultaneously (AVX2)
#[inline]
#[target_feature(enable = "avx2")]
unsafe fn repulsive_forces_avx2(
    positions: &[[f32; 2]],
    forces: &mut [[f32; 2]],
    k_squared: f32,
) {
    // 4-wide SIMD processing of Coulomb repulsion
    // F = k² / d
}
```

---

## 4. TUI Rendering Architecture (Mieruka - Visual Management)

### 4.1 Widget Hierarchy

```rust
// trueno-viz/src/tui/mod.rs

/// Graph visualization widget for ratatui
pub struct GraphWidget<'a, G: GraphData> {
    /// Source graph data
    graph: &'a G,
    /// Layout algorithm
    layout: LayoutAlgorithm,
    /// Visual style
    style: GraphStyle,
    /// Viewport for pan/zoom
    viewport: Viewport,
}

/// Layout algorithm selection
pub enum LayoutAlgorithm {
    ForceDirected(ForceDirectedConfig),
    Sugiyama(SugiyamaConfig),
    Radial(RadialConfig),
    Grid,
    Custom(Box<dyn Layout>),
}
```

### 4.2 Unicode Box-Drawing Characters

Terminal graph rendering uses Unicode for clean visualization:

| Element | Character | Unicode |
|---------|-----------|---------|
| Node (default) | ● | U+25CF |
| Node (selected) | ◉ | U+25C9 |
| Edge (horizontal) | ─ | U+2500 |
| Edge (vertical) | │ | U+2502 |
| Corner (top-left) | ┌ | U+250C |
| Arrow (right) | → | U+2192 |
| Arrow (down) | ↓ | U+2193 |

### 4.3 Color Semantics (Preattentive Processing)

Based on Healey's research on preattentive visual features [6]:

| Visual Feature | Semantic Meaning |
|----------------|------------------|
| **Hue (Red)** | Error, needs attention |
| **Hue (Green)** | Healthy, up-to-date |
| **Hue (Yellow)** | Warning, pending action |
| **Hue (Cyan)** | Information, selected |
| **Intensity** | Node importance (PageRank) |
| **Size** | Degree centrality |

---

## 5. API Design (Standardized Work)

### 5.1 Builder Pattern

```rust
use trueno_viz::tui::prelude::*;
use trueno_graph::Graph;

// Example: Visualizing publish-status dependencies
let graph = build_dependency_graph(&workspace)?;

let widget = GraphWidget::new(&graph)
    .layout(LayoutAlgorithm::Sugiyama(SugiyamaConfig::default()))
    .style(GraphStyle::default()
        .node_color(|node| match node.status {
            PublishAction::UpToDate => Color::Green,
            PublishAction::NeedsPublish => Color::Red,
            PublishAction::NeedsCommit => Color::Yellow,
            _ => Color::Gray,
        })
        .edge_style(EdgeStyle::Arrow))
    .viewport(Viewport::fit_content())
    .build();

// Render in ratatui frame
frame.render_widget(widget, area);
```

### 5.2 Trait Abstraction

```rust
/// Trait for types that can provide graph data for visualization
pub trait GraphData {
    type NodeId: Copy + Eq + Hash;
    type NodeData;
    type EdgeData;

    fn nodes(&self) -> impl Iterator<Item = (Self::NodeId, &Self::NodeData)>;
    fn edges(&self) -> impl Iterator<Item = (Self::NodeId, Self::NodeId, &Self::EdgeData)>;
    fn neighbors(&self, node: Self::NodeId) -> impl Iterator<Item = Self::NodeId>;
}

// Blanket implementation for trueno-graph types
impl<N, E> GraphData for trueno_graph::Graph<N, E> { ... }
```

---

## 6. Performance Targets (Takt Time)

### 6.1 Latency Requirements

| Operation | Target | Rationale |
|-----------|--------|-----------|
| Initial layout (≤100 nodes) | <100ms | Interactive feel |
| Initial layout (≤1000 nodes) | <1s | Acceptable wait |
| Incremental update | <16ms | 60 FPS animation |
| Pan/zoom | <8ms | Responsive interaction |

### 6.2 Memory Budget

| Graph Size | Memory Limit |
|------------|--------------|
| ≤100 nodes | <1 MB |
| ≤1000 nodes | <10 MB |
| ≤10000 nodes | <100 MB |

### 6.3 Cognitive Load Management

Based on Sweller's Cognitive Load Theory [7]:

- **Limit visible nodes**: Default to 50 nodes; use clustering for larger graphs
- **Progressive disclosure**: Show details on hover/selection
- **Reduce extraneous load**: No decorative elements; semantic color only
- **Chunking**: Group related nodes visually using Louvain communities

---

## 7. Integration Examples

### 7.1 Publish Status Visualization

```rust
// batuta/src/commands/stack_publish_status_viz.rs

pub fn render_dependency_graph(
    frame: &mut Frame,
    area: Rect,
    report: &PublishStatusReport,
) {
    // Build graph from publish status
    let mut graph = Graph::new();

    for crate_status in &report.crates {
        graph.add_node(crate_status.name.clone(), crate_status.clone());
    }

    // Add dependency edges (from Cargo.toml analysis)
    for (from, to) in &report.dependencies {
        graph.add_edge(from.clone(), to.clone(), ());
    }

    // Render with status-aware coloring
    let widget = GraphWidget::new(&graph)
        .layout(LayoutAlgorithm::Sugiyama(Default::default()))
        .style(publish_status_style())
        .build();

    frame.render_widget(widget, area);
}
```

### 7.2 ML Pipeline DAG

```rust
// batuta/src/commands/pipeline_viz.rs

pub fn render_pipeline(
    frame: &mut Frame,
    area: Rect,
    pipeline: &Pipeline,
) {
    let widget = GraphWidget::new(&pipeline.dag)
        .layout(LayoutAlgorithm::Sugiyama(SugiyamaConfig {
            layer_separation: 3,
            node_separation: 2,
            ..Default::default()
        }))
        .style(GraphStyle::default()
            .node_label(|stage| stage.name.clone())
            .node_shape(|stage| match stage.kind {
                StageKind::Input => NodeShape::Diamond,
                StageKind::Transform => NodeShape::Rectangle,
                StageKind::Output => NodeShape::Ellipse,
            }))
        .build();

    frame.render_widget(widget, area);
}
```

---

## 8. Testing Strategy (Jidoka)

### 8.1 Unit Tests

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fruchterman_reingold_convergence() {
        let graph = create_test_graph(10, 15);
        let layout = ForceDirectedLayout::new(&graph)
            .iterations(100)
            .compute();

        // Verify no overlapping nodes
        assert!(layout.no_overlaps());
        // Verify edge lengths are reasonable
        assert!(layout.edge_length_variance() < 0.5);
    }

    #[test]
    fn test_sugiyama_no_edge_crossings_simple_dag() {
        let dag = create_simple_dag();
        let layout = SugiyamaLayout::new(&dag).compute();

        // Simple DAGs should have zero crossings
        assert_eq!(layout.edge_crossings(), 0);
    }
}
```

### 8.2 Visual Regression Tests

```rust
#[test]
fn test_graph_widget_render_snapshot() {
    let graph = create_test_graph(5, 6);
    let widget = GraphWidget::new(&graph)
        .layout(LayoutAlgorithm::Grid)
        .build();

    let mut buffer = Buffer::empty(Rect::new(0, 0, 40, 20));
    widget.render(buffer.area, &mut buffer);

    insta::assert_snapshot!(buffer_to_string(&buffer));
}
```

---

## 9. Academic References

1. **Fruchterman, T. M. J., & Reingold, E. M.** (1991). "Graph drawing by force-directed placement." *Software: Practice and Experience*, 21(11), 1129-1164. DOI: [10.1002/spe.4380211102](https://doi.org/10.1002/spe.4380211102)

2. **Kamada, T., & Kawai, S.** (1989). "An algorithm for drawing general undirected graphs." *Information Processing Letters*, 31(1), 7-15. DOI: [10.1016/0020-0190(89)90102-6](https://doi.org/10.1016/0020-0190(89)90102-6)

3. **Walshaw, C.** (2003). "A multilevel algorithm for force-directed graph-drawing." *Journal of Graph Algorithms and Applications*, 7(3), 253-285. DOI: [10.7155/jgaa.00070](https://doi.org/10.7155/jgaa.00070)

4. **Sugiyama, K., Tagawa, S., & Toda, M.** (1981). "Methods for visual understanding of hierarchical system structures." *IEEE Transactions on Systems, Man, and Cybernetics*, 11(2), 109-125. DOI: [10.1109/TSMC.1981.4308636](https://doi.org/10.1109/TSMC.1981.4308636)

5. **Herman, I., Melançon, G., & Marshall, M. S.** (2000). "Graph visualization and navigation in information visualization: A survey." *IEEE Transactions on Visualization and Computer Graphics*, 6(1), 24-43. DOI: [10.1109/2945.841119](https://doi.org/10.1109/2945.841119)

6. **Healey, C. G., Booth, K. S., & Enns, J. T.** (1996). "High-speed visual estimation using preattentive processing." *ACM Transactions on Computer-Human Interaction*, 3(2), 107-135. DOI: [10.1145/230562.230563](https://doi.org/10.1145/230562.230563)

7. **Sweller, J.** (1988). "Cognitive load during problem solving: Effects on learning." *Cognitive Science*, 12(2), 257-285. DOI: [10.1016/0364-0213(88)90023-7](https://doi.org/10.1016/0364-0213(88)90023-7)

8. **von Landesberger, T., et al.** (2011). "Visual analysis of large graphs: State-of-the-art and future research challenges." *Computer Graphics Forum*, 30(6), 1719-1749. DOI: [10.1111/j.1467-8659.2011.01898.x](https://doi.org/10.1111/j.1467-8659.2011.01898.x)

9. **Morland, D. V.** (1983). "Human factors guidelines for terminal interface design." *Communications of the ACM*, 26(4), 484-494. DOI: [10.1145/358150.358156](https://doi.org/10.1145/358150.358156)

10. **Kobourov, S. G.** (2012). "Force-directed drawing algorithms." *Handbook of Graph Drawing and Visualization*, Chapter 12, CRC Press. Available: [Brown University](https://cs.brown.edu/people/rtamassi/gdhandbook/chapters/force-directed.pdf)

---

## 10. Toyota Way Alignment Summary

| Principle | Implementation |
|-----------|----------------|
| **Genchi Genbutsu** | Graph data stays at source (trueno-graph); no duplication |
| **Jidoka** | Built-in quality via layout convergence tests |
| **Just-in-Time** | Lazy layout computation; no precomputation |
| **Heijunka** | Single layout engine, multiple output backends |
| **Poka-Yoke** | Type-safe API prevents invalid configurations |
| **Kaizen** | Phased algorithm rollout; incremental improvement |
| **Mieruka** | Semantic colors for instant status recognition |
| **Standardized Work** | Consistent builder pattern across all widgets |

---

## 11. Implementation Roadmap

### Phase 1: Foundation (2 weeks)
- [ ] Add `tui` feature to trueno-viz
- [ ] Implement `GraphWidget` with Grid layout
- [ ] Basic ratatui integration

### Phase 2: Force-Directed (2 weeks)
- [ ] Fruchterman-Reingold implementation
- [ ] SIMD acceleration (AVX2/NEON)
- [ ] Interactive pan/zoom

### Phase 3: Hierarchical (2 weeks)
- [ ] Sugiyama algorithm for DAGs
- [ ] Radial layout for trees
- [ ] Layer-based positioning

### Phase 4: Integration (1 week)
- [ ] batuta stack publish-status visualization
- [ ] Pipeline DAG visualization
- [ ] Documentation and examples

---

## Appendix A: Feature Flag Configuration

```toml
# trueno-viz/Cargo.toml

[features]
default = []
tui = ["dep:ratatui", "dep:crossterm"]
graph = ["dep:trueno-graph"]
graph-tui = ["tui", "graph"]  # Combined feature for graph TUI

[dependencies]
ratatui = { version = "0.29", optional = true }
crossterm = { version = "0.28", optional = true }
trueno-graph = { version = "0.1.3", optional = true }
```

---

## Appendix B: Minimum Terminal Requirements

| Requirement | Value |
|-------------|-------|
| Minimum width | 80 columns |
| Minimum height | 24 rows |
| Color support | 256 colors (recommended: true color) |
| Unicode support | Required (UTF-8) |
| Mouse support | Optional (enhances interaction) |

---

## 12. Peer Review Annotations

The following annotations have been added during the architectural review cycle (2025-12-08) to ensure strict adherence to Toyota Way principles.

1.  **[Genchi Genbutsu] Section 1.1**: Confirmed via direct codebase inspection that `trueno-viz` currently lacks `tui` feature flags. The gap analysis is accurate; manual bridging in `batuta` would be *Muri* (overburden).
2.  **[Heijunka] Section 2.2**: Sharing the layout engine between WASM and TUI is a critical *Heijunka* implementation. It levels the engineering load, preventing "crunch" when adding new layout algorithms.
3.  **[Muda/Muri] Section 3.1**: While Fruchterman-Reingold is standard, its O(n²) complexity is potential *Muda* (waste). We must enforce a hard limit (e.g., n=500) in the TUI renderer to avoid *Muri* on the main thread.
4.  **[Jidoka] Section 3.3**: The AVX2 optimization is good, but we must ensure a software fallback exists for non-x86 architectures (e.g., Apple Silicon) to prevent build breaks (stopping the line).
5.  **[Standardized Work] Section 4.2**: The Unicode character set is well-chosen, but we should standardize on a fallback ASCII set for legacy terminals to ensure the visualization remains functional everywhere.
6.  **[Respect for People] Section 4.3**: Reliance on Red/Green hue for status violates accessibility standards for color-blind users. **Action**: Add a secondary visual cue (e.g., shape or icon) to status indicators.
7.  **[Poka-Yoke] Section 5.1**: The `GraphWidget` builder should ideally consume the `graph` reference in `new()` (as shown) rather than a builder method to prevent constructing an empty widget.
8.  **[Just-in-Time] Section 6.1**: The <16ms update target implies a 60FPS refresh, which is often limited by the TTY itself. We should synchronize render loops with the terminal's actual capability to avoid wasted cycles.
9.  **[Mieruka] Section 6.3**: "Limit visible nodes" is a key *Mieruka* strategy. The default should be aggressive (e.g., top 20 nodes by centrality) to ensure the most important information stands out immediately.
10. **[Kaizen] Section 8.2**: Visual regression tests are high-value. Integrate `insta` with CI artifacts so regressions are immediately visible to the team.
