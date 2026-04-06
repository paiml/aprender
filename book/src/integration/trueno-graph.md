# Trueno-Graph Visualization

Trueno-viz integrates with trueno-graph for network and graph visualization.

## Enabling Integration

```toml
[dependencies]
trueno-viz = { version = "0.1", features = ["graph"] }
trueno-graph = "0.1"
```

## Basic Graph Visualization

```rust
use trueno_graph::Graph;
use trueno_viz::interop::graph as viz;

let mut g = Graph::new();
g.add_edge(0, 1);
g.add_edge(1, 2);
g.add_edge(2, 0);

let plot = viz::draw_graph(&g)
    .layout(Layout::ForceDirected)
    .build();

plot.render_to_file("graph.png").unwrap();
```

## Layout Algorithms

### Force-Directed (Fruchterman-Reingold)

```rust
use trueno_viz::interop::graph::{draw_graph, Layout};

let plot = viz::draw_graph(&g)
    .layout(Layout::ForceDirected)
    .iterations(500)
    .build();
```

### Circular

```rust
let plot = viz::draw_graph(&g)
    .layout(Layout::Circular)
    .build();
```

### Hierarchical

```rust
let plot = viz::draw_graph(&g)
    .layout(Layout::Hierarchical)
    .root(0)
    .build();
```

### Spring

```rust
let plot = viz::draw_graph(&g)
    .layout(Layout::Spring)
    .k(0.5)  // Spring constant
    .build();
```

## Node Styling

```rust
use trueno_viz::prelude::*;

let plot = viz::draw_graph(&g)
    .node_color(Rgba::new(66, 133, 244, 255))
    .node_size(20.0)
    .node_border_color(Rgba::BLACK)
    .node_border_width(2.0)
    .build();
```

### Node Attributes

```rust
let sizes: Vec<f32> = g.nodes().map(|n| g.degree(n) as f32 * 5.0).collect();
let colors: Vec<Rgba> = g.nodes().map(|n| color_by_community(n)).collect();

let plot = viz::draw_graph(&g)
    .node_sizes(&sizes)
    .node_colors(&colors)
    .build();
```

## Edge Styling

```rust
let plot = viz::draw_graph(&g)
    .edge_color(Rgba::new(150, 150, 150, 255))
    .edge_width(1.0)
    .curved_edges(true)
    .build();
```

### Directed Edges

```rust
let plot = viz::draw_graph(&g)
    .directed(true)
    .arrow_size(10.0)
    .build();
```

### Edge Weights

```rust
let plot = viz::draw_graph(&g)
    .edge_weights(&weights)
    .edge_width_range(0.5, 5.0)  // Scale width by weight
    .build();
```

## Labels

```rust
let labels: Vec<&str> = vec!["A", "B", "C", "D", "E"];

let plot = viz::draw_graph(&g)
    .node_labels(&labels)
    .label_size(12.0)
    .label_color(Rgba::BLACK)
    .build();
```

## Community Detection Visualization

```rust
use trueno_graph::algorithms::community;

let communities = community::louvain(&g);

let plot = viz::draw_graph(&g)
    .color_by_community(&communities)
    .build();
```

## Large Graph Handling

For graphs with 1000+ nodes:

```rust
let plot = viz::draw_graph(&large_graph)
    .layout(Layout::ForceDirected)
    .sample_edges(0.1)  // Show 10% of edges
    .node_size(3.0)     // Smaller nodes
    .edge_alpha(0.3)    // Transparent edges
    .gpu(true)          // GPU acceleration
    .build();
```

## Complete Example

```rust
use trueno_graph::Graph;
use trueno_viz::prelude::*;
use trueno_viz::interop::graph as viz;

fn main() -> Result<()> {
    // Create social network
    let mut g = Graph::new();

    // Add connections
    let edges = [
        (0, 1), (0, 2), (0, 3),
        (1, 2), (1, 4),
        (2, 3), (2, 4), (2, 5),
        (3, 5),
        (4, 5), (4, 6),
        (5, 6),
    ];

    for (a, b) in edges {
        g.add_edge(a, b);
    }

    // Node labels
    let labels = ["Alice", "Bob", "Carol", "Dave", "Eve", "Frank", "Grace"];

    // Node sizes by degree
    let sizes: Vec<f32> = (0..7)
        .map(|n| 10.0 + g.degree(n) as f32 * 5.0)
        .collect();

    let plot = viz::draw_graph(&g)
        .layout(Layout::ForceDirected)
        .node_labels(&labels)
        .node_sizes(&sizes)
        .node_color(Rgba::new(66, 133, 244, 200))
        .edge_color(Rgba::new(100, 100, 100, 150))
        .title("Social Network")
        .build();

    plot.render_to_file("social_network.png")?;

    Ok(())
}
```

## Next Chapter

Continue to [Trueno-DB Queries](./trueno-db.md) for database integration.
