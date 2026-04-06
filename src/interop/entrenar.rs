//! Entrenar inference monitoring integration.
//!
//! Provides visualization extensions for decision paths and audit trails.
//!
//! # Features
//!
//! - **Feature Contribution Charts**: Bar charts showing feature importance
//! - **Decision Tree Visualization**: Tree path rendering as graphs
//! - **Hash Chain Provenance**: Timeline visualization of audit entries
//! - **Confidence Gauges**: Visual confidence indicators
//!
//! # Examples
//!
//! ```rust,ignore
//! use entrenar::monitor::inference::path::LinearPath;
//! use trueno_viz::interop::entrenar::DecisionPathViz;
//!
//! let path = LinearPath::new(vec![0.3, -0.2, 0.5], 0.1, 0.6, 0.75);
//! let fb = path.to_contribution_chart(&["age", "income", "score"])?;
//! ```

use batuta_common::display::WithDimensions;
use entrenar::monitor::inference::path::{
    DecisionPath, ForestPath, KNNPath, LinearPath, NeuralPath, TreePath, TreeSplit,
};
use entrenar::monitor::inference::{HashChainCollector, RingCollector};
use serde::Serialize;

use crate::color::Rgba;
use crate::error::{Error, Result};
use crate::framebuffer::Framebuffer;
use crate::plots::{
    ForceGraph, GraphEdge, GraphNode, Heatmap, HeatmapPalette, Histogram, LineChart, LineSeries,
    ScatterPlot,
};
use crate::render::{draw_circle, draw_line, draw_rect};

// =============================================================================
// Color Constants for Explainability
// =============================================================================

/// Positive contribution color (green)
const POSITIVE_COLOR: Rgba = Rgba::new(76, 175, 80, 255);
/// Negative contribution color (red)
const NEGATIVE_COLOR: Rgba = Rgba::new(244, 67, 54, 255);
/// Neutral color (gray)
const NEUTRAL_COLOR: Rgba = Rgba::new(158, 158, 158, 255);
/// High confidence color (blue)
const HIGH_CONFIDENCE_COLOR: Rgba = Rgba::new(33, 150, 243, 255);
/// Low confidence color (orange)
const LOW_CONFIDENCE_COLOR: Rgba = Rgba::new(255, 152, 0, 255);
/// Tree node color
const TREE_NODE_COLOR: Rgba = Rgba::new(103, 58, 183, 255);
/// Leaf node color
const LEAF_NODE_COLOR: Rgba = Rgba::new(0, 150, 136, 255);

// =============================================================================
// DecisionPath Visualization Trait
// =============================================================================

/// Visualization extensions for decision paths.
pub trait DecisionPathViz {
    /// Create a horizontal bar chart of feature contributions.
    ///
    /// Positive contributions are shown in green, negative in red.
    fn to_contribution_chart(&self, feature_names: &[&str]) -> Result<Framebuffer>;

    /// Create a contribution chart with custom dimensions.
    fn to_contribution_chart_with(
        &self,
        feature_names: &[&str],
        width: u32,
        height: u32,
    ) -> Result<Framebuffer>;

    /// Create a confidence gauge visualization.
    ///
    /// Shows confidence as a filled arc from 0% to 100%.
    fn to_confidence_gauge(&self) -> Result<Framebuffer>;

    /// Create a confidence gauge with custom dimensions.
    fn to_confidence_gauge_with(&self, width: u32, height: u32) -> Result<Framebuffer>;
}

impl DecisionPathViz for LinearPath {
    fn to_contribution_chart(&self, feature_names: &[&str]) -> Result<Framebuffer> {
        self.to_contribution_chart_with(feature_names, 600, 400)
    }

    fn to_contribution_chart_with(
        &self,
        feature_names: &[&str],
        width: u32,
        height: u32,
    ) -> Result<Framebuffer> {
        contribution_bar_chart(&self.contributions, feature_names, width, height)
    }

    fn to_confidence_gauge(&self) -> Result<Framebuffer> {
        self.to_confidence_gauge_with(200, 200)
    }

    fn to_confidence_gauge_with(&self, width: u32, height: u32) -> Result<Framebuffer> {
        confidence_gauge(self.confidence(), width, height)
    }
}

impl DecisionPathViz for NeuralPath {
    fn to_contribution_chart(&self, feature_names: &[&str]) -> Result<Framebuffer> {
        self.to_contribution_chart_with(feature_names, 600, 400)
    }

    fn to_contribution_chart_with(
        &self,
        feature_names: &[&str],
        width: u32,
        height: u32,
    ) -> Result<Framebuffer> {
        let contributions = self.feature_contributions();
        contribution_bar_chart(contributions, feature_names, width, height)
    }

    fn to_confidence_gauge(&self) -> Result<Framebuffer> {
        self.to_confidence_gauge_with(200, 200)
    }

    fn to_confidence_gauge_with(&self, width: u32, height: u32) -> Result<Framebuffer> {
        confidence_gauge(self.confidence(), width, height)
    }
}

impl DecisionPathViz for ForestPath {
    fn to_contribution_chart(&self, feature_names: &[&str]) -> Result<Framebuffer> {
        self.to_contribution_chart_with(feature_names, 600, 400)
    }

    fn to_contribution_chart_with(
        &self,
        feature_names: &[&str],
        width: u32,
        height: u32,
    ) -> Result<Framebuffer> {
        contribution_bar_chart(&self.feature_importance, feature_names, width, height)
    }

    fn to_confidence_gauge(&self) -> Result<Framebuffer> {
        self.to_confidence_gauge_with(200, 200)
    }

    fn to_confidence_gauge_with(&self, width: u32, height: u32) -> Result<Framebuffer> {
        confidence_gauge(self.confidence(), width, height)
    }
}

// =============================================================================
// Tree Path Visualization
// =============================================================================

/// Visualization extensions for tree-based decision paths.
pub trait TreePathViz {
    /// Render the decision path as a tree graph.
    fn to_tree_graph(&self) -> Result<Framebuffer>;

    /// Render with custom dimensions.
    fn to_tree_graph_with(&self, width: u32, height: u32) -> Result<Framebuffer>;

    /// Create a waterfall chart showing cumulative decision impact.
    fn to_waterfall_chart(&self, feature_names: &[&str]) -> Result<Framebuffer>;
}

impl TreePathViz for TreePath {
    fn to_tree_graph(&self) -> Result<Framebuffer> {
        self.to_tree_graph_with(600, 400)
    }

    fn to_tree_graph_with(&self, width: u32, height: u32) -> Result<Framebuffer> {
        tree_path_to_graph(&self.splits, &self.leaf, width, height)
    }

    fn to_waterfall_chart(&self, _feature_names: &[&str]) -> Result<Framebuffer> {
        let contributions = self.feature_contributions();
        waterfall_chart(contributions, 600, 400)
    }
}

// =============================================================================
// Forest Path Visualization
// =============================================================================

/// Visualization extensions for ensemble decision paths.
pub trait ForestPathViz {
    /// Create a histogram of tree predictions.
    fn to_prediction_histogram(&self) -> Result<Framebuffer>;

    /// Create a scatter plot of tree predictions vs tree index.
    fn to_tree_scatter(&self) -> Result<Framebuffer>;

    /// Visualize tree agreement as a bar chart.
    fn to_agreement_chart(&self) -> Result<Framebuffer>;
}

impl ForestPathViz for ForestPath {
    fn to_prediction_histogram(&self) -> Result<Framebuffer> {
        if self.tree_predictions.is_empty() {
            return Err(Error::EmptyData);
        }

        let plot = Histogram::new()
            .data(&self.tree_predictions)
            .color(TREE_NODE_COLOR)
            .dimensions(600, 400)
            .build()?;

        plot.to_framebuffer()
    }

    fn to_tree_scatter(&self) -> Result<Framebuffer> {
        if self.tree_predictions.is_empty() {
            return Err(Error::EmptyData);
        }

        let x: Vec<f32> = (0..self.tree_predictions.len()).map(|i| i as f32).collect();

        let plot = ScatterPlot::new()
            .x(&x)
            .y(&self.tree_predictions)
            .color(TREE_NODE_COLOR)
            .size(6.0)
            .dimensions(600, 400)
            .build()?;

        plot.to_framebuffer()
    }

    fn to_agreement_chart(&self) -> Result<Framebuffer> {
        // Create a simple bar showing agreement level
        let mut fb = Framebuffer::new(300, 100)?;
        fb.clear(Rgba::WHITE);

        let margin = 20;
        let bar_height = 30;
        let bar_width = 300 - 2 * margin;

        // Background bar
        draw_rect(&mut fb, margin as i32, 35, bar_width, bar_height, NEUTRAL_COLOR);

        // Filled portion based on agreement
        let filled_width = (bar_width as f32 * self.tree_agreement) as u32;
        let color = if self.tree_agreement >= 0.8 {
            HIGH_CONFIDENCE_COLOR
        } else if self.tree_agreement >= 0.5 {
            Rgba::new(255, 193, 7, 255) // Yellow
        } else {
            LOW_CONFIDENCE_COLOR
        };

        draw_rect(&mut fb, margin as i32, 35, filled_width, bar_height, color);

        Ok(fb)
    }
}

// =============================================================================
// KNN Path Visualization
// =============================================================================

/// Visualization extensions for KNN decision paths.
pub trait KNNPathViz {
    /// Create a distance-based scatter showing neighbors.
    fn to_neighbor_scatter(&self) -> Result<Framebuffer>;

    /// Create a vote distribution bar chart.
    fn to_vote_chart(&self) -> Result<Framebuffer>;
}

impl KNNPathViz for KNNPath {
    fn to_neighbor_scatter(&self) -> Result<Framebuffer> {
        if self.distances.is_empty() {
            return Err(Error::EmptyData);
        }

        // X-axis: neighbor rank (1, 2, 3, ...)
        let x: Vec<f32> = (1..=self.distances.len()).map(|i| i as f32).collect();

        // Color by label
        let mut fb = Framebuffer::new(600, 400)?;
        fb.clear(Rgba::WHITE);

        // Draw points with colors based on labels
        let margin = 50;
        let plot_width = 600 - 2 * margin;
        let plot_height = 400 - 2 * margin;

        let max_dist = self.distances.iter().copied().fold(0.0f32, f32::max).max(0.001);
        let max_x = self.distances.len() as f32;

        for (i, (&dist, &label)) in self.distances.iter().zip(&self.neighbor_labels).enumerate() {
            let px = margin as f32 + (x[i] / max_x) * plot_width as f32;
            let py = (400 - margin) as f32 - (dist / max_dist) * plot_height as f32;

            // Color based on label (cycle through palette)
            let color = label_to_color(label);
            draw_circle(&mut fb, px as i32, py as i32, 6, color);
        }

        Ok(fb)
    }

    fn to_vote_chart(&self) -> Result<Framebuffer> {
        if self.votes.is_empty() {
            return Err(Error::EmptyData);
        }

        let mut fb = Framebuffer::new(400, 300)?;
        fb.clear(Rgba::WHITE);

        let margin = 40;
        let bar_width = 40;
        let max_vote = self.votes.iter().map(|(_, c)| *c).max().unwrap_or(1);

        let spacing = if self.votes.len() > 1 {
            (400 - 2 * margin - bar_width as u32 * self.votes.len() as u32)
                / (self.votes.len() as u32 - 1).max(1)
        } else {
            0
        };

        for (i, (class, count)) in self.votes.iter().enumerate() {
            let x = margin + i as u32 * (bar_width as u32 + spacing);
            let bar_height = (*count as f32 / max_vote as f32 * 200.0) as u32;
            let y = 300 - margin - bar_height;

            let color = label_to_color(*class);
            draw_rect(&mut fb, x as i32, y as i32, bar_width as u32, bar_height, color);
        }

        Ok(fb)
    }
}

// =============================================================================
// Hash Chain Audit Trail Visualization
// =============================================================================

/// Visualization for hash chain audit trails.
pub trait HashChainViz<P: DecisionPath + Serialize> {
    /// Create a timeline visualization of audit entries.
    fn to_timeline(&self) -> Result<Framebuffer>;

    /// Create a confidence trend line over entries.
    fn to_confidence_trend(&self) -> Result<Framebuffer>;

    /// Create a provenance chain graph.
    fn to_chain_graph(&self) -> Result<Framebuffer>;
}

impl<P: DecisionPath + Serialize> HashChainViz<P> for HashChainCollector<P> {
    fn to_timeline(&self) -> Result<Framebuffer> {
        let entries = self.entries();
        if entries.is_empty() {
            return Err(Error::EmptyData);
        }

        let mut fb = Framebuffer::new(800, 200)?;
        fb.clear(Rgba::WHITE);

        let margin = 40;
        let timeline_y = 100;
        let n = entries.len();

        // Draw timeline line
        draw_line(&mut fb, margin, timeline_y, 800 - margin, timeline_y, NEUTRAL_COLOR);

        // Draw entry points
        for (i, entry) in entries.iter().enumerate() {
            let x = margin as f32 + (i as f32 / (n - 1).max(1) as f32) * (800 - 2 * margin) as f32;

            // Color based on verification
            let color = if entry.prev_hash == [0u8; 32] || i == 0 {
                HIGH_CONFIDENCE_COLOR // Genesis or first entry
            } else {
                POSITIVE_COLOR // Valid chain link
            };

            draw_circle(&mut fb, x as i32, timeline_y, 8, color);

            // Draw hash prefix indicator
            let hash_byte = entry.hash[0];
            let indicator_height = (f32::from(hash_byte) / 255.0 * 40.0) as i32;
            draw_line(
                &mut fb,
                x as i32,
                timeline_y + 15,
                x as i32,
                timeline_y + 15 + indicator_height,
                Rgba::new(hash_byte, 100, 200 - hash_byte, 180),
            );
        }

        Ok(fb)
    }

    fn to_confidence_trend(&self) -> Result<Framebuffer> {
        let entries = self.entries();
        if entries.is_empty() {
            return Err(Error::EmptyData);
        }

        let x: Vec<f32> = (0..entries.len()).map(|i| i as f32).collect();
        let y: Vec<f32> = entries.iter().map(|e| e.trace.path.confidence()).collect();

        let plot = LineChart::new()
            .add_series(LineSeries::new("confidence").data(&x, &y).color(HIGH_CONFIDENCE_COLOR))
            .dimensions(600, 300)
            .build()?;

        plot.to_framebuffer()
    }

    fn to_chain_graph(&self) -> Result<Framebuffer> {
        let entries = self.entries();
        if entries.is_empty() {
            return Err(Error::EmptyData);
        }

        // Limit to reasonable number for visualization
        let max_nodes = 20;
        let n = entries.len().min(max_nodes);

        let mut graph = ForceGraph::new().dimensions(600, 400).iterations(80);

        // Add nodes
        for i in 0..n {
            let entry = &entries[entries.len() - n + i];
            let confidence = entry.trace.path.confidence();

            // Color based on confidence
            let color = confidence_to_color(confidence);

            graph = graph.add_node(GraphNode::new(i).color(color).radius(8.0 + confidence * 4.0));
        }

        // Add edges (chain links)
        for i in 1..n {
            graph = graph.add_edge(GraphEdge::new(i - 1, i).weight(2.0));
        }

        let built = graph.build()?;
        built.to_framebuffer()
    }
}

// =============================================================================
// Ring Collector Visualization
// =============================================================================

/// Visualization for ring buffer collectors.
pub trait RingCollectorViz<P: DecisionPath, const N: usize> {
    /// Create an output trend line.
    fn to_output_trend(&self) -> Result<Framebuffer>;

    /// Create a confidence heatmap over recent predictions.
    fn to_confidence_heatmap(&self) -> Result<Framebuffer>;
}

impl<P: DecisionPath, const N: usize> RingCollectorViz<P, N> for RingCollector<P, N> {
    fn to_output_trend(&self) -> Result<Framebuffer> {
        let traces = self.all();
        if traces.is_empty() {
            return Err(Error::EmptyData);
        }

        let x: Vec<f32> = (0..traces.len()).map(|i| i as f32).collect();
        let y: Vec<f32> = traces.iter().map(|t| t.output).collect();

        let plot = LineChart::new()
            .add_series(LineSeries::new("output").data(&x, &y).color(TREE_NODE_COLOR))
            .dimensions(600, 300)
            .build()?;

        plot.to_framebuffer()
    }

    fn to_confidence_heatmap(&self) -> Result<Framebuffer> {
        let traces = self.all();
        if traces.is_empty() {
            return Err(Error::EmptyData);
        }

        // Create a 1xN heatmap of confidences
        let confidences: Vec<f32> = traces.iter().map(|t| t.path.confidence()).collect();
        let n = confidences.len();

        let plot = Heatmap::new()
            .data(&confidences, 1, n)
            .palette(HeatmapPalette::Viridis)
            .dimensions(600, 100)
            .build()?;

        plot.to_framebuffer()
    }
}

// =============================================================================
// Helper Functions
// =============================================================================

/// Create a horizontal bar chart for feature contributions.
fn contribution_bar_chart(
    contributions: &[f32],
    _feature_names: &[&str],
    width: u32,
    height: u32,
) -> Result<Framebuffer> {
    if contributions.is_empty() {
        return Err(Error::EmptyData);
    }

    let mut fb = Framebuffer::new(width, height)?;
    fb.clear(Rgba::WHITE);

    let n = contributions.len();
    let margin = 60;
    let bar_height = ((height - 2 * margin) / n as u32).min(30);
    let spacing = 5;

    let max_abs = contributions.iter().map(|c| c.abs()).fold(0.0f32, f32::max).max(0.001);

    let center_x = width / 2;
    let bar_max_width = (width / 2 - margin) as f32;

    for (i, &contrib) in contributions.iter().enumerate() {
        let y = margin + i as u32 * (bar_height + spacing);
        let bar_width = (contrib.abs() / max_abs * bar_max_width) as u32;

        let color = if contrib >= 0.0 { POSITIVE_COLOR } else { NEGATIVE_COLOR };

        if contrib >= 0.0 {
            draw_rect(&mut fb, center_x as i32, y as i32, bar_width, bar_height, color);
        } else {
            draw_rect(
                &mut fb,
                (center_x - bar_width) as i32,
                y as i32,
                bar_width,
                bar_height,
                color,
            );
        }

        // Draw center line
        draw_line(
            &mut fb,
            center_x as i32,
            margin as i32,
            center_x as i32,
            (height - margin) as i32,
            NEUTRAL_COLOR,
        );
    }

    Ok(fb)
}

/// Create a waterfall chart showing cumulative impact.
fn waterfall_chart(contributions: &[f32], width: u32, height: u32) -> Result<Framebuffer> {
    if contributions.is_empty() {
        return Err(Error::EmptyData);
    }

    let mut fb = Framebuffer::new(width, height)?;
    fb.clear(Rgba::WHITE);

    let n = contributions.len();
    let margin = 50;
    let bar_width = ((width - 2 * margin) / (n + 1) as u32).min(40);
    let spacing = 10;

    // Calculate cumulative values
    let mut cumulative = vec![0.0f32; n + 1];
    for (i, &c) in contributions.iter().enumerate() {
        cumulative[i + 1] = cumulative[i] + c;
    }

    let min_val = cumulative.iter().copied().fold(f32::INFINITY, f32::min);
    let max_val = cumulative.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let range = (max_val - min_val).max(0.001);

    let plot_height = (height - 2 * margin) as f32;
    let baseline_y = height - margin;

    for i in 0..n {
        let x = margin + i as u32 * (bar_width + spacing);

        let start_val = cumulative[i];
        let end_val = cumulative[i + 1];

        let start_y = baseline_y as f32 - ((start_val - min_val) / range * plot_height);
        let end_y = baseline_y as f32 - ((end_val - min_val) / range * plot_height);

        let (top_y, bar_h) =
            if end_y < start_y { (end_y, start_y - end_y) } else { (start_y, end_y - start_y) };

        let color = if contributions[i] >= 0.0 { POSITIVE_COLOR } else { NEGATIVE_COLOR };

        draw_rect(&mut fb, x as i32, top_y as i32, bar_width, bar_h.max(1.0) as u32, color);

        // Connect bars
        if i > 0 {
            let prev_x = margin + (i - 1) as u32 * (bar_width + spacing) + bar_width;
            let prev_y = baseline_y as f32 - ((cumulative[i] - min_val) / range * plot_height);
            draw_line(
                &mut fb,
                prev_x as i32,
                prev_y as i32,
                x as i32,
                start_y as i32,
                NEUTRAL_COLOR,
            );
        }
    }

    Ok(fb)
}

/// Create a confidence gauge visualization.
fn confidence_gauge(confidence: f32, width: u32, height: u32) -> Result<Framebuffer> {
    let mut fb = Framebuffer::new(width, height)?;
    fb.clear(Rgba::WHITE);

    let cx = (width / 2) as i32;
    let cy = (height / 2) as i32;
    let radius = (width.min(height) / 2 - 20) as i32;

    // Draw background arc (gray)
    draw_circle(&mut fb, cx, cy, radius, NEUTRAL_COLOR);
    draw_circle(&mut fb, cx, cy, radius - 10, Rgba::WHITE);

    // Draw filled portion based on confidence
    let color = confidence_to_color(confidence);

    // Approximate arc by drawing segments
    let segments = (confidence * 32.0) as i32;
    for i in 0..segments {
        let angle = std::f32::consts::PI * (1.0 - i as f32 / 32.0);
        let x = cx + (angle.cos() * (radius - 5) as f32) as i32;
        let y = cy - (angle.sin() * (radius - 5) as f32) as i32;
        draw_circle(&mut fb, x, y, 4, color);
    }

    // Draw center value indicator
    draw_circle(&mut fb, cx, cy, 8, color);

    Ok(fb)
}

/// Render tree path splits as a graph.
fn tree_path_to_graph(
    splits: &[TreeSplit],
    leaf: &entrenar::monitor::inference::path::LeafInfo,
    width: u32,
    height: u32,
) -> Result<Framebuffer> {
    if splits.is_empty() {
        // Just show leaf node
        let mut fb = Framebuffer::new(width, height)?;
        fb.clear(Rgba::WHITE);
        let cx = (width / 2) as i32;
        let cy = (height / 2) as i32;
        draw_circle(&mut fb, cx, cy, 20, LEAF_NODE_COLOR);
        return Ok(fb);
    }

    let mut graph = ForceGraph::new().dimensions(width, height).iterations(60).attraction(0.03);

    // Add split nodes
    for (i, _split) in splits.iter().enumerate() {
        graph = graph.add_node(GraphNode::new(i).color(TREE_NODE_COLOR).radius(12.0));
    }

    // Add leaf node
    let leaf_idx = splits.len();
    let leaf_radius = 10.0 + (leaf.n_samples as f32).log10() * 2.0;
    graph = graph.add_node(GraphNode::new(leaf_idx).color(LEAF_NODE_COLOR).radius(leaf_radius));

    // Add edges
    for i in 0..splits.len() {
        let target = if i == splits.len() - 1 { leaf_idx } else { i + 1 };
        let edge_color = if splits[i].went_left { POSITIVE_COLOR } else { NEGATIVE_COLOR };
        graph = graph.add_edge(GraphEdge::new(i, target).color(edge_color).weight(1.5));
    }

    let built = graph.build()?;
    built.to_framebuffer()
}

/// Map confidence to color gradient.
fn confidence_to_color(confidence: f32) -> Rgba {
    let c = confidence.clamp(0.0, 1.0);

    if c >= 0.8 {
        HIGH_CONFIDENCE_COLOR
    } else if c >= 0.5 {
        // Interpolate yellow to blue
        let t = (c - 0.5) / 0.3;
        Rgba::new(
            (255.0 * (1.0 - t) + 33.0 * t) as u8,
            (193.0 * (1.0 - t) + 150.0 * t) as u8,
            (7.0 * (1.0 - t) + 243.0 * t) as u8,
            255,
        )
    } else {
        // Interpolate orange to yellow
        let t = c / 0.5;
        Rgba::new(
            255,
            (152.0 * (1.0 - t) + 193.0 * t) as u8,
            (0.0 * (1.0 - t) + 7.0 * t) as u8,
            255,
        )
    }
}

/// Map label to color (cycling through palette).
fn label_to_color(label: usize) -> Rgba {
    const PALETTE: [Rgba; 8] = [
        Rgba::new(66, 133, 244, 255), // Blue
        Rgba::new(234, 67, 53, 255),  // Red
        Rgba::new(251, 188, 4, 255),  // Yellow
        Rgba::new(52, 168, 83, 255),  // Green
        Rgba::new(103, 58, 183, 255), // Purple
        Rgba::new(0, 150, 136, 255),  // Teal
        Rgba::new(255, 87, 34, 255),  // Deep Orange
        Rgba::new(121, 85, 72, 255),  // Brown
    ];

    PALETTE[label % PALETTE.len()]
}

// =============================================================================
// Convenience Functions
// =============================================================================

/// Create a feature contribution chart from a decision path.
pub fn feature_contributions<P: DecisionPath>(
    path: &P,
    feature_names: &[&str],
) -> Result<Framebuffer> {
    contribution_bar_chart(path.feature_contributions(), feature_names, 600, 400)
}

/// Create a confidence gauge from a decision path.
pub fn confidence_indicator<P: DecisionPath>(path: &P) -> Result<Framebuffer> {
    confidence_gauge(path.confidence(), 200, 200)
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;
    use entrenar::monitor::inference::path::LeafInfo;

    #[test]
    fn test_linear_path_contribution_chart() {
        let path = LinearPath::new(vec![0.3, -0.2, 0.5, -0.1], 0.1, 0.6, 0.75);
        let fb = path
            .to_contribution_chart(&["age", "income", "score", "tenure"])
            .expect("operation should succeed");
        assert_eq!(fb.width(), 600);
        assert_eq!(fb.height(), 400);
    }

    #[test]
    fn test_linear_path_confidence_gauge() {
        let path = LinearPath::new(vec![0.3], 0.0, 0.5, 0.7).with_probability(0.85);
        let fb = path.to_confidence_gauge().expect("operation should succeed");
        assert_eq!(fb.width(), 200);
        assert_eq!(fb.height(), 200);
    }

    #[test]
    fn test_neural_path_contribution_chart() {
        let path = NeuralPath::new(vec![0.1, -0.3, 0.2], 0.8, 0.9);
        let fb = path.to_contribution_chart(&["x1", "x2", "x3"]).expect("operation should succeed");
        assert!(fb.width() > 0);
    }

    #[test]
    fn test_forest_path_prediction_histogram() {
        let path = ForestPath::new(vec![], vec![0.5, 0.6, 0.55, 0.7, 0.45, 0.65]);
        let fb = path.to_prediction_histogram().expect("operation should succeed");
        assert!(fb.width() > 0);
    }

    #[test]
    fn test_forest_path_tree_scatter() {
        let path = ForestPath::new(vec![], vec![0.5, 0.6, 0.55, 0.7]);
        let fb = path.to_tree_scatter().expect("operation should succeed");
        assert!(fb.width() > 0);
    }

    #[test]
    fn test_forest_path_agreement_chart() {
        let path = ForestPath::new(vec![], vec![0.5, 0.5, 0.5]);
        let fb = path.to_agreement_chart().expect("operation should succeed");
        assert_eq!(fb.width(), 300);
    }

    #[test]
    fn test_tree_path_graph() {
        let splits = vec![
            TreeSplit { feature_idx: 0, threshold: 35.0, went_left: true, n_samples: 100 },
            TreeSplit { feature_idx: 1, threshold: 50000.0, went_left: false, n_samples: 60 },
        ];
        let leaf = LeafInfo { prediction: 0.8, n_samples: 30, class_distribution: None };

        let path = TreePath::new(splits, leaf);
        let fb = path.to_tree_graph().expect("operation should succeed");
        assert!(fb.width() > 0);
    }

    #[test]
    fn test_tree_path_empty_splits() {
        let leaf = LeafInfo { prediction: 0.5, n_samples: 100, class_distribution: None };
        let path = TreePath::new(vec![], leaf);
        let fb = path.to_tree_graph().expect("operation should succeed");
        assert!(fb.width() > 0);
    }

    #[test]
    fn test_knn_neighbor_scatter() {
        let path = KNNPath::new(
            vec![0, 5, 10, 15, 20],
            vec![0.1, 0.2, 0.3, 0.4, 0.5],
            vec![0, 1, 0, 1, 1],
            1.0,
        );
        let fb = path.to_neighbor_scatter().expect("operation should succeed");
        assert!(fb.width() > 0);
    }

    #[test]
    fn test_knn_vote_chart() {
        let path = KNNPath::new(
            vec![0, 1, 2, 3, 4],
            vec![0.1, 0.2, 0.3, 0.4, 0.5],
            vec![0, 0, 1, 1, 1],
            1.0,
        );
        let fb = path.to_vote_chart().expect("operation should succeed");
        assert!(fb.width() > 0);
    }

    #[test]
    fn test_confidence_to_color_bounds() {
        let low = confidence_to_color(0.0);
        let mid = confidence_to_color(0.5);
        let high = confidence_to_color(1.0);

        // Just verify they're different colors
        assert_ne!(low, high);
        assert_ne!(mid, high);
    }

    #[test]
    fn test_label_to_color_cycling() {
        let c0 = label_to_color(0);
        let c1 = label_to_color(1);
        let c8 = label_to_color(8);

        assert_ne!(c0, c1);
        assert_eq!(c0, c8); // Should cycle
    }

    #[test]
    fn test_empty_contributions_error() {
        let result = contribution_bar_chart(&[], &[], 600, 400);
        assert!(result.is_err());
    }

    #[test]
    fn test_waterfall_chart() {
        let contributions = vec![0.2, -0.1, 0.3, -0.05];
        let fb = waterfall_chart(&contributions, 600, 400).expect("operation should succeed");
        assert!(fb.width() > 0);
    }

    #[test]
    fn test_feature_contributions_convenience() {
        let path = LinearPath::new(vec![0.1, 0.2, 0.3], 0.0, 0.6, 0.6);
        let fb = feature_contributions(&path, &["a", "b", "c"]).expect("operation should succeed");
        assert!(fb.width() > 0);
    }

    #[test]
    fn test_confidence_indicator_convenience() {
        let path = LinearPath::new(vec![0.1], 0.0, 0.5, 0.5).with_probability(0.9);
        let fb = confidence_indicator(&path).expect("operation should succeed");
        assert_eq!(fb.width(), 200);
    }
}

#[cfg(test)]
mod proptests {
    use super::*;
    use proptest::prelude::*;

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(100))]

        #[test]
        fn prop_contribution_chart_any_values(
            contributions in prop::collection::vec(-100.0f32..100.0, 1..20)
        ) {
            let names: Vec<&str> = (0..contributions.len()).map(|_| "x").collect();
            let result = contribution_bar_chart(&contributions, &names, 600, 400);
            prop_assert!(result.is_ok());
        }

        #[test]
        fn prop_confidence_gauge_bounded(confidence in 0.0f32..1.0) {
            let result = confidence_gauge(confidence, 200, 200);
            prop_assert!(result.is_ok());
        }

        #[test]
        fn prop_confidence_color_always_valid(confidence in -1.0f32..2.0) {
            // Just verify it doesn't panic - u8 values are always valid
            let _color = confidence_to_color(confidence);
        }

        #[test]
        fn prop_label_color_never_panics(label in 0usize..1000) {
            let _color = label_to_color(label);
        }

        #[test]
        fn prop_linear_path_viz_works(
            contributions in prop::collection::vec(-10.0f32..10.0, 1..10),
            intercept in -1.0f32..1.0,
            logit in -5.0f32..5.0
        ) {
            let prediction = 1.0 / (1.0 + (-logit).exp());
            let path = LinearPath::new(contributions.clone(), intercept, logit, prediction);

            let names: Vec<&str> = (0..contributions.len()).map(|_| "f").collect();
            let chart = path.to_contribution_chart(&names);
            prop_assert!(chart.is_ok());

            let gauge = path.to_confidence_gauge();
            prop_assert!(gauge.is_ok());
        }

        #[test]
        fn prop_waterfall_chart_any_contributions(
            contributions in prop::collection::vec(-50.0f32..50.0, 1..15)
        ) {
            let result = waterfall_chart(&contributions, 600, 400);
            prop_assert!(result.is_ok());
        }
    }
}
