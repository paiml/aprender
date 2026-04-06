//! Core types for stack diagnostics
//!
//! This module contains the fundamental data types used throughout
//! the diagnostics system, including health status, component nodes,
//! metrics, and anomaly detection structures.

use crate::stack::quality::{QualityGrade, StackLayer};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

// ============================================================================
// Health Status (Andon System)
// ============================================================================

/// Health status for components (Andon-style visual control)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum HealthStatus {
    /// All systems healthy - normal operation
    Green,
    /// Attention needed - warnings present
    Yellow,
    /// Critical issues - stop-the-line
    Red,
    /// Not yet analyzed
    Unknown,
}

impl HealthStatus {
    /// Create from quality grade
    pub fn from_grade(grade: QualityGrade) -> Self {
        match grade {
            QualityGrade::APlus | QualityGrade::A => Self::Green,
            QualityGrade::AMinus | QualityGrade::BPlus => Self::Yellow,
            _ => Self::Red,
        }
    }

    /// Get display icon for status
    pub fn icon(&self) -> &'static str {
        match self {
            Self::Green => "🟢",
            Self::Yellow => "🟡",
            Self::Red => "🔴",
            Self::Unknown => "⚪",
        }
    }

    /// Get ASCII symbol for terminal without emoji support
    pub fn symbol(&self) -> &'static str {
        match self {
            Self::Green => "●",
            Self::Yellow => "◐",
            Self::Red => "○",
            Self::Unknown => "◌",
        }
    }
}

impl std::fmt::Display for HealthStatus {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.icon())
    }
}

// ============================================================================
// Component Node (Stack Knowledge Graph)
// ============================================================================

/// A component in the stack knowledge graph
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComponentNode {
    /// Component name (e.g., "trueno", "aprender")
    pub name: String,
    /// Semantic version
    pub version: String,
    /// Stack layer classification
    pub layer: StackLayer,
    /// Current health status
    pub health: HealthStatus,
    /// Quality metrics
    pub metrics: ComponentMetrics,
}

impl ComponentNode {
    /// Create a new component node
    pub fn new(name: impl Into<String>, version: impl Into<String>, layer: StackLayer) -> Self {
        Self {
            name: name.into(),
            version: version.into(),
            layer,
            health: HealthStatus::Unknown,
            metrics: ComponentMetrics::default(),
        }
    }

    /// Update health status from metrics
    pub fn update_health(&mut self) {
        self.health = HealthStatus::from_grade(self.metrics.grade);
    }
}

// ============================================================================
// Component Metrics
// ============================================================================

/// Quality and performance metrics for a component
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComponentMetrics {
    /// Demo score (0-100 normalized)
    pub demo_score: f64,
    /// Test coverage percentage
    pub coverage: f64,
    /// Mutation score percentage
    pub mutation_score: f64,
    /// Average cyclomatic complexity
    pub complexity_avg: f64,
    /// SATD (Self-Admitted Technical Debt) count
    pub satd_count: u32,
    /// Dead code percentage
    pub dead_code_pct: f64,
    /// Overall quality grade
    pub grade: QualityGrade,
}

impl Default for ComponentMetrics {
    fn default() -> Self {
        Self {
            demo_score: 0.0,
            coverage: 0.0,
            mutation_score: 0.0,
            complexity_avg: 0.0,
            satd_count: 0,
            dead_code_pct: 0.0,
            grade: QualityGrade::F, // Lowest grade as default
        }
    }
}

impl ComponentMetrics {
    /// Create metrics with demo score
    pub fn with_demo_score(demo_score: f64) -> Self {
        let grade = QualityGrade::from_sqi(demo_score);
        Self { demo_score, grade, ..Default::default() }
    }

    /// Check if metrics meet A- threshold
    pub fn meets_threshold(&self) -> bool {
        self.demo_score >= 85.0
    }
}

// ============================================================================
// Graph Metrics
// ============================================================================

/// Computed graph-level metrics
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct GraphMetrics {
    /// PageRank scores by node
    pub pagerank: HashMap<String, f64>,
    /// Betweenness centrality by node
    pub betweenness: HashMap<String, f64>,
    /// Clustering coefficient by node
    pub clustering: HashMap<String, f64>,
    /// Community assignments (node -> community_id)
    pub communities: HashMap<String, usize>,
    /// Depth from root nodes
    pub depth_map: HashMap<String, u32>,
    /// Total nodes in graph
    pub total_nodes: usize,
    /// Total edges in graph
    pub total_edges: usize,
    /// Graph density (edges / possible edges)
    pub density: f64,
    /// Average degree
    pub avg_degree: f64,
    /// Maximum depth
    pub max_depth: u32,
}

impl GraphMetrics {
    /// Get the most critical components by PageRank
    pub fn top_by_pagerank(&self, n: usize) -> Vec<(&String, f64)> {
        let mut scores: Vec<_> = self.pagerank.iter().map(|(k, v)| (k, *v)).collect();
        scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        scores.into_iter().take(n).collect()
    }

    /// Get bottleneck components (high betweenness)
    pub fn bottlenecks(&self, threshold: f64) -> Vec<&String> {
        self.betweenness.iter().filter(|(_, &v)| v > threshold).map(|(k, _)| k).collect()
    }
}

// ============================================================================
// Health Summary
// ============================================================================

/// Summary of stack health for dashboard
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HealthSummary {
    /// Total components in stack
    pub total_components: usize,
    /// Components with green status
    pub green_count: usize,
    /// Components with yellow status
    pub yellow_count: usize,
    /// Components with red status
    pub red_count: usize,
    /// Components with unknown status
    pub unknown_count: usize,
    /// Average demo score
    pub avg_demo_score: f64,
    /// Average test coverage
    pub avg_coverage: f64,
    /// Overall Andon status
    pub andon_status: AndonStatus,
}

impl HealthSummary {
    /// Check if all components are healthy
    pub fn all_healthy(&self) -> bool {
        self.red_count == 0 && self.yellow_count == 0 && self.green_count == self.total_components
    }

    /// Get percentage of healthy components
    pub fn health_percentage(&self) -> f64 {
        if self.total_components == 0 {
            return 0.0;
        }
        (self.green_count as f64 / self.total_components as f64) * 100.0
    }
}

// ============================================================================
// Andon Status (Overall Stack)
// ============================================================================

/// Andon board status for the entire stack
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum AndonStatus {
    /// All systems green - normal operation
    Green,
    /// Warnings present - attention needed
    Yellow,
    /// Critical issues - stop-the-line
    Red,
    /// Not yet analyzed
    Unknown,
}

impl AndonStatus {
    /// Get display message
    pub fn message(&self) -> &'static str {
        match self {
            Self::Green => "All systems healthy",
            Self::Yellow => "Attention needed",
            Self::Red => "Stop-the-line",
            Self::Unknown => "Analysis pending",
        }
    }
}

impl std::fmt::Display for AndonStatus {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let icon = match self {
            Self::Green => "🟢",
            Self::Yellow => "🟡",
            Self::Red => "🔴",
            Self::Unknown => "⚪",
        };
        write!(f, "{} {}", icon, self.message())
    }
}

// ============================================================================
// Anomaly Detection
// ============================================================================

/// Detected anomaly in the stack
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Anomaly {
    /// Component where anomaly was detected
    pub component: String,
    /// Anomaly score (0-1, higher = more anomalous)
    pub score: f64,
    /// Category of anomaly
    pub category: AnomalyCategory,
    /// Human-readable description
    pub description: String,
    /// Evidence supporting the anomaly
    pub evidence: Vec<String>,
    /// Suggested remediation
    pub recommendation: Option<String>,
}

impl Anomaly {
    /// Create a new anomaly
    pub fn new(
        component: impl Into<String>,
        score: f64,
        category: AnomalyCategory,
        description: impl Into<String>,
    ) -> Self {
        Self {
            component: component.into(),
            score,
            category,
            description: description.into(),
            evidence: Vec::new(),
            recommendation: None,
        }
    }

    /// Add evidence
    pub fn with_evidence(mut self, evidence: impl Into<String>) -> Self {
        self.evidence.push(evidence.into());
        self
    }

    /// Add recommendation
    pub fn with_recommendation(mut self, rec: impl Into<String>) -> Self {
        self.recommendation = Some(rec.into());
        self
    }

    /// Check if anomaly is critical (score > 0.8)
    pub fn is_critical(&self) -> bool {
        self.score > 0.8
    }
}

/// Categories of anomalies
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum AnomalyCategory {
    /// Quality score regression
    QualityRegression,
    /// Coverage drop
    CoverageDrop,
    /// Build time spike
    BuildTimeSpike,
    /// Dependency risk
    DependencyRisk,
    /// Complexity increase
    ComplexityIncrease,
    /// Other anomaly
    Other,
}

impl std::fmt::Display for AnomalyCategory {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::QualityRegression => write!(f, "Quality Regression"),
            Self::CoverageDrop => write!(f, "Coverage Drop"),
            Self::BuildTimeSpike => write!(f, "Build Time Spike"),
            Self::DependencyRisk => write!(f, "Dependency Risk"),
            Self::ComplexityIncrease => write!(f, "Complexity Increase"),
            Self::Other => write!(f, "Other"),
        }
    }
}
