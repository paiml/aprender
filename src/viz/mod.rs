//! Data Visualization Frameworks Integration
//!
//! Unified interface for visualization and ML demo frameworks:
//! - Gradio (ML demos, HuggingFace Spaces)
//! - Streamlit (Data apps, dashboards)
//! - Panel (HoloViz ecosystem, big data)
//! - Dash (Plotly enterprise visualization)
//!
//! ## Toyota Way Principles
//!
//! - Genchi Genbutsu: Direct visualization enables first-hand observation
//! - Poka-Yoke: Framework selection prevents platform lock-in
//! - Heijunka: Frame-rate limiting prevents GPU saturation
//! - Jidoka: Explicit component trees for predictable rendering
//! - Muda: Signal-based rendering eliminates wasted computation
//! - Kanban: Visual data flow with explicit signal graphs

pub mod dashboard;
pub mod tree;

// Re-export types for library users (used by lib.rs, not by main.rs binary)
#[allow(unused_imports)]
pub use dashboard::{DashboardBuilder, DashboardConfig};
#[allow(unused_imports)]
pub use tree::{
    Framework, FrameworkCategory, FrameworkComponent, IntegrationMapping, IntegrationType, VizTree,
};
