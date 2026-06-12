//! Ecosystem integrations.
//!
//! Provides native integration with:
//! - trueno-db: Query result visualization
//! - trueno-graph: Graph layout and visualization
//! - aprender: ML model and result visualization
//!
//! NOTE: the `entrenar` interop (inference-path / training-metrics visualization) was
//! removed for APR-MONO self-containment — entrenar (=aprender-train) depends on viz, so
//! viz→entrenar closed a train↔viz cycle. That visualization belongs in the aprender-train
//! crate (which already depends on viz), not the reverse; re-home it there if needed.

#[cfg(feature = "ml")]
#[cfg_attr(docsrs, doc(cfg(feature = "ml")))]
pub mod aprender;

#[cfg(feature = "graph")]
#[cfg_attr(docsrs, doc(cfg(feature = "graph")))]
pub mod trueno_graph;
