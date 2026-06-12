//! Visualization extensions for entrenar training & inference monitoring.
//!
//! These modules render `entrenar`-native types (decision paths, audit trails,
//! ring collectors) into `trueno_viz` framebuffers. The `train → viz` dependency
//! direction is correct (training plots via viz); this module was re-homed here
//! from `aprender-viz` (#1975 dropped the reverse `viz → entrenar` edge, which
//! closed a `train ↔ viz` cycle). See APR-MONO §S / #1978.

pub mod inference_path;
