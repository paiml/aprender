//! Media Generation Module (Spec: missing-features-in-pure-rust.md)
//!
//! Provides GIF, PNG, SVG, and video recording capabilities for test documentation.
//!
//! ## Toyota Way Principles
//!
//! - **Poka-Yoke**: Type-safe frame capture prevents format mismatches
//! - **Muda**: Lazy frame encoding reduces memory pressure
//! - **Jidoka**: Fail-fast on invalid configurations

mod gif_recorder;
mod png_exporter;
mod svg_exporter;
mod video_recorder;

pub use gif_recorder::{GifConfig, GifFrame, GifRecorder};
pub use png_exporter::{Annotation, CompressionLevel, PngExporter, PngMetadata};
pub use svg_exporter::{SvgCompression, SvgConfig, SvgExporter, SvgShape};
pub use video_recorder::{EncodedFrame, RecordingState, VideoCodec, VideoConfig, VideoRecorder};
