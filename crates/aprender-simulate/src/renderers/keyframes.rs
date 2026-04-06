//! Keyframe export for rmedia SVG animation pipeline.
//!
//! Captures per-frame element positions and properties from render commands,
//! then exports as JSON for consumption by resolve-pipeline's `animate.*`
//! properties on rmedia SVG producers.
//!
//! # Output Format
//!
//! ```json
//! {
//!   "fps": 60,
//!   "duration_frames": 600,
//!   "seed": 42,
//!   "domain": "orbit",
//!   "elements": {
//!     "earth": { "cx": [960.0, 962.1, ...], "cy": [540.0, 538.2, ...] },
//!     "trail": { "d": ["M960,540", "M960,540 L962,538", ...] }
//!   }
//! }
//! ```

use crate::orbit::render::{Camera, Color, RenderCommand};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::fmt::Write;

/// Per-element keyframe data: attribute name → frame values.
pub type ElementKeyframes = BTreeMap<String, Vec<KeyframeValue>>;

/// A single keyframe value (numeric, color, or path string).
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(untagged)]
pub enum KeyframeValue {
    /// Numeric value (position, radius, opacity).
    Number(f64),
    /// String value (SVG path data, text content).
    Text(String),
}

/// Complete keyframes export for a simulation run.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KeyframesExport {
    /// Frames per second.
    pub fps: u32,
    /// Total number of frames captured.
    pub duration_frames: usize,
    /// Simulation seed for reproducibility.
    pub seed: u64,
    /// Simulation domain name.
    pub domain: String,
    /// Per-element keyframe data.
    pub elements: BTreeMap<String, ElementKeyframes>,
}

/// Keyframe recorder that captures element state per frame.
#[derive(Debug, Clone)]
pub struct KeyframeRecorder {
    fps: u32,
    seed: u64,
    domain: String,
    camera: Camera,
    frame_count: usize,
    elements: BTreeMap<String, ElementKeyframes>,
    /// Element naming counters per command type.
    counters: BTreeMap<String, u32>,
}

impl KeyframeRecorder {
    /// Create a new keyframe recorder.
    #[must_use]
    pub fn new(fps: u32, seed: u64, domain: &str) -> Self {
        Self {
            fps,
            seed,
            domain: domain.to_string(),
            camera: Camera {
                width: 1920.0,
                height: 1080.0,
                ..Camera::default()
            },
            frame_count: 0,
            elements: BTreeMap::new(),
            counters: BTreeMap::new(),
        }
    }

    /// Record one frame of render commands.
    ///
    /// Elements are tracked by ID across frames. The first frame establishes
    /// the element set; subsequent frames append values to existing elements.
    pub fn record_frame(&mut self, commands: &[RenderCommand]) {
        self.counters.clear();

        for cmd in commands {
            match cmd {
                RenderCommand::SetCamera {
                    center_x,
                    center_y,
                    zoom,
                } => {
                    self.camera.center_x = *center_x;
                    self.camera.center_y = *center_y;
                    self.camera.zoom = *zoom;
                }
                RenderCommand::DrawCircle {
                    x,
                    y,
                    radius,
                    color,
                    ..
                } => {
                    let id = self.next_id("circle");
                    let (sx, sy) = self.camera.world_to_screen(*x, *y);
                    self.push_value(&id, "cx", KeyframeValue::Number(round2(sx)));
                    self.push_value(&id, "cy", KeyframeValue::Number(round2(sy)));
                    self.push_value(&id, "r", KeyframeValue::Number(round2(*radius)));
                    self.push_value(&id, "fill", KeyframeValue::Text(color_to_hex(*color)));
                }
                RenderCommand::DrawOrbitPath { points, color } => {
                    if points.len() < 2 {
                        continue;
                    }
                    let id = self.next_id("path");
                    let mut d = String::new();
                    for (i, (x, y)) in points.iter().enumerate() {
                        let (sx, sy) = self.camera.world_to_screen(*x, *y);
                        if i == 0 {
                            let _ = write!(d, "M{sx:.1},{sy:.1}");
                        } else {
                            let _ = write!(d, " L{sx:.1},{sy:.1}");
                        }
                    }
                    self.push_value(&id, "d", KeyframeValue::Text(d));
                    self.push_value(&id, "stroke", KeyframeValue::Text(color_to_hex(*color)));
                }
                RenderCommand::DrawText { x, y, text, color } => {
                    let id = self.next_id("text");
                    let (sx, sy) = self.camera.world_to_screen(*x, *y);
                    self.push_value(&id, "x", KeyframeValue::Number(round2(sx)));
                    self.push_value(&id, "y", KeyframeValue::Number(round2(sy)));
                    self.push_value(&id, "text", KeyframeValue::Text(text.clone()));
                    self.push_value(&id, "fill", KeyframeValue::Text(color_to_hex(*color)));
                }
                RenderCommand::DrawVelocity {
                    x,
                    y,
                    vx,
                    vy,
                    scale,
                    ..
                } => {
                    let id = self.next_id("velocity");
                    let (sx, sy) = self.camera.world_to_screen(*x, *y);
                    let ex = sx + vx * scale;
                    let ey = sy + vy * scale;
                    self.push_value(&id, "x1", KeyframeValue::Number(round2(sx)));
                    self.push_value(&id, "y1", KeyframeValue::Number(round2(sy)));
                    self.push_value(&id, "x2", KeyframeValue::Number(round2(ex)));
                    self.push_value(&id, "y2", KeyframeValue::Number(round2(ey)));
                }
                RenderCommand::DrawLine {
                    x1,
                    y1,
                    x2,
                    y2,
                    color,
                } => {
                    let id = self.next_id("line");
                    let (sx1, sy1) = self.camera.world_to_screen(*x1, *y1);
                    let (sx2, sy2) = self.camera.world_to_screen(*x2, *y2);
                    self.push_value(&id, "x1", KeyframeValue::Number(round2(sx1)));
                    self.push_value(&id, "y1", KeyframeValue::Number(round2(sy1)));
                    self.push_value(&id, "x2", KeyframeValue::Number(round2(sx2)));
                    self.push_value(&id, "y2", KeyframeValue::Number(round2(sy2)));
                    self.push_value(&id, "stroke", KeyframeValue::Text(color_to_hex(*color)));
                }
                RenderCommand::Clear { .. } | RenderCommand::HighlightBody { .. } => {}
            }
        }

        self.frame_count += 1;
    }

    /// Export captured keyframes as a serializable struct.
    #[must_use]
    pub fn export(&self) -> KeyframesExport {
        KeyframesExport {
            fps: self.fps,
            duration_frames: self.frame_count,
            seed: self.seed,
            domain: self.domain.clone(),
            elements: self.elements.clone(),
        }
    }

    /// Export keyframes as a JSON string.
    #[must_use]
    pub fn to_json(&self) -> String {
        serde_json::to_string_pretty(&self.export())
            .unwrap_or_else(|e| format!("{{\"error\": \"{e}\"}}"))
    }

    /// Get the number of recorded frames.
    #[must_use]
    pub fn frame_count(&self) -> usize {
        self.frame_count
    }

    /// Get the number of tracked elements.
    #[must_use]
    pub fn element_count(&self) -> usize {
        self.elements.len()
    }

    /// Generate next sequential ID for a command type.
    fn next_id(&mut self, prefix: &str) -> String {
        let counter = self.counters.entry(prefix.to_string()).or_insert(0);
        let id = format!("{prefix}-{counter}");
        *counter += 1;
        id
    }

    /// Push a keyframe value for an element attribute.
    fn push_value(&mut self, element_id: &str, attribute: &str, value: KeyframeValue) {
        self.elements
            .entry(element_id.to_string())
            .or_default()
            .entry(attribute.to_string())
            .or_default()
            .push(value);
    }
}

/// Convert RGBA color to hex string.
fn color_to_hex(color: Color) -> String {
    if color.a == 255 {
        format!("#{:02x}{:02x}{:02x}", color.r, color.g, color.b)
    } else {
        format!(
            "#{:02x}{:02x}{:02x}{:02x}",
            color.r, color.g, color.b, color.a
        )
    }
}

/// Round to 2 decimal places for compact JSON.
fn round2(v: f64) -> f64 {
    (v * 100.0).round() / 100.0
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::orbit::render::Color;

    #[test]
    fn test_recorder_new() {
        let recorder = KeyframeRecorder::new(60, 42, "orbit");
        assert_eq!(recorder.fps, 60);
        assert_eq!(recorder.seed, 42);
        assert_eq!(recorder.domain, "orbit");
        assert_eq!(recorder.frame_count(), 0);
        assert_eq!(recorder.element_count(), 0);
    }

    #[test]
    fn test_record_single_frame() {
        let mut recorder = KeyframeRecorder::new(60, 42, "orbit");
        recorder.record_frame(&[
            RenderCommand::SetCamera {
                center_x: 0.0,
                center_y: 0.0,
                zoom: 1.0,
            },
            RenderCommand::DrawCircle {
                x: 0.0,
                y: 0.0,
                radius: 10.0,
                color: Color::SUN,
                filled: true,
            },
        ]);

        assert_eq!(recorder.frame_count(), 1);
        assert_eq!(recorder.element_count(), 1);
        assert!(recorder.elements.contains_key("circle-0"));
    }

    #[test]
    fn test_record_multiple_frames() {
        let mut recorder = KeyframeRecorder::new(60, 42, "orbit");

        for i in 0..3 {
            recorder.record_frame(&[
                RenderCommand::SetCamera {
                    center_x: 0.0,
                    center_y: 0.0,
                    zoom: 1.0,
                },
                RenderCommand::DrawCircle {
                    x: i as f64 * 10.0,
                    y: 0.0,
                    radius: 5.0,
                    color: Color::EARTH,
                    filled: true,
                },
            ]);
        }

        assert_eq!(recorder.frame_count(), 3);
        let circle = &recorder.elements["circle-0"];
        let cx_values = &circle["cx"];
        assert_eq!(cx_values.len(), 3);
    }

    #[test]
    fn test_record_orbit_path() {
        let mut recorder = KeyframeRecorder::new(60, 42, "orbit");
        recorder.record_frame(&[RenderCommand::DrawOrbitPath {
            points: vec![(0.0, 0.0), (10.0, 10.0), (20.0, 0.0)],
            color: Color::EARTH,
        }]);

        assert!(recorder.elements.contains_key("path-0"));
        let path = &recorder.elements["path-0"];
        assert!(path.contains_key("d"));
        if let KeyframeValue::Text(d) = &path["d"][0] {
            assert!(d.starts_with('M'));
            assert!(d.contains('L'));
        } else {
            panic!("Expected Text value for path d");
        }
    }

    #[test]
    fn test_record_text() {
        let mut recorder = KeyframeRecorder::new(60, 42, "orbit");
        recorder.record_frame(&[RenderCommand::DrawText {
            x: 10.0,
            y: 20.0,
            text: "Jidoka: E=1e-9".to_string(),
            color: Color::GREEN,
        }]);

        assert!(recorder.elements.contains_key("text-0"));
        let text = &recorder.elements["text-0"];
        if let KeyframeValue::Text(t) = &text["text"][0] {
            assert_eq!(t, "Jidoka: E=1e-9");
        } else {
            panic!("Expected Text value");
        }
    }

    #[test]
    fn test_record_velocity() {
        let mut recorder = KeyframeRecorder::new(60, 42, "orbit");
        recorder.record_frame(&[
            RenderCommand::SetCamera {
                center_x: 0.0,
                center_y: 0.0,
                zoom: 1.0,
            },
            RenderCommand::DrawVelocity {
                x: 0.0,
                y: 0.0,
                vx: 50.0,
                vy: 30.0,
                scale: 1.0,
                color: Color::GREEN,
            },
        ]);

        assert!(recorder.elements.contains_key("velocity-0"));
        let vel = &recorder.elements["velocity-0"];
        assert!(vel.contains_key("x1"));
        assert!(vel.contains_key("y1"));
        assert!(vel.contains_key("x2"));
        assert!(vel.contains_key("y2"));
    }

    #[test]
    fn test_export_json() {
        let mut recorder = KeyframeRecorder::new(60, 42, "orbit");
        recorder.record_frame(&[RenderCommand::DrawCircle {
            x: 0.0,
            y: 0.0,
            radius: 5.0,
            color: Color::SUN,
            filled: true,
        }]);

        let json = recorder.to_json();
        assert!(json.contains("\"fps\": 60"));
        assert!(json.contains("\"seed\": 42"));
        assert!(json.contains("\"domain\": \"orbit\""));
        assert!(json.contains("\"duration_frames\": 1"));
        assert!(json.contains("circle-0"));
    }

    #[test]
    fn test_export_struct() {
        let mut recorder = KeyframeRecorder::new(30, 123, "monte_carlo");
        recorder.record_frame(&[RenderCommand::DrawCircle {
            x: 5.0,
            y: 5.0,
            radius: 3.0,
            color: Color::RED,
            filled: true,
        }]);

        let export = recorder.export();
        assert_eq!(export.fps, 30);
        assert_eq!(export.seed, 123);
        assert_eq!(export.domain, "monte_carlo");
        assert_eq!(export.duration_frames, 1);
        assert!(export.elements.contains_key("circle-0"));
    }

    #[test]
    fn test_multiple_elements_per_frame() {
        let mut recorder = KeyframeRecorder::new(60, 42, "orbit");
        recorder.record_frame(&[
            RenderCommand::DrawCircle {
                x: 0.0,
                y: 0.0,
                radius: 15.0,
                color: Color::SUN,
                filled: true,
            },
            RenderCommand::DrawCircle {
                x: 1.0,
                y: 0.0,
                radius: 5.0,
                color: Color::EARTH,
                filled: true,
            },
            RenderCommand::DrawText {
                x: 10.0,
                y: 10.0,
                text: "test".to_string(),
                color: Color::WHITE,
            },
        ]);

        assert_eq!(recorder.element_count(), 3);
        assert!(recorder.elements.contains_key("circle-0"));
        assert!(recorder.elements.contains_key("circle-1"));
        assert!(recorder.elements.contains_key("text-0"));
    }

    #[test]
    fn test_clear_and_highlight_ignored() {
        let mut recorder = KeyframeRecorder::new(60, 42, "orbit");
        recorder.record_frame(&[
            RenderCommand::Clear {
                color: Color::BLACK,
            },
            RenderCommand::HighlightBody {
                x: 0.0,
                y: 0.0,
                radius: 20.0,
                color: Color::RED,
            },
        ]);

        assert_eq!(recorder.element_count(), 0);
    }

    #[test]
    fn test_single_point_path_skipped() {
        let mut recorder = KeyframeRecorder::new(60, 42, "orbit");
        recorder.record_frame(&[RenderCommand::DrawOrbitPath {
            points: vec![(0.0, 0.0)],
            color: Color::EARTH,
        }]);

        assert_eq!(recorder.element_count(), 0);
    }

    #[test]
    fn test_round2() {
        assert!((round2(3.14159) - 3.14).abs() < 0.001);
        assert!((round2(0.0) - 0.0).abs() < f64::EPSILON);
        assert!((round2(-1.555) - (-1.56)).abs() < 0.001);
    }

    #[test]
    fn test_keyframe_value_serialize() {
        let num = KeyframeValue::Number(42.5);
        let json = serde_json::to_string(&num).unwrap();
        assert_eq!(json, "42.5");

        let text = KeyframeValue::Text("hello".to_string());
        let json = serde_json::to_string(&text).unwrap();
        assert_eq!(json, "\"hello\"");
    }
}
