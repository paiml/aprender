//! TUI Brick Traits (PROBAR-SPEC-009-P13)
//!
//! Implements the three-layer TUI brick architecture based on ttop patterns:
//! - `CollectorBrick`: Gathers system/audio metrics
//! - `AnalyzerBrick`: Produces insights from metrics
//! - `PanelBrick`: Renders TUI panels with state machine
//!
//! # Architecture
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────┐
//! │                     PanelBrick Layer                        │
//! │  (focus/explode state, layout, rendering)                   │
//! ├─────────────────────────────────────────────────────────────┤
//! │                    AnalyzerBrick Layer                      │
//! │  (RTF calculation, trend detection, alerts)                 │
//! ├─────────────────────────────────────────────────────────────┤
//! │                   CollectorBrick Layer                      │
//! │  (audio levels, system metrics, buffer stats)               │
//! └─────────────────────────────────────────────────────────────┘
//! ```
//!
//! # Example
//!
//! ```rust,ignore
//! use probar::brick::tui::{CollectorBrick, RingBuffer, PanelState};
//!
//! // Collect audio levels into ring buffer
//! let mut buffer: RingBuffer<f32> = RingBuffer::new(60);
//! buffer.push(0.5);
//! buffer.push(0.7);
//!
//! // Manage panel focus
//! let mut state = PanelState::default();
//! state.focus_next();
//! ```

use super::Brick;
use std::collections::VecDeque;
use std::time::Duration;

// ============================================================================
// Error Types
// ============================================================================

/// Error type for collector operations
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CollectorError {
    /// Feature not available on this platform
    NotAvailable,
    /// Collection failed with message
    Failed(String),
    /// Collector is disabled
    Disabled,
    /// Timeout during collection
    Timeout,
}

impl std::fmt::Display for CollectorError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::NotAvailable => write!(f, "Feature not available on this platform"),
            Self::Failed(msg) => write!(f, "Collection failed: {msg}"),
            Self::Disabled => write!(f, "Collector is disabled"),
            Self::Timeout => write!(f, "Collection timed out"),
        }
    }
}

impl std::error::Error for CollectorError {}

// ============================================================================
// Collector Brick
// ============================================================================

/// Trait for bricks that collect metrics from system or audio sources.
///
/// Collectors are feature-gated and can report availability.
///
/// # Example
///
/// ```rust,ignore
/// struct AudioLevelCollector {
///     sample_rate: u32,
/// }
///
/// impl CollectorBrick for AudioLevelCollector {
///     type Metrics = f32;
///
///     fn is_available(&self) -> bool { true }
///
///     fn collect(&mut self) -> Result<f32, CollectorError> {
///         Ok(0.5) // Return current audio level
///     }
/// }
/// ```
pub trait CollectorBrick: Brick + Send + Sync {
    /// Metrics type produced by this collector
    type Metrics;

    /// Check if collector is available on current platform.
    ///
    /// Returns `false` if the feature requires hardware or OS support
    /// that isn't present.
    fn is_available(&self) -> bool;

    /// Collect metrics.
    ///
    /// # Errors
    ///
    /// Returns `CollectorError` if collection fails.
    fn collect(&mut self) -> Result<Self::Metrics, CollectorError>;

    /// Optional feature gate name for conditional compilation.
    ///
    /// Returns `None` if always available.
    fn feature_gate(&self) -> Option<&'static str> {
        None
    }

    /// Collection interval hint for schedulers.
    fn collection_interval(&self) -> Duration {
        Duration::from_millis(100)
    }
}

// ============================================================================
// Analyzer Brick
// ============================================================================

/// Trait for bricks that analyze collected metrics.
///
/// Analyzers transform raw metrics into insights (trends, alerts, summaries).
///
/// # Example
///
/// ```rust,ignore
/// struct RtfAnalyzer;
///
/// impl AnalyzerBrick for RtfAnalyzer {
///     type Input = (f64, f64); // (audio_duration, processing_time)
///     type Output = RtfResult;
///
///     fn analyze(&self, input: &Self::Input) -> RtfResult {
///         let rtf = input.1 / input.0;
///         RtfResult { rtf, is_realtime: rtf < 1.0 }
///     }
/// }
/// ```
pub trait AnalyzerBrick: Brick + Send + Sync {
    /// Input metrics type
    type Input;
    /// Output analysis type
    type Output;

    /// Analyze metrics and produce insights.
    fn analyze(&self, input: &Self::Input) -> Self::Output;

    /// Check if analysis is stale and needs refresh.
    fn is_stale(&self, _age: Duration) -> bool {
        false
    }
}

// ============================================================================
// Panel Brick
// ============================================================================

/// Trait for bricks that render TUI panels.
///
/// Panels support focus/explode behavior for keyboard navigation.
pub trait PanelBrick: Brick + Send + Sync {
    /// Render panel content to the given area.
    ///
    /// Returns lines of text to display.
    fn render(&self, width: u16, height: u16) -> Vec<String>;

    /// Panel title for border display.
    fn title(&self) -> &str;

    /// Whether this panel can be focused.
    fn focusable(&self) -> bool {
        true
    }

    /// Whether this panel can be exploded to full screen.
    fn explodable(&self) -> bool {
        true
    }

    /// Minimum height required for meaningful display.
    fn min_height(&self) -> u16 {
        3
    }

    /// Preferred height as fraction of available space (0.0-1.0).
    fn preferred_height_fraction(&self) -> f32 {
        0.25
    }
}

// ============================================================================
// Panel State Machine
// ============================================================================

/// Panel type identifier for focus/explode tracking.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PanelId {
    /// Waveform display
    Waveform,
    /// Spectrogram visualization
    Spectrogram,
    /// Transcription text
    Transcription,
    /// Performance metrics
    Metrics,
    /// VU meter
    VuMeter,
    /// Status bar
    Status,
    /// Custom panel with ID
    Custom(u32),
}

/// Panel state for focus/explode behavior.
///
/// Implements the ttop pattern for keyboard-navigable TUI panels.
#[derive(Debug, Clone)]
pub struct PanelState {
    /// Currently focused panel
    pub focused: Option<PanelId>,
    /// Currently exploded (full-screen) panel
    pub exploded: Option<PanelId>,
    /// Visible panels in display order
    pub visible: Vec<PanelId>,
}

impl Default for PanelState {
    fn default() -> Self {
        Self {
            focused: None,
            exploded: None,
            visible: vec![
                PanelId::Waveform,
                PanelId::Transcription,
                PanelId::Metrics,
                PanelId::Status,
            ],
        }
    }
}

impl PanelState {
    /// Create with custom panel list.
    #[must_use]
    pub fn with_panels(panels: Vec<PanelId>) -> Self {
        Self {
            focused: panels.first().copied(),
            exploded: None,
            visible: panels,
        }
    }

    /// Focus next panel in list.
    pub fn focus_next(&mut self) {
        if self.visible.is_empty() {
            self.focused = None;
            return;
        }

        let current_idx = self
            .focused
            .and_then(|f| self.visible.iter().position(|p| *p == f));

        let next_idx = current_idx
            .map(|i| (i + 1) % self.visible.len())
            .unwrap_or(0);

        self.focused = self.visible.get(next_idx).copied();
    }

    /// Focus previous panel in list.
    pub fn focus_prev(&mut self) {
        if self.visible.is_empty() {
            self.focused = None;
            return;
        }

        let current_idx = self
            .focused
            .and_then(|f| self.visible.iter().position(|p| *p == f));

        let prev_idx = current_idx
            .map(|i| {
                if i == 0 {
                    self.visible.len() - 1
                } else {
                    i - 1
                }
            })
            .unwrap_or(0);

        self.focused = self.visible.get(prev_idx).copied();
    }

    /// Toggle exploded state for focused panel.
    pub fn toggle_explode(&mut self) {
        if self.exploded.is_some() {
            self.exploded = None;
        } else {
            self.exploded = self.focused;
        }
    }

    /// Check if a panel is focused.
    #[must_use]
    pub fn is_focused(&self, panel: PanelId) -> bool {
        self.focused == Some(panel)
    }

    /// Check if a panel is exploded.
    #[must_use]
    pub fn is_exploded(&self, panel: PanelId) -> bool {
        self.exploded == Some(panel)
    }

    /// Check if any panel is exploded.
    #[must_use]
    pub fn has_exploded(&self) -> bool {
        self.exploded.is_some()
    }

    /// Focus a specific panel.
    pub fn focus(&mut self, panel: PanelId) {
        if self.visible.contains(&panel) {
            self.focused = Some(panel);
        }
    }

    /// Add a panel to the visible list.
    pub fn add_panel(&mut self, panel: PanelId) {
        if !self.visible.contains(&panel) {
            self.visible.push(panel);
        }
    }

    /// Remove a panel from the visible list.
    pub fn remove_panel(&mut self, panel: PanelId) {
        self.visible.retain(|p| *p != panel);
        if self.focused == Some(panel) {
            self.focused = self.visible.first().copied();
        }
        if self.exploded == Some(panel) {
            self.exploded = None;
        }
    }
}

// ============================================================================
// Ring Buffer
// ============================================================================

/// Ring buffer for time-series data.
///
/// Implements the ttop pattern for efficient sliding window storage.
/// Oldest values are evicted when capacity is reached.
///
/// # Example
///
/// ```rust,ignore
/// use probar::brick::tui::RingBuffer;
///
/// let mut buf: RingBuffer<f32> = RingBuffer::new(60);
/// for i in 0..100 {
///     buf.push(i as f32);
/// }
/// assert_eq!(buf.len(), 60);
/// assert_eq!(*buf.last().unwrap(), 99.0);
/// ```
#[derive(Debug, Clone)]
pub struct RingBuffer<T> {
    data: VecDeque<T>,
    capacity: usize,
}

impl<T> RingBuffer<T> {
    /// Create a new ring buffer with given capacity.
    #[must_use]
    pub fn new(capacity: usize) -> Self {
        Self {
            data: VecDeque::with_capacity(capacity),
            capacity,
        }
    }

    /// Push a value, evicting oldest if at capacity.
    pub fn push(&mut self, value: T) {
        if self.data.len() >= self.capacity {
            self.data.pop_front();
        }
        self.data.push_back(value);
    }

    /// Get iterator over values (oldest first).
    pub fn iter(&self) -> impl Iterator<Item = &T> {
        self.data.iter()
    }

    /// Get mutable iterator over values.
    pub fn iter_mut(&mut self) -> impl Iterator<Item = &mut T> {
        self.data.iter_mut()
    }

    /// Get number of elements.
    #[must_use]
    pub fn len(&self) -> usize {
        self.data.len()
    }

    /// Check if buffer is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    /// Get capacity.
    #[must_use]
    pub fn capacity(&self) -> usize {
        self.capacity
    }

    /// Get most recent value.
    #[must_use]
    pub fn last(&self) -> Option<&T> {
        self.data.back()
    }

    /// Get oldest value.
    #[must_use]
    pub fn first(&self) -> Option<&T> {
        self.data.front()
    }

    /// Get value at index (0 = oldest).
    #[must_use]
    pub fn get(&self, index: usize) -> Option<&T> {
        self.data.get(index)
    }

    /// Clear the buffer.
    pub fn clear(&mut self) {
        self.data.clear();
    }

    /// Check if buffer is at capacity.
    #[must_use]
    pub fn is_full(&self) -> bool {
        self.data.len() >= self.capacity
    }
}

impl<T: Clone> RingBuffer<T> {
    /// Make buffer contiguous and return as Vec.
    #[must_use]
    pub fn to_vec(&self) -> Vec<T> {
        self.data.iter().cloned().collect()
    }
}

impl<T: Copy + Default> RingBuffer<T> {
    /// Fill with default values to capacity.
    pub fn fill_default(&mut self) {
        while self.data.len() < self.capacity {
            self.data.push_back(T::default());
        }
    }
}

impl<T: Copy + Into<f64>> RingBuffer<T> {
    /// Calculate average of all values.
    #[must_use]
    pub fn average(&self) -> f64 {
        if self.is_empty() {
            return 0.0;
        }
        let sum: f64 = self.data.iter().map(|v| (*v).into()).sum();
        sum / self.data.len() as f64
    }

    /// Calculate min value.
    #[must_use]
    pub fn min(&self) -> Option<f64> {
        self.data
            .iter()
            .map(|v| (*v).into())
            .min_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
    }

    /// Calculate max value.
    #[must_use]
    pub fn max(&self) -> Option<f64> {
        self.data
            .iter()
            .map(|v| (*v).into())
            .max_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
    }
}

// ============================================================================
// CIELAB Color
// ============================================================================

/// CIELAB color for perceptually uniform gradients.
///
/// CIELAB (L*a*b*) is designed so that equal distances in the color space
/// correspond to equal perceived color differences.
///
/// # Example
///
/// ```rust,ignore
/// use probar::brick::tui::CielabColor;
///
/// // Create gradient from green (0%) to red (100%)
/// let green = CielabColor::percent_gradient(0.0);
/// let red = CielabColor::percent_gradient(1.0);
///
/// // Interpolate for 50%
/// let yellow = green.lerp(&red, 0.5);
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CielabColor {
    /// Lightness (0-100)
    pub l: f32,
    /// Green-red axis (approx -128 to 127)
    pub a: f32,
    /// Blue-yellow axis (approx -128 to 127)
    pub b: f32,
}

impl CielabColor {
    /// Create a CIELAB color.
    #[must_use]
    pub const fn new(l: f32, a: f32, b: f32) -> Self {
        Self { l, a, b }
    }

    /// Interpolate between two colors.
    #[must_use]
    pub fn lerp(&self, other: &Self, t: f32) -> Self {
        let t = t.clamp(0.0, 1.0);
        Self {
            l: self.l + (other.l - self.l) * t,
            a: self.a + (other.a - self.a) * t,
            b: self.b + (other.b - self.b) * t,
        }
    }

    /// Convert to approximate sRGB (0-255).
    ///
    /// Note: This is a simplified conversion. For accurate results,
    /// use a proper color management library.
    #[must_use]
    #[allow(
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss,
        clippy::many_single_char_names // x,y,z,r,g,b are standard color space variables
    )]
    pub fn to_rgb(&self) -> (u8, u8, u8) {
        // Convert L*a*b* to XYZ (D65 illuminant)
        let fy = (self.l + 16.0) / 116.0;
        let fx = self.a / 500.0 + fy;
        let fz = fy - self.b / 200.0;

        let xr = if fx.powi(3) > 0.008856 {
            fx.powi(3)
        } else {
            (116.0 * fx - 16.0) / 903.3
        };
        let yr = if self.l > 7.9996 {
            fy.powi(3)
        } else {
            self.l / 903.3
        };
        let zr = if fz.powi(3) > 0.008856 {
            fz.powi(3)
        } else {
            (116.0 * fz - 16.0) / 903.3
        };

        // D65 reference white
        let x = xr * 0.95047;
        let y = yr * 1.0;
        let z = zr * 1.08883;

        // XYZ to sRGB
        let r = x * 3.2406 - y * 1.5372 - z * 0.4986;
        let g = -x * 0.9689 + y * 1.8758 + z * 0.0415;
        let b = x * 0.0557 - y * 0.2040 + z * 1.0570;

        // Gamma correction
        let gamma = |c: f32| -> f32 {
            if c > 0.0031308 {
                1.055 * c.powf(1.0 / 2.4) - 0.055
            } else {
                12.92 * c
            }
        };

        let r = (gamma(r) * 255.0).clamp(0.0, 255.0) as u8;
        let g = (gamma(g) * 255.0).clamp(0.0, 255.0) as u8;
        let b = (gamma(b) * 255.0).clamp(0.0, 255.0) as u8;

        (r, g, b)
    }

    /// Convert to hex string (e.g., "#ff0000").
    #[must_use]
    pub fn to_hex(&self) -> String {
        let (r, g, b) = self.to_rgb();
        format!("#{r:02x}{g:02x}{b:02x}")
    }

    /// Create perceptually uniform gradient from green to red.
    ///
    /// Uses green -> yellow -> red transition in CIELAB space.
    #[must_use]
    pub fn percent_gradient(percent: f32) -> Self {
        let percent = percent.clamp(0.0, 1.0);

        // Perceptually uniform green/yellow/red
        let green = Self::new(87.0, -86.0, 83.0);
        let yellow = Self::new(97.0, -21.0, 94.0);
        let red = Self::new(53.0, 80.0, 67.0);

        if percent < 0.5 {
            green.lerp(&yellow, percent * 2.0)
        } else {
            yellow.lerp(&red, (percent - 0.5) * 2.0)
        }
    }

    /// Create gradient for meter display (blue -> green -> yellow -> red).
    #[must_use]
    pub fn meter_gradient(level: f32) -> Self {
        let level = level.clamp(0.0, 1.0);

        let blue = Self::new(50.0, -10.0, -50.0);
        let green = Self::new(87.0, -86.0, 83.0);
        let yellow = Self::new(97.0, -21.0, 94.0);
        let red = Self::new(53.0, 80.0, 67.0);

        if level < 0.33 {
            blue.lerp(&green, level * 3.0)
        } else if level < 0.66 {
            green.lerp(&yellow, (level - 0.33) * 3.0)
        } else {
            yellow.lerp(&red, (level - 0.66) * 3.0)
        }
    }
}

impl Default for CielabColor {
    fn default() -> Self {
        Self::new(50.0, 0.0, 0.0) // Neutral gray
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;

    // RingBuffer tests
    #[test]
    fn test_ring_buffer_basic() {
        let mut buf: RingBuffer<i32> = RingBuffer::new(3);
        buf.push(1);
        buf.push(2);
        buf.push(3);
        buf.push(4); // Should evict 1

        let values: Vec<_> = buf.iter().copied().collect();
        assert_eq!(values, vec![2, 3, 4]);
    }

    #[test]
    fn test_ring_buffer_capacity() {
        let mut buf: RingBuffer<i32> = RingBuffer::new(5);
        for i in 0..10 {
            buf.push(i);
        }
        assert_eq!(buf.len(), 5);
        assert_eq!(*buf.last().unwrap(), 9);
        assert_eq!(*buf.first().unwrap(), 5);
    }

    #[test]
    fn test_ring_buffer_to_vec() {
        let mut buf: RingBuffer<i32> = RingBuffer::new(3);
        buf.push(1);
        buf.push(2);
        buf.push(3);
        buf.push(4);

        assert_eq!(buf.to_vec(), vec![2, 3, 4]);
    }

    #[test]
    fn test_ring_buffer_average() {
        let mut buf: RingBuffer<f32> = RingBuffer::new(4);
        buf.push(1.0);
        buf.push(2.0);
        buf.push(3.0);
        buf.push(4.0);

        assert!((buf.average() - 2.5).abs() < 0.001);
    }

    #[test]
    fn test_ring_buffer_min_max() {
        let mut buf: RingBuffer<f32> = RingBuffer::new(5);
        buf.push(3.0);
        buf.push(1.0);
        buf.push(4.0);
        buf.push(1.5);
        buf.push(9.0);

        assert!((buf.min().unwrap() - 1.0).abs() < 0.001);
        assert!((buf.max().unwrap() - 9.0).abs() < 0.001);
    }

    // PanelState tests
    #[test]
    fn test_panel_state_focus() {
        let mut state = PanelState::default();
        state.focused = Some(PanelId::Waveform);

        state.focus_next();
        assert_eq!(state.focused, Some(PanelId::Transcription));

        state.focus_next();
        assert_eq!(state.focused, Some(PanelId::Metrics));

        state.focus_prev();
        assert_eq!(state.focused, Some(PanelId::Transcription));
    }

    #[test]
    fn test_panel_state_focus_wrap() {
        let mut state = PanelState::with_panels(vec![PanelId::Waveform, PanelId::Metrics]);
        state.focused = Some(PanelId::Metrics);

        state.focus_next();
        assert_eq!(state.focused, Some(PanelId::Waveform));

        state.focus_prev();
        assert_eq!(state.focused, Some(PanelId::Metrics));
    }

    #[test]
    fn test_panel_state_explode() {
        let mut state = PanelState::default();
        state.focused = Some(PanelId::Transcription);

        assert!(!state.has_exploded());

        state.toggle_explode();
        assert!(state.is_exploded(PanelId::Transcription));
        assert!(state.has_exploded());

        state.toggle_explode();
        assert!(!state.has_exploded());
    }

    #[test]
    fn test_panel_state_add_remove() {
        let mut state = PanelState::default();
        let custom = PanelId::Custom(42);

        state.add_panel(custom);
        assert!(state.visible.contains(&custom));

        state.focus(custom);
        assert_eq!(state.focused, Some(custom));

        state.remove_panel(custom);
        assert!(!state.visible.contains(&custom));
        assert_ne!(state.focused, Some(custom));
    }

    // CIELAB tests
    #[test]
    fn test_cielab_lerp() {
        let green = CielabColor::new(87.0, -86.0, 83.0);
        let red = CielabColor::new(53.0, 80.0, 67.0);

        let mid = green.lerp(&red, 0.5);
        assert!((mid.l - 70.0).abs() < 0.1);
        assert!((mid.a - (-3.0)).abs() < 0.1);
    }

    #[test]
    fn test_cielab_gradient() {
        let start = CielabColor::percent_gradient(0.0);
        let end = CielabColor::percent_gradient(1.0);

        // Start should be greenish (negative a)
        assert!(start.a < 0.0);
        // End should be reddish (positive a)
        assert!(end.a > 0.0);
    }

    #[test]
    fn test_cielab_to_rgb() {
        let white = CielabColor::new(100.0, 0.0, 0.0);
        let (r, g, b) = white.to_rgb();
        // Should be close to white
        assert!(r > 250);
        assert!(g > 250);
        assert!(b > 250);

        let black = CielabColor::new(0.0, 0.0, 0.0);
        let (r, g, b) = black.to_rgb();
        // Should be close to black
        assert!(r < 5);
        assert!(g < 5);
        assert!(b < 5);
    }

    #[test]
    fn test_cielab_to_hex() {
        let color = CielabColor::new(50.0, 0.0, 0.0);
        let hex = color.to_hex();
        assert!(hex.starts_with('#'));
        assert_eq!(hex.len(), 7);
    }

    #[test]
    fn test_cielab_meter_gradient() {
        let low = CielabColor::meter_gradient(0.0);
        let mid = CielabColor::meter_gradient(0.5);
        let high = CielabColor::meter_gradient(1.0);

        // Low should be bluish (negative b)
        assert!(low.b < 0.0);
        // High should be reddish (positive a)
        assert!(high.a > 0.0);
        // Mid should be greenish/yellowish
        assert!(mid.l > 80.0);
    }

    // CollectorError tests
    #[test]
    fn test_collector_error_display() {
        assert_eq!(
            CollectorError::NotAvailable.to_string(),
            "Feature not available on this platform"
        );
        assert_eq!(
            CollectorError::Failed("test".into()).to_string(),
            "Collection failed: test"
        );
    }

    // ========================================================================
    // Additional comprehensive tests for 95%+ coverage
    // ========================================================================

    #[test]
    fn test_collector_error_disabled() {
        assert_eq!(
            CollectorError::Disabled.to_string(),
            "Collector is disabled"
        );
    }

    #[test]
    fn test_collector_error_timeout() {
        assert_eq!(CollectorError::Timeout.to_string(), "Collection timed out");
    }

    #[test]
    fn test_collector_error_eq() {
        assert_eq!(CollectorError::NotAvailable, CollectorError::NotAvailable);
        assert_eq!(CollectorError::Disabled, CollectorError::Disabled);
        assert_eq!(CollectorError::Timeout, CollectorError::Timeout);
        assert_eq!(
            CollectorError::Failed("a".into()),
            CollectorError::Failed("a".into())
        );
        assert_ne!(
            CollectorError::Failed("a".into()),
            CollectorError::Failed("b".into())
        );
    }

    #[test]
    fn test_collector_error_clone() {
        let err = CollectorError::Failed("test".into());
        let cloned = err.clone();
        assert_eq!(err, cloned);
    }

    #[test]
    fn test_collector_error_is_error() {
        let err = CollectorError::Timeout;
        let _: &dyn std::error::Error = &err;
    }

    #[test]
    fn test_ring_buffer_is_empty() {
        let buf: RingBuffer<i32> = RingBuffer::new(5);
        assert!(buf.is_empty());

        let mut buf2: RingBuffer<i32> = RingBuffer::new(5);
        buf2.push(1);
        assert!(!buf2.is_empty());
    }

    #[test]
    fn test_ring_buffer_capacity_getter() {
        let buf: RingBuffer<i32> = RingBuffer::new(10);
        assert_eq!(buf.capacity(), 10);
    }

    #[test]
    fn test_ring_buffer_first_last_empty() {
        let buf: RingBuffer<i32> = RingBuffer::new(5);
        assert!(buf.first().is_none());
        assert!(buf.last().is_none());
    }

    #[test]
    fn test_ring_buffer_get() {
        let mut buf: RingBuffer<i32> = RingBuffer::new(5);
        buf.push(10);
        buf.push(20);
        buf.push(30);

        assert_eq!(buf.get(0), Some(&10));
        assert_eq!(buf.get(1), Some(&20));
        assert_eq!(buf.get(2), Some(&30));
        assert_eq!(buf.get(3), None);
    }

    #[test]
    fn test_ring_buffer_clear() {
        let mut buf: RingBuffer<i32> = RingBuffer::new(5);
        buf.push(1);
        buf.push(2);
        buf.push(3);

        assert_eq!(buf.len(), 3);
        buf.clear();
        assert_eq!(buf.len(), 0);
        assert!(buf.is_empty());
    }

    #[test]
    fn test_ring_buffer_is_full() {
        let mut buf: RingBuffer<i32> = RingBuffer::new(3);
        assert!(!buf.is_full());

        buf.push(1);
        assert!(!buf.is_full());

        buf.push(2);
        assert!(!buf.is_full());

        buf.push(3);
        assert!(buf.is_full());

        buf.push(4); // Evicts oldest
        assert!(buf.is_full());
    }

    #[test]
    fn test_ring_buffer_iter_mut() {
        let mut buf: RingBuffer<i32> = RingBuffer::new(5);
        buf.push(1);
        buf.push(2);
        buf.push(3);

        for val in buf.iter_mut() {
            *val *= 2;
        }

        let values: Vec<_> = buf.iter().copied().collect();
        assert_eq!(values, vec![2, 4, 6]);
    }

    #[test]
    fn test_ring_buffer_fill_default() {
        let mut buf: RingBuffer<i32> = RingBuffer::new(5);
        buf.push(1);
        buf.push(2);

        assert_eq!(buf.len(), 2);
        buf.fill_default();
        assert_eq!(buf.len(), 5);
        assert!(buf.is_full());
    }

    #[test]
    fn test_ring_buffer_average_empty() {
        let buf: RingBuffer<f32> = RingBuffer::new(5);
        assert!((buf.average() - 0.0).abs() < 0.001);
    }

    #[test]
    fn test_ring_buffer_min_max_empty() {
        let buf: RingBuffer<f32> = RingBuffer::new(5);
        assert!(buf.min().is_none());
        assert!(buf.max().is_none());
    }

    #[test]
    fn test_ring_buffer_clone() {
        let mut buf: RingBuffer<i32> = RingBuffer::new(3);
        buf.push(1);
        buf.push(2);

        let cloned = buf.clone();
        assert_eq!(buf.len(), cloned.len());
        assert_eq!(buf.to_vec(), cloned.to_vec());
    }

    #[test]
    fn test_panel_id_eq() {
        assert_eq!(PanelId::Waveform, PanelId::Waveform);
        assert_eq!(PanelId::Custom(1), PanelId::Custom(1));
        assert_ne!(PanelId::Custom(1), PanelId::Custom(2));
        assert_ne!(PanelId::Waveform, PanelId::Spectrogram);
    }

    #[test]
    fn test_panel_id_hash() {
        use std::collections::HashSet;
        let mut set = HashSet::new();
        set.insert(PanelId::Waveform);
        set.insert(PanelId::Spectrogram);
        set.insert(PanelId::Waveform); // Duplicate
        assert_eq!(set.len(), 2);
    }

    #[test]
    fn test_panel_id_all_variants() {
        let ids = [
            PanelId::Waveform,
            PanelId::Spectrogram,
            PanelId::Transcription,
            PanelId::Metrics,
            PanelId::VuMeter,
            PanelId::Status,
            PanelId::Custom(0),
        ];
        for id in ids {
            // Just verify they can be created and compared
            assert_eq!(id, id);
        }
    }

    #[test]
    fn test_panel_state_default() {
        let state = PanelState::default();
        assert!(state.focused.is_none());
        assert!(state.exploded.is_none());
        assert_eq!(state.visible.len(), 4);
        assert!(state.visible.contains(&PanelId::Waveform));
        assert!(state.visible.contains(&PanelId::Transcription));
        assert!(state.visible.contains(&PanelId::Metrics));
        assert!(state.visible.contains(&PanelId::Status));
    }

    #[test]
    fn test_panel_state_with_panels() {
        let panels = vec![PanelId::Spectrogram, PanelId::VuMeter];
        let state = PanelState::with_panels(panels);
        assert_eq!(state.visible.len(), 2);
        assert_eq!(state.focused, Some(PanelId::Spectrogram));
    }

    #[test]
    fn test_panel_state_with_panels_empty() {
        let state = PanelState::with_panels(vec![]);
        assert!(state.visible.is_empty());
        assert!(state.focused.is_none());
    }

    #[test]
    fn test_panel_state_focus_next_empty() {
        let mut state = PanelState::with_panels(vec![]);
        state.focus_next();
        assert!(state.focused.is_none());
    }

    #[test]
    fn test_panel_state_focus_prev_empty() {
        let mut state = PanelState::with_panels(vec![]);
        state.focus_prev();
        assert!(state.focused.is_none());
    }

    #[test]
    fn test_panel_state_focus_next_no_current_focus() {
        let mut state = PanelState::with_panels(vec![PanelId::Waveform, PanelId::Metrics]);
        state.focused = None;

        state.focus_next();
        assert_eq!(state.focused, Some(PanelId::Waveform));
    }

    #[test]
    fn test_panel_state_focus_prev_no_current_focus() {
        let mut state = PanelState::with_panels(vec![PanelId::Waveform, PanelId::Metrics]);
        state.focused = None;

        state.focus_prev();
        assert_eq!(state.focused, Some(PanelId::Waveform));
    }

    #[test]
    fn test_panel_state_focus_prev_at_start() {
        let mut state = PanelState::with_panels(vec![PanelId::Waveform, PanelId::Metrics]);
        state.focused = Some(PanelId::Waveform);

        state.focus_prev();
        assert_eq!(state.focused, Some(PanelId::Metrics)); // Wraps to end
    }

    #[test]
    fn test_panel_state_toggle_explode_no_focus() {
        let mut state = PanelState::default();
        state.focused = None;

        state.toggle_explode();
        // exploded should be set to None (focused)
        assert!(state.exploded.is_none());
    }

    #[test]
    fn test_panel_state_is_focused() {
        let mut state = PanelState::default();
        state.focused = Some(PanelId::Waveform);

        assert!(state.is_focused(PanelId::Waveform));
        assert!(!state.is_focused(PanelId::Metrics));
    }

    #[test]
    fn test_panel_state_is_exploded() {
        let mut state = PanelState::default();
        state.exploded = Some(PanelId::Transcription);

        assert!(state.is_exploded(PanelId::Transcription));
        assert!(!state.is_exploded(PanelId::Waveform));
    }

    #[test]
    fn test_panel_state_focus_invalid_panel() {
        let mut state = PanelState::default();
        state.focused = Some(PanelId::Waveform);

        // Try to focus a panel not in visible list
        state.focus(PanelId::VuMeter);
        // Focus should remain unchanged
        assert_eq!(state.focused, Some(PanelId::Waveform));
    }

    #[test]
    fn test_panel_state_add_panel_duplicate() {
        let mut state = PanelState::default();
        let initial_len = state.visible.len();

        state.add_panel(PanelId::Waveform); // Already in list
        assert_eq!(state.visible.len(), initial_len);
    }

    #[test]
    fn test_panel_state_remove_panel_updates_focus() {
        let mut state = PanelState::default();
        state.focused = Some(PanelId::Metrics);

        state.remove_panel(PanelId::Metrics);
        // Focus should move to first visible
        assert_eq!(state.focused, Some(PanelId::Waveform));
    }

    #[test]
    fn test_panel_state_remove_panel_clears_exploded() {
        let mut state = PanelState::default();
        state.exploded = Some(PanelId::Metrics);

        state.remove_panel(PanelId::Metrics);
        assert!(state.exploded.is_none());
    }

    #[test]
    fn test_panel_state_clone() {
        let state = PanelState::default();
        let cloned = state.clone();
        assert_eq!(state.visible.len(), cloned.visible.len());
    }

    #[test]
    fn test_cielab_default() {
        let color = CielabColor::default();
        assert!((color.l - 50.0).abs() < 0.001);
        assert!((color.a - 0.0).abs() < 0.001);
        assert!((color.b - 0.0).abs() < 0.001);
    }

    #[test]
    fn test_cielab_lerp_clamp() {
        let c1 = CielabColor::new(0.0, 0.0, 0.0);
        let c2 = CielabColor::new(100.0, 100.0, 100.0);

        // t < 0 should clamp to 0
        let result = c1.lerp(&c2, -0.5);
        assert!((result.l - 0.0).abs() < 0.001);

        // t > 1 should clamp to 1
        let result = c1.lerp(&c2, 1.5);
        assert!((result.l - 100.0).abs() < 0.001);
    }

    #[test]
    fn test_cielab_percent_gradient_clamp() {
        // Below 0 should clamp
        let color = CielabColor::percent_gradient(-0.5);
        assert!(color.a < 0.0); // Should be green

        // Above 1 should clamp
        let color = CielabColor::percent_gradient(1.5);
        assert!(color.a > 0.0); // Should be red
    }

    #[test]
    fn test_cielab_percent_gradient_midpoint() {
        // Test the transition point (0.5)
        let color = CielabColor::percent_gradient(0.5);
        // At 0.5, should be yellow-ish
        assert!(color.l > 90.0);
    }

    #[test]
    fn test_cielab_meter_gradient_clamp() {
        let low = CielabColor::meter_gradient(-1.0);
        let high = CielabColor::meter_gradient(2.0);
        // Should clamp to valid range
        assert!(low.b < 0.0); // Blue
        assert!(high.a > 0.0); // Red
    }

    #[test]
    fn test_cielab_meter_gradient_transitions() {
        // Test at boundary points
        let at_033 = CielabColor::meter_gradient(0.33);
        let at_066 = CielabColor::meter_gradient(0.66);

        // Both should be valid colors
        assert!(at_033.l > 0.0);
        assert!(at_066.l > 0.0);
    }

    #[test]
    fn test_cielab_to_rgb_edge_cases() {
        // Test with low L (dark)
        let dark = CielabColor::new(5.0, 0.0, 0.0);
        let (r, g, b) = dark.to_rgb();
        // Should be very dark
        assert!(r < 20);
        assert!(g < 20);
        assert!(b < 20);
    }

    #[test]
    fn test_cielab_eq() {
        let c1 = CielabColor::new(50.0, 10.0, -20.0);
        let c2 = CielabColor::new(50.0, 10.0, -20.0);
        let c3 = CielabColor::new(51.0, 10.0, -20.0);

        assert_eq!(c1, c2);
        assert_ne!(c1, c3);
    }

    #[test]
    fn test_cielab_clone() {
        let c1 = CielabColor::new(75.0, 25.0, -50.0);
        let c2 = c1;
        assert_eq!(c1, c2);
    }

    #[test]
    fn test_ring_buffer_f64() {
        let mut buf: RingBuffer<f64> = RingBuffer::new(3);
        buf.push(1.5);
        buf.push(2.5);
        buf.push(3.5);

        assert!((buf.average() - 2.5).abs() < 0.001);
        assert!((buf.min().unwrap() - 1.5).abs() < 0.001);
        assert!((buf.max().unwrap() - 3.5).abs() < 0.001);
    }
}
