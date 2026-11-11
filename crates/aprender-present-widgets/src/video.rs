//! Video widget: playback state, layout and draw commands.
//!
//! This is the pure (non-browser) half of the widget. It owns the playback
//! state machine and turns it into draw commands; the real `<video>` element
//! on `wasm32` is bound to this state by a later ticket.
//!
//! Contract: `contracts/present-video-widget-v1.yaml`.

use presentar_core::{
    widget::{AccessibleRole, LayoutResult},
    Brick, BrickAssertion, BrickBudget, BrickVerification, Canvas, Color, Constraints, Event, Key,
    MouseButton, Point, Rect, Size, TypeId, Widget,
};
use serde::{Deserialize, Serialize};
use std::any::Any;
use std::time::Duration;

/// Play glyph height as a fraction of the shorter side of the bounds.
const GLYPH_SCALE: f32 = 0.25;
/// Play glyph width over height (an equilateral triangle pointing right).
const GLYPH_ASPECT: f32 = 0.866;
/// Default intrinsic width when none is set (16:9).
const DEFAULT_WIDTH: f32 = 320.0;
/// Default intrinsic height when none is set (16:9).
const DEFAULT_HEIGHT: f32 = 180.0;
/// Largest magnitude accepted for any coordinate or extent. Larger finite
/// values are clamped so that `x + width` can never overflow to infinity.
const COORD_LIMIT: f32 = 1.0e15;

/// A finite coordinate: NaN and infinities become 0, huge values are clamped.
fn sane(value: f32) -> f32 {
    if value.is_finite() {
        value.clamp(-COORD_LIMIT, COORD_LIMIT)
    } else {
        0.0
    }
}

/// `bounds` made safe to compute with: every component finite and within
/// `COORD_LIMIT`, and the size not negative. Valid bounds in range pass through
/// unchanged.
fn sane_bounds(bounds: Rect) -> Rect {
    Rect::new(
        sane(bounds.x),
        sane(bounds.y),
        sane(bounds.width).max(0.0),
        sane(bounds.height).max(0.0),
    )
}

/// Whether `bounds` is finite with a size that is not negative.
fn valid_bounds(bounds: Rect) -> bool {
    [bounds.x, bounds.y, bounds.width, bounds.height]
        .iter()
        .all(|v| v.is_finite())
        && bounds.width >= 0.0
        && bounds.height >= 0.0
}

/// A usable intrinsic extent: finite values are clamped into
/// `[0, COORD_LIMIT]`; NaN and infinities count as "not set".
fn sane_extent(value: f32) -> Option<f32> {
    value.is_finite().then(|| value.clamp(0.0, COORD_LIMIT))
}

/// Fit `value` between `min` and `max` and keep it finite. Unlike
/// `f32::clamp` this never panics, whatever the constraints (NaN, infinite or
/// `min > max`); a lower bound that exceeds the upper one loses.
fn fit(value: f32, min: f32, max: f32) -> f32 {
    sane(value.max(min).min(max))
}

/// `point` pulled onto `bounds` (NaN-safe; never panics).
fn within(bounds: Rect, point: Point) -> Point {
    Point::new(
        point.x.max(bounds.x).min(bounds.x + bounds.width),
        point.y.max(bounds.y).min(bounds.y + bounds.height),
    )
}

/// Playback state of a [`Video`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
pub enum VideoState {
    /// Not playing; the poster and the play glyph are shown
    #[default]
    Paused,
    /// Playing; the current frame is shown
    Playing,
    /// Playback reached the end; the poster and the play glyph are shown
    Ended,
}

/// Message emitted when the user toggles playback.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VideoToggled {
    /// Whether the video is playing after the toggle
    pub playing: bool,
}

/// Video widget (state, layout and draw commands).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Video {
    /// Video source URI
    src: String,
    /// Poster image URI shown while not playing
    poster: Option<String>,
    /// Alternative text for accessibility
    alt: String,
    /// Intrinsic width
    width: Option<f32>,
    /// Intrinsic height
    height: Option<f32>,
    /// Start playing as soon as the widget is created
    autoplay: bool,
    /// Show the native controls
    controls: bool,
    /// Playback state
    state: VideoState,
    /// Accessible name override
    accessible_name_value: Option<String>,
    /// Test ID
    test_id_value: Option<String>,
    /// Mouse is pressed inside the widget
    #[serde(skip)]
    pressed: bool,
    /// Cached bounds
    #[serde(skip)]
    bounds: Rect,
}

impl Default for Video {
    fn default() -> Self {
        Self {
            src: String::new(),
            poster: None,
            alt: String::new(),
            width: None,
            height: None,
            autoplay: false,
            controls: true,
            state: VideoState::Paused,
            accessible_name_value: None,
            test_id_value: None,
            pressed: false,
            bounds: Rect::default(),
        }
    }
}

impl Video {
    /// Create a new video with a source.
    #[must_use]
    pub fn new(src: impl Into<String>) -> Self {
        Self {
            src: src.into(),
            ..Self::default()
        }
    }

    /// Set the poster image.
    #[must_use]
    pub fn poster(mut self, url: impl Into<String>) -> Self {
        self.poster = Some(url.into());
        self
    }

    /// Set the alt text.
    #[must_use]
    pub fn alt(mut self, alt: impl Into<String>) -> Self {
        self.alt = alt.into();
        self
    }

    /// Set the intrinsic width.
    ///
    /// Negative values become 0 and values above 1e15 are clamped to it; NaN
    /// and infinities count as "not set", so `measure` falls back to the
    /// default width instead of reporting a non-finite size.
    #[must_use]
    pub fn width(mut self, width: f32) -> Self {
        self.width = sane_extent(width);
        self
    }

    /// Set the intrinsic height (same rules as [`Video::width`]).
    #[must_use]
    pub fn height(mut self, height: f32) -> Self {
        self.height = sane_extent(height);
        self
    }

    /// Set both width and height.
    #[must_use]
    pub fn size(self, width: f32, height: f32) -> Self {
        self.width(width).height(height)
    }

    /// Set the accessible name.
    #[must_use]
    pub fn accessible_name(mut self, name: impl Into<String>) -> Self {
        self.accessible_name_value = Some(name.into());
        self
    }

    /// Set the test ID.
    #[must_use]
    pub fn test_id(mut self, id: impl Into<String>) -> Self {
        self.test_id_value = Some(id.into());
        self
    }

    /// Set autoplay; the video starts `Playing` when true, `Paused` when false.
    #[must_use]
    pub const fn autoplay(mut self, autoplay: bool) -> Self {
        self.autoplay = autoplay;
        self.state = if autoplay {
            VideoState::Playing
        } else {
            VideoState::Paused
        };
        self
    }

    /// Set whether the controls are shown.
    #[must_use]
    pub const fn controls(mut self, controls: bool) -> Self {
        self.controls = controls;
        self
    }

    /// Get the source.
    #[must_use]
    pub fn get_src(&self) -> &str {
        &self.src
    }

    /// Get the poster URI.
    #[must_use]
    pub fn get_poster(&self) -> Option<&str> {
        self.poster.as_deref()
    }

    /// Get the alt text.
    #[must_use]
    pub fn get_alt(&self) -> &str {
        &self.alt
    }

    /// Get the intrinsic width.
    #[must_use]
    pub const fn get_width(&self) -> Option<f32> {
        self.width
    }

    /// Get the intrinsic height.
    #[must_use]
    pub const fn get_height(&self) -> Option<f32> {
        self.height
    }

    /// Whether autoplay is set.
    #[must_use]
    pub const fn is_autoplay(&self) -> bool {
        self.autoplay
    }

    /// Whether controls are shown.
    #[must_use]
    pub const fn has_controls(&self) -> bool {
        self.controls
    }

    /// Get the playback state.
    #[must_use]
    pub const fn state(&self) -> VideoState {
        self.state
    }

    /// Whether the video is playing.
    #[must_use]
    pub const fn is_playing(&self) -> bool {
        matches!(self.state, VideoState::Playing)
    }

    /// Start (or restart) playback.
    pub fn play(&mut self) {
        self.state = VideoState::Playing;
    }

    /// Pause playback. A video that has already ended stays `Ended`.
    pub fn pause(&mut self) {
        if self.is_playing() {
            self.state = VideoState::Paused;
        }
    }

    /// Toggle between playing and not playing.
    pub fn toggle(&mut self) {
        if self.is_playing() {
            self.pause();
        } else {
            self.play();
        }
    }

    /// Mark playback as having reached the end.
    pub fn ended(&mut self) {
        self.state = VideoState::Ended;
    }

    /// The three vertices of the play glyph, centred in the laid-out bounds.
    ///
    /// The bounding box of the triangle has the same centre as the bounds.
    /// Every vertex is finite and inside the laid-out bounds (which `layout`
    /// has sanitised), whatever rectangle was passed to `layout`.
    #[must_use]
    pub fn play_glyph(&self) -> [Point; 3] {
        let bounds = self.bounds;
        let centre = bounds.center();
        let height = bounds.width.min(bounds.height).max(0.0) * GLYPH_SCALE;
        let width = height * GLYPH_ASPECT;
        [
            Point::new(centre.x - width / 2.0, centre.y - height / 2.0),
            Point::new(centre.x - width / 2.0, centre.y + height / 2.0),
            Point::new(centre.x + width / 2.0, centre.y),
        ]
        .map(|vertex| within(bounds, vertex))
    }

    /// Colour of the frame while playing.
    fn frame_color() -> Color {
        Color::rgb(0.0, 0.0, 0.0)
    }

    /// Colour of the placeholder shown when no poster is set.
    fn placeholder_color() -> Color {
        Color::rgb(0.15, 0.15, 0.18)
    }

    /// Colour standing in for the poster image.
    fn poster_color() -> Color {
        Color::rgb(0.2, 0.2, 0.28)
    }

    /// Toggle playback and build the message describing the new state.
    fn toggled_message(&mut self) -> Option<Box<dyn Any + Send>> {
        self.toggle();
        Some(Box::new(VideoToggled {
            playing: self.is_playing(),
        }))
    }

    /// Non-empty accessible name, if any.
    fn effective_name(&self) -> Option<&str> {
        self.accessible_name_value
            .as_deref()
            .filter(|name| !name.trim().is_empty())
            .or((!self.alt.trim().is_empty()).then_some(self.alt.as_str()))
    }
}

/// Escape a string for use inside a double-quoted HTML attribute.
fn escape_attr(value: &str) -> String {
    value
        .replace('&', "&amp;")
        .replace('"', "&quot;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
}

/// Record one verification outcome.
fn record(
    passed: &mut Vec<BrickAssertion>,
    failed: &mut Vec<(BrickAssertion, String)>,
    ok: bool,
    assertion: BrickAssertion,
    reason: &str,
) {
    if ok {
        passed.push(assertion);
    } else {
        failed.push((assertion, reason.to_string()));
    }
}

impl Widget for Video {
    fn type_id(&self) -> TypeId {
        TypeId::of::<Self>()
    }

    fn measure(&self, constraints: Constraints) -> Size {
        // Not `Constraints::constrain`: its `f32::clamp` panics on NaN or
        // inverted constraints and passes an infinite minimum through.
        Size::new(
            fit(
                self.width.unwrap_or(DEFAULT_WIDTH),
                constraints.min_width,
                constraints.max_width,
            ),
            fit(
                self.height.unwrap_or(DEFAULT_HEIGHT),
                constraints.min_height,
                constraints.max_height,
            ),
        )
    }

    /// Lay the video out in `bounds`.
    ///
    /// The bounds are sanitised first (NaN and infinities become 0, components
    /// are clamped to +-1e15, a negative size becomes 0), so every rect and
    /// glyph vertex `paint` emits is finite and inside the stored bounds. Bounds
    /// that are already finite, not negative and in range are kept exactly.
    fn layout(&mut self, bounds: Rect) -> LayoutResult {
        self.bounds = sane_bounds(bounds);
        LayoutResult {
            size: self.bounds.size(),
        }
    }

    fn paint(&self, canvas: &mut dyn Canvas) {
        if self.is_playing() {
            canvas.fill_rect(self.bounds, Self::frame_color());
            return;
        }
        let backdrop = if self.poster.is_some() {
            Self::poster_color()
        } else {
            Self::placeholder_color()
        };
        canvas.fill_rect(self.bounds, backdrop);
        canvas.fill_polygon(&self.play_glyph(), Color::WHITE);
    }

    fn event(&mut self, event: &Event) -> Option<Box<dyn Any + Send>> {
        match event {
            Event::MouseDown {
                position,
                button: MouseButton::Left,
            } => {
                self.pressed = self.bounds.contains_point(position);
                None
            }
            Event::MouseUp {
                position,
                button: MouseButton::Left,
            } => {
                let was_pressed = std::mem::take(&mut self.pressed);
                if was_pressed && self.bounds.contains_point(position) {
                    self.toggled_message()
                } else {
                    None
                }
            }
            Event::MouseLeave => {
                self.pressed = false;
                None
            }
            Event::KeyUp {
                key: Key::Enter | Key::Space,
                ..
            } => self.toggled_message(),
            _ => None,
        }
    }

    fn children(&self) -> &[Box<dyn Widget>] {
        &[]
    }

    fn children_mut(&mut self) -> &mut [Box<dyn Widget>] {
        &mut []
    }

    fn is_interactive(&self) -> bool {
        true
    }

    fn is_focusable(&self) -> bool {
        true
    }

    fn accessible_name(&self) -> Option<&str> {
        self.effective_name()
    }

    fn accessible_role(&self) -> AccessibleRole {
        AccessibleRole::Generic
    }

    fn bounds(&self) -> Rect {
        self.bounds
    }

    fn test_id(&self) -> Option<&str> {
        self.test_id_value.as_deref()
    }
}

// PROBAR-SPEC-009: Brick Architecture - Tests define interface
impl Brick for Video {
    fn brick_name(&self) -> &'static str {
        "Video"
    }

    fn assertions(&self) -> &[BrickAssertion] {
        &[BrickAssertion::MaxLatencyMs(16), BrickAssertion::Focusable]
    }

    fn budget(&self) -> BrickBudget {
        BrickBudget::uniform(16)
    }

    fn verify(&self) -> BrickVerification {
        let mut passed = self.assertions().to_vec();
        let mut failed = Vec::new();

        record(
            &mut passed,
            &mut failed,
            !self.src.is_empty(),
            BrickAssertion::ElementPresent("video[src]".into()),
            "Video has no src",
        );
        record(
            &mut passed,
            &mut failed,
            self.effective_name().is_some(),
            BrickAssertion::ElementPresent("accessible-name".into()),
            "Video has no accessible name",
        );
        record(
            &mut passed,
            &mut failed,
            valid_bounds(self.bounds),
            BrickAssertion::ElementPresent("valid-bounds".into()),
            "Video bounds are not finite and >= 0",
        );

        BrickVerification {
            passed,
            failed,
            verification_time: Duration::from_micros(10),
        }
    }

    fn to_html(&self) -> String {
        let poster = self.poster.as_deref().map_or_else(String::new, |url| {
            format!(r#" poster="{}""#, escape_attr(url))
        });
        let controls = if self.controls { " controls" } else { "" };
        let autoplay = if self.autoplay { " autoplay" } else { "" };
        format!(
            r#"<video class="brick-video" src="{}"{}{}{} aria-label="{}"></video>"#,
            escape_attr(&self.src),
            poster,
            controls,
            autoplay,
            escape_attr(self.effective_name().unwrap_or(""))
        )
    }

    fn to_css(&self) -> String {
        ".brick-video { display: block; background: #000; }".to_string()
    }

    fn test_id(&self) -> Option<&str> {
        self.test_id_value.as_deref()
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::disallowed_methods)]
mod tests {
    use super::*;
    use crate::modal::Modal;
    use presentar_core::draw::DrawCommand;
    use presentar_core::RecordingCanvas;
    use proptest::prelude::*;

    const BOUNDS: Rect = Rect {
        x: 10.0,
        y: 20.0,
        width: 320.0,
        height: 180.0,
    };

    fn named(src: &str) -> Video {
        Video::new(src).accessible_name("Lecture video")
    }

    fn laid_out(mut video: Video) -> Video {
        video.layout(BOUNDS);
        video
    }

    fn click(video: &mut Video, at: Point) -> Option<Box<dyn Any + Send>> {
        video.event(&Event::MouseDown {
            position: at,
            button: MouseButton::Left,
        });
        video.event(&Event::MouseUp {
            position: at,
            button: MouseButton::Left,
        })
    }

    fn toggled(msg: &Option<Box<dyn Any + Send>>) -> Option<VideoToggled> {
        msg.as_ref()
            .and_then(|m| m.downcast_ref::<VideoToggled>())
            .cloned()
    }

    fn paint_commands(video: &Video) -> Vec<DrawCommand> {
        let mut canvas = RecordingCanvas::new();
        video.paint(&mut canvas);
        canvas.commands().to_vec()
    }

    fn contains(outer: &Rect, p: &Point, eps: f32) -> bool {
        p.x >= outer.x - eps
            && p.x <= outer.x + outer.width + eps
            && p.y >= outer.y - eps
            && p.y <= outer.y + outer.height + eps
    }

    fn rect_inside(outer: &Rect, inner: &Rect, eps: f32) -> bool {
        contains(outer, &Point::new(inner.x, inner.y), eps)
            && contains(
                outer,
                &Point::new(inner.x + inner.width, inner.y + inner.height),
                eps,
            )
    }

    fn glyph_centre(points: &[Point]) -> Point {
        let min_x = points.iter().map(|p| p.x).fold(f32::INFINITY, f32::min);
        let max_x = points.iter().map(|p| p.x).fold(f32::NEG_INFINITY, f32::max);
        let min_y = points.iter().map(|p| p.y).fold(f32::INFINITY, f32::min);
        let max_y = points.iter().map(|p| p.y).fold(f32::NEG_INFINITY, f32::max);
        Point::new((min_x + max_x) / 2.0, (min_y + max_y) / 2.0)
    }

    // ===== Construction =====

    #[test]
    fn video_new_has_documented_defaults() {
        let v = Video::new("movie.mp4");
        assert_eq!(v.get_src(), "movie.mp4");
        assert!(v.get_poster().is_none());
        assert!(v.get_alt().is_empty());
        assert!(v.get_width().is_none());
        assert!(v.get_height().is_none());
        assert!(!v.is_autoplay());
        assert!(v.has_controls());
        assert_eq!(v.state(), VideoState::Paused);
        assert!(!v.is_playing());
    }

    #[test]
    fn video_default_is_empty_and_paused() {
        let v = Video::default();
        assert!(v.get_src().is_empty());
        assert_eq!(v.state(), VideoState::default());
        assert_eq!(VideoState::default(), VideoState::Paused);
    }

    #[test]
    fn video_builder_sets_every_field() {
        let v = Video::new("a.mp4")
            .poster("a.png")
            .alt("Intro")
            .size(640.0, 360.0)
            .accessible_name("Intro video")
            .test_id("hero-video")
            .autoplay(true)
            .controls(false);
        assert_eq!(v.get_poster(), Some("a.png"));
        assert_eq!(v.get_alt(), "Intro");
        assert_eq!(v.get_width(), Some(640.0));
        assert_eq!(v.get_height(), Some(360.0));
        assert_eq!(Widget::accessible_name(&v), Some("Intro video"));
        assert_eq!(Widget::test_id(&v), Some("hero-video"));
        assert_eq!(Brick::test_id(&v), Some("hero-video"));
        assert!(v.is_autoplay());
        assert!(!v.has_controls());
    }

    #[test]
    fn video_negative_size_clamps_to_zero() {
        let v = Video::default().width(-5.0).height(-7.0);
        assert_eq!(v.get_width(), Some(0.0));
        assert_eq!(v.get_height(), Some(0.0));
    }

    #[test]
    fn video_autoplay_decides_initial_state() {
        assert_eq!(Video::new("a").autoplay(true).state(), VideoState::Playing);
        assert!(Video::new("a").autoplay(true).is_playing());
        assert_eq!(
            Video::new("a").autoplay(true).autoplay(false).state(),
            VideoState::Paused
        );
    }

    // ===== State machine =====

    #[test]
    fn video_play_pause_toggle_ended_transitions() {
        let mut v = Video::new("a");
        v.play();
        assert_eq!(v.state(), VideoState::Playing);
        v.pause();
        assert_eq!(v.state(), VideoState::Paused);
        v.toggle();
        assert!(v.is_playing());
        v.toggle();
        assert!(!v.is_playing());
        v.play();
        v.ended();
        assert_eq!(v.state(), VideoState::Ended);
        assert!(!v.is_playing());
    }

    #[test]
    fn video_pause_keeps_ended_and_toggle_replays() {
        let mut v = Video::new("a");
        v.ended();
        v.pause();
        assert_eq!(v.state(), VideoState::Ended);
        v.toggle();
        assert_eq!(v.state(), VideoState::Playing);
    }

    // ===== Events =====

    #[test]
    fn video_click_inside_toggles_and_emits() {
        let mut v = laid_out(named("a"));
        let inside = BOUNDS.center();

        let first = click(&mut v, inside);
        assert_eq!(toggled(&first), Some(VideoToggled { playing: true }));
        assert!(v.is_playing());

        let second = click(&mut v, inside);
        assert_eq!(toggled(&second), Some(VideoToggled { playing: false }));
        assert!(!v.is_playing());
    }

    #[test]
    fn video_click_outside_does_nothing() {
        let mut v = laid_out(named("a"));
        let outside = Point::new(BOUNDS.x - 1.0, BOUNDS.y - 1.0);
        assert!(click(&mut v, outside).is_none());
        assert_eq!(v.state(), VideoState::Paused);
    }

    #[test]
    fn video_press_and_release_must_both_be_inside() {
        let mut v = laid_out(named("a"));
        let inside = BOUNDS.center();
        let outside = Point::new(BOUNDS.x + BOUNDS.width + 5.0, BOUNDS.y);

        // Press inside, release outside.
        v.event(&Event::MouseDown {
            position: inside,
            button: MouseButton::Left,
        });
        let released = v.event(&Event::MouseUp {
            position: outside,
            button: MouseButton::Left,
        });
        assert!(released.is_none());

        // Press outside, release inside.
        v.event(&Event::MouseDown {
            position: outside,
            button: MouseButton::Left,
        });
        let released = v.event(&Event::MouseUp {
            position: inside,
            button: MouseButton::Left,
        });
        assert!(released.is_none());
        assert!(!v.is_playing());
    }

    #[test]
    fn video_non_left_button_and_mouse_leave_do_not_toggle() {
        let mut v = laid_out(named("a"));
        let inside = BOUNDS.center();

        v.event(&Event::MouseDown {
            position: inside,
            button: MouseButton::Right,
        });
        let up = v.event(&Event::MouseUp {
            position: inside,
            button: MouseButton::Right,
        });
        assert!(up.is_none());

        v.event(&Event::MouseDown {
            position: inside,
            button: MouseButton::Left,
        });
        assert!(v.event(&Event::MouseLeave).is_none());
        let up = v.event(&Event::MouseUp {
            position: inside,
            button: MouseButton::Left,
        });
        assert!(up.is_none());
        assert!(!v.is_playing());
    }

    #[test]
    fn video_space_and_enter_toggle_other_events_ignored() {
        let mut v = laid_out(named("a"));

        let space = v.event(&Event::key_up(Key::Space));
        assert_eq!(toggled(&space), Some(VideoToggled { playing: true }));
        let enter = v.event(&Event::key_up(Key::Enter));
        assert_eq!(toggled(&enter), Some(VideoToggled { playing: false }));

        assert!(v.event(&Event::key_down(Key::Space)).is_none());
        assert!(v.event(&Event::key_up(Key::A)).is_none());
        assert!(!v.is_playing());
    }

    // ===== Paint =====

    #[test]
    fn video_paint_paused_draws_placeholder_and_glyph() {
        let v = laid_out(named("a"));
        let cmds = paint_commands(&v);
        assert_eq!(cmds.len(), 2);
        assert!(matches!(
            &cmds[0],
            DrawCommand::Rect { bounds, style, .. }
                if *bounds == BOUNDS && style.fill == Some(Video::placeholder_color())
        ));
        assert!(matches!(
            &cmds[1],
            DrawCommand::Path { points, closed, .. } if *closed && points.len() == 3
        ));
    }

    #[test]
    fn video_paint_with_poster_uses_poster_colour() {
        let v = laid_out(named("a").poster("p.png"));
        assert!(matches!(
            &paint_commands(&v)[0],
            DrawCommand::Rect { style, .. } if style.fill == Some(Video::poster_color())
        ));
    }

    #[test]
    fn video_paint_playing_draws_frame_and_no_glyph() {
        let mut v = laid_out(named("a"));
        v.play();
        let cmds = paint_commands(&v);
        assert_eq!(cmds.len(), 1);
        assert!(matches!(
            &cmds[0],
            DrawCommand::Rect { bounds, style, .. }
                if *bounds == BOUNDS && style.fill == Some(Video::frame_color())
        ));
    }

    #[test]
    fn video_paint_ended_shows_glyph_again() {
        let mut v = laid_out(named("a"));
        v.play();
        v.ended();
        let cmds = paint_commands(&v);
        assert_eq!(cmds.len(), 2);
        assert!(matches!(cmds[1], DrawCommand::Path { .. }));
    }

    #[test]
    fn video_glyph_is_centred_and_inside_bounds() {
        let v = laid_out(named("a"));
        let glyph = v.play_glyph();
        let centre = glyph_centre(&glyph);
        assert!((centre.x - BOUNDS.center().x).abs() < 0.5);
        assert!((centre.y - BOUNDS.center().y).abs() < 0.5);
        for p in &glyph {
            assert!(contains(&BOUNDS, p, 0.0));
        }
    }

    #[test]
    fn video_glyph_in_degenerate_bounds_collapses_to_centre() {
        let mut v = named("a");
        v.layout(Rect::new(5.0, 5.0, 0.0, 0.0));
        let glyph = v.play_glyph();
        for p in &glyph {
            assert!((p.x - 5.0).abs() < f32::EPSILON);
            assert!((p.y - 5.0).abs() < f32::EPSILON);
        }
        let mut v = named("a");
        v.layout(Rect::new(0.0, 0.0, -10.0, 40.0));
        let glyph = v.play_glyph();
        assert!((glyph[0].x - glyph[2].x).abs() < f32::EPSILON);
    }

    // ===== Widget =====

    #[test]
    fn video_measure_uses_default_and_explicit_size() {
        let loose = Constraints::loose(Size::new(1000.0, 1000.0));
        assert_eq!(Video::new("a").measure(loose), Size::new(320.0, 180.0));
        assert_eq!(
            Video::new("a").size(200.0, 100.0).measure(loose),
            Size::new(200.0, 100.0)
        );
        let tight = Constraints::loose(Size::new(150.0, 90.0));
        assert_eq!(Video::new("a").measure(tight), Size::new(150.0, 90.0));
    }

    #[test]
    fn video_layout_stores_bounds_and_returns_size() {
        let mut v = named("a");
        let result = v.layout(BOUNDS);
        assert_eq!(result.size, BOUNDS.size());
        assert_eq!(v.bounds, BOUNDS);
    }

    #[test]
    fn video_widget_metadata() {
        let mut v = named("a");
        assert_eq!(Widget::type_id(&v), TypeId::of::<Video>());
        assert!(v.children().is_empty());
        assert!(v.children_mut().is_empty());
        assert!(v.is_interactive());
        assert!(v.is_focusable());
        assert_eq!(v.accessible_role(), AccessibleRole::Generic);
        assert_eq!(Widget::test_id(&v), None);
    }

    #[test]
    fn video_accessible_name_falls_back_to_alt() {
        let v = Video::new("a").alt("Alt text");
        assert_eq!(Widget::accessible_name(&v), Some("Alt text"));
        let v = Video::new("a").alt("Alt text").accessible_name("Named");
        assert_eq!(Widget::accessible_name(&v), Some("Named"));
        let v = Video::new("a").alt("Alt text").accessible_name("");
        assert_eq!(Widget::accessible_name(&v), Some("Alt text"));
        assert_eq!(Widget::accessible_name(&Video::new("a")), None);
    }

    // ===== Brick =====

    #[test]
    fn video_brick_metadata() {
        let v = named("a");
        assert_eq!(v.brick_name(), "Video");
        assert!(v.assertions().contains(&BrickAssertion::MaxLatencyMs(16)));
        assert_eq!(v.budget(), BrickBudget::uniform(16));
    }

    #[test]
    fn video_verify_fails_on_empty_src() {
        let result = named("").verify();
        assert!(!result.is_valid());
        assert_eq!(result.failed.len(), 1);
        assert!(result.failed[0].1.contains("src"));
    }

    #[test]
    fn video_verify_fails_without_accessible_name() {
        let result = Video::new("a.mp4").verify();
        assert!(!result.is_valid());
        assert_eq!(result.failed.len(), 1);
        assert!(result.failed[0].1.contains("accessible name"));
        // An empty name is no name.
        assert!(!Video::new("a.mp4").accessible_name("").verify().is_valid());
        // A blank name or blank alt text is no name either.
        assert!(!Video::new("a.mp4")
            .accessible_name("   ")
            .verify()
            .is_valid());
        assert!(!Video::new("a.mp4").alt("  ").verify().is_valid());
        assert_eq!(
            result.failed[0].0,
            BrickAssertion::ElementPresent("accessible-name".into())
        );
        // The widget reports the bounds it was laid out with.
        let mut v = Video::new("a.mp4");
        v.layout(Rect::new(3.0, 4.0, 50.0, 60.0));
        assert_eq!(Widget::bounds(&v), Rect::new(3.0, 4.0, 50.0, 60.0));
    }

    #[test]
    fn video_verify_reports_both_failures() {
        let result = Video::default().verify();
        assert_eq!(result.failed.len(), 2);
    }

    #[test]
    fn video_verify_passes_in_every_state() {
        let mut v = named("a.mp4");
        assert!(v.verify().is_valid());
        v.play();
        assert!(v.verify().is_valid());
        v.ended();
        assert!(v.verify().is_valid());
        v.pause();
        assert!(v.verify().is_valid());
        // Alt text alone satisfies the name requirement.
        assert!(Video::new("a.mp4").alt("Intro").verify().is_valid());
    }

    #[test]
    fn video_to_html_reflects_configuration() {
        let v = Video::new("a.mp4")
            .poster("a.png")
            .autoplay(true)
            .accessible_name("Intro");
        let html = v.to_html();
        assert!(html.contains(r#"src="a.mp4""#));
        assert!(html.contains(r#"poster="a.png""#));
        assert!(html.contains(" controls"));
        assert!(html.contains(" autoplay"));
        assert!(html.contains(r#"aria-label="Intro""#));

        let bare = Video::new("b.mp4").controls(false).to_html();
        assert!(!bare.contains("poster"));
        assert!(!bare.contains("controls"));
        assert!(!bare.contains("autoplay"));
        assert!(bare.contains(r#"aria-label="""#));
    }

    #[test]
    fn video_to_html_escapes_attributes() {
        let html = Video::new(r#"a"b&c<d>.mp4"#).to_html();
        assert!(html.contains("a&quot;b&amp;c&lt;d&gt;.mp4"));
        let with_poster = named("a").poster(r#"p"><x>.png"#).to_html();
        assert!(with_poster.contains(r#"poster="p&quot;&gt;&lt;x&gt;.png""#));
        let with_name = Video::new("a").accessible_name(r#"n"<b>"#).to_html();
        assert!(with_name.contains("n&quot;&lt;b&gt;"));
    }

    #[test]
    fn video_to_css_declares_the_class() {
        assert!(named("a").to_css().contains(".brick-video"));
    }

    // ===== Serde / derives =====

    #[test]
    fn video_serde_roundtrip_keeps_state_and_drops_runtime_fields() {
        let mut v = named("a.mp4").poster("p.png").size(10.0, 20.0);
        v.layout(BOUNDS);
        v.play();
        let json = serde_json::to_string(&v).unwrap();
        let back: Video = serde_json::from_str(&json).unwrap();
        assert_eq!(back.get_src(), "a.mp4");
        assert_eq!(back.get_poster(), Some("p.png"));
        assert_eq!(back.state(), VideoState::Playing);
        assert_eq!(back.bounds, Rect::default());

        let state_json = serde_json::to_string(&VideoState::Ended).unwrap();
        let state: VideoState = serde_json::from_str(&state_json).unwrap();
        assert_eq!(state, VideoState::Ended);
    }

    #[test]
    fn video_debug_and_clone_are_usable() {
        let v = named("a.mp4");
        let copy = v.clone();
        assert!(format!("{copy:?}").contains("a.mp4"));
        let msg = VideoToggled { playing: true };
        assert_eq!(msg.clone(), msg);
        assert!(format!("{msg:?}").contains("playing"));
    }

    // ===== Composition =====

    #[test]
    fn video_inside_modal_lays_out_and_paints_inside_bounds() {
        // Modal::content takes any `impl Widget`, so the Video is placed in
        // the modal's content slot directly.
        let mut modal = Modal::new().content(named("a.mp4"));
        modal.show();
        let viewport = Rect::new(0.0, 0.0, 800.0, 600.0);
        modal.layout(viewport);

        let mut canvas = RecordingCanvas::new();
        modal.paint(&mut canvas);
        let cmds = canvas.commands();

        let content = modal.content_bounds();
        let glyph_idx = cmds
            .iter()
            .position(|c| matches!(c, DrawCommand::Path { .. }))
            .expect("the paused Video must paint a glyph path");
        let points = cmds
            .iter()
            .find_map(|c| match c {
                DrawCommand::Path { points, .. } => Some(points),
                _ => None,
            })
            .expect("the glyph path");
        // The last rect painted before the glyph is the video backdrop.
        let frame = cmds[..=glyph_idx]
            .iter()
            .filter_map(|c| match c {
                DrawCommand::Rect { bounds, .. } => Some(bounds),
                _ => None,
            })
            .next_back()
            .expect("the video backdrop rect");

        assert!(rect_inside(&content, frame, 0.001));
        for p in points {
            assert!(contains(frame, p, 0.001));
        }
        let centre = glyph_centre(points);
        assert!((centre.x - frame.center().x).abs() < 0.5);
        assert!((centre.y - frame.center().y).abs() < 0.5);
    }

    // ===== Overflow-proof size and bounds (APR-6897 review, defect 3) =====

    /// Documented coordinate limit: independent of the implementation's constant.
    const LIMIT: f32 = 1.0e15;

    fn oracle_coord(value: f32) -> f32 {
        if value.is_finite() {
            value.clamp(-LIMIT, LIMIT)
        } else {
            0.0
        }
    }

    /// The rectangle every emitted draw must stay in.
    fn oracle_bounds(b: Rect) -> Rect {
        Rect::new(
            oracle_coord(b.x),
            oracle_coord(b.y),
            oracle_coord(b.width).max(0.0),
            oracle_coord(b.height).max(0.0),
        )
    }

    fn point_finite_inside(p: &Point, outer: &Rect) -> bool {
        p.x.is_finite() && p.y.is_finite() && contains(outer, p, 0.0)
    }

    fn rect_finite_inside(r: &Rect, outer: &Rect) -> bool {
        [r.x, r.y, r.width, r.height].iter().all(|v| v.is_finite()) && rect_inside(outer, r, 0.0)
    }

    /// Every rect and every glyph vertex in `cmds` is finite and inside `outer`.
    fn commands_finite_inside(cmds: &[DrawCommand], outer: &Rect) -> bool {
        cmds.iter().all(|cmd| match cmd {
            DrawCommand::Rect { bounds, .. } => rect_finite_inside(bounds, outer),
            DrawCommand::Path { points, .. } => {
                !points.is_empty() && points.iter().all(|p| point_finite_inside(p, outer))
            }
            _ => false,
        })
    }

    #[test]
    fn commands_helper_rejects_anything_but_rects_and_paths() {
        let circle = DrawCommand::filled_circle(Point::new(1.0, 1.0), 1.0, Color::WHITE);
        assert!(!commands_finite_inside(&[circle], &BOUNDS));
        let empty_path = DrawCommand::Path {
            points: Vec::new(),
            closed: true,
            style: presentar_core::draw::StrokeStyle::default(),
        };
        assert!(!commands_finite_inside(&[empty_path], &BOUNDS));
    }

    /// Lay `bounds` out, then check the glyph and the paint in both states.
    fn layout_paint_is_finite_inside(bounds: Rect) -> bool {
        let expected = oracle_bounds(bounds);
        let mut v = named("a");
        let result = v.layout(bounds);
        let mut ok = result.size.width.is_finite()
            && result.size.height.is_finite()
            && v.play_glyph()
                .iter()
                .all(|p| point_finite_inside(p, &expected));
        ok &= commands_finite_inside(&paint_commands(&v), &expected);
        v.play();
        ok &= commands_finite_inside(&paint_commands(&v), &expected);
        ok
    }

    #[test]
    fn video_reviewer_repro_f32_max_bounds_paint_finite_and_inside() {
        let bounds = Rect::new(f32::MAX, f32::MAX, f32::MAX, f32::MAX);
        let mut v = named("a");
        v.layout(bounds);
        let expected = oracle_bounds(bounds);
        assert_eq!(Widget::bounds(&v), expected);
        assert!(v
            .play_glyph()
            .iter()
            .all(|p| point_finite_inside(p, &expected)));
        let cmds = paint_commands(&v);
        assert_eq!(cmds.len(), 2);
        assert!(commands_finite_inside(&cmds, &expected), "{cmds:?}");
    }

    #[test]
    fn video_reviewer_repro_negative_extents_collapse_inside_the_origin() {
        let mut v = named("a");
        let result = v.layout(Rect::new(10.0, 20.0, -50.0, -40.0));
        let zero = Rect::new(10.0, 20.0, 0.0, 0.0);
        assert_eq!(result.size, Size::new(0.0, 0.0));
        assert_eq!(Widget::bounds(&v), zero);
        for p in v.play_glyph() {
            assert_eq!(p, Point::new(10.0, 20.0));
        }
        let cmds = paint_commands(&v);
        assert!(matches!(
            &cmds[0],
            DrawCommand::Rect { bounds, .. } if *bounds == zero
        ));
        assert!(commands_finite_inside(&cmds, &zero));
    }

    #[test]
    fn video_layout_extreme_bounds_table_paints_finite_and_inside() {
        let extremes = [
            f32::MAX,
            f32::MIN,
            f32::INFINITY,
            f32::NEG_INFINITY,
            f32::NAN,
            f32::MIN_POSITIVE,
            -0.0,
            LIMIT,
            -LIMIT,
            3.0e15,
            -50.0,
        ];
        for &a in &extremes {
            for &b in &extremes {
                for bounds in [
                    Rect::new(a, a, b, b),
                    Rect::new(a, b, b, a),
                    Rect::new(10.0, 20.0, a, b),
                    Rect::new(a, b, 100.0, 50.0),
                ] {
                    assert!(layout_paint_is_finite_inside(bounds), "{bounds:?}");
                }
            }
        }
    }

    #[test]
    fn video_layout_keeps_ordinary_bounds_untouched() {
        assert!(layout_paint_is_finite_inside(BOUNDS));
        let mut v = named("a");
        assert_eq!(v.layout(BOUNDS).size, BOUNDS.size());
        assert_eq!(Widget::bounds(&v), BOUNDS);
    }

    #[test]
    fn video_non_finite_size_counts_as_absent() {
        for bad in [f32::INFINITY, f32::NEG_INFINITY, f32::NAN] {
            let v = Video::new("a").size(bad, bad);
            assert_eq!(v.get_width(), None, "{bad}");
            assert_eq!(v.get_height(), None, "{bad}");
            assert_eq!(
                v.measure(Constraints::unbounded()),
                Size::new(320.0, 180.0),
                "{bad}"
            );
        }
    }

    #[test]
    fn video_huge_size_is_clamped_to_the_coordinate_limit() {
        let v = Video::new("a").size(f32::MAX, 1.0e30);
        assert_eq!(v.get_width(), Some(LIMIT));
        assert_eq!(v.get_height(), Some(LIMIT));
        assert_eq!(v.measure(Constraints::unbounded()), Size::new(LIMIT, LIMIT));
    }

    #[test]
    fn video_measure_is_finite_for_degenerate_constraints() {
        let v = Video::new("a").size(f32::MAX, 10.0);
        for c in [
            Constraints::unbounded(),
            Constraints::new(f32::INFINITY, f32::INFINITY, 0.0, 5.0),
            Constraints::new(0.0, f32::NAN, f32::NAN, f32::INFINITY),
            Constraints::new(9.0, 3.0, 9.0, 3.0),
        ] {
            let s = v.measure(c);
            assert!(
                s.width.is_finite() && s.height.is_finite(),
                "{c:?} -> {s:?}"
            );
        }
    }

    #[test]
    fn video_verify_flags_invalid_bounds() {
        for bad in [
            Rect::new(f32::NAN, 0.0, 10.0, 10.0),
            Rect::new(0.0, f32::INFINITY, 10.0, 10.0),
            Rect::new(0.0, 0.0, -1.0, 10.0),
            Rect::new(0.0, 0.0, 10.0, -1.0),
            Rect::new(0.0, 0.0, f32::NAN, 10.0),
            Rect::new(0.0, 0.0, 10.0, f32::NEG_INFINITY),
        ] {
            let mut v = named("a.mp4");
            v.bounds = bad;
            let result = v.verify();
            assert!(!result.is_valid(), "{bad:?}");
            assert_eq!(result.failed.len(), 1, "{bad:?}");
            assert_eq!(
                result.failed[0].0,
                BrickAssertion::ElementPresent("valid-bounds".into())
            );
            assert!(result.failed[0].1.contains("bounds"));
        }
    }

    #[test]
    fn video_verify_accepts_laid_out_bounds_even_when_the_input_was_wild() {
        let mut v = named("a.mp4");
        v.layout(BOUNDS);
        let result = v.verify();
        assert!(result.is_valid());
        assert!(result
            .passed
            .contains(&BrickAssertion::ElementPresent("valid-bounds".into())));
        // layout sanitises, so wild input still ends in valid bounds.
        v.layout(Rect::new(f32::NAN, f32::MAX, -1.0, f32::INFINITY));
        assert!(v.verify().is_valid());
    }

    proptest! {
        #[test]
        fn video_layout_paint_finite_and_inside_any_f32_bounds(
            x in proptest::num::f32::ANY,
            y in proptest::num::f32::ANY,
            w in proptest::num::f32::ANY,
            h in proptest::num::f32::ANY,
        ) {
            prop_assert!(layout_paint_is_finite_inside(Rect::new(x, y, w, h)));
        }

        #[test]
        fn video_measure_finite_for_any_size_and_constraints(
            w in proptest::num::f32::ANY,
            h in proptest::num::f32::ANY,
            a in proptest::num::f32::ANY,
            b in proptest::num::f32::ANY,
            c in proptest::num::f32::ANY,
            d in proptest::num::f32::ANY,
        ) {
            let s = Video::new("a").size(w, h).measure(Constraints::new(a, b, c, d));
            prop_assert!(s.width.is_finite() && s.height.is_finite());
        }
    }

    // ===== Properties =====

    proptest! {
        #[test]
        fn video_paint_stays_inside_any_bounds(
            x in -1000.0f32..1000.0,
            y in -1000.0f32..1000.0,
            w in 0.0f32..2000.0,
            h in 0.0f32..2000.0,
        ) {
            let bounds = Rect::new(x, y, w, h);
            let mut v = named("a");
            v.layout(bounds);

            // Not playing: backdrop + centred glyph, all inside bounds.
            let cmds = paint_commands(&v);
            prop_assert_eq!(cmds.len(), 2);
            for cmd in &cmds {
                match cmd {
                    DrawCommand::Rect { bounds: r, .. } => {
                        prop_assert!(rect_inside(&bounds, r, 1e-3));
                    }
                    DrawCommand::Path { points, .. } => {
                        for p in points {
                            prop_assert!(contains(&bounds, p, 1e-3));
                        }
                        let centre = glyph_centre(points);
                        prop_assert!((centre.x - bounds.center().x).abs() < 0.5);
                        prop_assert!((centre.y - bounds.center().y).abs() < 0.5);
                    }
                    other => prop_assert!(false, "unexpected command {:?}", other),
                }
            }

            // Playing: exactly the frame rect, no glyph.
            v.play();
            let cmds = paint_commands(&v);
            prop_assert_eq!(cmds.len(), 1);
            match &cmds[0] {
                DrawCommand::Rect { bounds: r, .. } => {
                    prop_assert!(rect_inside(&bounds, r, 1e-3));
                }
                other => prop_assert!(false, "unexpected command {:?}", other),
            }
        }
    }
}
