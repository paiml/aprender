//! Rust-side mount: a widget tree, its input events and its frame schedule, with no script
//! on the page (aprender#6893, aprender#6892).
//!
//! `Mounted` is the part that decides: it takes events, tracks whether a repaint is owed
//! and produces the draw commands of one frame. It touches no browser API, so it is tested
//! natively. The wasm32 shell that feeds it from DOM listeners and
//! `requestAnimationFrame` is `browser::entry`.
//!
//! Contract: `contracts/present-mount-frame-v1.yaml`.

use presentar_core::draw::DrawCommand;
use presentar_core::{Constraints, Event, RecordingCanvas, Rect, Size, Widget};
use std::any::Any;

/// What the frame loop does at one tick.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FrameAction {
    /// Paint this tick and ask for another one.
    Repaint,
    /// Paint nothing and stop asking for frames until an event arrives.
    Stop,
}

/// The whole scheduling decision: repaint while something is dirty or an animation runs,
/// otherwise stop.
#[must_use]
pub const fn schedule(dirty: bool, animating: bool) -> FrameAction {
    if dirty || animating {
        FrameAction::Repaint
    } else {
        FrameAction::Stop
    }
}

/// The outcome of one frame tick.
#[derive(Debug, Clone, PartialEq)]
pub enum Frame {
    /// Nothing was owed; the loop stops.
    Idle,
    /// The tree was laid out and painted; these are its draw commands.
    Painted(Vec<DrawCommand>),
    /// The root brick failed its assertions, so nothing was painted (PROBAR-SPEC-009).
    Blocked(String),
}

impl Frame {
    /// Whether the loop asks for another tick after this one.
    #[must_use]
    pub const fn wants_next(&self) -> bool {
        !matches!(self, Self::Idle)
    }
}

/// A widget tree mounted on a surface of a known size.
pub struct Mounted {
    root: Box<dyn Widget>,
    size: Size,
    dirty: bool,
    animating: bool,
    messages: Vec<Box<dyn Any + Send>>,
    painted: u64,
}

impl Mounted {
    /// Mount `root` on a `width` x `height` surface. The tree is laid out at once, so the
    /// first event already hits the right bounds, and the first frame paints.
    #[must_use]
    pub fn new(root: Box<dyn Widget>, width: f32, height: f32) -> Self {
        let mut mounted = Self {
            root,
            size: Size::new(width, height),
            dirty: true,
            animating: false,
            messages: Vec::new(),
            painted: 0,
        };
        mounted.relayout();
        mounted
    }

    fn relayout(&mut self) {
        let wanted = self.root.measure(Constraints::loose(self.size));
        self.root
            .layout(Rect::new(0.0, 0.0, wanted.width, wanted.height));
    }

    /// Give one input event to the tree. Any message a widget emits is queued for
    /// `take_messages`. Every event owes a repaint, because widgets change hover and
    /// pressed state without emitting a message. Returns whether a message was emitted.
    pub fn dispatch(&mut self, event: &Event) -> bool {
        if let Event::Resize { width, height } = event {
            self.size = Size::new(*width, *height);
            self.relayout();
        }
        self.dirty = true;
        match self.root.event(event) {
            Some(message) => {
                self.messages.push(message);
                true
            }
            None => false,
        }
    }

    /// Run one frame tick.
    pub fn frame(&mut self) -> Frame {
        if schedule(self.dirty, self.animating) == FrameAction::Stop {
            return Frame::Idle;
        }
        self.dirty = false;
        if !self.root.can_render() {
            let reasons: Vec<String> = self
                .root
                .verify()
                .failed
                .iter()
                .map(|(assertion, reason)| format!("{assertion:?}: {reason}"))
                .collect();
            return Frame::Blocked(format!(
                "brick '{}' failed verification: {}",
                self.root.brick_name(),
                reasons.join(", ")
            ));
        }
        self.relayout();
        let mut canvas = RecordingCanvas::new();
        self.root.paint(&mut canvas);
        self.painted += 1;
        Frame::Painted(canvas.take_commands())
    }

    /// Repaint on every tick (a waveform, a video overlay) or only when dirty.
    pub fn set_animating(&mut self, animating: bool) {
        self.animating = animating;
    }

    /// Owe a repaint without an event (the app changed the tree).
    pub fn request_repaint(&mut self) {
        self.dirty = true;
    }

    /// Whether the next tick would paint.
    #[must_use]
    pub const fn needs_frame(&self) -> bool {
        matches!(schedule(self.dirty, self.animating), FrameAction::Repaint)
    }

    /// Messages emitted by widgets since the last call, oldest first.
    pub fn take_messages(&mut self) -> Vec<Box<dyn Any + Send>> {
        std::mem::take(&mut self.messages)
    }

    /// Frames painted so far.
    #[must_use]
    pub const fn frames_painted(&self) -> u64 {
        self.painted
    }

    /// The mounted tree, for the app to change between frames.
    pub fn root_mut(&mut self) -> &mut dyn Widget {
        self.root.as_mut()
    }

    /// Replace the tree (the app rebuilt it from new state); lays it out and owes a repaint.
    pub fn set_root(&mut self, root: Box<dyn Widget>) {
        self.root = root;
        self.relayout();
        self.dirty = true;
    }

    /// The surface size the tree is laid out for.
    #[must_use]
    pub const fn size(&self) -> Size {
        self.size
    }
}

/// The tree of the built-in counter page: a title, the count and one "+" button.
#[must_use]
pub fn counter_tree(count: i32) -> Box<dyn Widget> {
    use presentar_widgets::{Button, Column, Text};
    Box::new(
        Column::new()
            .gap(20.0)
            .child(Text::new("Counter").font_size(28.0))
            .child(Text::new(format!("Count: {count}")).font_size(48.0))
            .child(Button::new("+").padding(12.0)),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use presentar_core::{MouseButton, Point};
    use presentar_widgets::button::ButtonClicked;
    use presentar_widgets::{Button, Column, Text};

    fn tree() -> Box<dyn Widget> {
        Box::new(
            Column::new()
                .child(Text::new("Counter"))
                .child(Button::new("+")),
        )
    }

    /// Find the button by clicking a grid. `Widget::bounds` is not overridden by `Button`,
    /// so the tree cannot be asked. The grid starts at (2, 2): a tree that was never laid
    /// out has a zero rect at the origin, which this scan must not hit.
    fn find_button(mounted: &mut Mounted) -> Option<Point> {
        for y in (2..600).step_by(4) {
            for x in (2..800).step_by(4) {
                let at = Point::new(x as f32, y as f32);
                if click(mounted, at) {
                    let _ = mounted.take_messages();
                    return Some(at);
                }
            }
        }
        None
    }

    fn click(mounted: &mut Mounted, at: Point) -> bool {
        let down = mounted.dispatch(&Event::MouseDown {
            position: at,
            button: MouseButton::Left,
        });
        let up = mounted.dispatch(&Event::MouseUp {
            position: at,
            button: MouseButton::Left,
        });
        down || up
    }

    #[test]
    fn the_button_probe_finds_nothing_on_a_tree_without_a_button() {
        let mut mounted = Mounted::new(Box::new(Text::new("no button here")), 800.0, 600.0);
        assert!(find_button(&mut mounted).is_none());
    }

    #[test]
    fn schedule_truth_table() {
        assert_eq!(schedule(false, false), FrameAction::Stop);
        assert_eq!(schedule(true, false), FrameAction::Repaint);
        assert_eq!(schedule(false, true), FrameAction::Repaint);
        assert_eq!(schedule(true, true), FrameAction::Repaint);
    }

    #[test]
    fn first_frame_paints_then_the_loop_stops() {
        let mut mounted = Mounted::new(tree(), 800.0, 600.0);
        assert!(mounted.needs_frame());
        let first = mounted.frame();
        assert!(matches!(&first, Frame::Painted(commands) if !commands.is_empty()));
        assert!(first.wants_next());
        assert_eq!(mounted.frame(), Frame::Idle);
        assert!(!Frame::Idle.wants_next());
        assert!(!mounted.needs_frame());
        assert_eq!(mounted.frames_painted(), 1);
    }

    #[test]
    fn a_click_on_the_button_emits_its_message_and_owes_a_repaint() {
        let mut mounted = Mounted::new(tree(), 800.0, 600.0);
        let _ = mounted.frame();
        let at = find_button(&mut mounted).expect("the button is on the surface");
        assert!(click(&mut mounted, at));
        let messages = mounted.take_messages();
        assert_eq!(messages.len(), 1);
        assert!(messages[0].downcast_ref::<ButtonClicked>().is_some());
        assert!(mounted.take_messages().is_empty());
        assert!(matches!(mounted.frame(), Frame::Painted(_)));
        assert_eq!(mounted.frames_painted(), 2);
    }

    #[test]
    fn a_click_beside_the_button_emits_nothing_but_still_repaints() {
        let mut mounted = Mounted::new(tree(), 800.0, 600.0);
        let _ = mounted.frame();
        assert!(!click(&mut mounted, Point::new(790.0, 590.0)));
        assert!(mounted.take_messages().is_empty());
        assert!(mounted.needs_frame());
    }

    #[test]
    fn the_tree_is_laid_out_before_the_first_frame() {
        let mut mounted = Mounted::new(tree(), 800.0, 600.0);
        assert!(find_button(&mut mounted).is_some());
    }

    #[test]
    fn animating_repaints_every_tick_until_switched_off() {
        let mut mounted = Mounted::new(tree(), 800.0, 600.0);
        mounted.set_animating(true);
        for _ in 0..3 {
            assert!(matches!(mounted.frame(), Frame::Painted(_)));
        }
        assert_eq!(mounted.frames_painted(), 3);
        mounted.set_animating(false);
        assert_eq!(mounted.frame(), Frame::Idle);
    }

    #[test]
    fn request_repaint_owes_exactly_one_frame() {
        let mut mounted = Mounted::new(tree(), 800.0, 600.0);
        let _ = mounted.frame();
        mounted.request_repaint();
        assert!(matches!(mounted.frame(), Frame::Painted(_)));
        assert_eq!(mounted.frame(), Frame::Idle);
    }

    #[test]
    fn resize_changes_the_surface_and_repaints() {
        let mut mounted = Mounted::new(tree(), 800.0, 600.0);
        let _ = mounted.frame();
        assert!(!mounted.dispatch(&Event::Resize {
            width: 320.0,
            height: 240.0,
        }));
        assert_eq!(mounted.size(), Size::new(320.0, 240.0));
        assert_eq!(mounted.root_mut().children().len(), 2);
        assert!(matches!(mounted.frame(), Frame::Painted(_)));
    }

    #[test]
    fn set_root_swaps_the_tree_and_owes_a_repaint() {
        let mut mounted = Mounted::new(counter_tree(0), 800.0, 600.0);
        let before = mounted.frame();
        assert_eq!(mounted.frame(), Frame::Idle);
        mounted.set_root(counter_tree(1));
        let after = mounted.frame();
        assert!(matches!(after, Frame::Painted(_)));
        assert_ne!(
            before, after,
            "the count text changed, so the commands differ"
        );
    }

    #[test]
    fn a_swapped_tree_is_laid_out_before_its_first_event() {
        let mut mounted = Mounted::new(counter_tree(0), 800.0, 600.0);
        let _ = mounted.frame();
        mounted.set_root(counter_tree(1));
        assert!(
            find_button(&mut mounted).is_some(),
            "a click right after the swap, before any frame, reaches the new button"
        );
    }

    #[test]
    fn a_brick_that_fails_verification_is_not_painted_and_does_not_spin() {
        let mut mounted = Mounted::new(Box::new(Button::new("")), 800.0, 600.0);
        let blocked = mounted.frame();
        assert!(
            matches!(&blocked, Frame::Blocked(why) if why.contains("no label")),
            "{blocked:?}"
        );
        assert!(blocked.wants_next());
        assert_eq!(mounted.frames_painted(), 0);
        assert_eq!(mounted.frame(), Frame::Idle);
    }
}
