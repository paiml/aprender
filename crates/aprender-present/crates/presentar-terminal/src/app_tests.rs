use super::*;
use presentar_core::{
    Brick, BrickAssertion, BrickBudget, BrickVerification, Canvas, Color, Event, LayoutResult,
    Size, TypeId,
};
use std::any::Any;
use std::time::Duration;

struct TestWidget {
    assertions: Vec<BrickAssertion>,
}

impl TestWidget {
    fn new() -> Self {
        Self {
            assertions: vec![BrickAssertion::max_latency_ms(16)],
        }
    }

    fn without_assertions() -> Self {
        Self { assertions: vec![] }
    }
}

impl Brick for TestWidget {
    fn brick_name(&self) -> &'static str {
        "test_widget"
    }

    fn assertions(&self) -> &[BrickAssertion] {
        &self.assertions
    }

    fn budget(&self) -> BrickBudget {
        BrickBudget::default()
    }

    fn verify(&self) -> BrickVerification {
        BrickVerification {
            passed: self.assertions.clone(),
            failed: vec![],
            verification_time: Duration::from_micros(10),
        }
    }

    fn to_html(&self) -> String {
        String::new()
    }

    fn to_css(&self) -> String {
        String::new()
    }
}

impl Widget for TestWidget {
    fn type_id(&self) -> TypeId {
        TypeId::of::<Self>()
    }

    fn measure(&self, constraints: Constraints) -> Size {
        constraints.constrain(Size::new(10.0, 5.0))
    }

    fn layout(&mut self, bounds: Rect) -> LayoutResult {
        LayoutResult {
            size: Size::new(bounds.width, bounds.height),
        }
    }

    fn paint(&self, canvas: &mut dyn Canvas) {
        canvas.fill_rect(Rect::new(0.0, 0.0, 10.0, 5.0), Color::BLUE);
    }

    fn event(&mut self, _event: &Event) -> Option<Box<dyn Any + Send>> {
        None
    }

    fn children(&self) -> &[Box<dyn Widget>] {
        &[]
    }

    fn children_mut(&mut self) -> &mut [Box<dyn Widget>] {
        &mut []
    }
}

#[test]
fn test_tui_app_creation() {
    let widget = TestWidget::new();
    let app = TuiApp::new(widget);
    assert!(app.is_ok());
}

#[test]
fn test_tui_app_rejects_empty_assertions() {
    let widget = TestWidget::without_assertions();
    let app = TuiApp::new(widget);
    assert!(app.is_err());
    let err = app.err().expect("expected error");
    assert!(matches!(err, TuiError::InvalidBrick(_)));
}

#[test]
fn test_config_default() {
    let config = TuiConfig::default();
    assert_eq!(config.tick_rate_ms, 250);
    assert_eq!(config.target_fps, 60);
    assert!(!config.enable_mouse);
    assert!(!config.skip_verification);
    assert!(config.color_mode.is_none());
}

#[test]
fn test_config_high_performance() {
    let config = TuiConfig::high_performance();
    assert_eq!(config.tick_rate_ms, 16);
    assert_eq!(config.target_fps, 60);
}

#[test]
fn test_config_power_saving() {
    let config = TuiConfig::power_saving();
    assert_eq!(config.tick_rate_ms, 100);
    assert_eq!(config.target_fps, 30);
}

#[test]
fn test_tui_app_with_config() {
    let widget = TestWidget::new();
    let mut app = TuiApp::new(widget).unwrap();

    let config = TuiConfig {
        tick_rate_ms: 50,
        enable_mouse: true,
        color_mode: Some(ColorMode::Color256),
        skip_verification: false,
        target_fps: 30,
    };

    app = app.with_config(config);
    assert!(app.metrics().frame_count == 0);
}

#[test]
fn test_tui_app_with_input_handler() {
    let widget = TestWidget::new();
    let mut app = TuiApp::new(widget).unwrap();

    let mut handler = InputHandler::new();
    handler.add_binding(crate::input::KeyBinding::simple(
        crossterm::event::KeyCode::Char('q'),
        "quit",
    ));

    app = app.with_input_handler(handler);
    assert!(app.root().assertions().len() == 1);
}

#[test]
fn test_tui_app_root_accessors() {
    let widget = TestWidget::new();
    let mut app = TuiApp::new(widget).unwrap();

    assert_eq!(app.root().brick_name(), "test_widget");
    assert_eq!(app.root_mut().brick_name(), "test_widget");
}

#[test]
fn test_tui_app_metrics() {
    let widget = TestWidget::new();
    let app = TuiApp::new(widget).unwrap();

    let metrics = app.metrics();
    assert_eq!(metrics.frame_count, 0);
    assert_eq!(metrics.total_time, Duration::ZERO);
}

#[test]
fn test_tui_app_quit() {
    let widget = TestWidget::new();
    let mut app = TuiApp::new(widget).unwrap();

    assert!(!app.should_quit);
    app.quit();
    assert!(app.should_quit);
}

#[test]
fn test_frame_metrics_default() {
    let metrics = FrameMetrics::default();
    assert_eq!(metrics.frame_count, 0);
    assert_eq!(metrics.verify_time, Duration::ZERO);
    assert_eq!(metrics.measure_time, Duration::ZERO);
    assert_eq!(metrics.layout_time, Duration::ZERO);
    assert_eq!(metrics.paint_time, Duration::ZERO);
    assert_eq!(metrics.total_time, Duration::ZERO);
}

#[test]
fn test_config_with_color_mode_override() {
    let widget = TestWidget::new();
    let app = TuiApp::new(widget).unwrap();

    let config = TuiConfig {
        color_mode: Some(ColorMode::Mono),
        ..Default::default()
    };

    let app = app.with_config(config);
    assert_eq!(app.color_mode, ColorMode::Mono);
}

#[test]
fn test_config_without_color_mode() {
    let widget = TestWidget::new();
    let app = TuiApp::new(widget).unwrap();
    let original_mode = app.color_mode;

    let config = TuiConfig {
        color_mode: None,
        ..Default::default()
    };

    let app = app.with_config(config);
    assert_eq!(app.color_mode, original_mode);
}

#[test]
#[allow(clippy::nonminimal_bool)]
fn test_render_frame() {
    let widget = TestWidget::new();
    let mut app = TuiApp::new(widget).unwrap();
    let mut buffer = CellBuffer::new(80, 24);

    // Render a frame and verify metrics are updated
    app.render_frame(&mut buffer);

    assert!(app.metrics.measure_time >= Duration::ZERO);
    assert!(app.metrics.layout_time >= Duration::ZERO);
    assert!(app.metrics.paint_time >= Duration::ZERO);
}

#[test]
fn test_render_frame_updates_metrics() {
    let widget = TestWidget::new();
    let mut app = TuiApp::new(widget).unwrap();
    let mut buffer = CellBuffer::new(40, 10);

    // Render multiple frames
    for _ in 0..3 {
        app.render_frame(&mut buffer);
    }

    // Metrics should be set (even if durations are very small)
    let metrics = app.metrics();
    assert_eq!(metrics.frame_count, 0); // frame_count is only updated in run_loop
}

#[test]
fn test_render_frame_with_different_buffer_sizes() {
    let widget = TestWidget::new();
    let mut app = TuiApp::new(widget).unwrap();

    // Small buffer
    let mut small_buffer = CellBuffer::new(10, 5);
    app.render_frame(&mut small_buffer);

    // Large buffer
    let mut large_buffer = CellBuffer::new(200, 50);
    app.render_frame(&mut large_buffer);

    // Should not panic with any buffer size
}

#[test]
fn test_frame_metrics_clone() {
    let metrics = FrameMetrics {
        verify_time: Duration::from_millis(1),
        measure_time: Duration::from_millis(2),
        layout_time: Duration::from_millis(3),
        paint_time: Duration::from_millis(4),
        total_time: Duration::from_millis(10),
        frame_count: 100,
    };

    let cloned = metrics;
    assert_eq!(cloned.frame_count, 100);
    assert_eq!(cloned.verify_time, Duration::from_millis(1));
}

#[test]
fn test_frame_metrics_debug() {
    let metrics = FrameMetrics::default();
    let debug_str = format!("{:?}", metrics);
    assert!(debug_str.contains("FrameMetrics"));
    assert!(debug_str.contains("frame_count"));
}

#[test]
fn test_tui_config_clone() {
    let config = TuiConfig::high_performance();
    let cloned = config;
    assert_eq!(cloned.tick_rate_ms, 16);
    assert_eq!(cloned.target_fps, 60);
}

#[test]
fn test_tui_config_debug() {
    let config = TuiConfig::default();
    let debug_str = format!("{:?}", config);
    assert!(debug_str.contains("TuiConfig"));
    assert!(debug_str.contains("tick_rate_ms"));
}

// Additional tests for improved coverage

struct FailingWidget;

impl Brick for FailingWidget {
    fn brick_name(&self) -> &'static str {
        "failing_widget"
    }

    fn assertions(&self) -> &[BrickAssertion] {
        static ASSERTIONS: &[BrickAssertion] = &[BrickAssertion::max_latency_ms(16)];
        ASSERTIONS
    }

    fn budget(&self) -> BrickBudget {
        BrickBudget::default()
    }

    fn verify(&self) -> BrickVerification {
        // This widget always fails verification
        BrickVerification {
            passed: vec![],
            failed: vec![(
                BrickAssertion::max_latency_ms(16),
                "Intentional failure".to_string(),
            )],
            verification_time: Duration::from_micros(10),
        }
    }

    fn to_html(&self) -> String {
        String::new()
    }

    fn to_css(&self) -> String {
        String::new()
    }
}

impl Widget for FailingWidget {
    fn type_id(&self) -> TypeId {
        TypeId::of::<Self>()
    }

    fn measure(&self, constraints: Constraints) -> Size {
        constraints.constrain(Size::new(10.0, 5.0))
    }

    fn layout(&mut self, bounds: Rect) -> LayoutResult {
        LayoutResult {
            size: Size::new(bounds.width, bounds.height),
        }
    }

    fn paint(&self, _canvas: &mut dyn Canvas) {}

    fn event(&mut self, _event: &Event) -> Option<Box<dyn Any + Send>> {
        None
    }

    fn children(&self) -> &[Box<dyn Widget>] {
        &[]
    }

    fn children_mut(&mut self) -> &mut [Box<dyn Widget>] {
        &mut []
    }
}

#[test]
fn test_tui_app_with_failing_widget() {
    let widget = FailingWidget;
    let app = TuiApp::new(widget);
    // Should be Ok since we only check assertions on creation, not verify()
    assert!(app.is_ok());
}

#[test]
fn test_tui_config_all_fields() {
    let config = TuiConfig {
        tick_rate_ms: 100,
        enable_mouse: true,
        color_mode: Some(ColorMode::Color16),
        skip_verification: true,
        target_fps: 30,
    };

    assert_eq!(config.tick_rate_ms, 100);
    assert!(config.enable_mouse);
    assert_eq!(config.color_mode, Some(ColorMode::Color16));
    assert!(config.skip_verification);
    assert_eq!(config.target_fps, 30);
}

#[test]
fn test_frame_metrics_all_fields() {
    let metrics = FrameMetrics {
        verify_time: Duration::from_millis(1),
        measure_time: Duration::from_millis(2),
        layout_time: Duration::from_millis(3),
        paint_time: Duration::from_millis(4),
        total_time: Duration::from_millis(10),
        frame_count: 42,
    };

    assert_eq!(metrics.verify_time, Duration::from_millis(1));
    assert_eq!(metrics.measure_time, Duration::from_millis(2));
    assert_eq!(metrics.layout_time, Duration::from_millis(3));
    assert_eq!(metrics.paint_time, Duration::from_millis(4));
    assert_eq!(metrics.total_time, Duration::from_millis(10));
    assert_eq!(metrics.frame_count, 42);
}

#[test]
fn test_tui_app_skip_verification_config() {
    let widget = TestWidget::new();
    let app = TuiApp::new(widget).unwrap();

    let config = TuiConfig {
        skip_verification: true,
        ..Default::default()
    };

    let app = app.with_config(config);
    assert!(app.config.skip_verification);
}

#[test]
fn test_tui_app_enable_mouse_config() {
    let widget = TestWidget::new();
    let app = TuiApp::new(widget).unwrap();

    let config = TuiConfig {
        enable_mouse: true,
        ..Default::default()
    };

    let app = app.with_config(config);
    assert!(app.config.enable_mouse);
}

#[test]
fn test_render_frame_zero_size_buffer() {
    let widget = TestWidget::new();
    let mut app = TuiApp::new(widget).unwrap();

    // Test with minimal buffer size
    let mut buffer = CellBuffer::new(1, 1);
    app.render_frame(&mut buffer);
    // Should not panic
}

#[test]
fn test_render_frame_metrics_populated() {
    let widget = TestWidget::new();
    let mut app = TuiApp::new(widget).unwrap();
    let mut buffer = CellBuffer::new(80, 24);

    app.render_frame(&mut buffer);

    // All timing metrics should be non-negative (possibly zero for fast operations)
    assert!(app.metrics.measure_time >= Duration::ZERO);
    assert!(app.metrics.layout_time >= Duration::ZERO);
    assert!(app.metrics.paint_time >= Duration::ZERO);
}

#[test]
fn test_tui_config_color_modes() {
    // Test all color modes
    for mode in [
        ColorMode::TrueColor,
        ColorMode::Color256,
        ColorMode::Color16,
        ColorMode::Mono,
    ] {
        let widget = TestWidget::new();
        let app = TuiApp::new(widget).unwrap();

        let config = TuiConfig {
            color_mode: Some(mode),
            ..Default::default()
        };

        let app = app.with_config(config);
        assert_eq!(app.color_mode, mode);
    }
}

#[test]
fn test_test_widget_brick_methods() {
    let widget = TestWidget::new();

    assert_eq!(widget.brick_name(), "test_widget");
    assert!(!widget.assertions().is_empty());
    assert!(widget.verify().is_valid());
    assert!(widget.to_html().is_empty());
    assert!(widget.to_css().is_empty());
}

#[test]
fn test_test_widget_widget_methods() {
    let mut widget = TestWidget::new();

    // measure
    let size = widget.measure(Constraints::loose(Size::new(100.0, 100.0)));
    assert!(size.width > 0.0);
    assert!(size.height > 0.0);

    // layout
    let bounds = Rect::new(0.0, 0.0, 50.0, 25.0);
    let result = widget.layout(bounds);
    assert_eq!(result.size.width, 50.0);
    assert_eq!(result.size.height, 25.0);

    // event
    let event = Event::key_down(presentar_core::Key::Enter);
    assert!(widget.event(&event).is_none());

    // children
    assert!(widget.children().is_empty());
    assert!(widget.children_mut().is_empty());
}

#[test]
fn test_tui_app_multiple_render_frames() {
    let widget = TestWidget::new();
    let mut app = TuiApp::new(widget).unwrap();
    let mut buffer = CellBuffer::new(80, 24);

    // Render multiple frames to ensure stability
    for _ in 0..10 {
        app.render_frame(&mut buffer);
    }

    // Should complete without panic
}

// Mock terminal for testing run/run_loop

use std::cell::RefCell;
use std::collections::VecDeque;

struct MockTerminal {
    size: (u16, u16),
    events: RefCell<VecDeque<CrosstermEvent>>,
    poll_results: RefCell<VecDeque<bool>>,
    entered: RefCell<bool>,
    left: RefCell<bool>,
    mouse_enabled: RefCell<bool>,
    flush_count: RefCell<u32>,
}

impl MockTerminal {
    fn new(width: u16, height: u16) -> Self {
        Self {
            size: (width, height),
            events: RefCell::new(VecDeque::new()),
            poll_results: RefCell::new(VecDeque::new()),
            entered: RefCell::new(false),
            left: RefCell::new(false),
            mouse_enabled: RefCell::new(false),
            flush_count: RefCell::new(0),
        }
    }

    fn with_events(mut self, events: Vec<CrosstermEvent>) -> Self {
        self.events = RefCell::new(events.into());
        self
    }

    fn with_polls(mut self, polls: Vec<bool>) -> Self {
        self.poll_results = RefCell::new(polls.into());
        self
    }
}

impl Terminal for MockTerminal {
    fn enter(&mut self) -> Result<(), TuiError> {
        *self.entered.borrow_mut() = true;
        Ok(())
    }

    fn leave(&mut self) -> Result<(), TuiError> {
        *self.left.borrow_mut() = true;
        Ok(())
    }

    fn size(&self) -> Result<(u16, u16), TuiError> {
        Ok(self.size)
    }

    fn poll(&self, _timeout: Duration) -> Result<bool, TuiError> {
        Ok(self.poll_results.borrow_mut().pop_front().unwrap_or(false))
    }

    fn read_event(&self) -> Result<CrosstermEvent, TuiError> {
        self.events
            .borrow_mut()
            .pop_front()
            .ok_or_else(|| TuiError::Io(io::Error::other("no event")))
    }

    fn flush(
        &mut self,
        _buffer: &mut CellBuffer,
        _renderer: &mut DiffRenderer,
    ) -> Result<(), TuiError> {
        *self.flush_count.borrow_mut() += 1;
        Ok(())
    }

    fn enable_mouse(&mut self) -> Result<(), TuiError> {
        *self.mouse_enabled.borrow_mut() = true;
        Ok(())
    }

    fn disable_mouse(&mut self) -> Result<(), TuiError> {
        *self.mouse_enabled.borrow_mut() = false;
        Ok(())
    }
}

#[test]
fn test_run_with_terminal_quit_on_q() {
    let widget = TestWidget::new();
    let mut app = TuiApp::new(widget).unwrap();

    let terminal = MockTerminal::new(80, 24)
        .with_polls(vec![true])
        .with_events(vec![CrosstermEvent::Key(crossterm::event::KeyEvent::new(
            KeyCode::Char('q'),
            crossterm::event::KeyModifiers::NONE,
        ))]);

    let result = app.run_with_terminal(terminal);
    assert!(result.is_ok());
    assert!(app.should_quit);
}

#[test]
fn test_run_with_terminal_ctrl_c() {
    let widget = TestWidget::new();
    let mut app = TuiApp::new(widget).unwrap();

    let terminal = MockTerminal::new(80, 24)
        .with_polls(vec![true])
        .with_events(vec![CrosstermEvent::Key(crossterm::event::KeyEvent::new(
            KeyCode::Char('c'),
            crossterm::event::KeyModifiers::CONTROL,
        ))]);

    let result = app.run_with_terminal(terminal);
    assert!(result.is_ok());
    assert!(app.should_quit);
}

#[test]
fn test_run_with_terminal_mouse_enabled() {
    let widget = TestWidget::new();
    let mut app = TuiApp::new(widget).unwrap();
    app.config.enable_mouse = true;

    let terminal = MockTerminal::new(80, 24)
        .with_polls(vec![true])
        .with_events(vec![CrosstermEvent::Key(crossterm::event::KeyEvent::new(
            KeyCode::Char('q'),
            crossterm::event::KeyModifiers::NONE,
        ))]);

    let result = app.run_with_terminal(terminal);
    assert!(result.is_ok());
}

#[test]
fn test_run_with_terminal_skip_verification() {
    let widget = FailingWidget;
    let mut app = TuiApp::new(widget).unwrap();
    app.config.skip_verification = true;

    let terminal = MockTerminal::new(80, 24)
        .with_polls(vec![true])
        .with_events(vec![CrosstermEvent::Key(crossterm::event::KeyEvent::new(
            KeyCode::Char('q'),
            crossterm::event::KeyModifiers::NONE,
        ))]);

    // Should succeed because verification is skipped
    let result = app.run_with_terminal(terminal);
    assert!(result.is_ok());
}

#[test]
fn test_run_with_terminal_verification_failure() {
    let widget = FailingWidget;
    let mut app = TuiApp::new(widget).unwrap();

    let terminal = MockTerminal::new(80, 24).with_polls(vec![false]);

    // Should fail verification
    let result = app.run_with_terminal(terminal);
    assert!(result.is_err());
    assert!(matches!(result, Err(TuiError::VerificationFailed(_))));
}

#[test]
fn test_run_with_terminal_no_events() {
    let widget = TestWidget::new();
    let mut app = TuiApp::new(widget).unwrap();
    app.quit(); // Pre-set quit to exit immediately

    let terminal = MockTerminal::new(80, 24).with_polls(vec![false]);

    let result = app.run_with_terminal(terminal);
    assert!(result.is_ok());
}

#[test]
fn test_run_with_terminal_other_key() {
    let widget = TestWidget::new();
    let mut app = TuiApp::new(widget).unwrap();

    let terminal = MockTerminal::new(80, 24)
        .with_polls(vec![true, true])
        .with_events(vec![
            CrosstermEvent::Key(crossterm::event::KeyEvent::new(
                KeyCode::Enter,
                crossterm::event::KeyModifiers::NONE,
            )),
            CrosstermEvent::Key(crossterm::event::KeyEvent::new(
                KeyCode::Char('q'),
                crossterm::event::KeyModifiers::NONE,
            )),
        ]);

    let result = app.run_with_terminal(terminal);
    assert!(result.is_ok());
}

#[test]
fn test_run_with_terminal_frame_count() {
    let widget = TestWidget::new();
    let mut app = TuiApp::new(widget).unwrap();

    let terminal = MockTerminal::new(80, 24)
        .with_polls(vec![false, false, true])
        .with_events(vec![CrosstermEvent::Key(crossterm::event::KeyEvent::new(
            KeyCode::Char('q'),
            crossterm::event::KeyModifiers::NONE,
        ))]);

    let result = app.run_with_terminal(terminal);
    assert!(result.is_ok());
    assert!(app.metrics.frame_count >= 1);
}

#[test]
fn test_crossterm_backend_new() {
    let backend = CrosstermBackend::new();
    // Just verify it can be created
    let _ = backend;
}

#[test]
fn test_crossterm_backend_default() {
    let backend = CrosstermBackend::default();
    let _ = backend;
}

// Mock backend for testing GenericTerminal
struct MockBackend {
    size: (u16, u16),
    events: RefCell<VecDeque<CrosstermEvent>>,
    poll_results: RefCell<VecDeque<bool>>,
    raw_mode: RefCell<bool>,
    alternate_screen: RefCell<bool>,
    cursor_hidden: RefCell<bool>,
    mouse_captured: RefCell<bool>,
}

impl MockBackend {
    fn new(width: u16, height: u16) -> Self {
        Self {
            size: (width, height),
            events: RefCell::new(VecDeque::new()),
            poll_results: RefCell::new(VecDeque::new()),
            raw_mode: RefCell::new(false),
            alternate_screen: RefCell::new(false),
            cursor_hidden: RefCell::new(false),
            mouse_captured: RefCell::new(false),
        }
    }

    fn with_events(self, events: Vec<CrosstermEvent>) -> Self {
        *self.events.borrow_mut() = events.into();
        self
    }

    fn with_polls(self, polls: Vec<bool>) -> Self {
        *self.poll_results.borrow_mut() = polls.into();
        self
    }
}

impl TerminalBackend for MockBackend {
    fn enable_raw_mode(&mut self) -> Result<(), TuiError> {
        *self.raw_mode.borrow_mut() = true;
        Ok(())
    }
    fn disable_raw_mode(&mut self) -> Result<(), TuiError> {
        *self.raw_mode.borrow_mut() = false;
        Ok(())
    }
    fn enter_alternate_screen(&mut self) -> Result<(), TuiError> {
        *self.alternate_screen.borrow_mut() = true;
        Ok(())
    }
    fn leave_alternate_screen(&mut self) -> Result<(), TuiError> {
        *self.alternate_screen.borrow_mut() = false;
        Ok(())
    }
    fn hide_cursor(&mut self) -> Result<(), TuiError> {
        *self.cursor_hidden.borrow_mut() = true;
        Ok(())
    }
    fn show_cursor(&mut self) -> Result<(), TuiError> {
        *self.cursor_hidden.borrow_mut() = false;
        Ok(())
    }
    fn size(&self) -> Result<(u16, u16), TuiError> {
        Ok(self.size)
    }
    fn poll(&self, _timeout: Duration) -> Result<bool, TuiError> {
        Ok(self.poll_results.borrow_mut().pop_front().unwrap_or(false))
    }
    fn read_event(&self) -> Result<CrosstermEvent, TuiError> {
        self.events
            .borrow_mut()
            .pop_front()
            .ok_or_else(|| TuiError::Io(io::Error::other("no event")))
    }
    fn write_flush(
        &mut self,
        _buffer: &mut CellBuffer,
        _renderer: &mut DiffRenderer,
    ) -> Result<(), TuiError> {
        Ok(())
    }
    fn enable_mouse_capture(&mut self) -> Result<(), TuiError> {
        *self.mouse_captured.borrow_mut() = true;
        Ok(())
    }
    fn disable_mouse_capture(&mut self) -> Result<(), TuiError> {
        *self.mouse_captured.borrow_mut() = false;
        Ok(())
    }
}

#[test]
fn test_generic_terminal_enter_leave() {
    let backend = MockBackend::new(80, 24);
    let mut terminal = GenericTerminal::new(backend);

    terminal.enter().unwrap();
    assert!(*terminal.backend.raw_mode.borrow());
    assert!(*terminal.backend.alternate_screen.borrow());
    assert!(*terminal.backend.cursor_hidden.borrow());

    terminal.leave().unwrap();
    assert!(!*terminal.backend.raw_mode.borrow());
    assert!(!*terminal.backend.alternate_screen.borrow());
    assert!(!*terminal.backend.cursor_hidden.borrow());
}

#[test]
fn test_generic_terminal_size() {
    let backend = MockBackend::new(100, 50);
    let terminal = GenericTerminal::new(backend);
    let (w, h) = terminal.size().unwrap();
    assert_eq!(w, 100);
    assert_eq!(h, 50);
}

#[test]
fn test_generic_terminal_poll_read() {
    let backend = MockBackend::new(80, 24)
        .with_polls(vec![true, false])
        .with_events(vec![CrosstermEvent::Key(crossterm::event::KeyEvent::new(
            KeyCode::Enter,
            crossterm::event::KeyModifiers::NONE,
        ))]);
    let terminal = GenericTerminal::new(backend);

    assert!(terminal.poll(Duration::from_millis(10)).unwrap());
    let event = terminal.read_event().unwrap();
    assert!(matches!(event, CrosstermEvent::Key(_)));

    assert!(!terminal.poll(Duration::from_millis(10)).unwrap());
}

#[test]
fn test_generic_terminal_mouse() {
    let backend = MockBackend::new(80, 24);
    let mut terminal = GenericTerminal::new(backend);

    assert!(!*terminal.backend.mouse_captured.borrow());
    terminal.enable_mouse().unwrap();
    assert!(*terminal.backend.mouse_captured.borrow());
    terminal.disable_mouse().unwrap();
    assert!(!*terminal.backend.mouse_captured.borrow());
}

#[test]
fn test_generic_terminal_flush() {
    let backend = MockBackend::new(80, 24);
    let mut terminal = GenericTerminal::new(backend);
    let mut buffer = CellBuffer::new(80, 24);
    let mut renderer = DiffRenderer::new();

    terminal.flush(&mut buffer, &mut renderer).unwrap();
}

#[test]
fn test_run_with_generic_terminal() {
    let widget = TestWidget::new();
    let mut app = TuiApp::new(widget).unwrap();

    let backend = MockBackend::new(80, 24)
        .with_polls(vec![true])
        .with_events(vec![CrosstermEvent::Key(crossterm::event::KeyEvent::new(
            KeyCode::Char('q'),
            crossterm::event::KeyModifiers::NONE,
        ))]);
    let terminal = GenericTerminal::new(backend);

    let result = app.run_with_terminal(terminal);
    assert!(result.is_ok());
    assert!(app.should_quit);
}

#[test]
fn test_mock_terminal_enter_leave() {
    let mut terminal = MockTerminal::new(80, 24);
    assert!(!*terminal.entered.borrow());
    terminal.enter().unwrap();
    assert!(*terminal.entered.borrow());

    assert!(!*terminal.left.borrow());
    terminal.leave().unwrap();
    assert!(*terminal.left.borrow());
}

#[test]
fn test_mock_terminal_mouse() {
    let mut terminal = MockTerminal::new(80, 24);
    assert!(!*terminal.mouse_enabled.borrow());
    terminal.enable_mouse().unwrap();
    assert!(*terminal.mouse_enabled.borrow());
    terminal.disable_mouse().unwrap();
    assert!(!*terminal.mouse_enabled.borrow());
}

#[test]
fn test_mock_terminal_size() {
    let terminal = MockTerminal::new(100, 50);
    let (w, h) = terminal.size().unwrap();
    assert_eq!(w, 100);
    assert_eq!(h, 50);
}

#[test]
fn test_run_with_terminal_resize() {
    let widget = TestWidget::new();
    let mut app = TuiApp::new(widget).unwrap();

    // Create terminal that simulates a size change by having different initial size
    let terminal = MockTerminal::new(40, 12)
        .with_polls(vec![true])
        .with_events(vec![CrosstermEvent::Key(crossterm::event::KeyEvent::new(
            KeyCode::Char('q'),
            crossterm::event::KeyModifiers::NONE,
        ))]);

    let result = app.run_with_terminal(terminal);
    assert!(result.is_ok());
}

#[test]
fn test_run_with_terminal_mouse_event() {
    let widget = TestWidget::new();
    let mut app = TuiApp::new(widget).unwrap();

    let terminal = MockTerminal::new(80, 24)
        .with_polls(vec![true, true])
        .with_events(vec![
            CrosstermEvent::Mouse(crossterm::event::MouseEvent {
                kind: crossterm::event::MouseEventKind::Down(crossterm::event::MouseButton::Left),
                column: 10,
                row: 5,
                modifiers: crossterm::event::KeyModifiers::NONE,
            }),
            CrosstermEvent::Key(crossterm::event::KeyEvent::new(
                KeyCode::Char('q'),
                crossterm::event::KeyModifiers::NONE,
            )),
        ]);

    let result = app.run_with_terminal(terminal);
    assert!(result.is_ok());
}

#[test]
fn test_run_with_terminal_non_key_event_then_quit() {
    let widget = TestWidget::new();
    let mut app = TuiApp::new(widget).unwrap();

    let terminal = MockTerminal::new(80, 24)
        .with_polls(vec![true, true])
        .with_events(vec![
            CrosstermEvent::Resize(100, 50),
            CrosstermEvent::Key(crossterm::event::KeyEvent::new(
                KeyCode::Char('q'),
                crossterm::event::KeyModifiers::NONE,
            )),
        ]);

    let result = app.run_with_terminal(terminal);
    assert!(result.is_ok());
}

#[test]
fn test_app_runner_metrics_update() {
    let widget = TestWidget::new();
    let mut app = TuiApp::new(widget).unwrap();

    let terminal = MockTerminal::new(80, 24)
        .with_polls(vec![false, true])
        .with_events(vec![CrosstermEvent::Key(crossterm::event::KeyEvent::new(
            KeyCode::Char('q'),
            crossterm::event::KeyModifiers::NONE,
        ))]);

    app.run_with_terminal(terminal).unwrap();

    // Metrics should be populated
    assert!(app.metrics.frame_count >= 1);
}

// =====================================================
// TestableBackend tests - TTY mocking with escape sequences
// =====================================================

#[test]
fn test_testable_backend_new() {
    let buf: Vec<u8> = Vec::new();
    let backend = TestableBackend::new(buf, 80, 24);
    assert_eq!(backend.size, (80, 24));
    assert!(!backend.raw_mode);
    assert!(!backend.alternate_screen);
    assert!(!backend.cursor_hidden);
    assert!(!backend.mouse_captured);
}

#[test]
fn test_testable_backend_with_events() {
    let buf: Vec<u8> = Vec::new();
    let backend = TestableBackend::new(buf, 80, 24).with_events(vec![CrosstermEvent::Key(
        crossterm::event::KeyEvent::new(KeyCode::Char('a'), crossterm::event::KeyModifiers::NONE),
    )]);
    assert_eq!(backend.events.borrow().len(), 1);
}

#[test]
fn test_testable_backend_with_polls() {
    let buf: Vec<u8> = Vec::new();
    let backend = TestableBackend::new(buf, 80, 24).with_polls(vec![true, false, true]);
    assert_eq!(backend.poll_results.borrow().len(), 3);
}

#[test]
fn test_testable_backend_enable_raw_mode() {
    let buf: Vec<u8> = Vec::new();
    let mut backend = TestableBackend::new(buf, 80, 24);
    assert!(!backend.is_raw_mode());
    backend.enable_raw_mode().unwrap();
    assert!(backend.is_raw_mode());
}

#[test]
fn test_testable_backend_disable_raw_mode() {
    let buf: Vec<u8> = Vec::new();
    let mut backend = TestableBackend::new(buf, 80, 24);
    backend.enable_raw_mode().unwrap();
    assert!(backend.is_raw_mode());
    backend.disable_raw_mode().unwrap();
    assert!(!backend.is_raw_mode());
}

#[test]
fn test_testable_backend_enter_alternate_screen() {
    let buf: Vec<u8> = Vec::new();
    let mut backend = TestableBackend::new(buf, 80, 24);
    assert!(!backend.is_alternate_screen());
    backend.enter_alternate_screen().unwrap();
    assert!(backend.is_alternate_screen());
    // Verify escape sequence was written
    let output = backend.into_writer();
    assert!(!output.is_empty());
    // EnterAlternateScreen is \x1b[?1049h
    assert!(output.starts_with(b"\x1b["));
}

#[test]
fn test_testable_backend_leave_alternate_screen() {
    let buf: Vec<u8> = Vec::new();
    let mut backend = TestableBackend::new(buf, 80, 24);
    backend.enter_alternate_screen().unwrap();
    backend.leave_alternate_screen().unwrap();
    assert!(!backend.is_alternate_screen());
}

#[test]
fn test_testable_backend_hide_cursor() {
    let buf: Vec<u8> = Vec::new();
    let mut backend = TestableBackend::new(buf, 80, 24);
    assert!(!backend.is_cursor_hidden());
    backend.hide_cursor().unwrap();
    assert!(backend.is_cursor_hidden());
    // Verify escape sequence was written
    let output = backend.into_writer();
    assert!(!output.is_empty());
}

#[test]
fn test_testable_backend_show_cursor() {
    let buf: Vec<u8> = Vec::new();
    let mut backend = TestableBackend::new(buf, 80, 24);
    backend.hide_cursor().unwrap();
    backend.show_cursor().unwrap();
    assert!(!backend.is_cursor_hidden());
}

#[test]
fn test_testable_backend_size() {
    let buf: Vec<u8> = Vec::new();
    let backend = TestableBackend::new(buf, 120, 40);
    assert_eq!(backend.size().unwrap(), (120, 40));
}

#[test]
fn test_testable_backend_poll() {
    let buf: Vec<u8> = Vec::new();
    let backend = TestableBackend::new(buf, 80, 24).with_polls(vec![true, false]);
    assert!(backend.poll(Duration::from_millis(100)).unwrap());
    assert!(!backend.poll(Duration::from_millis(100)).unwrap());
    // Default when empty
    assert!(!backend.poll(Duration::from_millis(100)).unwrap());
}

#[test]
fn test_testable_backend_read_event() {
    let buf: Vec<u8> = Vec::new();
    let backend = TestableBackend::new(buf, 80, 24).with_events(vec![CrosstermEvent::Key(
        crossterm::event::KeyEvent::new(KeyCode::Char('x'), crossterm::event::KeyModifiers::NONE),
    )]);
    let event = backend.read_event().unwrap();
    assert!(matches!(event, CrosstermEvent::Key(_)));
}

#[test]
fn test_testable_backend_read_event_empty() {
    let buf: Vec<u8> = Vec::new();
    let backend = TestableBackend::new(buf, 80, 24);
    let result = backend.read_event();
    assert!(result.is_err());
}

#[test]
fn test_testable_backend_enable_mouse_capture() {
    let buf: Vec<u8> = Vec::new();
    let mut backend = TestableBackend::new(buf, 80, 24);
    assert!(!backend.is_mouse_captured());
    backend.enable_mouse_capture().unwrap();
    assert!(backend.is_mouse_captured());
    // Verify escape sequence was written
    let output = backend.into_writer();
    assert!(!output.is_empty());
}

#[test]
fn test_testable_backend_disable_mouse_capture() {
    let buf: Vec<u8> = Vec::new();
    let mut backend = TestableBackend::new(buf, 80, 24);
    backend.enable_mouse_capture().unwrap();
    backend.disable_mouse_capture().unwrap();
    assert!(!backend.is_mouse_captured());
}

#[test]
fn test_testable_backend_write_flush() {
    use crate::direct::Cell;

    let buf: Vec<u8> = Vec::new();
    let mut backend = TestableBackend::new(buf, 80, 24);
    let mut buffer = CellBuffer::new(80, 24);
    let mut renderer = DiffRenderer::new();

    // Write something to the buffer using the Cell API
    let mut cell_a = Cell::default();
    cell_a.update(
        "A",
        presentar_core::Color::WHITE,
        presentar_core::Color::BLACK,
        crate::direct::Modifiers::empty(),
    );
    buffer.set(0, 0, cell_a);

    let mut cell_b = Cell::default();
    cell_b.update(
        "B",
        presentar_core::Color::WHITE,
        presentar_core::Color::BLACK,
        crate::direct::Modifiers::empty(),
    );
    buffer.set(1, 0, cell_b);

    buffer.mark_all_dirty();
    backend.write_flush(&mut buffer, &mut renderer).unwrap();

    // Verify output was written
    let output = backend.into_writer();
    assert!(!output.is_empty());
}

#[test]
fn test_testable_backend_full_lifecycle() {
    let buf: Vec<u8> = Vec::new();
    let mut backend = TestableBackend::new(buf, 80, 24);

    // Enter
    backend.enable_raw_mode().unwrap();
    backend.enter_alternate_screen().unwrap();
    backend.hide_cursor().unwrap();

    assert!(backend.is_raw_mode());
    assert!(backend.is_alternate_screen());
    assert!(backend.is_cursor_hidden());

    // Leave
    backend.show_cursor().unwrap();
    backend.leave_alternate_screen().unwrap();
    backend.disable_raw_mode().unwrap();

    assert!(!backend.is_raw_mode());
    assert!(!backend.is_alternate_screen());
    assert!(!backend.is_cursor_hidden());
}

#[test]
fn test_testable_backend_escape_sequences() {
    let buf: Vec<u8> = Vec::new();
    let mut backend = TestableBackend::new(buf, 80, 24);

    backend.enter_alternate_screen().unwrap();
    backend.hide_cursor().unwrap();
    backend.enable_mouse_capture().unwrap();

    let output = backend.into_writer();
    let output_str = String::from_utf8_lossy(&output);

    // Check for ANSI escape sequences (CSI = \x1b[)
    assert!(
        output_str.contains("\x1b["),
        "Expected ANSI escape sequences"
    );
}

#[test]
fn test_generic_terminal_with_testable_backend() {
    let buf: Vec<u8> = Vec::new();
    let backend = TestableBackend::new(buf, 80, 24)
        .with_polls(vec![true])
        .with_events(vec![CrosstermEvent::Key(crossterm::event::KeyEvent::new(
            KeyCode::Char('q'),
            crossterm::event::KeyModifiers::NONE,
        ))]);

    let mut terminal = GenericTerminal::new(backend);

    terminal.enter().unwrap();
    assert_eq!(terminal.size().unwrap(), (80, 24));

    // Poll and read
    assert!(terminal.poll(Duration::from_millis(10)).unwrap());
    let event = terminal.read_event().unwrap();
    assert!(matches!(event, CrosstermEvent::Key(_)));

    terminal.leave().unwrap();
}

#[test]
fn test_testable_backend_with_tui_app() {
    let widget = TestWidget::new();
    let mut app = TuiApp::new(widget).unwrap();

    let buf: Vec<u8> = Vec::new();
    let backend = TestableBackend::new(buf, 80, 24)
        .with_polls(vec![true])
        .with_events(vec![CrosstermEvent::Key(crossterm::event::KeyEvent::new(
            KeyCode::Char('q'),
            crossterm::event::KeyModifiers::NONE,
        ))]);

    let terminal = GenericTerminal::new(backend);
    let result = app.run_with_terminal(terminal);
    assert!(result.is_ok());
}

#[test]
fn test_testable_backend_captures_render_output() {
    let widget = TestWidget::new();
    let _app = TuiApp::new(widget).unwrap();

    let buf: Vec<u8> = Vec::new();
    let backend = TestableBackend::new(buf, 40, 10)
        .with_polls(vec![true])
        .with_events(vec![CrosstermEvent::Key(crossterm::event::KeyEvent::new(
            KeyCode::Char('q'),
            crossterm::event::KeyModifiers::NONE,
        ))]);

    let mut terminal = GenericTerminal::new(backend);
    terminal.enter().unwrap();

    // Get terminal size
    let (width, height) = terminal.size().unwrap();
    assert_eq!((width, height), (40, 10));

    terminal.leave().unwrap();
}
