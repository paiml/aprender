//! WASM application entry point.

use super::canvas2d::Canvas2DRenderer;
use super::events::{keyboard_event_to_presentar, mouse_event_to_presentar};
use presentar_core::draw::DrawCommand;
use presentar_core::{Brick, Constraints, Event, RecordingCanvas, Rect, Size, Widget};
use wasm_bindgen::prelude::*;
use wasm_bindgen::JsCast;
use web_sys::{window, HtmlCanvasElement, KeyboardEvent, MouseEvent};

/// Main application runner for browser.
#[wasm_bindgen]
pub struct App {
    renderer: Canvas2DRenderer,
    canvas: HtmlCanvasElement,
    width: f32,
    height: f32,
    click_callback: Option<Closure<dyn FnMut(MouseEvent)>>,
    mousemove_callback: Option<Closure<dyn FnMut(MouseEvent)>>,
    keydown_callback: Option<Closure<dyn FnMut(KeyboardEvent)>>,
}

#[wasm_bindgen]
impl App {
    /// Create a new app attached to a canvas element by ID.
    #[wasm_bindgen(constructor)]
    pub fn new(canvas_id: &str) -> Result<App, JsValue> {
        console_error_panic_hook::set_once();

        let document = window()
            .ok_or("No window")?
            .document()
            .ok_or("No document")?;

        let canvas = document
            .get_element_by_id(canvas_id)
            .ok_or_else(|| format!("Canvas '{}' not found", canvas_id))?
            .dyn_into::<HtmlCanvasElement>()
            .map_err(|_| "Element is not a canvas")?;

        let width = canvas.width() as f32;
        let height = canvas.height() as f32;

        let renderer = Canvas2DRenderer::new(canvas.clone()).map_err(|e| JsValue::from_str(&e))?;

        Ok(Self {
            renderer,
            canvas,
            width,
            height,
            click_callback: None,
            mousemove_callback: None,
            keydown_callback: None,
        })
    }

    /// Register a click handler that receives event JSON.
    pub fn on_click(&mut self, callback: js_sys::Function) {
        let cb = Closure::new(move |e: MouseEvent| {
            let event = mouse_event_to_presentar(&e, "click");
            let json = serde_json::to_string(&event).unwrap_or_default();
            let _ = callback.call1(&JsValue::NULL, &JsValue::from_str(&json));
        });
        self.canvas
            .add_event_listener_with_callback("click", cb.as_ref().unchecked_ref())
            .ok();
        self.click_callback = Some(cb);
    }

    /// Register a mousemove handler.
    pub fn on_mousemove(&mut self, callback: js_sys::Function) {
        let cb = Closure::new(move |e: MouseEvent| {
            let event = mouse_event_to_presentar(&e, "mousemove");
            let json = serde_json::to_string(&event).unwrap_or_default();
            let _ = callback.call1(&JsValue::NULL, &JsValue::from_str(&json));
        });
        self.canvas
            .add_event_listener_with_callback("mousemove", cb.as_ref().unchecked_ref())
            .ok();
        self.mousemove_callback = Some(cb);
    }

    /// Register a keydown handler.
    pub fn on_keydown(&mut self, callback: js_sys::Function) {
        let document = window().and_then(|w| w.document());
        if let Some(doc) = document {
            let cb = Closure::new(move |e: KeyboardEvent| {
                let event = keyboard_event_to_presentar(&e, "keydown");
                let json = serde_json::to_string(&event).unwrap_or_default();
                let _ = callback.call1(&JsValue::NULL, &JsValue::from_str(&json));
            });
            doc.add_event_listener_with_callback("keydown", cb.as_ref().unchecked_ref())
                .ok();
            self.keydown_callback = Some(cb);
        }
    }

    /// Get canvas width.
    pub fn width(&self) -> f32 {
        self.width
    }

    /// Get canvas height.
    pub fn height(&self) -> f32 {
        self.height
    }

    /// Clear the canvas.
    pub fn clear(&self) {
        self.renderer.clear();
    }

    /// Render draw commands from JSON.
    pub fn render_json(&self, json: &str) -> Result<(), JsValue> {
        contract_pre_render!();
        contract_post_configuration!(&"ok");
        let commands: Vec<DrawCommand> = serde_json::from_str(json)
            .map_err(|e| JsValue::from_str(&format!("JSON parse error: {}", e)))?;
        self.renderer.render(&commands);
        Ok(())
    }

    /// Render a counter widget demo (WASM export).
    pub fn render_counter(&self, count: i32) {
        contract_pre_render!();
        contract_post_configuration!(&"ok");
        use presentar_core::Color;
        use presentar_widgets::{Button, Column, Row, Text};

        let mut widget = Column::new()
            .gap(20.0)
            .child(
                Text::new("Counter Demo")
                    .font_size(28.0)
                    .color(Color::from_hex("#111827").unwrap_or(Color::BLACK)),
            )
            .child(
                Text::new(&format!("Count: {}", count))
                    .font_size(48.0)
                    .color(Color::from_hex("#6366f1").unwrap_or(Color::BLUE)),
            )
            .child(
                Row::new()
                    .gap(16.0)
                    .child(Button::new("-").padding(12.0))
                    .child(Button::new("+").padding(12.0)),
            );

        self.render_widget_internal(&mut widget);
    }

    /// Render a dashboard widget (WASM export).
    pub fn render_dashboard(&self, title: &str, value: f64, progress: f64) {
        contract_pre_render!();
        contract_post_configuration!(&"ok");
        use presentar_core::Color;
        use presentar_widgets::{Column, ProgressBar, Text};

        let mut widget = Column::new()
            .gap(16.0)
            .child(
                Text::new(title)
                    .font_size(24.0)
                    .color(Color::from_hex("#111827").unwrap_or(Color::BLACK)),
            )
            .child(
                Text::new(&format!("${:.2}M", value / 1_000_000.0))
                    .font_size(36.0)
                    .color(Color::from_hex("#059669").unwrap_or(Color::GREEN)),
            )
            .child(
                Column::new()
                    .gap(8.0)
                    .child(Text::new("Progress").font_size(14.0))
                    .child(
                        ProgressBar::new()
                            .value(progress as f32)
                            .fill_color(Color::from_hex("#6366f1").unwrap_or(Color::BLUE)),
                    ),
            );

        self.render_widget_internal(&mut widget);
    }
}

impl App {
    /// Render a widget tree (internal Rust API).
    ///
    /// # PROBAR-SPEC-009: Brick Architecture Enforcement
    ///
    /// This function enforces the Brick Architecture by calling `can_render()`
    /// before `paint()`. If verification fails, rendering is blocked (JIDOKA).
    fn render_widget_internal<W: Widget>(&self, widget: &mut W) {
        // PROBAR-SPEC-009: Verify Brick assertions before rendering
        if !widget.can_render() {
            let verification = widget.verify();
            let errors: Vec<String> = verification
                .failed
                .iter()
                .map(|(assertion, reason)| format!("{:?}: {}", assertion, reason))
                .collect();
            web_sys::console::error_1(&wasm_bindgen::JsValue::from_str(&format!(
                "JIDOKA: Brick '{}' failed verification - rendering blocked: {}",
                widget.brick_name(),
                errors.join(", ")
            )));
            return; // Block rendering if verification fails
        }

        let constraints = Constraints::loose(Size::new(self.width, self.height));
        let size = widget.measure(constraints);
        let bounds = Rect::new(0.0, 0.0, size.width, size.height);
        widget.layout(bounds);

        let mut canvas = RecordingCanvas::new();
        widget.paint(&mut canvas);

        self.renderer.clear();
        self.renderer.render(canvas.commands());
    }

    /// Render a widget tree (public Rust API).
    pub fn render_widget<W: Widget>(&self, widget: &mut W) {
        contract_pre_render!();
        contract_post_configuration!(&"ok");
        self.render_widget_internal(widget);
    }

    /// Render raw draw commands.
    pub fn render_commands(&self, commands: &[DrawCommand]) {
        self.renderer.clear();
        self.renderer.render(commands);
    }

    /// Handle a presentar event (returns true if needs repaint).
    pub fn handle_event(&mut self, _event: &Event) -> bool {
        // Event handling to be implemented with state management
        false
    }
}

/// Initialize panic hook for better error messages.
#[wasm_bindgen(start)]
pub fn init() {
    console_error_panic_hook::set_once();
}

/// Log to browser console.
#[wasm_bindgen]
pub fn log(msg: &str) {
    web_sys::console::log_1(&JsValue::from_str(msg));
}
