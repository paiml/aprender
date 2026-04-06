//! Orbit WASM Application - Zero JavaScript Architecture
//!
//! This module implements the ENTIRE Orbit demo in Rust/WASM.
//! JavaScript is reduced to a single initialization line.
//!
//! # Architecture (OR-001 Compliant)
//!
//! ```text
//! HTML: <script type="module">import init, { initOrbitApp } from './pkg/simular.js'; init().then(initOrbitApp);</script>
//!       ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
//!                                              ONE LINE OF JAVASCRIPT
//! ```
//!
//! All logic lives in Rust:
//! - DOM manipulation via web-sys
//! - Canvas rendering via web-sys
//! - Event handling via closures
//! - Animation loop via requestAnimationFrame
//! - State management in Rust structs

use std::cell::RefCell;
use std::rc::Rc;
use wasm_bindgen::prelude::*;
use wasm_bindgen::JsCast;

use crate::orbit::wasm::OrbitSimulation;

/// Maximum trail length for orbit visualization
const MAX_TRAIL: usize = 500;

/// App state for the orbit simulation
struct OrbitAppState {
    sim: OrbitSimulation,
    canvas: web_sys::HtmlCanvasElement,
    ctx: web_sys::CanvasRenderingContext2d,
    trail: Vec<(f64, f64)>,
    speed: u32,
    running: bool,
    frame_count: u32,
}

impl OrbitAppState {
    fn new(canvas: web_sys::HtmlCanvasElement) -> Self {
        let ctx = canvas
            .get_context("2d")
            .unwrap()
            .unwrap()
            .dyn_into::<web_sys::CanvasRenderingContext2d>()
            .unwrap();

        Self {
            sim: OrbitSimulation::new(),
            canvas,
            ctx,
            trail: Vec::with_capacity(MAX_TRAIL),
            speed: 10,
            running: true,
            frame_count: 0,
        }
    }

    fn toggle_running(&mut self) -> bool {
        self.running = !self.running;
        self.running
    }

    fn set_speed(&mut self, speed: u32) {
        self.speed = speed;
    }

    fn reset(&mut self) {
        self.sim.reset();
        self.trail.clear();
        self.frame_count = 0;
    }

    fn step_single(&mut self) {
        self.sim.step_days(1.0);
        self.frame_count += 1;
    }

    fn tick(&mut self) {
        if self.running {
            // Step simulation based on speed setting
            for _ in 0..self.speed {
                self.sim.step(3600.0); // 1 hour per step
            }
            self.frame_count += 1;

            // Add Earth position to trail
            let earth_x = self.sim.body_x_au(1);
            let earth_y = self.sim.body_y_au(1);
            self.trail.push((earth_x, earth_y));
            if self.trail.len() > MAX_TRAIL {
                self.trail.remove(0);
            }
        }
    }

    fn render(&self) {
        let w = self.canvas.width() as f64;
        let h = self.canvas.height() as f64;
        let cx = w / 2.0;
        let cy = h / 2.0;
        let scale = w.min(h) / 3.0; // AU to pixels

        // Clear background
        self.ctx.set_fill_style_str("#0f0f23");
        self.ctx.fill_rect(0.0, 0.0, w, h);

        // Draw grid circles (1 AU and 2 AU)
        self.ctx.set_stroke_style_str("#1a1a2e");
        self.ctx.set_line_width(1.0);
        for i in 1..=2 {
            self.ctx.begin_path();
            self.ctx
                .arc(
                    cx,
                    cy,
                    scale * f64::from(i),
                    0.0,
                    std::f64::consts::PI * 2.0,
                )
                .unwrap();
            self.ctx.stroke();
        }

        // Get positions in AU
        let positions = self.sim.positions_au_flat();

        // Sun position
        let sun_x = cx + positions[0] * scale;
        let sun_y = cy + positions[1] * scale;

        // Draw Sun
        self.ctx.set_fill_style_str("#ffd93d");
        self.ctx.begin_path();
        self.ctx
            .arc(sun_x, sun_y, 15.0, 0.0, std::f64::consts::PI * 2.0)
            .unwrap();
        self.ctx.fill();

        // Sun glow (simplified - no gradient for WASM compatibility)
        self.ctx.set_global_alpha(0.3);
        self.ctx.set_fill_style_str("#ffd93d");
        self.ctx.begin_path();
        self.ctx
            .arc(sun_x, sun_y, 25.0, 0.0, std::f64::consts::PI * 2.0)
            .unwrap();
        self.ctx.fill();
        self.ctx.set_global_alpha(1.0);

        // Earth position
        let earth_x = cx + positions[3] * scale;
        let earth_y = cy + positions[4] * scale;

        // Draw trail
        if self.trail.len() > 1 {
            self.ctx.set_stroke_style_str("#4ecdc4");
            self.ctx.set_line_width(1.0);
            self.ctx.begin_path();

            let first = &self.trail[0];
            self.ctx.move_to(cx + first.0 * scale, cy + first.1 * scale);

            for (i, (tx, ty)) in self.trail.iter().enumerate().skip(1) {
                let alpha = (i as f64) / (self.trail.len() as f64) * 0.5;
                self.ctx.set_global_alpha(alpha);
                self.ctx.line_to(cx + tx * scale, cy + ty * scale);
            }
            self.ctx.set_global_alpha(1.0);
            self.ctx.stroke();
        }

        // Draw Earth
        self.ctx.set_fill_style_str("#4ecdc4");
        self.ctx.begin_path();
        self.ctx
            .arc(earth_x, earth_y, 8.0, 0.0, std::f64::consts::PI * 2.0)
            .unwrap();
        self.ctx.fill();

        // Earth label
        self.ctx.set_fill_style_str("#888888");
        self.ctx.set_font("12px monospace");
        self.ctx
            .fill_text("Earth", earth_x + 12.0, earth_y + 4.0)
            .unwrap();
    }

    fn update_stats(&self, document: &web_sys::Document) {
        let time_days = self.sim.sim_time_days();
        set_text(document, "sim-time", &format!("{time_days:.1} days"));
        set_text(document, "frame-count", &self.frame_count.to_string());
        set_text(document, "body-count", &self.sim.num_bodies().to_string());

        let energy = self.sim.total_energy();
        set_text(document, "energy", &format!("{energy:.2e} J"));

        let ang_mom = self.sim.angular_momentum();
        set_text(
            document,
            "angular-momentum",
            &format!("{ang_mom:.2e} kg*m2/s"),
        );

        // Earth position
        let earth_x = self.sim.body_x_au(1);
        let earth_y = self.sim.body_y_au(1);
        let earth_r = (earth_x * earth_x + earth_y * earth_y).sqrt();

        set_text(document, "earth-x", &format!("{earth_x:.3} AU"));
        set_text(document, "earth-y", &format!("{earth_y:.3} AU"));
        set_text(document, "earth-r", &format!("{earth_r:.3} AU"));

        // Jidoka status
        if let Ok(status) =
            serde_json::from_str::<serde_json::Value>(&self.sim.jidoka_status_json())
        {
            let energy_ok = status
                .get("energy_ok")
                .and_then(|v| v.as_bool())
                .unwrap_or(false);
            let angular_ok = status
                .get("angular_momentum_ok")
                .and_then(|v| v.as_bool())
                .unwrap_or(false);
            let finite_ok = status
                .get("finite_ok")
                .and_then(|v| v.as_bool())
                .unwrap_or(false);

            set_indicator_class(document, "energy-indicator", energy_ok);
            set_indicator_class(document, "angular-indicator", angular_ok);
            set_indicator_class(document, "finite-indicator", finite_ok);
        }
    }
}

fn set_text(document: &web_sys::Document, id: &str, text: &str) {
    if let Some(el) = document.get_element_by_id(id) {
        el.set_text_content(Some(text));
    }
}

fn set_indicator_class(document: &web_sys::Document, id: &str, is_ok: bool) {
    if let Some(el) = document.get_element_by_id(id) {
        el.set_class_name(if is_ok {
            "indicator-dot"
        } else {
            "indicator-dot error"
        });
    }
}

fn setup_button<F>(document: &web_sys::Document, id: &str, mut callback: F) -> Result<(), JsValue>
where
    F: FnMut() + 'static,
{
    if let Some(btn) = document.get_element_by_id(id) {
        let closure = Closure::wrap(Box::new(move |_: web_sys::Event| {
            callback();
        }) as Box<dyn FnMut(_)>);

        btn.add_event_listener_with_callback("click", closure.as_ref().unchecked_ref())?;
        closure.forget();
    }
    Ok(())
}

fn request_animation_frame(f: &Closure<dyn FnMut()>) {
    web_sys::window()
        .unwrap()
        .request_animation_frame(f.as_ref().unchecked_ref())
        .unwrap();
}

/// Initialize the Orbit WASM app - call from JavaScript
#[wasm_bindgen(js_name = initOrbitApp)]
pub fn init_orbit_app() -> Result<(), JsValue> {
    // Set panic hook for better error messages
    console_error_panic_hook::set_once();

    let window = web_sys::window().expect("no window");
    let document = window.document().expect("no document");

    // Get canvas
    let canvas = document
        .get_element_by_id("orbit-canvas")
        .expect("no canvas")
        .dyn_into::<web_sys::HtmlCanvasElement>()?;

    // Resize canvas to container
    if let Some(container) = canvas.parent_element() {
        canvas.set_width(container.client_width() as u32);
        canvas.set_height(container.client_height() as u32);
    }

    // Create app state
    let state = Rc::new(RefCell::new(OrbitAppState::new(canvas)));

    // Initial render
    {
        let s = state.borrow();
        s.render();
        s.update_stats(&document);
    }

    // Hide loading
    if let Some(loading) = document.get_element_by_id("loading") {
        loading.set_attribute("style", "display: none")?;
    }

    // Pause/Resume button
    setup_button(&document, "pause-btn", {
        let state = Rc::clone(&state);
        let doc = document.clone();
        move || {
            let mut s = state.borrow_mut();
            let running = s.toggle_running();
            if let Some(btn) = doc.get_element_by_id("pause-btn") {
                btn.set_text_content(Some(if running { "Pause" } else { "Resume" }));
            }
        }
    })?;

    // Reset button
    setup_button(&document, "reset-btn", {
        let state = Rc::clone(&state);
        let doc = document.clone();
        move || {
            let mut s = state.borrow_mut();
            s.reset();
            s.render();
            s.update_stats(&doc);
        }
    })?;

    // Step button
    setup_button(&document, "step-btn", {
        let state = Rc::clone(&state);
        let doc = document.clone();
        move || {
            let mut s = state.borrow_mut();
            if !s.running {
                s.step_single();
                s.render();
                s.update_stats(&doc);
            }
        }
    })?;

    // Speed slider
    setup_speed_slider(&document, &state)?;

    // Animation loop
    start_animation_loop(document, &state);

    Ok(())
}

fn setup_speed_slider(
    document: &web_sys::Document,
    state: &Rc<RefCell<OrbitAppState>>,
) -> Result<(), JsValue> {
    if let Some(slider) = document.get_element_by_id("speed-slider") {
        let state = Rc::clone(state);
        let doc = document.clone();
        let closure = Closure::wrap(Box::new(move |e: web_sys::Event| {
            if let Some(target) = e.target() {
                if let Some(input) = target.dyn_ref::<web_sys::HtmlInputElement>() {
                    if let Ok(speed) = input.value().parse::<u32>() {
                        state.borrow_mut().set_speed(speed);
                        set_text(&doc, "speed-value", &format!("{speed}x"));
                    }
                }
            }
        }) as Box<dyn FnMut(_)>);

        slider.add_event_listener_with_callback("input", closure.as_ref().unchecked_ref())?;
        closure.forget();
    }
    Ok(())
}

fn start_animation_loop(document: web_sys::Document, state: &Rc<RefCell<OrbitAppState>>) {
    let state = Rc::clone(state);

    #[allow(clippy::type_complexity)]
    let f: Rc<RefCell<Option<Closure<dyn FnMut()>>>> = Rc::new(RefCell::new(None));
    let g = Rc::clone(&f);

    *g.borrow_mut() = Some(Closure::new(move || {
        {
            let mut s = state.borrow_mut();
            s.tick();
            s.render();
            s.update_stats(&document);
        }
        request_animation_frame(f.borrow().as_ref().unwrap());
    }));

    request_animation_frame(g.borrow().as_ref().unwrap());
}
