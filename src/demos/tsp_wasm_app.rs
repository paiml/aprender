//! TSP WASM Application - Zero JavaScript Architecture
//!
//! This module implements the ENTIRE TSP demo in Rust/WASM.
//! JavaScript is reduced to a single initialization line.
//!
//! # Architecture (OR-001 Compliant)
//!
//! ```text
//! HTML: <script type="module">import init from './pkg/simular.js'; init();</script>
//!       ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
//!                           ONE LINE OF JAVASCRIPT
//! ```
//!
//! All logic lives in Rust:
//! - DOM manipulation via web-sys
//! - Canvas rendering via web-sys
//! - Event handling via closures
//! - State management in Rust structs

use std::cell::RefCell;
use std::rc::Rc;
use wasm_bindgen::prelude::*;
use wasm_bindgen::JsCast;

use crate::demos::tsp_grasp::TspGraspDemo;
use crate::demos::tsp_instance::TspInstanceYaml;

/// Embedded Bay Area TSP instance - same as TUI uses
const BAY_AREA_YAML: &str = include_str!("../../examples/experiments/bay_area_tsp.yaml");

/// Global app state - wrapped in `RefCell` for interior mutability
struct TspAppState {
    tsp: TspGraspDemo,
    canvas: web_sys::HtmlCanvasElement,
    ctx: web_sys::CanvasRenderingContext2d,
    convergence: Vec<f64>,
    seed: u32,
    running: bool,
    frame_count: u32,
}

impl TspAppState {
    fn new(canvas: web_sys::HtmlCanvasElement, _n: usize, seed: u32) -> Self {
        let ctx = canvas
            .get_context("2d")
            .unwrap()
            .unwrap()
            .dyn_into::<web_sys::CanvasRenderingContext2d>()
            .unwrap();

        // Load Bay Area TSP from embedded YAML - same as TUI
        let instance =
            TspInstanceYaml::from_yaml(BAY_AREA_YAML).expect("embedded YAML should be valid");
        let tsp = TspGraspDemo::from_instance(&instance);

        Self {
            tsp,
            canvas,
            ctx,
            convergence: Vec::new(),
            seed,
            running: false,
            frame_count: 0,
        }
    }

    fn toggle_running(&mut self) -> bool {
        self.running = !self.running;
        self.running
    }

    fn render(&self) {
        let w = self.canvas.width() as f64;
        let h = self.canvas.height() as f64;
        let pad = 50.0;

        // Clear
        self.ctx.set_fill_style_str("#0f0f23");
        self.ctx.fill_rect(0.0, 0.0, w, h);

        // Get data - x=lon, y=lat for geographic coordinates
        let cities: Vec<[f64; 2]> = self.tsp.cities.iter().map(|c| [c.x, c.y]).collect();
        let tour = &self.tsp.best_tour;

        if cities.is_empty() {
            return;
        }

        // Compute bounds for normalization (Bay Area coordinates)
        let min_x = cities.iter().map(|c| c[0]).fold(f64::INFINITY, f64::min);
        let max_x = cities
            .iter()
            .map(|c| c[0])
            .fold(f64::NEG_INFINITY, f64::max);
        let min_y = cities.iter().map(|c| c[1]).fold(f64::INFINITY, f64::min);
        let max_y = cities
            .iter()
            .map(|c| c[1])
            .fold(f64::NEG_INFINITY, f64::max);

        let range_x = (max_x - min_x).max(0.001);
        let range_y = (max_y - min_y).max(0.001);

        // Scale to fit canvas while maintaining aspect ratio
        let scale_x = (w - pad * 2.0) / range_x;
        let scale_y = (h - pad * 2.0) / range_y;
        let scale = scale_x.min(scale_y);

        // Center the drawing
        let offset_x = (w - range_x * scale) / 2.0;
        let offset_y = (h - range_y * scale) / 2.0;

        // Transform function: geographic -> canvas (flip Y for lat)
        let transform = |lon: f64, lat: f64| -> (f64, f64) {
            let x = offset_x + (lon - min_x) * scale;
            let y = offset_y + (max_y - lat) * scale; // Flip Y so north is up
            (x, y)
        };

        // Draw tour
        if !tour.is_empty() {
            self.ctx.set_stroke_style_str("#4ecdc4");
            self.ctx.set_line_width(2.0);
            self.ctx.begin_path();

            for (i, &idx) in tour.iter().enumerate() {
                let (x, y) = transform(cities[idx][0], cities[idx][1]);
                if i == 0 {
                    self.ctx.move_to(x, y);
                } else {
                    self.ctx.line_to(x, y);
                }
            }
            // Close the tour
            let first = tour[0];
            let (x, y) = transform(cities[first][0], cities[first][1]);
            self.ctx.line_to(x, y);
            self.ctx.stroke();
        }

        // Draw cities
        for (i, city) in cities.iter().enumerate() {
            let (x, y) = transform(city[0], city[1]);

            // City dot
            self.ctx.set_fill_style_str("#ffd93d");
            self.ctx.begin_path();
            self.ctx
                .arc(x, y, 6.0, 0.0, std::f64::consts::PI * 2.0)
                .unwrap();
            self.ctx.fill();

            // City index label
            self.ctx.set_fill_style_str("#e0e0e0");
            self.ctx.set_font("10px 'JetBrains Mono', monospace");
            self.ctx
                .fill_text(&format!("{i}"), x + 8.0, y + 4.0)
                .unwrap();
        }

        // Draw title
        self.ctx.set_fill_style_str("#4ecdc4");
        self.ctx.set_font("bold 14px 'JetBrains Mono', monospace");
        self.ctx
            .fill_text("Bay Area TSP - 20 Cities", 10.0, 20.0)
            .unwrap();
    }

    fn update_stats(&self, document: &web_sys::Document) {
        let n = self.tsp.cities.len();
        let best = self.tsp.best_tour_length;
        let lb = self.tsp.lower_bound;
        let gap = self.tsp.optimality_gap();
        let restarts = self.tsp.restarts;
        let two_opt = self.tsp.two_opt_improvements;
        let cv = self.tsp.restart_cv();
        let units = &self.tsp.units;

        set_text(document, "stat-n", &n.to_string());
        set_text(document, "stat-best", &format!("{best:.1} {units}"));
        set_text(document, "stat-lb", &format!("{lb:.1} {units}"));
        set_text(document, "stat-gap", &format!("{:.1}%", gap * 100.0));
        set_text(document, "stat-restarts", &restarts.to_string());
        set_text(document, "stat-2opt", &two_opt.to_string());

        set_text(document, "eq-tour", &format!("L = {best:.1} {units}"));
        set_text(document, "eq-lb", &format!("LB = {lb:.1} {units}"));

        let gap_ok = gap <= 0.20;
        let cv_ok = cv <= 0.05 || restarts < 2;

        set_text(document, "fals-gap", &format!("{:.1}%", gap * 100.0));
        set_class(
            document,
            "fals-gap",
            &format!("stat-value {}", if gap_ok { "ok" } else { "error" }),
        );

        set_text(document, "fals-cv", &format!("{:.1}%", cv * 100.0));
        set_class(
            document,
            "fals-cv",
            &format!("stat-value {}", if cv_ok { "ok" } else { "warn" }),
        );

        let verified = gap_ok && cv_ok;
        set_text(
            document,
            "fals-status",
            if verified { "VERIFIED" } else { "PENDING" },
        );
        set_class(
            document,
            "fals-status",
            &format!("stat-value {}", if verified { "ok" } else { "warn" }),
        );
    }

    fn step(&mut self) {
        self.tsp.grasp_iteration();
        self.convergence.push(self.tsp.best_tour_length);
        if self.convergence.len() > 100 {
            self.convergence.remove(0);
        }
    }

    fn run(&mut self, n: usize) {
        self.tsp.run_grasp(n);
        self.convergence.push(self.tsp.best_tour_length);
        if self.convergence.len() > 100 {
            self.convergence.remove(0);
        }
    }

    fn reset(&mut self, _n: usize) {
        self.seed = (js_sys::Math::random() * 1_000_000.0) as u32;
        // Reload from embedded YAML - same cities, new seed
        let instance =
            TspInstanceYaml::from_yaml(BAY_AREA_YAML).expect("embedded YAML should be valid");
        self.tsp = TspGraspDemo::from_instance(&instance);
        self.tsp.seed = u64::from(self.seed);
        self.convergence.clear();
        self.running = false;
        self.frame_count = 0;
    }

    fn tick(&mut self) {
        if self.running {
            self.frame_count += 1;
            // Step every 3rd frame (~20 iterations/sec at 60fps)
            if self.frame_count % 3 == 0 {
                self.step();
            }
        }
    }
}

fn set_text(document: &web_sys::Document, id: &str, text: &str) {
    if let Some(el) = document.get_element_by_id(id) {
        el.set_text_content(Some(text));
    }
}

fn set_class(document: &web_sys::Document, id: &str, class: &str) {
    if let Some(el) = document.get_element_by_id(id) {
        el.set_class_name(class);
    }
}

fn get_value(document: &web_sys::Document, id: &str) -> Option<String> {
    document
        .get_element_by_id(id)?
        .dyn_ref::<web_sys::HtmlInputElement>()
        .map(|el| el.value())
}

/// Initialize the TSP WASM app - call from JavaScript
#[wasm_bindgen(js_name = initTspApp)]
pub fn init_tsp_app() -> Result<(), JsValue> {
    // Set panic hook for better error messages
    console_error_panic_hook::set_once();

    let window = web_sys::window().expect("no window");
    let document = window.document().expect("no document");

    // Get canvas
    let canvas = document
        .get_element_by_id("tsp-canvas")
        .expect("no canvas")
        .dyn_into::<web_sys::HtmlCanvasElement>()?;

    // Resize canvas to container
    if let Some(container) = canvas.parent_element() {
        canvas.set_width(container.client_width() as u32);
        canvas.set_height(container.client_height() as u32);
    }

    // Create app state
    let state = Rc::new(RefCell::new(TspAppState::new(canvas, 25, 42)));

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

    // Setup button handlers
    setup_button(&document, "btn-step", {
        let state = Rc::clone(&state);
        let doc = document.clone();
        move || {
            let mut s = state.borrow_mut();
            s.step();
            s.render();
            s.update_stats(&doc);
        }
    })?;

    setup_button(&document, "btn-run10", {
        let state = Rc::clone(&state);
        let doc = document.clone();
        move || {
            let mut s = state.borrow_mut();
            s.run(10);
            s.render();
            s.update_stats(&doc);
        }
    })?;

    setup_button(&document, "btn-run100", {
        let state = Rc::clone(&state);
        let doc = document.clone();
        move || {
            let mut s = state.borrow_mut();
            s.run(100);
            s.render();
            s.update_stats(&doc);
        }
    })?;

    setup_button(&document, "btn-reset", {
        let state = Rc::clone(&state);
        let doc = document.clone();
        move || {
            let n: usize = get_value(&doc, "slider-n")
                .and_then(|v| v.parse().ok())
                .unwrap_or(25);
            let mut s = state.borrow_mut();
            s.reset(n);
            s.render();
            s.update_stats(&doc);
        }
    })?;

    // Slider handler
    if let Some(slider) = document.get_element_by_id("slider-n") {
        let state = Rc::clone(&state);
        let doc = document.clone();
        let closure = Closure::wrap(Box::new(move |_: web_sys::Event| {
            let n: usize = get_value(&doc, "slider-n")
                .and_then(|v| v.parse().ok())
                .unwrap_or(25);
            set_text(&doc, "val-n", &n.to_string());
            let mut s = state.borrow_mut();
            s.reset(n);
            s.render();
            s.update_stats(&doc);
        }) as Box<dyn FnMut(_)>);

        slider.add_event_listener_with_callback("change", closure.as_ref().unchecked_ref())?;
        closure.forget();
    }

    // Play/Pause button handler
    setup_button(&document, "btn-play", {
        let state = Rc::clone(&state);
        let doc = document.clone();
        move || {
            let mut s = state.borrow_mut();
            let running = s.toggle_running();
            // Update button text
            if let Some(btn) = doc.get_element_by_id("btn-play") {
                btn.set_text_content(Some(if running { "⏸ Pause" } else { "▶ Play" }));
            }
        }
    })?;

    // Animation loop using requestAnimationFrame
    {
        let state = Rc::clone(&state);
        let doc = document.clone();

        // Create a reference-counted closure for the animation loop
        #[allow(clippy::type_complexity)]
        let f: Rc<RefCell<Option<Closure<dyn FnMut()>>>> = Rc::new(RefCell::new(None));
        let g = Rc::clone(&f);

        *g.borrow_mut() = Some(Closure::new(move || {
            {
                let mut s = state.borrow_mut();
                if s.running {
                    s.tick();
                    s.render();
                    s.update_stats(&doc);
                }
            }
            // Request next frame
            request_animation_frame(f.borrow().as_ref().unwrap());
        }));

        // Start the loop
        request_animation_frame(g.borrow().as_ref().unwrap());
    }

    // Tab switching
    setup_tabs(&document)?;

    Ok(())
}

fn request_animation_frame(f: &Closure<dyn FnMut()>) {
    web_sys::window()
        .unwrap()
        .request_animation_frame(f.as_ref().unchecked_ref())
        .unwrap();
}

fn setup_tabs(document: &web_sys::Document) -> Result<(), JsValue> {
    let tabs = match document.query_selector_all(".tab") {
        Ok(t) => t,
        Err(_) => return Ok(()),
    };

    for i in 0..tabs.length() {
        let Some(tab) = tabs.get(i) else { continue };
        let doc = document.clone();
        let closure = Closure::wrap(Box::new(move |e: web_sys::Event| {
            handle_tab_click(&doc, &e);
        }) as Box<dyn FnMut(_)>);

        tab.add_event_listener_with_callback("click", closure.as_ref().unchecked_ref())?;
        closure.forget();
    }
    Ok(())
}

fn handle_tab_click(doc: &web_sys::Document, e: &web_sys::Event) {
    let Some(target) = e.target() else { return };
    let Some(el) = target.dyn_ref::<web_sys::Element>() else {
        return;
    };

    // Remove active from all tabs
    remove_class_from_all(doc, ".tab", "active");
    // Remove active from all mains
    remove_class_from_all(doc, "main", "active");
    // Add active to clicked tab
    let _ = el.class_list().add_1("active");
    // Add active to corresponding view
    if let Some(view) = el.get_attribute("data-view") {
        if let Some(main) = doc.get_element_by_id(&format!("view-{view}")) {
            let _ = main.class_list().add_1("active");
        }
    }
}

fn remove_class_from_all(doc: &web_sys::Document, selector: &str, class: &str) {
    if let Ok(elements) = doc.query_selector_all(selector) {
        for j in 0..elements.length() {
            if let Some(el) = elements.get(j) {
                if let Some(el_ref) = el.dyn_ref::<web_sys::Element>() {
                    let _ = el_ref.class_list().remove_1(class);
                }
            }
        }
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
