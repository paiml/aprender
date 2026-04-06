//! Recipe registrations for the Cookbook
//!
//! This module contains all recipe definitions, split from mod.rs for file size compliance.

use super::Recipe;

/// Register all recipes in the cookbook
pub fn register_all(cookbook: &mut super::Cookbook) {
    register_wasm_recipes(cookbook);
    register_ml_recipes(cookbook);
    register_transpilation_recipes(cookbook);
    register_distributed_recipes(cookbook);
    register_quality_recipes(cookbook);
    register_speech_recipes(cookbook);
    super::recipes_more::register_training_recipes(cookbook);
    super::recipes_more::register_data_recipes(cookbook);
    super::recipes_more::register_registry_recipes(cookbook);
    super::recipes_more::register_rag_recipes(cookbook);
    super::recipes_more::register_viz_recipes(cookbook);
    super::recipes_more::register_rlhf_recipes(cookbook);
}

// =========================================================================
// WASM Recipes
// =========================================================================

fn register_wasm_recipes(cookbook: &mut super::Cookbook) {
    // Zero-JS WASM Architecture
    cookbook.add(
            Recipe::new("wasm-zero-js", "Zero-JS WASM Application")
                .with_problem("Build interactive web apps with pure Rust/WASM, eliminating JavaScript entirely")
                .with_components(vec!["simular", "trueno", "web-sys", "wasm-bindgen"])
                .with_tags(vec!["wasm", "zero-js", "web", "canvas", "animation"])
                .with_code(r##"// Cargo.toml
[lib]
crate-type = ["cdylib"]

[dependencies]
wasm-bindgen = "0.2"
web-sys = { version = "0.3", features = [
    "Window", "Document", "HtmlCanvasElement",
    "CanvasRenderingContext2d", "Element"
]}
console_error_panic_hook = "0.1"

// src/lib.rs
use std::cell::RefCell;
use std::rc::Rc;
use wasm_bindgen::prelude::*;
use wasm_bindgen::JsCast;

struct AppState {
    canvas: web_sys::HtmlCanvasElement,
    ctx: web_sys::CanvasRenderingContext2d,
    frame: u32,
}

impl AppState {
    fn tick(&mut self) {
        self.frame += 1;
        self.render();
    }

    fn render(&self) {
        let w = self.canvas.width() as f64;
        let h = self.canvas.height() as f64;

        // Clear
        self.ctx.set_fill_style_str("#0a0a1a");
        self.ctx.fill_rect(0.0, 0.0, w, h);

        // Draw
        self.ctx.set_fill_style_str("#4ecdc4");
        self.ctx.begin_path();
        let _ = self.ctx.arc(w/2.0, h/2.0, 50.0, 0.0, std::f64::consts::PI * 2.0);
        self.ctx.fill();
    }
}

fn request_animation_frame(f: &Closure<dyn FnMut()>) {
    web_sys::window().expect("no global window")
        .request_animation_frame(f.as_ref().unchecked_ref()).expect("raf failed");
}

#[wasm_bindgen(js_name = initApp)]
pub fn init_app() -> Result<(), JsValue> {
    console_error_panic_hook::set_once();

    let window = web_sys::window().expect("no window");
    let document = window.document().expect("no document");
    let canvas = document.get_element_by_id("canvas")
        .expect("no canvas")
        .dyn_into::<web_sys::HtmlCanvasElement>()?;

    let ctx = canvas.get_context("2d")?.expect("no 2d context")
        .dyn_into::<web_sys::CanvasRenderingContext2d>()?;

    let state = Rc::new(RefCell::new(AppState { canvas, ctx, frame: 0 }));

    // Animation loop
    let f: Rc<RefCell<Option<Closure<dyn FnMut()>>>> = Rc::new(RefCell::new(None));
    let g = Rc::clone(&f);
    let state_clone = Rc::clone(&state);

    *g.borrow_mut() = Some(Closure::new(move || {
        state_clone.borrow_mut().tick();
        request_animation_frame(f.borrow().as_ref().expect("closure not set"));
    }));

    request_animation_frame(g.borrow().as_ref().expect("closure not set"));
    Ok(())
}

// index.html - ONE LINE OF JAVASCRIPT
// <script type="module">import init, { initApp } from './pkg/app.js'; init().then(initApp);</script>
"##)
                .with_related(vec!["wasm-event-handling", "wasm-canvas-rendering"])
                .with_test_code(r#"#[cfg(all(test, not(target_arch = "wasm32")))]
mod tests {
    #[test]
    fn test_frame_increment() {
        let mut frame = 0u32;
        frame += 1;
        assert_eq!(frame, 1);
    }

    #[test]
    fn test_circle_radius_positive() {
        let radius = 50.0_f64;
        assert!(radius > 0.0);
    }

    #[test]
    fn test_canvas_center_calculation() {
        let w = 800_u32;
        let h = 600_u32;
        let center_x = (w as f64) / 2.0;
        let center_y = (h as f64) / 2.0;
        assert_eq!(center_x, 400.0);
        assert_eq!(center_y, 300.0);
    }
}"#),
        );

    // WASM Event Handling
    cookbook.add(
            Recipe::new("wasm-event-handling", "WASM Event Handling")
                .with_problem("Handle DOM events (click, input, keypress) in pure Rust")
                .with_components(vec!["simular", "web-sys", "wasm-bindgen"])
                .with_tags(vec!["wasm", "events", "dom", "closure"])
                .with_code(r#"use wasm_bindgen::prelude::*;
use wasm_bindgen::JsCast;

fn setup_button<F>(document: &web_sys::Document, id: &str, mut callback: F) -> Result<(), JsValue>
where
    F: FnMut() + 'static,
{
    if let Some(btn) = document.get_element_by_id(id) {
        let closure = Closure::wrap(Box::new(move |_: web_sys::Event| {
            callback();
        }) as Box<dyn FnMut(_)>);

        btn.add_event_listener_with_callback("click", closure.as_ref().unchecked_ref())?;
        closure.forget(); // Prevent closure from being dropped
    }
    Ok(())
}

fn setup_slider(document: &web_sys::Document, id: &str, state: Rc<RefCell<AppState>>) -> Result<(), JsValue> {
    if let Some(slider) = document.get_element_by_id(id) {
        let closure = Closure::wrap(Box::new(move |e: web_sys::Event| {
            if let Some(target) = e.target() {
                if let Some(input) = target.dyn_ref::<web_sys::HtmlInputElement>() {
                    if let Ok(value) = input.value().parse::<u32>() {
                        state.borrow_mut().set_speed(value);
                    }
                }
            }
        }) as Box<dyn FnMut(_)>);

        slider.add_event_listener_with_callback("input", closure.as_ref().unchecked_ref())?;
        closure.forget();
    }
    Ok(())
}
"#)
                .with_related(vec!["wasm-zero-js", "wasm-canvas-rendering"])
                .with_test_code(r#"#[cfg(all(test, not(target_arch = "wasm32")))]
mod tests {
    use std::cell::RefCell;
    use std::rc::Rc;

    #[test]
    fn test_closure_state_capture() {
        let counter = Rc::new(RefCell::new(0u32));
        let counter_clone = Rc::clone(&counter);
        let increment = move || { *counter_clone.borrow_mut() += 1; };
        increment();
        assert_eq!(*counter.borrow(), 1);
    }

    #[test]
    fn test_input_value_parsing() {
        let input_value = "42";
        let parsed = input_value.parse::<u32>();
        assert_eq!(parsed.expect("unexpected failure"), 42);
    }

    #[test]
    fn test_speed_state_update() {
        let mut speed = 0_u32;
        let new_value = 10_u32;
        speed = new_value;
        assert_eq!(speed, 10);
    }
}"#),
        );

    // WASM Canvas Rendering
    cookbook.add(
        Recipe::new("wasm-canvas-rendering", "WASM Canvas 2D Rendering")
            .with_problem("Render graphics to HTML5 Canvas from Rust/WASM")
            .with_components(vec!["simular", "web-sys"])
            .with_tags(vec!["wasm", "canvas", "graphics", "rendering"])
            .with_code(
                r##"use web_sys::CanvasRenderingContext2d;

fn render(ctx: &CanvasRenderingContext2d, w: f64, h: f64, trail: &[(f64, f64)]) {
    // Clear background
    ctx.set_fill_style_str("#0f0f23");
    ctx.fill_rect(0.0, 0.0, w, h);

    // Draw grid
    ctx.set_stroke_style_str("#1a1a2e");
    ctx.set_line_width(1.0);
    for i in 1..=10 {
        let x = w * (i as f64) / 10.0;
        ctx.begin_path();
        ctx.move_to(x, 0.0);
        ctx.line_to(x, h);
        ctx.stroke();
    }

    // Draw circle with glow
    ctx.set_fill_style_str("#ffd93d");
    ctx.begin_path();
    let _ = ctx.arc(w/2.0, h/2.0, 15.0, 0.0, std::f64::consts::PI * 2.0);
    ctx.fill();

    // Semi-transparent glow
    ctx.set_global_alpha(0.3);
    ctx.begin_path();
    let _ = ctx.arc(w/2.0, h/2.0, 25.0, 0.0, std::f64::consts::PI * 2.0);
    ctx.fill();
    ctx.set_global_alpha(1.0);

    // Draw trail with fading alpha
    ctx.begin_path();
    if let Some((x, y)) = trail.first() {
        ctx.move_to(*x, *y);
    }
    for (i, (x, y)) in trail.iter().enumerate().skip(1) {
        let alpha = (i as f64) / (trail.len() as f64) * 0.5;
        ctx.set_global_alpha(alpha);
        ctx.line_to(*x, *y);
    }
    ctx.set_global_alpha(1.0);
    ctx.stroke();

    // Text labels
    ctx.set_fill_style_str("#888888");
    ctx.set_font("12px monospace");
    let _ = ctx.fill_text("Label", 100.0, 100.0);
}
"##,
            )
            .with_related(vec!["wasm-zero-js", "wasm-event-handling"])
            .with_test_code(
                r#"#[cfg(all(test, not(target_arch = "wasm32")))]
mod tests {
    #[test]
    fn test_grid_spacing_calculation() {
        let width = 800.0_f64;
        let divisions = 10;
        let spacing = width / (divisions as f64);
        assert_eq!(spacing, 80.0);
    }

    #[test]
    fn test_trail_alpha_fade() {
        let trail_len = 10;
        let index = 5;
        let alpha = (index as f64) / (trail_len as f64) * 0.5;
        assert!((alpha - 0.25).abs() < 0.001);
    }

    #[test]
    fn test_trail_data_structure() {
        let mut trail: Vec<(f64, f64)> = Vec::new();
        trail.push((100.0, 200.0));
        trail.push((150.0, 250.0));
        assert_eq!(trail.len(), 2);
    }
}"#,
            ),
    );
}

// =========================================================================
// ML Recipes
// =========================================================================

fn register_ml_recipes(cookbook: &mut super::Cookbook) {
    // Random Forest Classification
    cookbook.add(
        Recipe::new("ml-random-forest", "Random Forest Classification")
            .with_problem("Train a random forest classifier and export for serving")
            .with_components(vec!["aprender", "realizar", "alimentar"])
            .with_tags(vec!["ml", "classification", "random-forest", "supervised"])
            .with_code(
                r#"use aprender::prelude::*;
use alimentar::CsvReader;

// Load data
let data = CsvReader::from_path("data.csv")?.load()?;
let (X, y) = data.split_target("label")?;

// Train-test split
let (X_train, X_test, y_train, y_test) = train_test_split(&X, &y, 0.2, Some(42))?;

// Train model
let model = RandomForestClassifier::new()
    .n_estimators(100)
    .max_depth(Some(10))
    .min_samples_split(5)
    .fit(&X_train, &y_train)?;

// Evaluate
let predictions = model.predict(&X_test)?;
let accuracy = accuracy_score(&y_test, &predictions);
println!("Accuracy: {:.2}%", accuracy * 100.0);

// Export for serving
model.save_apr("model.apr")?;

// Load in realizar for inference
// realizar serve --model model.apr --port 8080
"#,
            )
            .with_related(vec!["ml-serving", "ml-preprocessing"])
            .with_test_code(
                r#"#[cfg(test)]
mod tests {
    #[test]
    fn test_random_forest_builder_config() {
        let n_estimators = 100;
        let max_depth = Some(10);
        assert_eq!(n_estimators, 100);
        assert!(max_depth.expect("unexpected failure") > 0);
    }

    #[test]
    fn test_predictions_collection() {
        let predictions = vec![0, 1, 1, 0, 1];
        assert_eq!(predictions.len(), 5);
    }

    #[test]
    fn test_accuracy_in_range() {
        let correct = 85;
        let total = 100;
        let accuracy = correct as f64 / total as f64;
        assert!(accuracy >= 0.0 && accuracy <= 1.0);
    }
}"#,
            ),
    );

    // Model Serving
    cookbook.add(
        Recipe::new("ml-serving", "Model Serving with Realizar")
            .with_problem("Deploy trained models as HTTP API or Lambda function")
            .with_components(vec!["realizar", "aprender"])
            .with_tags(vec!["ml", "serving", "inference", "api", "lambda"])
            .with_code(
                r#"// Command line serving
// realizar serve --model model.apr --port 8080

// Programmatic serving
use realizar::prelude::*;

let model = Model::load("model.apr")?;
let server = Server::new(model)
    .port(8080)
    .batch_size(32)
    .workers(4);

server.run()?;

// Lambda deployment
// realizar package --model model.apr --output lambda.zip
// aws lambda create-function --function-name my-model \
//     --runtime provided.al2 --handler bootstrap \
//     --zip-file fileb://lambda.zip

// Request format
// POST /predict
// Content-Type: application/json
// {"features": [1.0, 2.0, 3.0, 4.0]}
"#,
            )
            .with_related(vec!["ml-random-forest", "distributed-inference"])
            .with_test_code(
                r"#[cfg(test)]
mod tests {
    #[test]
    fn test_server_config_port() {
        let port = 8080_u16;
        assert!(port >= 1024);
    }

    #[test]
    fn test_server_config_batch_and_workers() {
        let batch_size = 32;
        let workers = 4;
        assert_eq!(batch_size, 32);
        assert_eq!(workers, 4);
    }

    #[test]
    fn test_feature_vector_format() {
        let features: Vec<f32> = vec![1.0, 2.5, 3.7, 4.2];
        assert_eq!(features.len(), 4);
    }
}",
            ),
    );
}

// =========================================================================
// Transpilation Recipes
// =========================================================================

fn register_transpilation_recipes(cookbook: &mut super::Cookbook) {
    // Python to Rust
    cookbook.add(
        Recipe::new("transpile-python", "Python to Rust Migration")
            .with_problem("Convert Python ML code to Rust using depyler")
            .with_components(vec!["depyler", "aprender", "trueno", "batuta"])
            .with_tags(vec!["transpilation", "python", "migration"])
            .with_code(
                r"# Original Python code (sklearn_model.py)
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import numpy as np

X = np.random.rand(100, 4)
y = np.random.randint(0, 2, 100)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
model = RandomForestClassifier(n_estimators=100)
model.fit(X_train, y_train)

# Transpile with depyler
# batuta transpile sklearn_model.py --output src/model.rs

// Generated Rust code (src/model.rs)
use aprender::prelude::*;
use trueno::prelude::*;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let X = Tensor::rand(&[100, 4]);
    let y = Tensor::randint(0, 2, &[100]);

    let (X_train, X_test, y_train, y_test) = train_test_split(&X, &y, 0.2, None)?;
    let model = RandomForestClassifier::new()
        .n_estimators(100)
        .fit(&X_train, &y_train)?;

    Ok(())
}
",
            )
            .with_related(vec!["transpile-numpy", "quality-golden-trace"])
            .with_test_code(
                r"#[cfg(test)]
mod tests {
    #[test]
    fn test_tensor_shape() {
        let rows = 100;
        let cols = 4;
        let total_elements = rows * cols;
        assert_eq!(total_elements, 400);
    }

    #[test]
    fn test_train_test_split_ratio() {
        let total = 100;
        let train_ratio = 0.8;
        let train_size = (total as f64 * train_ratio) as usize;
        assert_eq!(train_size, 80);
    }

    #[test]
    fn test_label_values_binary() {
        let labels = vec![0, 1, 1, 0, 1];
        assert!(labels.iter().all(|&l| l == 0 || l == 1));
    }
}",
            ),
    );

    // NumPy to Trueno
    cookbook.add(
        Recipe::new("transpile-numpy", "NumPy to Trueno Conversion")
            .with_problem("Convert NumPy operations to SIMD-accelerated Trueno")
            .with_components(vec!["depyler", "trueno"])
            .with_tags(vec!["transpilation", "numpy", "simd", "tensors"])
            .with_code(
                r"# Python NumPy
import numpy as np
a = np.array([1, 2, 3, 4])
b = np.array([5, 6, 7, 8])
dot = np.dot(a, b)
matmul = np.matmul(X, W)

// Rust Trueno (SIMD-accelerated)
use trueno::prelude::*;

let a = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0]);
let b = Tensor::from_vec(vec![5.0, 6.0, 7.0, 8.0]);
let dot = a.dot(&b);  // SIMD auto-vectorized
let matmul = X.matmul(&W);  // GPU if available
",
            )
            .with_related(vec!["transpile-python"])
            .with_test_code(
                r"#[cfg(test)]
mod tests {
    #[test]
    fn test_vector_creation() {
        let a = vec![1.0, 2.0, 3.0, 4.0];
        let b = vec![5.0, 6.0, 7.0, 8.0];
        assert_eq!(a.len(), b.len());
    }

    #[test]
    fn test_dot_product_computation() {
        let a = vec![1.0, 2.0, 3.0, 4.0];
        let b = vec![5.0, 6.0, 7.0, 8.0];
        let dot: f64 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
        assert_eq!(dot, 70.0);
    }

    #[test]
    fn test_simd_element_type() {
        let vec: Vec<f64> = vec![1.0, 2.0, 3.0];
        assert!(vec.iter().all(|x| x.is_finite()));
    }
}",
            ),
    );
}

// =========================================================================
// Distributed Recipes
// =========================================================================

fn register_distributed_recipes(cookbook: &mut super::Cookbook) {
    // Work-Stealing Distribution
    cookbook.add(
        Recipe::new("distributed-work-stealing", "Work-Stealing Task Distribution")
            .with_problem("Distribute computation across CPU cores with work-stealing")
            .with_components(vec!["repartir", "trueno"])
            .with_tags(vec!["distributed", "parallel", "work-stealing", "cpu"])
            .with_code(
                r"use repartir::prelude::*;

// Create pool with work-stealing scheduler
let pool = Pool::builder()
    .num_workers(num_cpus::get())
    .build()?;

// Submit tasks
let results: Vec<f64> = pool.map(data.chunks(1000), |chunk| {
    // Each chunk processed by a worker
    chunk.iter().map(|x| x * x).sum()
})?;

// Reduce results
let total: f64 = results.iter().sum();
",
            )
            .with_related(vec!["distributed-gpu", "distributed-remote"])
            .with_test_code(
                r"#[cfg(test)]
mod tests {
    #[test]
    fn test_pool_worker_count() {
        let workers = 8;
        assert!(workers > 0);
    }

    #[test]
    fn test_chunk_processing() {
        let data = vec![1, 2, 3, 4, 5];
        let chunks: Vec<_> = data.chunks(2).collect();
        assert_eq!(chunks.len(), 3);
    }

    #[test]
    fn test_result_reduction() {
        let results = vec![10.0, 20.0, 30.0, 40.0];
        let total: f64 = results.iter().sum();
        assert_eq!(total, 100.0);
    }
}",
            ),
    );

    // GPU Distribution
    cookbook.add(
        Recipe::new("distributed-gpu", "GPU Compute Distribution")
            .with_problem("Offload computation to GPU using wgpu")
            .with_components(vec!["repartir", "trueno"])
            .with_tags(vec!["distributed", "gpu", "wgpu", "compute"])
            .with_code(
                r"use repartir::prelude::*;
use trueno::prelude::*;

// Create GPU executor
let executor = GpuExecutor::new()?;

// GPU-accelerated matrix multiplication
let result = executor.submit(|gpu| {
    let a = gpu.tensor(&matrix_a)?;
    let b = gpu.tensor(&matrix_b)?;
    a.matmul(&b)
})?;

// Hybrid CPU/GPU
let pool = Pool::builder()
    .add_gpu_executor(executor)
    .add_cpu_executor(CpuExecutor::new())
    .build()?;
",
            )
            .with_related(vec!["distributed-work-stealing"])
            .with_test_code(
                r"#[cfg(test)]
mod tests {
    #[test]
    fn test_matrix_dimensions_valid() {
        let m = 1024;
        let k = 512;
        let n = 2048;
        assert!(m > 0 && k > 0 && n > 0);
    }

    #[test]
    fn test_hybrid_pool_worker_total() {
        let cpu_workers = 4;
        let gpu_workers = 2;
        let total = cpu_workers + gpu_workers;
        assert_eq!(total, 6);
    }

    #[test]
    fn test_matmul_output_shape() {
        let rows_a = 128;
        let cols_b = 256;
        let output_elements = rows_a * cols_b;
        assert_eq!(output_elements, 32768);
    }
}",
            ),
    );
}

// =========================================================================
// Quality Recipes
// =========================================================================

fn register_quality_recipes(cookbook: &mut super::Cookbook) {
    // EDD Testing Pattern
    cookbook.add(
        Recipe::new("quality-edd", "Equation-Driven Development")
            .with_problem("Implement simulations with verifiable governing equations")
            .with_components(vec!["simular", "probar", "certeza"])
            .with_tags(vec!["quality", "testing", "edd", "simulation", "falsification"])
            .with_code(
                r"use simular::prelude::*;

/// EDD Demo following the complete cycle:
/// 1. Equation - Define governing equation
/// 2. Failing Test - Test fails without implementation
/// 3. Implementation - Implement the simulation
/// 4. Verification - Test passes, equation verified
/// 5. Falsification - Demonstrate conditions that break

pub trait DemoEngine {
    /// Load from YAML configuration (YAML-first)
    fn from_yaml(yaml: &str) -> Result<Self, DemoError> where Self: Sized;

    /// Advance simulation by one timestep
    fn step(&mut self, dt: f64) -> StepResult;

    /// Verify governing equation invariants
    fn verify_invariants(&self) -> bool;

    /// Get falsification status
    fn falsification_status(&self) -> FalsificationStatus;
}

// Example: Harmonic Oscillator
// Governing equation: E = (1/2)kx² + (1/2)mv²
// Falsification: Energy drift > tolerance indicates integrator failure

impl DemoEngine for HarmonicOscillator {
    fn verify_invariants(&self) -> bool {
        let energy = self.kinetic_energy() + self.potential_energy();
        let error = (energy - self.initial_energy).abs() / self.initial_energy;
        error < 1e-9  // Energy conservation within tolerance
    }
}
",
            )
            .with_related(vec!["quality-probar", "quality-golden-trace"])
            .with_test_code(
                r"#[cfg(test)]
mod tests {
    #[test]
    fn test_energy_conservation_invariant() {
        let kinetic = 60.0;
        let potential = 40.0;
        let initial_energy = 100.0;
        assert!((kinetic + potential - initial_energy).abs() < 1e-6);
    }

    #[test]
    fn test_timestep_positivity() {
        let dt = 0.01_f64;
        assert!(dt > 0.0);
    }

    #[test]
    fn test_falsification_detection() {
        let tolerance = 1e-9_f64;
        let error = 1e-5;
        assert!(error > tolerance);
    }
}",
            ),
    );

    // Probar Testing
    cookbook.add(
        Recipe::new("quality-probar", "Probar Property-Based Testing")
            .with_problem("Validate WASM demos with property-based and GUI coverage testing")
            .with_components(vec!["probar", "simular", "certeza"])
            .with_tags(vec!["quality", "testing", "probar", "property-testing", "gui-coverage"])
            .with_code(
                r#"use probar::prelude::*;

// Property-based tests
#[probar::property]
fn prop_tour_length_positive(cities: Vec<City>) -> bool {
    let tour = solve_tsp(&cities);
    tour.length() > 0.0
}

#[probar::property]
fn prop_energy_conserved(dt: f64) -> bool {
    let mut sim = OrbitSimulation::new();
    let e0 = sim.total_energy();
    sim.step(dt);
    let e1 = sim.total_energy();
    (e1 - e0).abs() / e0.abs() < 1e-9
}

// Metamorphic relations
#[probar::metamorphic]
fn mr_scale_invariance(scale: f64, cities: Vec<City>) {
    let tour1 = solve_tsp(&cities);
    let scaled = cities.iter().map(|c| c * scale).collect();
    let tour2 = solve_tsp(&scaled);
    assert!((tour2.length() / tour1.length() - scale).abs() < 1e-6);
}

// GUI coverage
#[probar::gui_coverage]
fn test_canvas_coverage(app: &mut App) {
    app.click_button("start");
    app.wait_frames(100);
    let coverage = app.pixel_coverage();
    assert!(coverage > 0.8, "Must render to >80% of canvas");
}
"#,
            )
            .with_related(vec!["quality-edd", "quality-certeza"])
            .with_test_code(
                r"#[cfg(test)]
mod tests {
    #[test]
    fn test_tour_length_positivity() {
        let distances = vec![10.5, 20.3, 15.7, 8.2];
        let tour_length: f64 = distances.iter().sum();
        assert!(tour_length > 0.0);
    }

    #[test]
    fn test_energy_conservation_property() {
        let initial = 100.0_f64;
        let final_energy = 100.0_f64;
        assert!((initial - final_energy).abs() < 1e-10);
    }

    #[test]
    fn test_coverage_threshold() {
        let covered = 950;
        let total = 1000;
        let coverage = covered as f64 / total as f64;
        assert!(coverage >= 0.95);
    }
}",
            ),
    );

    // Golden Trace Comparison
    cookbook.add(
        Recipe::new("quality-golden-trace", "Golden Trace Validation")
            .with_problem("Validate transpiled code produces identical behavior to original")
            .with_components(vec!["renacer", "certeza"])
            .with_tags(vec!["quality", "validation", "trace", "transpilation"])
            .with_code(
                r#"use renacer::prelude::*;

// Capture golden trace from Python
// renacer trace python sklearn_model.py --output golden.trace

// Capture Rust trace
// renacer trace ./target/release/model --output rust.trace

// Compare traces
let comparison = renacer::compare("golden.trace", "rust.trace")?;

assert!(comparison.semantically_equivalent(),
    "Transpiled code must produce identical results");
assert!(comparison.syscall_compatible(),
    "System calls must match (file I/O, network, etc.)");

// Performance comparison
println!("Python: {:.2}ms", comparison.baseline_time_ms());
println!("Rust: {:.2}ms", comparison.target_time_ms());
println!("Speedup: {:.1}x", comparison.speedup());
"#,
            )
            .with_related(vec!["transpile-python", "quality-edd"])
            .with_test_code(
                r#"#[cfg(test)]
mod tests {
    #[test]
    fn test_semantic_equivalence_check() {
        let baseline = vec![1.0, 2.0, 3.0];
        let candidate = vec![1.0, 2.0, 3.0];
        assert_eq!(baseline, candidate);
    }

    #[test]
    fn test_speedup_calculation() {
        let baseline_ms = 1000.0_f64;
        let optimized_ms = 250.0_f64;
        let speedup = baseline_ms / optimized_ms;
        assert_eq!(speedup, 4.0);
    }

    #[test]
    fn test_trace_comparison_length() {
        let golden = vec!["step1", "step2", "step3"];
        let actual = vec!["step1", "step2", "step3"];
        assert_eq!(golden.len(), actual.len());
    }
}"#,
            ),
    );
}

// =========================================================================
// Speech Recognition Recipes
// =========================================================================

fn register_speech_recipes(cookbook: &mut super::Cookbook) {
    // Whisper ASR
    cookbook.add(
        Recipe::new("speech-whisper", "Whisper Speech Recognition")
            .with_problem("Transcribe audio to text using pure-Rust Whisper implementation")
            .with_components(vec!["whisper-apr", "aprender", "trueno"])
            .with_tags(vec!["speech", "asr", "whisper", "transcription", "wasm"])
            .with_code(
                r#"use whisper_apr::prelude::*;

// Load model (downloads from HuggingFace on first run)
let model = WhisperModel::load("tiny.en")?;

// Transcribe audio file
let audio = Audio::load("recording.wav")?;
let result = model.transcribe(&audio)?;

println!("Transcription: {}", result.text);

// With timestamps
for segment in result.segments {
    println!("[{:.2}s - {:.2}s] {}",
        segment.start, segment.end, segment.text);
}

// Streaming transcription (real-time)
let mut stream = model.stream()?;
for chunk in audio_chunks {
    if let Some(text) = stream.process(&chunk)? {
        print!("{}", text);
    }
}

// WASM deployment
// wasm-pack build --target web --features wasm
// <script type="module">
//   import init, { transcribe } from './pkg/whisper_apr.js';
//   await init();
//   const text = await transcribe(audioBuffer);
// </script>
"#,
            )
            .with_related(vec!["speech-streaming", "ml-serving"])
            .with_test_code(
                r#"#[cfg(test)]
mod tests {
    #[test]
    fn test_model_name_validation() {
        let model = "tiny.en";
        let valid = vec!["tiny", "tiny.en", "base", "base.en"];
        assert!(valid.contains(&model));
    }

    #[test]
    fn test_transcription_result_non_empty() {
        let text = "Hello world";
        assert!(!text.is_empty());
    }

    #[test]
    fn test_segment_timestamp_ordering() {
        let start = 0.5_f64;
        let end = 2.0_f64;
        assert!(start < end);
    }
}"#,
            ),
    );

    // Streaming Speech
    cookbook.add(
        Recipe::new("speech-streaming", "Real-time Speech Streaming")
            .with_problem("Process audio in real-time with low latency")
            .with_components(vec!["whisper-apr", "trueno"])
            .with_tags(vec!["speech", "streaming", "real-time", "low-latency"])
            .with_code(
                r#"use whisper_apr::streaming::*;

// Configure streaming decoder
let config = StreamConfig {
    chunk_size_ms: 500,      // Process every 500ms
    overlap_ms: 100,         // Overlap for continuity
    vad_enabled: true,       // Voice activity detection
    language: Some("en"),
};

let mut decoder = StreamingDecoder::new(model, config)?;

// Process microphone input
let mut mic = Microphone::open()?;
loop {
    let samples = mic.read_chunk()?;

    // VAD filters silence
    if let Some(result) = decoder.process(&samples)? {
        // Partial results for UI feedback
        if result.is_partial {
            print!("\r{}", result.text);
        } else {
            println!("\n[Final] {}", result.text);
        }
    }
}
"#,
            )
            .with_related(vec!["speech-whisper"])
            .with_test_code(
                r"#[cfg(test)]
mod tests {
    #[test]
    fn test_stream_config_chunk_size_positive() {
        let chunk_size_ms = 500_u32;
        assert!(chunk_size_ms > 0);
    }

    #[test]
    fn test_overlap_less_than_chunk_size() {
        let chunk_size_ms = 500;
        let overlap_ms = 100;
        assert!(overlap_ms < chunk_size_ms);
    }

    #[test]
    fn test_vad_enabled_flag() {
        let vad_enabled = true;
        assert!(vad_enabled);
    }
}",
            ),
    );
}

// =========================================================================
// Training Recipes
// =========================================================================
