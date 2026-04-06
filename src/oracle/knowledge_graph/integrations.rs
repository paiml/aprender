//! Integration patterns between stack components.

use super::super::types::IntegrationPattern;
use super::types::KnowledgeGraph;

impl KnowledgeGraph {
    /// Register integration patterns
    pub(crate) fn register_integration_patterns(&mut self) {
        // ML Pipeline patterns
        self.integrations.push(IntegrationPattern {
            from: "aprender".into(),
            to: "realizar".into(),
            pattern_name: "model_export".into(),
            description: "Export trained model for serving".into(),
            code_template: Some(
                r#"// Train with aprender
let model = RandomForest::new().fit(&X, &y)?;
// Export for realizar
model.save_apr("model.apr")?;"#
                    .into(),
            ),
        });

        self.integrations.push(IntegrationPattern {
            from: "aprender".into(),
            to: "trueno".into(),
            pattern_name: "tensor_backend".into(),
            description: "Use trueno as compute backend for aprender".into(),
            code_template: None,
        });

        self.integrations.push(IntegrationPattern {
            from: "entrenar".into(),
            to: "realizar".into(),
            pattern_name: "training_to_inference".into(),
            description: "Export fine-tuned model for inference".into(),
            code_template: None,
        });

        // Transpilation patterns
        self.integrations.push(IntegrationPattern {
            from: "depyler".into(),
            to: "aprender".into(),
            pattern_name: "sklearn_convert".into(),
            description: "Convert sklearn code to aprender".into(),
            code_template: Some(
                r"// depyler converts:
// from sklearn.ensemble import RandomForestClassifier
// to:
use aprender::tree::RandomForestClassifier;"
                    .into(),
            ),
        });

        self.integrations.push(IntegrationPattern {
            from: "depyler".into(),
            to: "trueno".into(),
            pattern_name: "numpy_convert".into(),
            description: "Convert numpy operations to trueno".into(),
            code_template: None,
        });

        self.integrations.push(IntegrationPattern {
            from: "batuta".into(),
            to: "depyler".into(),
            pattern_name: "orchestrated_transpilation".into(),
            description: "Batuta orchestrates depyler for Python projects".into(),
            code_template: None,
        });

        // Distribution patterns
        self.integrations.push(IntegrationPattern {
            from: "repartir".into(),
            to: "entrenar".into(),
            pattern_name: "distributed_training".into(),
            description: "Distribute training across nodes".into(),
            code_template: None,
        });

        self.integrations.push(IntegrationPattern {
            from: "repartir".into(),
            to: "trueno".into(),
            pattern_name: "distributed_compute".into(),
            description: "Distribute tensor operations".into(),
            code_template: None,
        });

        // Data patterns
        self.integrations.push(IntegrationPattern {
            from: "alimentar".into(),
            to: "aprender".into(),
            pattern_name: "data_loading".into(),
            description: "Load data for ML training".into(),
            code_template: Some(
                r#"use alimentar::CsvReader;
use aprender::Dataset;

let data = CsvReader::from_path("data.csv")?.load()?;
let dataset = Dataset::from(data);"#
                    .into(),
            ),
        });

        // Ruchy patterns
        self.integrations.push(IntegrationPattern {
            from: "ruchy".into(),
            to: "batuta".into(),
            pattern_name: "scripted_orchestration".into(),
            description: "Use ruchy scripts to drive batuta pipelines".into(),
            code_template: Some(
                r#"// ruchy script for pipeline automation
let result = batuta::analyze("./project")?;
let transpiled = batuta::transpile(&result)?;
batuta::validate(&transpiled)?;"#
                    .into(),
            ),
        });

        self.integrations.push(IntegrationPattern {
            from: "ruchy".into(),
            to: "renacer".into(),
            pattern_name: "scripted_tracing".into(),
            description: "Automate syscall tracing with ruchy scripts".into(),
            code_template: None,
        });

        // Bashrs patterns
        self.integrations.push(IntegrationPattern {
            from: "bashrs".into(),
            to: "batuta".into(),
            pattern_name: "bootstrap_generation".into(),
            description: "Generate shell bootstrap scripts from Rust".into(),
            code_template: Some(
                r#"// Generate portable install script
bashrs::transpile! {
    check_rust_installed();
    cargo_install("batuta");
    batuta_init("./project");
}"#
                .into(),
            ),
        });

        self.integrations.push(IntegrationPattern {
            from: "bashrs".into(),
            to: "repartir".into(),
            pattern_name: "cluster_bootstrap".into(),
            description: "Generate shell scripts for cluster node setup".into(),
            code_template: None,
        });

        // Quality patterns
        self.integrations.push(IntegrationPattern {
            from: "renacer".into(),
            to: "certeza".into(),
            pattern_name: "trace_validation".into(),
            description: "Validate performance with golden traces".into(),
            code_template: None,
        });

        self.integrations.push(IntegrationPattern {
            from: "pmat".into(),
            to: "batuta".into(),
            pattern_name: "quality_gating".into(),
            description: "Gate pipeline on quality metrics".into(),
            code_template: None,
        });

        // Zero-JS WASM patterns (simular)
        self.integrations.push(IntegrationPattern {
            from: "simular".into(),
            to: "trueno".into(),
            pattern_name: "zero_js_wasm".into(),
            description: "Zero-JS WASM architecture for interactive demos".into(),
            code_template: Some(
                r#"// HTML: ONE LINE OF JAVASCRIPT
// <script type="module">import init, { initApp } from './pkg/app.js'; init().then(initApp);</script>

// Rust: Full app in WASM
use wasm_bindgen::prelude::*;
use web_sys::{Document, HtmlCanvasElement, CanvasRenderingContext2d};

#[wasm_bindgen(js_name = initApp)]
pub fn init_app() -> Result<(), JsValue> {
    let window = web_sys::window().expect("no window");
    let document = window.document().expect("no document");
    let canvas = document.get_element_by_id("canvas")
        .expect("no canvas")
        .dyn_into::<HtmlCanvasElement>()?;

    // All DOM, rendering, animation in Rust
    let ctx = canvas.get_context("2d")?.expect("unexpected failure")
        .dyn_into::<CanvasRenderingContext2d>()?;

    // requestAnimationFrame loop in Rust
    // Event handlers in Rust closures
    Ok(())
}"#
                    .into(),
            ),
        });

        self.integrations.push(IntegrationPattern {
            from: "simular".into(),
            to: "probar".into(),
            pattern_name: "edd_testing".into(),
            description: "Equation-Driven Development with Probar validation".into(),
            code_template: Some(
                r"// EDD Demo pattern
pub trait DemoEngine {
    fn from_yaml(yaml: &str) -> Result<Self, DemoError>;
    fn step(&mut self, dt: f64) -> StepResult;
    fn verify_invariants(&self) -> bool;
    fn falsification_status(&self) -> FalsificationStatus;
}

// Probar validates:
// - Governing equations hold within tolerance
// - Invariants never violated
// - Deterministic replay produces identical results"
                    .into(),
            ),
        });

        self.integrations.push(IntegrationPattern {
            from: "probar".into(),
            to: "certeza".into(),
            pattern_name: "gui_coverage".into(),
            description: "GUI/pixel coverage extends code coverage".into(),
            code_template: None,
        });

        self.integrations.push(IntegrationPattern {
            from: "simular".into(),
            to: "repartir".into(),
            pattern_name: "distributed_simulation".into(),
            description: "Distribute Monte Carlo simulations across nodes".into(),
            code_template: None,
        });

        // APR QA patterns
        self.integrations.push(IntegrationPattern {
            from: "apr-qa".into(),
            to: "aprender".into(),
            pattern_name: "model_validation".into(),
            description: "Validate APR models trained with aprender".into(),
            code_template: Some(
                r#"// Generate and run QA tests for APR model
use apr_qa_gen::TestGenerator;
use apr_qa_runner::Runner;

let tests = TestGenerator::for_model("model.apr")?.generate()?;
let results = Runner::new().run(&tests)?;"#
                    .into(),
            ),
        });

        self.integrations.push(IntegrationPattern {
            from: "apr-qa".into(),
            to: "realizar".into(),
            pattern_name: "inference_validation".into(),
            description: "Validate APR model inference with realizar".into(),
            code_template: None,
        });

        self.integrations.push(IntegrationPattern {
            from: "apr-qa".into(),
            to: "certeza".into(),
            pattern_name: "qa_quality_gates".into(),
            description: "Integrate APR QA into quality gate pipeline".into(),
            code_template: None,
        });

        // Provable contracts patterns
        self.integrations.push(IntegrationPattern {
            from: "provable-contracts".into(),
            to: "trueno".into(),
            pattern_name: "kernel_verification".into(),
            description: "Verify trueno SIMD kernel correctness via YAML contracts".into(),
            code_template: None,
        });

        self.integrations.push(IntegrationPattern {
            from: "provable-contracts".into(),
            to: "realizar".into(),
            pattern_name: "inference_kernel_verification".into(),
            description: "Verify realizar Q4K/Q6K kernel contracts".into(),
            code_template: None,
        });

        self.integrations.push(IntegrationPattern {
            from: "provable-contracts".into(),
            to: "probar".into(),
            pattern_name: "contract_to_property_tests".into(),
            description: "Generate probar property tests from YAML contracts".into(),
            code_template: None,
        });

        self.integrations.push(IntegrationPattern {
            from: "provable-contracts".into(),
            to: "certeza".into(),
            pattern_name: "formal_quality_gates".into(),
            description: "Kani verification as quality gate".into(),
            code_template: None,
        });

        // Tiny model ground truth patterns
        self.integrations.push(IntegrationPattern {
            from: "tiny-model-ground-truth".into(),
            to: "realizar".into(),
            pattern_name: "inference_parity".into(),
            description: "Validate realizar inference against HuggingFace oracle".into(),
            code_template: None,
        });

        self.integrations.push(IntegrationPattern {
            from: "tiny-model-ground-truth".into(),
            to: "aprender".into(),
            pattern_name: "conversion_parity".into(),
            description: "Validate APR format conversions".into(),
            code_template: None,
        });

        // Forjar patterns
        self.integrations.push(IntegrationPattern {
            from: "forjar".into(),
            to: "bashrs".into(),
            pattern_name: "bootstrap_provisioning".into(),
            description: "Generate shell provisioning scripts".into(),
            code_template: None,
        });

        self.integrations.push(IntegrationPattern {
            from: "forjar".into(),
            to: "repartir".into(),
            pattern_name: "cluster_provisioning".into(),
            description: "Provision distributed compute nodes".into(),
            code_template: None,
        });

        // Media Production patterns (rmedia)
        self.integrations.push(IntegrationPattern {
            from: "whisper-apr".into(),
            to: "rmedia".into(),
            pattern_name: "transcription_pipeline".into(),
            description:
                "Transcribe course audio with whisper-apr, feed into rmedia subtitle pipeline"
                    .into(),
            code_template: Some(
                r#"// 1. Transcribe audio with whisper-apr
let model = WhisperModel::from_apr("whisper-base.apr")?;
let transcript = model.transcribe(&audio)?;

// 2. Burn subtitles into video with rmedia
rmedia::subtitle::burn_in("lecture.mp4", &transcript.srt(), "output.mp4")?;"#
                    .into(),
            ),
        });

        self.integrations.push(IntegrationPattern {
            from: "rmedia".into(),
            to: "whisper-apr".into(),
            pattern_name: "audio_extraction".into(),
            description: "Extract audio from video for transcription".into(),
            code_template: None,
        });

        self.integrations.push(IntegrationPattern {
            from: "rmedia".into(),
            to: "certeza".into(),
            pattern_name: "course_quality_gate".into(),
            description:
                "Score course artifacts against quality dimensions, enforce critical defect gates"
                    .into(),
            code_template: None,
        });

        self.integrations.push(IntegrationPattern {
            from: "rmedia".into(),
            to: "batuta".into(),
            pattern_name: "orchestrated_rendering".into(),
            description: "Batuta orchestrates rmedia for course video production pipelines".into(),
            code_template: None,
        });
    }
}
