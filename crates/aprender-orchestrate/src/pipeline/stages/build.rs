//! Build stage - compiles to final binary.

use anyhow::{Context as AnyhowContext, Result};

#[cfg(feature = "native")]
use tracing::info;

// Stub macros for WASM build
#[cfg(not(feature = "native"))]
macro_rules! info {
    ($($arg:tt)*) => {{}};
}

use crate::pipeline::types::{PipelineContext, PipelineStage, ValidationResult};

/// Build stage - compiles to final binary
pub struct BuildStage {
    pub(crate) release: bool,
    pub(crate) target: Option<String>,
    pub(crate) wasm: bool,
}

impl BuildStage {
    pub fn new(release: bool, target: Option<String>, wasm: bool) -> Self {
        Self { release, target, wasm }
    }
}

#[async_trait::async_trait]
impl PipelineStage for BuildStage {
    fn name(&self) -> &'static str {
        "Build"
    }

    async fn execute(&self, mut ctx: PipelineContext) -> Result<PipelineContext> {
        info!("Building Rust project (release: {})", self.release);

        // Check if Cargo.toml exists in output directory
        let cargo_toml = ctx.output_path.join("Cargo.toml");
        if !cargo_toml.exists() {
            anyhow::bail!("No Cargo.toml found in output directory");
        }

        // Build cargo arguments
        let mut args = vec!["build"];

        if self.release {
            args.push("--release");
        }

        if let Some(target) = &self.target {
            args.push("--target");
            args.push(target);
        }

        // For WASM, use special target
        if self.wasm {
            args.push("--target");
            args.push("wasm32-unknown-unknown");
        }

        // Run cargo build
        let output = std::process::Command::new("cargo")
            .args(&args)
            .current_dir(&ctx.output_path)
            .output()
            .context("Failed to run cargo build")?;

        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            anyhow::bail!("Cargo build failed: {}", stderr);
        }

        ctx.metadata.insert(
            "build_mode".to_string(),
            serde_json::json!(if self.release { "release" } else { "debug" }),
        );

        if self.wasm {
            ctx.metadata.insert("wasm_build".to_string(), serde_json::json!(true));
        }

        Ok(ctx)
    }

    fn validate(&self, ctx: &PipelineContext) -> Result<ValidationResult> {
        // Check that build artifacts exist
        let build_dir = if self.release {
            ctx.output_path.join("target/release")
        } else {
            ctx.output_path.join("target/debug")
        };

        let passed = build_dir.exists();

        Ok(ValidationResult {
            stage: self.name().to_string(),
            passed,
            message: if passed {
                "Build artifacts found".to_string()
            } else {
                "Build directory not found".to_string()
            },
            details: None,
        })
    }
}
