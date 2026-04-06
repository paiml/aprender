//! Transpilation stage - converts source to Rust.

use anyhow::Result;

#[cfg(feature = "native")]
use tracing::{info, warn};

// Stub macros for WASM build
#[cfg(not(feature = "native"))]
macro_rules! info {
    ($($arg:tt)*) => {{}};
}

#[cfg(not(feature = "native"))]
macro_rules! warn {
    ($($arg:tt)*) => {{}};
}

use crate::pipeline::types::{PipelineContext, PipelineStage, ValidationResult};
use crate::pipeline_analysis::LibraryAnalyzer;

/// Transpilation stage - converts source to Rust
pub struct TranspilationStage {
    pub(crate) incremental: bool,
    pub(crate) cache: bool,
    pub(crate) library_analyzer: LibraryAnalyzer,
}

impl TranspilationStage {
    pub fn new(incremental: bool, cache: bool) -> Self {
        Self { incremental, cache, library_analyzer: LibraryAnalyzer::new() }
    }
}

#[async_trait::async_trait]
impl PipelineStage for TranspilationStage {
    fn name(&self) -> &'static str {
        "Transpilation"
    }

    async fn execute(&self, mut ctx: PipelineContext) -> Result<PipelineContext> {
        info!(
            "Transpiling {} to Rust",
            ctx.primary_language
                .as_ref()
                .map(|l| format!("{}", l))
                .unwrap_or_else(|| "unknown".to_string())
        );

        // Create output directory structure
        std::fs::create_dir_all(&ctx.output_path)?;
        std::fs::create_dir_all(ctx.output_path.join("src"))?;

        // If Python project, analyze NumPy usage
        #[cfg(feature = "native")]
        if let Some(crate::types::Language::Python) = ctx.primary_language {
            self.analyze_python_libraries(&mut ctx)?;
        }

        // Detect available tools
        let tools = crate::tools::ToolRegistry::detect();

        // Get transpiler for primary language and run transpilation
        if let Some(lang) = &ctx.primary_language {
            use crate::types::Language;

            info!("Starting transpilation for language: {}", lang);

            let result = match lang {
                Language::Python => {
                    if tools.depyler.is_some() {
                        info!("Using Depyler for Python transpilation");
                        crate::tools::transpile_python(&ctx.input_path, &ctx.output_path)
                    } else {
                        anyhow::bail!("Depyler not available. Install with: cargo install depyler");
                    }
                }
                Language::Shell => {
                    if tools.bashrs.is_some() {
                        info!("Using Bashrs for Shell transpilation");
                        crate::tools::transpile_shell(&ctx.input_path, &ctx.output_path)
                    } else {
                        anyhow::bail!("Bashrs not available. Install with: cargo install bashrs");
                    }
                }
                Language::C | Language::Cpp => {
                    if tools.decy.is_some() {
                        info!("Using Decy for C/C++ transpilation");
                        crate::tools::transpile_c_cpp(&ctx.input_path, &ctx.output_path)
                    } else {
                        anyhow::bail!("Decy not available. Install with: cargo install decy");
                    }
                }
                _ => {
                    anyhow::bail!("No transpiler available for language: {}", lang);
                }
            };

            match result {
                Ok(output) => {
                    info!("Transpilation completed successfully");
                    info!("Output: {}", output);

                    ctx.metadata
                        .insert("transpiler".to_string(), serde_json::json!(format!("{}", lang)));
                    ctx.metadata
                        .insert("transpilation_output".to_string(), serde_json::json!(output));
                }
                Err(e) => {
                    warn!("Transpilation failed: {}", e);
                    anyhow::bail!("Transpilation failed: {}", e);
                }
            }
        } else {
            anyhow::bail!("No primary language detected");
        }

        Ok(ctx)
    }

    fn validate(&self, ctx: &PipelineContext) -> Result<ValidationResult> {
        // Check that output directory exists and has content
        let src_dir = ctx.output_path.join("src");
        let passed = src_dir.exists() && src_dir.read_dir()?.next().is_some();

        Ok(ValidationResult {
            stage: self.name().to_string(),
            passed,
            message: if passed {
                "Transpilation output validated".to_string()
            } else {
                "No transpiled files found".to_string()
            },
            details: None,
        })
    }
}

impl TranspilationStage {
    #[cfg(feature = "native")]
    #[allow(clippy::cognitive_complexity)]
    fn analyze_python_libraries(&self, ctx: &mut PipelineContext) -> Result<()> {
        // Analyze NumPy usage
        info!("Analyzing NumPy usage for conversion guidance");
        match self.library_analyzer.analyze_numpy_usage(&ctx.input_path) {
            Ok(recommendations) => {
                if !recommendations.is_empty() {
                    info!("Found {} NumPy operations to convert:", recommendations.len());
                    for rec in &recommendations {
                        info!("  - {}", rec);
                    }

                    ctx.metadata.insert(
                        "numpy_conversions".to_string(),
                        serde_json::json!(recommendations),
                    );

                    ctx.metadata.insert("numpy_detected".to_string(), serde_json::json!(true));
                } else {
                    info!("No NumPy usage detected");
                    ctx.metadata.insert("numpy_detected".to_string(), serde_json::json!(false));
                }
            }
            Err(e) => {
                warn!("NumPy analysis failed: {}", e);
            }
        }

        // Analyze sklearn usage
        info!("Analyzing sklearn usage for conversion guidance");
        match self.library_analyzer.analyze_sklearn_usage(&ctx.input_path) {
            Ok(recommendations) => {
                if !recommendations.is_empty() {
                    info!("Found {} sklearn algorithms to convert:", recommendations.len());
                    for rec in &recommendations {
                        info!("  - {}", rec);
                    }

                    ctx.metadata.insert(
                        "sklearn_conversions".to_string(),
                        serde_json::json!(recommendations),
                    );

                    ctx.metadata.insert("sklearn_detected".to_string(), serde_json::json!(true));
                } else {
                    info!("No sklearn usage detected");
                    ctx.metadata.insert("sklearn_detected".to_string(), serde_json::json!(false));
                }
            }
            Err(e) => {
                warn!("sklearn analysis failed: {}", e);
            }
        }

        // Analyze PyTorch usage
        info!("Analyzing PyTorch usage for conversion guidance");
        match self.library_analyzer.analyze_pytorch_usage(&ctx.input_path) {
            Ok(recommendations) => {
                if !recommendations.is_empty() {
                    info!("Found {} PyTorch operations to convert:", recommendations.len());
                    for rec in &recommendations {
                        info!("  - {}", rec);
                    }

                    ctx.metadata.insert(
                        "pytorch_conversions".to_string(),
                        serde_json::json!(recommendations),
                    );

                    ctx.metadata.insert("pytorch_detected".to_string(), serde_json::json!(true));
                } else {
                    info!("No PyTorch usage detected");
                    ctx.metadata.insert("pytorch_detected".to_string(), serde_json::json!(false));
                }
            }
            Err(e) => {
                warn!("PyTorch analysis failed: {}", e);
            }
        }

        Ok(())
    }
}
