//! Minimal JavaScript Generation (Zero-JavaScript Policy)
//!
//! Generates minimal JavaScript for WASM loading ONLY.
//! Enforces strict limit: under 20 lines of JavaScript.

use crate::result::{ProbarError, ProbarResult};
use serde::{Deserialize, Serialize};

/// Maximum allowed lines of JavaScript
pub const MAX_JS_LINES: usize = 20;

/// Generated JavaScript output
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GeneratedJs {
    /// JavaScript content
    pub content: String,
    /// Number of lines
    pub line_count: usize,
    /// Functions defined
    pub functions: Vec<String>,
}

impl GeneratedJs {
    /// Check if JS is within line limit
    #[must_use]
    pub fn within_limit(&self) -> bool {
        self.line_count <= MAX_JS_LINES
    }
}

/// WASM module configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WasmConfig {
    /// Path to WASM file
    pub path: String,
    /// Initial memory pages (64KB each)
    pub memory_initial: u32,
    /// Maximum memory pages
    pub memory_maximum: u32,
}

impl Default for WasmConfig {
    fn default() -> Self {
        Self {
            path: "app.wasm".to_string(),
            memory_initial: 256,  // 16 MB
            memory_maximum: 1024, // 64 MB
        }
    }
}

/// Minimal JavaScript builder for WASM loading
#[derive(Debug, Clone)]
pub struct JsBuilder {
    wasm_path: String,
    canvas_id: String,
    memory_initial: u32,
    memory_maximum: u32,
    entry_point: String,
}

impl JsBuilder {
    /// Create a new JS builder
    #[must_use]
    pub fn new(wasm_path: &str, canvas_id: &str) -> Self {
        Self {
            wasm_path: wasm_path.to_string(),
            canvas_id: canvas_id.to_string(),
            memory_initial: 256,
            memory_maximum: 1024,
            entry_point: "main".to_string(),
        }
    }

    /// Set memory configuration
    #[must_use]
    pub fn memory(mut self, initial: u32, maximum: u32) -> Self {
        self.memory_initial = initial;
        self.memory_maximum = maximum;
        self
    }

    /// Set entry point function name
    #[must_use]
    pub fn entry_point(mut self, name: &str) -> Self {
        self.entry_point = name.to_string();
        self
    }

    /// Build the minimal JavaScript loader
    ///
    /// # Errors
    ///
    /// Returns error if generated JS exceeds line limit
    pub fn build(self) -> ProbarResult<GeneratedJs> {
        // Generate minimal WASM loader
        let content = format!(
            r#"(async()=>{{
const c=document.getElementById('{canvas_id}');
const m=new WebAssembly.Memory({{initial:{mem_init},maximum:{mem_max}}});
const i={{env:{{memory:m,canvas:c}}}};
const{{instance:w}}=await WebAssembly.instantiateStreaming(fetch('{wasm_path}'),i);
w.exports.{entry}();
}})();"#,
            canvas_id = self.canvas_id,
            wasm_path = self.wasm_path,
            mem_init = self.memory_initial,
            mem_max = self.memory_maximum,
            entry = self.entry_point,
        );

        let line_count = content.lines().count();

        // Enforce line limit
        if line_count > MAX_JS_LINES {
            return Err(ProbarError::JsGeneration(format!(
                "Generated JavaScript exceeds {MAX_JS_LINES} line limit: {line_count} lines"
            )));
        }

        Ok(GeneratedJs {
            content,
            line_count,
            functions: vec!["main".to_string()],
        })
    }
}

/// Extended JS builder for additional minimal functionality
#[allow(dead_code)]
#[derive(Debug, Clone)]
pub struct ExtendedJsBuilder {
    base: JsBuilder,
    error_handler: bool,
    loading_indicator: bool,
}

#[allow(dead_code)]
impl ExtendedJsBuilder {
    /// Create from base builder
    #[must_use]
    pub fn new(wasm_path: &str, canvas_id: &str) -> Self {
        Self {
            base: JsBuilder::new(wasm_path, canvas_id),
            error_handler: false,
            loading_indicator: false,
        }
    }

    /// Enable error handling
    #[must_use]
    pub fn with_error_handler(mut self) -> Self {
        self.error_handler = true;
        self
    }

    /// Enable loading indicator
    #[must_use]
    pub fn with_loading_indicator(mut self) -> Self {
        self.loading_indicator = true;
        self
    }

    /// Set memory configuration
    #[must_use]
    pub fn memory(mut self, initial: u32, maximum: u32) -> Self {
        self.base = self.base.memory(initial, maximum);
        self
    }

    /// Build the JavaScript with optional features
    ///
    /// # Errors
    ///
    /// Returns error if generated JS exceeds line limit
    pub fn build(self) -> ProbarResult<GeneratedJs> {
        let mut lines = Vec::new();

        lines.push("(async()=>{".to_string());

        // Loading indicator
        if self.loading_indicator {
            lines.push("const l=document.querySelector('.loading');".to_string());
        }

        lines.push(format!(
            "const c=document.getElementById('{}');",
            self.base.canvas_id
        ));
        lines.push(format!(
            "const m=new WebAssembly.Memory({{initial:{},maximum:{}}});",
            self.base.memory_initial, self.base.memory_maximum
        ));

        // Error handler wrapper
        if self.error_handler {
            lines.push("try{".to_string());
        }

        lines.push(format!(
            "const{{instance:w}}=await WebAssembly.instantiateStreaming(fetch('{}'),{{env:{{memory:m,canvas:c}}}});",
            self.base.wasm_path
        ));

        // Hide loading indicator
        if self.loading_indicator {
            lines.push("if(l)l.style.display='none';".to_string());
        }

        lines.push(format!("w.exports.{}();", self.base.entry_point));

        // Error handler catch
        if self.error_handler {
            lines.push("}catch(e){console.error('WASM Error:',e);}".to_string());
        }

        lines.push("})();".to_string());

        let content = lines.join("\n");
        let line_count = lines.len();

        // Enforce line limit
        if line_count > MAX_JS_LINES {
            return Err(ProbarError::JsGeneration(format!(
                "Generated JavaScript exceeds {MAX_JS_LINES} line limit: {line_count} lines"
            )));
        }

        let mut functions = vec!["main".to_string()];
        if self.error_handler {
            functions.push("error_handler".to_string());
        }

        Ok(GeneratedJs {
            content,
            line_count,
            functions,
        })
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;

    // =========================================================================
    // H₀-JS-01: JsBuilder creation
    // =========================================================================

    #[test]
    fn h0_js_01_builder_new() {
        let builder = JsBuilder::new("app.wasm", "canvas");
        assert_eq!(builder.wasm_path, "app.wasm");
        assert_eq!(builder.canvas_id, "canvas");
    }

    #[test]
    fn h0_js_02_builder_memory() {
        let builder = JsBuilder::new("app.wasm", "c").memory(512, 2048);
        assert_eq!(builder.memory_initial, 512);
        assert_eq!(builder.memory_maximum, 2048);
    }

    #[test]
    fn h0_js_03_builder_entry_point() {
        let builder = JsBuilder::new("app.wasm", "c").entry_point("start");
        assert_eq!(builder.entry_point, "start");
    }

    // =========================================================================
    // H₀-JS-04: Generated JavaScript validation
    // =========================================================================

    #[test]
    fn h0_js_04_build_success() {
        let js = JsBuilder::new("test.wasm", "canvas").build().unwrap();

        assert!(js.within_limit());
        assert!(js.line_count <= MAX_JS_LINES);
    }

    #[test]
    fn h0_js_05_contains_wasm_loading() {
        let js = JsBuilder::new("game.wasm", "game").build().unwrap();

        assert!(js.content.contains("WebAssembly.instantiateStreaming"));
        assert!(js.content.contains("fetch('game.wasm')"));
    }

    #[test]
    fn h0_js_06_contains_canvas_reference() {
        let js = JsBuilder::new("app.wasm", "myCanvas").build().unwrap();

        assert!(js.content.contains("getElementById('myCanvas')"));
    }

    #[test]
    fn h0_js_07_contains_memory_config() {
        let js = JsBuilder::new("app.wasm", "c")
            .memory(128, 512)
            .build()
            .unwrap();

        assert!(js.content.contains("initial:128"));
        assert!(js.content.contains("maximum:512"));
    }

    #[test]
    fn h0_js_08_contains_entry_point() {
        let js = JsBuilder::new("app.wasm", "c")
            .entry_point("init")
            .build()
            .unwrap();

        assert!(js.content.contains(".init()"));
    }

    // =========================================================================
    // H₀-JS-09: Line limit enforcement
    // =========================================================================

    #[test]
    fn h0_js_09_under_20_lines() {
        let js = JsBuilder::new("app.wasm", "canvas").build().unwrap();

        assert!(
            js.line_count <= 20,
            "JS must be under 20 lines, got {}",
            js.line_count
        );
    }

    #[test]
    fn h0_js_10_within_limit_check() {
        let js = GeneratedJs {
            content: "test".to_string(),
            line_count: 10,
            functions: vec![],
        };
        assert!(js.within_limit());

        let over_limit = GeneratedJs {
            content: "test".to_string(),
            line_count: 25,
            functions: vec![],
        };
        assert!(!over_limit.within_limit());
    }

    // =========================================================================
    // H₀-JS-11: ExtendedJsBuilder
    // =========================================================================

    #[test]
    fn h0_js_11_extended_builder() {
        let js = ExtendedJsBuilder::new("app.wasm", "canvas")
            .build()
            .unwrap();

        assert!(js.within_limit());
    }

    #[test]
    fn h0_js_12_extended_with_error_handler() {
        let js = ExtendedJsBuilder::new("app.wasm", "canvas")
            .with_error_handler()
            .build()
            .unwrap();

        assert!(js.content.contains("try{"));
        assert!(js.content.contains("catch(e)"));
        assert!(js.within_limit());
    }

    #[test]
    fn h0_js_13_extended_with_loading() {
        let js = ExtendedJsBuilder::new("app.wasm", "canvas")
            .with_loading_indicator()
            .build()
            .unwrap();

        assert!(js.content.contains(".loading"));
        assert!(js.content.contains("display='none'"));
        assert!(js.within_limit());
    }

    #[test]
    fn h0_js_14_extended_all_features() {
        let js = ExtendedJsBuilder::new("app.wasm", "canvas")
            .with_error_handler()
            .with_loading_indicator()
            .memory(256, 1024)
            .build()
            .unwrap();

        assert!(js.content.contains("try{"));
        assert!(js.content.contains(".loading"));
        assert!(js.within_limit(), "Must stay under {} lines", MAX_JS_LINES);
    }

    // =========================================================================
    // H₀-JS-15: WasmConfig
    // =========================================================================

    #[test]
    fn h0_js_15_wasm_config_default() {
        let config = WasmConfig::default();
        assert_eq!(config.path, "app.wasm");
        assert_eq!(config.memory_initial, 256);
        assert_eq!(config.memory_maximum, 1024);
    }

    // =========================================================================
    // H₀-JS-16: Generated structure
    // =========================================================================

    #[test]
    fn h0_js_16_generated_js_fields() {
        let js = JsBuilder::new("test.wasm", "c").build().unwrap();

        assert!(!js.content.is_empty());
        assert!(js.line_count > 0);
        assert!(!js.functions.is_empty());
    }
}
