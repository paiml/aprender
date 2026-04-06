//! WebAssembly API for Batuta
//!
//! Provides JavaScript-friendly interfaces for core Batuta functionality
//! that operates on in-memory code without file system access.
//!
//! # Features
//!
//! - **Language Detection**: Analyze code snippets to detect languages
//! - **Backend Selection**: Recommend optimal compute backend (SIMD/GPU)
//! - **NumPy Conversion**: Convert NumPy operations to Trueno
//! - **sklearn Conversion**: Convert sklearn algorithms to Aprender
//! - **PyTorch Conversion**: Convert PyTorch operations to Realizar
//! - **Code Analysis**: PARF pattern detection and analysis
//!
//! # Example (JavaScript)
//!
//! ```javascript
//! import init, { analyze_code, convert_numpy, backend_recommend } from './batuta.js';
//!
//! await init();
//!
//! // Detect language
//! const analysis = analyze_code("import numpy as np\nx = np.array([1, 2, 3])");
//! console.log(analysis.language); // "Python"
//!
//! // Convert NumPy to Trueno
//! const conversion = convert_numpy("np.add(a, b)");
//! console.log(conversion.rust_code);
//!
//! // Get backend recommendation
//! const backend = backend_recommend("matmul", 1024);
//! console.log(backend); // "SIMD" or "GPU"
//! ```

#[cfg(feature = "wasm")]
use wasm_bindgen::prelude::*;

#[cfg(feature = "wasm")]
use serde::{Deserialize, Serialize};

#[cfg(feature = "wasm")]
use crate::backend::{BackendSelector, OpComplexity};
#[cfg(feature = "wasm")]
use crate::numpy_converter::{NumPyConverter, NumPyOp};
#[cfg(feature = "wasm")]
use crate::pytorch_converter::{PyTorchConverter, PyTorchOperation};
#[cfg(feature = "wasm")]
use crate::sklearn_converter::{SklearnAlgorithm, SklearnConverter};

/// Initialize the WASM module (sets panic hook for better error messages)
#[cfg(feature = "wasm")]
#[wasm_bindgen(start)]
pub fn wasm_init() {
    // Set panic hook for better error messages in browser console
    // Note: console_error_panic_hook is not included as a feature
    // Add feature to Cargo.toml if needed: console_error_panic_hook = "0.1"

    web_sys::console::log_1(&"Batuta WASM module initialized".into());
}

/// Analysis result for code snippets
#[cfg(feature = "wasm")]
#[wasm_bindgen]
#[allow(clippy::unsafe_derive_deserialize)]
#[derive(Serialize, Deserialize)]
pub struct AnalysisResult {
    language: String,
    has_numpy: bool,
    has_sklearn: bool,
    has_pytorch: bool,
    lines_of_code: usize,
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
impl AnalysisResult {
    #[wasm_bindgen(getter)]
    pub fn language(&self) -> String {
        self.language.clone()
    }
    #[wasm_bindgen(getter)]
    pub fn has_numpy(&self) -> bool {
        self.has_numpy
    }
    #[wasm_bindgen(getter)]
    pub fn has_sklearn(&self) -> bool {
        self.has_sklearn
    }
    #[wasm_bindgen(getter)]
    pub fn has_pytorch(&self) -> bool {
        self.has_pytorch
    }
    #[wasm_bindgen(getter)]
    pub fn lines_of_code(&self) -> usize {
        self.lines_of_code
    }

    /// Get JSON representation
    #[wasm_bindgen(js_name = toJSON)]
    pub fn to_json(&self) -> Result<String, JsValue> {
        serde_json::to_string(self)
            .map_err(|e| JsValue::from_str(&format!("Serialization error: {}", e)))
    }
}

/// Conversion result for code transformations
#[cfg(feature = "wasm")]
#[wasm_bindgen]
#[allow(clippy::unsafe_derive_deserialize)]
#[derive(Serialize, Deserialize)]
pub struct ConversionResult {
    original_code: String,
    rust_code: String,
    imports: String,
    backend_recommendation: String,
    complexity: String,
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
impl ConversionResult {
    #[wasm_bindgen(getter)]
    pub fn original_code(&self) -> String {
        self.original_code.clone()
    }
    #[wasm_bindgen(getter)]
    pub fn rust_code(&self) -> String {
        self.rust_code.clone()
    }
    #[wasm_bindgen(getter)]
    pub fn imports(&self) -> String {
        self.imports.clone()
    }
    #[wasm_bindgen(getter)]
    pub fn backend_recommendation(&self) -> String {
        self.backend_recommendation.clone()
    }
    #[wasm_bindgen(getter)]
    pub fn complexity(&self) -> String {
        self.complexity.clone()
    }

    /// Get JSON representation
    #[wasm_bindgen(js_name = toJSON)]
    pub fn to_json(&self) -> Result<String, JsValue> {
        serde_json::to_string(self)
            .map_err(|e| JsValue::from_str(&format!("Serialization error: {}", e)))
    }
}

/// Analyze code snippet and detect language and dependencies
///
/// # Arguments
/// * `code` - Source code to analyze
///
/// # Returns
/// Analysis result with language detection and dependency info
#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn analyze_code(code: &str) -> Result<AnalysisResult, JsValue> {
    let lines: Vec<&str> = code.lines().collect();

    const LANG_PATTERNS: &[(&[&str], &str)] = &[
        (&["import ", "def ", "class "], "Python"),
        (&["#include", "int main"], "C/C++"),
        (&["fn ", "struct "], "Rust"),
        (&["#!/bin/bash", "#!/bin/sh"], "Shell"),
    ];
    let language = LANG_PATTERNS
        .iter()
        .find(|(pats, _)| pats.iter().any(|p| code.contains(p)))
        .map(|(_, lang)| *lang)
        .unwrap_or("Unknown");

    // Detect ML libraries
    let has_numpy = code.contains("numpy") || code.contains("np.");
    let has_sklearn = code.contains("sklearn");
    let has_pytorch = code.contains("torch") || code.contains("transformers");

    Ok(AnalysisResult {
        language: language.to_string(),
        has_numpy,
        has_sklearn,
        has_pytorch,
        lines_of_code: lines.len(),
    })
}

/// Convert NumPy operation to Trueno Rust code
///
/// # Arguments
/// * `numpy_code` - NumPy operation string (e.g., "np.add(a, b)")
/// * `data_size` - Optional data size for backend recommendation
///
/// # Returns
/// Conversion result with Rust code and backend recommendation
#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn convert_numpy(
    numpy_code: &str,
    data_size: Option<usize>,
) -> Result<ConversionResult, JsValue> {
    let converter = NumPyConverter::new();

    // Data-driven pattern matching for NumPy operations
    const NUMPY_PATTERNS: &[(&[&str], NumPyOp)] = &[
        (&["np.add", "numpy.add"], NumPyOp::Add),
        (&["np.dot", "numpy.dot"], NumPyOp::Dot),
        (&["np.sum", "numpy.sum"], NumPyOp::Sum),
        (&["np.mean", "numpy.mean"], NumPyOp::Mean),
        (&["np.array", "numpy.array"], NumPyOp::Array),
        (&["reshape"], NumPyOp::Reshape),
        (&["transpose", ".T"], NumPyOp::Transpose),
    ];
    let op = NUMPY_PATTERNS
        .iter()
        .find(|(pats, _)| pats.iter().any(|p| numpy_code.contains(p)))
        .map(|(_, op)| op.clone())
        .ok_or_else(|| JsValue::from_str("Unsupported NumPy operation"))?;

    let trueno_op = converter
        .convert(&op)
        .ok_or_else(|| JsValue::from_str("Conversion failed: operation not supported"))?;

    let size = data_size.unwrap_or(1000);
    let backend = converter.recommend_backend(&op, size);

    Ok(ConversionResult {
        original_code: numpy_code.to_string(),
        rust_code: trueno_op.code_template.clone(),
        imports: trueno_op.imports.join("\n"),
        backend_recommendation: format!("{:?}", backend),
        complexity: format!("{:?}", trueno_op.complexity),
    })
}

/// Convert sklearn algorithm to Aprender Rust code
///
/// # Arguments
/// * `sklearn_code` - sklearn algorithm string (e.g., "LinearRegression()")
/// * `data_size` - Optional data size for backend recommendation
///
/// # Returns
/// Conversion result with Rust code and backend recommendation
#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn convert_sklearn(
    sklearn_code: &str,
    data_size: Option<usize>,
) -> Result<ConversionResult, JsValue> {
    let converter = SklearnConverter::new();

    const SKLEARN_PATTERNS: &[(&str, SklearnAlgorithm)] = &[
        ("LinearRegression", SklearnAlgorithm::LinearRegression),
        ("LogisticRegression", SklearnAlgorithm::LogisticRegression),
        ("KMeans", SklearnAlgorithm::KMeans),
        ("DecisionTreeClassifier", SklearnAlgorithm::DecisionTreeClassifier),
        ("RandomForestClassifier", SklearnAlgorithm::RandomForestClassifier),
        ("StandardScaler", SklearnAlgorithm::StandardScaler),
    ];
    let algo = SKLEARN_PATTERNS
        .iter()
        .find(|(pat, _)| sklearn_code.contains(pat))
        .map(|(_, algo)| algo.clone())
        .ok_or_else(|| JsValue::from_str("Unsupported sklearn algorithm"))?;

    let aprender_algo = converter
        .convert(&algo)
        .ok_or_else(|| JsValue::from_str("Conversion failed: algorithm not supported"))?;

    let size = data_size.unwrap_or(1000);
    let backend = converter.recommend_backend(&algo, size);

    Ok(ConversionResult {
        original_code: sklearn_code.to_string(),
        rust_code: aprender_algo.code_template.clone(),
        imports: aprender_algo.imports.join("\n"),
        backend_recommendation: format!("{:?}", backend),
        complexity: format!("{:?}", aprender_algo.complexity),
    })
}

/// Convert PyTorch operation to Realizar Rust code
///
/// # Arguments
/// * `pytorch_code` - PyTorch operation string (e.g., "model.generate()")
/// * `data_size` - Optional data size for backend recommendation
///
/// # Returns
/// Conversion result with Rust code and backend recommendation
#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn convert_pytorch(
    pytorch_code: &str,
    data_size: Option<usize>,
) -> Result<ConversionResult, JsValue> {
    let converter = PyTorchConverter::new();

    const PYTORCH_PATTERNS: &[(&[&str], PyTorchOperation)] = &[
        (&["generate"], PyTorchOperation::Generate),
        (&["forward"], PyTorchOperation::Forward),
        (&["encode"], PyTorchOperation::Encode),
        (&["decode"], PyTorchOperation::Decode),
        (&["nn.Linear"], PyTorchOperation::Linear),
        (&["Attention"], PyTorchOperation::Attention),
    ];
    // LoadModel needs special compound check (both "load" AND "model")
    let op = if pytorch_code.contains("load") && pytorch_code.contains("model") {
        PyTorchOperation::LoadModel
    } else {
        PYTORCH_PATTERNS
            .iter()
            .find(|(pats, _)| pats.iter().any(|p| pytorch_code.contains(p)))
            .map(|(_, op)| op.clone())
            .ok_or_else(|| JsValue::from_str("Unsupported PyTorch operation"))?
    };

    let realizar_op = converter
        .convert(&op)
        .ok_or_else(|| JsValue::from_str("Conversion failed: operation not supported"))?;

    let size = data_size.unwrap_or(1000000); // LLMs are large by default
    let backend = converter.recommend_backend(&op, size);

    Ok(ConversionResult {
        original_code: pytorch_code.to_string(),
        rust_code: realizar_op.code_template.clone(),
        imports: realizar_op.imports.join("\n"),
        backend_recommendation: format!("{:?}", backend),
        complexity: format!("{:?}", realizar_op.complexity),
    })
}

/// Get backend recommendation for an operation
///
/// # Arguments
/// * `operation_type` - Type of operation ("element-wise", "reduction", "matmul")
/// * `data_size` - Size of data to process
///
/// # Returns
/// Recommended backend as string ("Scalar", "SIMD", "GPU")
#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn backend_recommend(operation_type: &str, data_size: usize) -> Result<String, JsValue> {
    let selector = BackendSelector::new();

    let backend = match operation_type.to_lowercase().as_str() {
        "element-wise" | "elementwise" => selector.select_for_elementwise(data_size),
        "reduction" | "reduce" => {
            let complexity = if data_size < 10_000 {
                OpComplexity::Low
            } else if data_size < 100_000 {
                OpComplexity::Medium
            } else {
                OpComplexity::High
            };
            selector.select_with_moe(complexity, data_size)
        }
        "matmul" | "matrix-multiply" => {
            let n = (data_size as f64).sqrt() as usize;
            selector.select_for_matmul(n, n, n)
        }
        _ => return Err(JsValue::from_str("Unknown operation type")),
    };

    Ok(format!("{:?}", backend))
}

/// Get Batuta version
#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn version() -> String {
    env!("CARGO_PKG_VERSION").to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    // ============================================================================
    // ANALYSIS RESULT TESTS (Native-compatible)
    // ============================================================================

    #[test]
    #[cfg(feature = "wasm")]
    fn test_analysis_result_direct_construction() {
        let result = AnalysisResult {
            language: "Python".to_string(),
            has_numpy: true,
            has_sklearn: false,
            has_pytorch: true,
            lines_of_code: 42,
        };

        // Test direct field access (not through wasm_bindgen getters)
        assert_eq!(result.language, "Python");
        assert!(result.has_numpy);
        assert!(!result.has_sklearn);
        assert!(result.has_pytorch);
        assert_eq!(result.lines_of_code, 42);
    }

    #[test]
    #[cfg(feature = "wasm")]
    fn test_analysis_result_serialization() {
        let result = AnalysisResult {
            language: "C/C++".to_string(),
            has_numpy: false,
            has_sklearn: false,
            has_pytorch: false,
            lines_of_code: 50,
        };

        let json = serde_json::to_string(&result).expect("json serialize failed");
        let deserialized: AnalysisResult =
            serde_json::from_str(&json).expect("json deserialize failed");

        assert_eq!(result.language, deserialized.language);
        assert_eq!(result.has_numpy, deserialized.has_numpy);
        assert_eq!(result.lines_of_code, deserialized.lines_of_code);
    }

    #[test]
    #[cfg(feature = "wasm")]
    fn test_analysis_result_multiple_languages() {
        let rust_result = AnalysisResult {
            language: "Rust".to_string(),
            has_numpy: false,
            has_sklearn: false,
            has_pytorch: false,
            lines_of_code: 100,
        };
        assert_eq!(rust_result.language, "Rust");

        let python_result = AnalysisResult {
            language: "Python".to_string(),
            has_numpy: true,
            has_sklearn: true,
            has_pytorch: false,
            lines_of_code: 200,
        };
        assert_eq!(python_result.language, "Python");
        assert!(python_result.has_numpy);
        assert!(python_result.has_sklearn);
    }

    // ============================================================================
    // CONVERSION RESULT TESTS (Native-compatible)
    // ============================================================================

    #[test]
    #[cfg(feature = "wasm")]
    fn test_conversion_result_direct_construction() {
        let result = ConversionResult {
            original_code: "np.add(a, b)".to_string(),
            rust_code: "a.add(&b)".to_string(),
            imports: "use trueno::Vector;".to_string(),
            backend_recommendation: "SIMD".to_string(),
            complexity: "Low".to_string(),
        };

        assert_eq!(result.original_code, "np.add(a, b)");
        assert_eq!(result.rust_code, "a.add(&b)");
        assert_eq!(result.imports, "use trueno::Vector;");
        assert_eq!(result.backend_recommendation, "SIMD");
        assert_eq!(result.complexity, "Low");
    }

    #[test]
    #[cfg(feature = "wasm")]
    fn test_conversion_result_serialization() {
        let result = ConversionResult {
            original_code: "original".to_string(),
            rust_code: "rust".to_string(),
            imports: "imports".to_string(),
            backend_recommendation: "SIMD".to_string(),
            complexity: "Medium".to_string(),
        };

        let json = serde_json::to_string(&result).expect("json serialize failed");
        let deserialized: ConversionResult =
            serde_json::from_str(&json).expect("json deserialize failed");

        assert_eq!(result.original_code, deserialized.original_code);
        assert_eq!(result.rust_code, deserialized.rust_code);
        assert_eq!(result.backend_recommendation, deserialized.backend_recommendation);
    }

    #[test]
    #[cfg(feature = "wasm")]
    fn test_conversion_result_all_backends() {
        let scalar_result = ConversionResult {
            original_code: "test".to_string(),
            rust_code: "test_rust".to_string(),
            imports: "".to_string(),
            backend_recommendation: "Scalar".to_string(),
            complexity: "Low".to_string(),
        };
        assert_eq!(scalar_result.backend_recommendation, "Scalar");

        let simd_result = ConversionResult {
            original_code: "test".to_string(),
            rust_code: "test_rust".to_string(),
            imports: "".to_string(),
            backend_recommendation: "SIMD".to_string(),
            complexity: "Medium".to_string(),
        };
        assert_eq!(simd_result.backend_recommendation, "SIMD");

        let gpu_result = ConversionResult {
            original_code: "test".to_string(),
            rust_code: "test_rust".to_string(),
            imports: "".to_string(),
            backend_recommendation: "GPU".to_string(),
            complexity: "High".to_string(),
        };
        assert_eq!(gpu_result.backend_recommendation, "GPU");
    }

    // NOTE: Tests for wasm_bindgen functions (analyze_code, convert_numpy, etc.)
    // cannot run on native targets. They require wasm32 target and wasm-bindgen-test.
    // The tests above cover the data structures (AnalysisResult, ConversionResult)
    // which is what can be tested natively. For full WASM function testing, use:
    // wasm-pack test --node --features wasm
}
