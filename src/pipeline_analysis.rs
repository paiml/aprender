//! Pipeline Library Analysis
//!
//! Provides analysis of Python ML library usage for conversion guidance.
//! Extracts NumPy, scikit-learn, and PyTorch operations to map to Rust equivalents.

use anyhow::Result;
use std::path::Path;

#[cfg(feature = "native")]
use tracing::info;

#[cfg(feature = "native")]
use walkdir::WalkDir;

use crate::numpy_converter::{NumPyConverter, NumPyOp};
use crate::pytorch_converter::{PyTorchConverter, PyTorchOperation};
use crate::sklearn_converter::{SklearnAlgorithm, SklearnConverter};

/// Analyzer for ML library usage in Python code
pub struct LibraryAnalyzer {
    numpy_converter: NumPyConverter,
    sklearn_converter: SklearnConverter,
    pytorch_converter: PyTorchConverter,
}

impl Default for LibraryAnalyzer {
    fn default() -> Self {
        Self::new()
    }
}

impl LibraryAnalyzer {
    /// Create a new library analyzer
    pub fn new() -> Self {
        Self {
            numpy_converter: NumPyConverter::new(),
            sklearn_converter: SklearnConverter::new(),
            pytorch_converter: PyTorchConverter::new(),
        }
    }

    /// Analyze Python source for NumPy usage and provide conversion guidance
    #[cfg(feature = "native")]
    pub fn analyze_numpy_usage(&self, input_path: &Path) -> Result<Vec<String>> {
        contract_pre_analyze!(input_path);
        let converter = &self.numpy_converter;
        analyze_library(input_path, &["import numpy", "from numpy"], "NumPy", |path, content| {
            let operations = [
                ("np.add", NumPyOp::Add),
                ("np.subtract", NumPyOp::Subtract),
                ("np.multiply", NumPyOp::Multiply),
                ("np.dot", NumPyOp::Dot),
                ("np.sum", NumPyOp::Sum),
                ("np.array", NumPyOp::Array),
            ];
            operations
                .iter()
                .filter_map(|(pattern, op)| {
                    if content.contains(pattern) {
                        converter.convert(op).map(|r| {
                            format!("{}: {} → {}", path.display(), pattern, r.code_template)
                        })
                    } else {
                        None
                    }
                })
                .collect()
        })
    }

    /// Stub for WASM build
    #[cfg(not(feature = "native"))]
    pub fn analyze_numpy_usage(&self, _input_path: &Path) -> Result<Vec<String>> {
        Ok(Vec::new())
    }

    /// Analyze Python source for sklearn usage and provide conversion guidance
    #[cfg(feature = "native")]
    pub fn analyze_sklearn_usage(&self, input_path: &Path) -> Result<Vec<String>> {
        contract_pre_analyze!(input_path);
        let converter = &self.sklearn_converter;
        analyze_library(
            input_path,
            &["import sklearn", "from sklearn"],
            "sklearn",
            |path, content| {
                let algorithms = [
                    ("LinearRegression", SklearnAlgorithm::LinearRegression),
                    ("LogisticRegression", SklearnAlgorithm::LogisticRegression),
                    ("KMeans", SklearnAlgorithm::KMeans),
                    ("DecisionTreeClassifier", SklearnAlgorithm::DecisionTreeClassifier),
                    ("RandomForestClassifier", SklearnAlgorithm::RandomForestClassifier),
                    ("StandardScaler", SklearnAlgorithm::StandardScaler),
                    ("train_test_split", SklearnAlgorithm::TrainTestSplit),
                ];
                algorithms
                    .iter()
                    .filter(|(pattern, _)| content.contains(*pattern))
                    .filter_map(|(pattern, alg)| {
                        converter.convert(alg).map(|r| {
                            format!(
                                "{}: {} ({}) → {}",
                                path.display(),
                                pattern,
                                alg.sklearn_module(),
                                r.code_template
                            )
                        })
                    })
                    .collect()
            },
        )
    }

    /// Stub for WASM build
    #[cfg(not(feature = "native"))]
    pub fn analyze_sklearn_usage(&self, _input_path: &Path) -> Result<Vec<String>> {
        Ok(Vec::new())
    }

    /// Analyze Python source for PyTorch usage and provide conversion guidance
    #[cfg(feature = "native")]
    pub fn analyze_pytorch_usage(&self, input_path: &Path) -> Result<Vec<String>> {
        contract_pre_analyze!(input_path);
        let converter = &self.pytorch_converter;
        analyze_library(
            input_path,
            &["import torch", "from torch", "from transformers"],
            "PyTorch",
            |path, content| {
                let operations = [
                    ("torch.load", PyTorchOperation::LoadModel),
                    ("from_pretrained", PyTorchOperation::LoadModel),
                    ("AutoTokenizer", PyTorchOperation::LoadTokenizer),
                    (".forward(", PyTorchOperation::Forward),
                    (".generate(", PyTorchOperation::Generate),
                    ("nn.Linear", PyTorchOperation::Linear),
                    ("MultiheadAttention", PyTorchOperation::Attention),
                    ("tokenizer.encode", PyTorchOperation::Encode),
                    ("tokenizer.decode", PyTorchOperation::Decode),
                ];
                operations
                    .iter()
                    .filter(|(pattern, _)| content.contains(*pattern))
                    .filter_map(|(pattern, op)| {
                        converter.convert(op).map(|r| {
                            format!(
                                "{}: {} ({}) → {}",
                                path.display(),
                                pattern,
                                op.pytorch_module(),
                                r.code_template
                            )
                        })
                    })
                    .collect()
            },
        )
    }

    /// Stub for WASM build
    #[cfg(not(feature = "native"))]
    pub fn analyze_pytorch_usage(&self, _input_path: &Path) -> Result<Vec<String>> {
        Ok(Vec::new())
    }
}

/// Shared helper: walk Python files matching import patterns and apply conversion logic
#[cfg(feature = "native")]
fn analyze_library<F>(
    input_path: &Path,
    import_patterns: &[&str],
    lib_name: &str,
    match_content: F,
) -> Result<Vec<String>>
where
    F: Fn(&Path, &str) -> Vec<String>,
{
    let mut recommendations = Vec::new();
    for entry in WalkDir::new(input_path).follow_links(true).into_iter().filter_map(|e| e.ok()) {
        let Some(ext) = entry.path().extension() else {
            continue;
        };
        if ext != "py" {
            continue;
        }
        let Ok(content) = std::fs::read_to_string(entry.path()) else {
            continue;
        };
        if !import_patterns.iter().any(|p| content.contains(p)) {
            continue;
        }
        info!("  Found {} usage in: {}", lib_name, entry.path().display());
        recommendations.extend(match_content(entry.path(), &content));
    }
    Ok(recommendations)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    fn setup_dir(name: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(name);
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).expect("mkdir failed");
        dir
    }

    fn cleanup(dir: &Path) {
        let _ = std::fs::remove_dir_all(dir);
    }

    #[test]
    fn test_library_analyzer_creation() {
        let _analyzer = LibraryAnalyzer::new();
    }

    #[test]
    fn test_library_analyzer_default() {
        let _analyzer = LibraryAnalyzer::default();
    }

    // ===== Nonexistent paths =====

    #[cfg(feature = "native")]
    #[test]
    fn test_analyze_numpy_nonexistent_path() {
        let analyzer = LibraryAnalyzer::new();
        let result = analyzer.analyze_numpy_usage(Path::new("/nonexistent/path"));
        assert!(result.is_ok());
        assert!(result.expect("operation failed").is_empty());
    }

    #[cfg(feature = "native")]
    #[test]
    fn test_analyze_sklearn_nonexistent_path() {
        let analyzer = LibraryAnalyzer::new();
        let result = analyzer.analyze_sklearn_usage(Path::new("/nonexistent/path"));
        assert!(result.is_ok());
        assert!(result.expect("operation failed").is_empty());
    }

    #[cfg(feature = "native")]
    #[test]
    fn test_analyze_pytorch_nonexistent_path() {
        let analyzer = LibraryAnalyzer::new();
        let result = analyzer.analyze_pytorch_usage(Path::new("/nonexistent/path"));
        assert!(result.is_ok());
        assert!(result.expect("operation failed").is_empty());
    }

    // ===== NumPy with real files =====

    #[cfg(feature = "native")]
    #[test]
    fn test_analyze_numpy_with_matching_file() {
        let dir = setup_dir("test_pa_numpy");
        std::fs::write(
            dir.join("model.py"),
            "import numpy as np\nx = np.array([1,2,3])\ny = np.dot(x, x)\nz = np.sum(y)\n",
        )
        .expect("unexpected failure");
        let analyzer = LibraryAnalyzer::new();
        let results = analyzer.analyze_numpy_usage(&dir).expect("unexpected failure");
        assert!(!results.is_empty());
        assert!(results.iter().any(|r| r.contains("np.array")));
        assert!(results.iter().any(|r| r.contains("np.dot")));
        assert!(results.iter().any(|r| r.contains("np.sum")));
        cleanup(&dir);
    }

    #[cfg(feature = "native")]
    #[test]
    fn test_analyze_numpy_no_import() {
        let dir = setup_dir("test_pa_numpy_noimport");
        std::fs::write(dir.join("script.py"), "x = [1, 2, 3]\nprint(sum(x))\n")
            .expect("fs write failed");
        let analyzer = LibraryAnalyzer::new();
        let results = analyzer.analyze_numpy_usage(&dir).expect("unexpected failure");
        assert!(results.is_empty());
        cleanup(&dir);
    }

    #[cfg(feature = "native")]
    #[test]
    fn test_analyze_numpy_non_python_files_ignored() {
        let dir = setup_dir("test_pa_numpy_nonpy");
        std::fs::write(dir.join("data.txt"), "import numpy as np\nnp.array([1])\n")
            .expect("fs write failed");
        let analyzer = LibraryAnalyzer::new();
        let results = analyzer.analyze_numpy_usage(&dir).expect("unexpected failure");
        assert!(results.is_empty());
        cleanup(&dir);
    }

    #[cfg(feature = "native")]
    #[test]
    fn test_analyze_numpy_add_subtract_multiply() {
        let dir = setup_dir("test_pa_numpy_ops");
        std::fs::write(
            dir.join("ops.py"),
            "import numpy as np\na = np.add(x, y)\nb = np.subtract(x, y)\nc = np.multiply(x, y)\n",
        )
        .expect("unexpected failure");
        let analyzer = LibraryAnalyzer::new();
        let results = analyzer.analyze_numpy_usage(&dir).expect("unexpected failure");
        assert!(results.iter().any(|r| r.contains("np.add")));
        assert!(results.iter().any(|r| r.contains("np.subtract")));
        assert!(results.iter().any(|r| r.contains("np.multiply")));
        cleanup(&dir);
    }

    // ===== sklearn with real files =====

    #[cfg(feature = "native")]
    #[test]
    fn test_analyze_sklearn_with_matching_file() {
        let dir = setup_dir("test_pa_sklearn");
        std::fs::write(
            dir.join("train.py"),
            "from sklearn.linear_model import LinearRegression\nfrom sklearn.cluster import KMeans\nmodel = LinearRegression()\nkm = KMeans(n_clusters=3)\n",
        )
        .expect("unexpected failure");
        let analyzer = LibraryAnalyzer::new();
        let results = analyzer.analyze_sklearn_usage(&dir).expect("unexpected failure");
        assert!(!results.is_empty());
        assert!(results.iter().any(|r| r.contains("LinearRegression")));
        assert!(results.iter().any(|r| r.contains("KMeans")));
        cleanup(&dir);
    }

    #[cfg(feature = "native")]
    #[test]
    fn test_analyze_sklearn_no_import() {
        let dir = setup_dir("test_pa_sklearn_noimport");
        std::fs::write(dir.join("script.py"), "print('hello')\n").expect("fs write failed");
        let analyzer = LibraryAnalyzer::new();
        let results = analyzer.analyze_sklearn_usage(&dir).expect("unexpected failure");
        assert!(results.is_empty());
        cleanup(&dir);
    }

    #[cfg(feature = "native")]
    #[test]
    fn test_analyze_sklearn_more_algorithms() {
        // Only algorithms registered in SklearnConverter::new() produce output
        let dir = setup_dir("test_pa_sklearn_more");
        std::fs::write(
            dir.join("ml.py"),
            "from sklearn.tree import DecisionTreeClassifier\nfrom sklearn.preprocessing import StandardScaler\nfrom sklearn.model_selection import train_test_split\nfrom sklearn.linear_model import LogisticRegression\n",
        )
        .expect("unexpected failure");
        let analyzer = LibraryAnalyzer::new();
        let results = analyzer.analyze_sklearn_usage(&dir).expect("unexpected failure");
        assert!(results.iter().any(|r| r.contains("DecisionTreeClassifier")));
        assert!(results.iter().any(|r| r.contains("StandardScaler")));
        assert!(results.iter().any(|r| r.contains("train_test_split")));
        assert!(results.iter().any(|r| r.contains("LogisticRegression")));
        cleanup(&dir);
    }

    // ===== PyTorch with real files =====

    #[cfg(feature = "native")]
    #[test]
    fn test_analyze_pytorch_with_matching_file() {
        let dir = setup_dir("test_pa_pytorch");
        std::fs::write(
            dir.join("infer.py"),
            "import torch\nmodel = torch.load('model.pt')\nout = model.forward(x)\n",
        )
        .expect("unexpected failure");
        let analyzer = LibraryAnalyzer::new();
        let results = analyzer.analyze_pytorch_usage(&dir).expect("unexpected failure");
        assert!(!results.is_empty());
        assert!(results.iter().any(|r| r.contains("torch.load")));
        assert!(results.iter().any(|r| r.contains(".forward(")));
        cleanup(&dir);
    }

    #[cfg(feature = "native")]
    #[test]
    fn test_analyze_pytorch_no_import() {
        let dir = setup_dir("test_pa_pytorch_noimport");
        std::fs::write(dir.join("app.py"), "print('hello')\n").expect("fs write failed");
        let analyzer = LibraryAnalyzer::new();
        let results = analyzer.analyze_pytorch_usage(&dir).expect("unexpected failure");
        assert!(results.is_empty());
        cleanup(&dir);
    }

    #[cfg(feature = "native")]
    #[test]
    fn test_analyze_pytorch_transformers() {
        let dir = setup_dir("test_pa_pytorch_hf");
        std::fs::write(
            dir.join("hf.py"),
            "from transformers import AutoTokenizer\ntokenizer = AutoTokenizer.from_pretrained('bert')\nids = tokenizer.encode('hello')\ntext = tokenizer.decode(ids)\n",
        )
        .expect("unexpected failure");
        let analyzer = LibraryAnalyzer::new();
        let results = analyzer.analyze_pytorch_usage(&dir).expect("unexpected failure");
        assert!(results.iter().any(|r| r.contains("AutoTokenizer")));
        assert!(results.iter().any(|r| r.contains("from_pretrained")));
        assert!(results.iter().any(|r| r.contains("tokenizer.encode")));
        assert!(results.iter().any(|r| r.contains("tokenizer.decode")));
        cleanup(&dir);
    }

    #[cfg(feature = "native")]
    #[test]
    fn test_analyze_pytorch_nn_modules() {
        let dir = setup_dir("test_pa_pytorch_nn");
        std::fs::write(
            dir.join("model.py"),
            "import torch\nimport torch.nn as nn\nlayer = nn.Linear(10, 5)\nattn = nn.MultiheadAttention(512, 8)\nout = model.generate(ids)\n",
        )
        .expect("unexpected failure");
        let analyzer = LibraryAnalyzer::new();
        let results = analyzer.analyze_pytorch_usage(&dir).expect("unexpected failure");
        assert!(results.iter().any(|r| r.contains("nn.Linear")));
        assert!(results.iter().any(|r| r.contains("MultiheadAttention")));
        assert!(results.iter().any(|r| r.contains(".generate(")));
        cleanup(&dir);
    }

    // ===== Subdirectory traversal =====

    #[cfg(feature = "native")]
    #[test]
    fn test_analyze_numpy_recursive() {
        let dir = setup_dir("test_pa_numpy_recurse");
        let sub = dir.join("pkg").join("sub");
        std::fs::create_dir_all(&sub).expect("mkdir failed");
        std::fs::write(sub.join("deep.py"), "from numpy import array\nx = np.array([1])\n")
            .expect("unexpected failure");
        let analyzer = LibraryAnalyzer::new();
        let results = analyzer.analyze_numpy_usage(&dir).expect("unexpected failure");
        assert!(results.iter().any(|r| r.contains("np.array")));
        cleanup(&dir);
    }

    // ===== Empty directory =====

    #[cfg(feature = "native")]
    #[test]
    fn test_analyze_all_empty_dir() {
        let dir = setup_dir("test_pa_all_empty");
        let analyzer = LibraryAnalyzer::new();
        assert!(analyzer.analyze_numpy_usage(&dir).expect("unexpected failure").is_empty());
        assert!(analyzer.analyze_sklearn_usage(&dir).expect("unexpected failure").is_empty());
        assert!(analyzer.analyze_pytorch_usage(&dir).expect("unexpected failure").is_empty());
        cleanup(&dir);
    }
}
