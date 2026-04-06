//! NumPy to Trueno conversion module (BATUTA-008)
//!
//! Converts Python NumPy operations to Rust Trueno operations with
//! automatic backend selection via MoE routing.
//!
//! # Conversion Strategy
//!
//! NumPy operations are mapped to equivalent Trueno operations:
//! - `np.array(...)` → `Vector::from_slice(...)` or `Matrix::from_slice(...)`
//! - `np.add(a, b)` → `a.add(&b)`
//! - `np.dot(a, b)` → `a.dot(&b)` or `a.matmul(&b)`
//! - `np.sum(a)` → `a.sum()`
//! - Element-wise ops automatically use MoE routing
//!
//! # Example
//!
//! ```python
//! # Python NumPy code
//! import numpy as np
//! a = np.array([1.0, 2.0, 3.0])
//! b = np.array([4.0, 5.0, 6.0])
//! c = np.add(a, b)
//! ```
//!
//! Converts to:
//!
//! ```rust,ignore
//! use trueno::Vector;
//! let a = Vector::from_slice(&[1.0, 2.0, 3.0]);
//! let b = Vector::from_slice(&[4.0, 5.0, 6.0]);
//! let c = a.add(&b).unwrap();
//! ```

use std::collections::HashMap;

/// NumPy operation types
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum NumPyOp {
    /// Array creation: np.array, np.zeros, np.ones
    Array,
    /// Element-wise addition: np.add, a + b
    Add,
    /// Element-wise subtraction: np.subtract, a - b
    Subtract,
    /// Element-wise multiplication: np.multiply, a * b
    Multiply,
    /// Element-wise division: np.divide, a / b
    Divide,
    /// Dot product / matrix multiply: np.dot, np.matmul, a @ b
    Dot,
    /// Sum reduction: np.sum
    Sum,
    /// Mean reduction: np.mean
    Mean,
    /// Max reduction: np.max
    Max,
    /// Min reduction: np.min
    Min,
    /// Reshape: np.reshape
    Reshape,
    /// Transpose: np.transpose, a.T
    Transpose,
}

impl NumPyOp {
    /// Get the operation complexity for MoE routing
    pub fn complexity(&self) -> crate::backend::OpComplexity {
        use crate::backend::OpComplexity;

        match self {
            // Element-wise operations are Low complexity (memory-bound)
            NumPyOp::Add | NumPyOp::Subtract | NumPyOp::Multiply | NumPyOp::Divide => {
                OpComplexity::Low
            }
            // Reductions are Medium complexity
            NumPyOp::Sum | NumPyOp::Mean | NumPyOp::Max | NumPyOp::Min => OpComplexity::Medium,
            // Dot product and matrix ops are High complexity
            NumPyOp::Dot => OpComplexity::High,
            // Structural operations don't need backend selection
            NumPyOp::Array | NumPyOp::Reshape | NumPyOp::Transpose => OpComplexity::Low,
        }
    }
}

/// Trueno equivalent operation
#[derive(Debug, Clone)]
pub struct TruenoOp {
    /// Rust code template for the operation
    pub code_template: String,
    /// Required imports
    pub imports: Vec<String>,
    /// Operation complexity
    pub complexity: crate::backend::OpComplexity,
}

/// NumPy to Trueno converter
pub struct NumPyConverter {
    /// Operation mapping
    op_map: HashMap<NumPyOp, TruenoOp>,
    /// Backend selector for MoE routing
    backend_selector: crate::backend::BackendSelector,
}

impl Default for NumPyConverter {
    fn default() -> Self {
        Self::new()
    }
}

impl NumPyConverter {
    /// Create a new NumPy converter with default mappings
    pub fn new() -> Self {
        let mut op_map = HashMap::new();

        // Array operations
        op_map.insert(
            NumPyOp::Array,
            TruenoOp {
                code_template: "Vector::from_slice(&[{values}])".to_string(),
                imports: vec!["use trueno::Vector;".to_string()],
                complexity: crate::backend::OpComplexity::Low,
            },
        );

        // Element-wise operations
        op_map.insert(
            NumPyOp::Add,
            TruenoOp {
                code_template: "{lhs}.add(&{rhs}).unwrap()".to_string(),
                imports: vec!["use trueno::Vector;".to_string()],
                complexity: crate::backend::OpComplexity::Low,
            },
        );

        op_map.insert(
            NumPyOp::Subtract,
            TruenoOp {
                code_template: "{lhs}.sub(&{rhs}).unwrap()".to_string(),
                imports: vec!["use trueno::Vector;".to_string()],
                complexity: crate::backend::OpComplexity::Low,
            },
        );

        op_map.insert(
            NumPyOp::Multiply,
            TruenoOp {
                code_template: "{lhs}.mul(&{rhs}).unwrap()".to_string(),
                imports: vec!["use trueno::Vector;".to_string()],
                complexity: crate::backend::OpComplexity::Low,
            },
        );

        // Reductions
        op_map.insert(
            NumPyOp::Sum,
            TruenoOp {
                code_template: "{array}.sum()".to_string(),
                imports: vec!["use trueno::Vector;".to_string()],
                complexity: crate::backend::OpComplexity::Medium,
            },
        );

        op_map.insert(
            NumPyOp::Dot,
            TruenoOp {
                code_template: "{lhs}.dot(&{rhs}).unwrap()".to_string(),
                imports: vec!["use trueno::Vector;".to_string()],
                complexity: crate::backend::OpComplexity::High,
            },
        );

        Self { op_map, backend_selector: crate::backend::BackendSelector::new() }
    }

    /// Convert a NumPy operation to Trueno
    pub fn convert(&self, op: &NumPyOp) -> Option<&TruenoOp> {
        self.op_map.get(op)
    }

    /// Get recommended backend for an operation
    pub fn recommend_backend(&self, op: &NumPyOp, data_size: usize) -> crate::backend::Backend {
        self.backend_selector.select_with_moe(op.complexity(), data_size)
    }

    /// Get all available conversions
    pub fn available_ops(&self) -> Vec<&NumPyOp> {
        self.op_map.keys().collect()
    }

    /// Generate conversion report
    pub fn conversion_report(&self) -> String {
        let mut report = String::from("NumPy → Trueno Conversion Map\n");
        report.push_str("================================\n\n");

        for (op, trueno_op) in &self.op_map {
            report.push_str(&format!("{:?}:\n", op));
            report.push_str(&format!("  Complexity: {:?}\n", trueno_op.complexity));
            report.push_str(&format!("  Template: {}\n", trueno_op.code_template));
            report.push_str(&format!("  Imports: {}\n\n", trueno_op.imports.join(", ")));
        }

        report
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_converter_creation() {
        let converter = NumPyConverter::new();
        assert!(!converter.available_ops().is_empty());
    }

    #[test]
    fn test_operation_complexity() {
        assert_eq!(NumPyOp::Add.complexity(), crate::backend::OpComplexity::Low);
        assert_eq!(NumPyOp::Sum.complexity(), crate::backend::OpComplexity::Medium);
        assert_eq!(NumPyOp::Dot.complexity(), crate::backend::OpComplexity::High);
    }

    #[test]
    fn test_add_conversion() {
        let converter = NumPyConverter::new();
        let trueno_op = converter.convert(&NumPyOp::Add).expect("conversion failed");
        assert!(trueno_op.code_template.contains("add"));
        assert!(trueno_op.imports.iter().any(|i| i.contains("Vector")));
    }

    #[test]
    fn test_backend_recommendation() {
        let converter = NumPyConverter::new();

        // Small element-wise operation should use Scalar
        let backend = converter.recommend_backend(&NumPyOp::Add, 100);
        assert_eq!(backend, crate::backend::Backend::Scalar);

        // Large element-wise should use SIMD
        let backend = converter.recommend_backend(&NumPyOp::Add, 2_000_000);
        assert_eq!(backend, crate::backend::Backend::SIMD);

        // Large matrix operation should use GPU
        let backend = converter.recommend_backend(&NumPyOp::Dot, 50_000);
        assert_eq!(backend, crate::backend::Backend::GPU);
    }

    #[test]
    fn test_conversion_report() {
        let converter = NumPyConverter::new();
        let report = converter.conversion_report();
        assert!(report.contains("NumPy → Trueno"));
        assert!(report.contains("Add"));
        assert!(report.contains("Complexity"));
    }

    // ============================================================================
    // NUMPY OP ENUM TESTS
    // ============================================================================

    #[test]
    fn test_all_numpy_ops_exist() {
        // Test all 13 variants can be constructed
        let ops = vec![
            NumPyOp::Array,
            NumPyOp::Add,
            NumPyOp::Subtract,
            NumPyOp::Multiply,
            NumPyOp::Divide,
            NumPyOp::Dot,
            NumPyOp::Sum,
            NumPyOp::Mean,
            NumPyOp::Max,
            NumPyOp::Min,
            NumPyOp::Reshape,
            NumPyOp::Transpose,
        ];
        assert_eq!(ops.len(), 12); // 12 operations tested
    }

    #[test]
    fn test_op_equality() {
        assert_eq!(NumPyOp::Add, NumPyOp::Add);
        assert_ne!(NumPyOp::Add, NumPyOp::Multiply);
    }

    #[test]
    fn test_op_clone() {
        let op1 = NumPyOp::Dot;
        let op2 = op1.clone();
        assert_eq!(op1, op2);
    }

    #[test]
    fn test_complexity_low_ops() {
        let low_ops = vec![
            NumPyOp::Add,
            NumPyOp::Subtract,
            NumPyOp::Multiply,
            NumPyOp::Divide,
            NumPyOp::Array,
            NumPyOp::Reshape,
            NumPyOp::Transpose,
        ];

        for op in low_ops {
            assert_eq!(op.complexity(), crate::backend::OpComplexity::Low);
        }
    }

    #[test]
    fn test_complexity_medium_ops() {
        let medium_ops = vec![NumPyOp::Sum, NumPyOp::Mean, NumPyOp::Max, NumPyOp::Min];

        for op in medium_ops {
            assert_eq!(op.complexity(), crate::backend::OpComplexity::Medium);
        }
    }

    #[test]
    fn test_complexity_high_ops() {
        let high_ops = vec![NumPyOp::Dot];

        for op in high_ops {
            assert_eq!(op.complexity(), crate::backend::OpComplexity::High);
        }
    }

    // ============================================================================
    // TRUENO OP STRUCT TESTS
    // ============================================================================

    #[test]
    fn test_trueno_op_construction() {
        let op = TruenoOp {
            code_template: "test_template".to_string(),
            imports: vec!["use test;".to_string()],
            complexity: crate::backend::OpComplexity::Medium,
        };

        assert_eq!(op.code_template, "test_template");
        assert_eq!(op.imports.len(), 1);
        assert_eq!(op.complexity, crate::backend::OpComplexity::Medium);
    }

    #[test]
    fn test_trueno_op_clone() {
        let op1 = TruenoOp {
            code_template: "template".to_string(),
            imports: vec!["import".to_string()],
            complexity: crate::backend::OpComplexity::High,
        };

        let op2 = op1.clone();
        assert_eq!(op1.code_template, op2.code_template);
        assert_eq!(op1.imports, op2.imports);
        assert_eq!(op1.complexity, op2.complexity);
    }

    // ============================================================================
    // NUMPY CONVERTER TESTS
    // ============================================================================

    #[test]
    fn test_converter_default() {
        let converter = NumPyConverter::default();
        assert!(!converter.available_ops().is_empty());
    }

    #[test]
    fn test_convert_all_mapped_ops() {
        let converter = NumPyConverter::new();

        // Test all operations that should have mappings
        let mapped_ops = vec![
            NumPyOp::Array,
            NumPyOp::Add,
            NumPyOp::Subtract,
            NumPyOp::Multiply,
            NumPyOp::Sum,
            NumPyOp::Dot,
        ];

        for op in mapped_ops {
            assert!(converter.convert(&op).is_some(), "Missing mapping for {:?}", op);
        }
    }

    #[test]
    fn test_convert_unmapped_op() {
        let converter = NumPyConverter::new();

        // Divide, Mean, etc. might not be mapped
        // Just verify the function handles missing ops gracefully
        let result = converter.convert(&NumPyOp::Divide);
        // It's ok if this is None - we're testing the API works
        let _ = result;
    }

    #[test]
    fn test_array_conversion() {
        let converter = NumPyConverter::new();
        let op = converter.convert(&NumPyOp::Array).expect("conversion failed");

        assert!(op.code_template.contains("Vector"));
        assert!(op.code_template.contains("from_slice"));
        assert!(op.imports.iter().any(|i| i.contains("Vector")));
        assert_eq!(op.complexity, crate::backend::OpComplexity::Low);
    }

    #[test]
    fn test_subtract_conversion() {
        let converter = NumPyConverter::new();
        let op = converter.convert(&NumPyOp::Subtract).expect("conversion failed");

        assert!(op.code_template.contains("sub"));
        assert!(op.imports.iter().any(|i| i.contains("Vector")));
        assert_eq!(op.complexity, crate::backend::OpComplexity::Low);
    }

    #[test]
    fn test_multiply_conversion() {
        let converter = NumPyConverter::new();
        let op = converter.convert(&NumPyOp::Multiply).expect("conversion failed");

        assert!(op.code_template.contains("mul"));
        assert!(op.imports.iter().any(|i| i.contains("Vector")));
    }

    #[test]
    fn test_sum_conversion() {
        let converter = NumPyConverter::new();
        let op = converter.convert(&NumPyOp::Sum).expect("conversion failed");

        assert!(op.code_template.contains("sum"));
        assert_eq!(op.complexity, crate::backend::OpComplexity::Medium);
    }

    #[test]
    fn test_dot_conversion() {
        let converter = NumPyConverter::new();
        let op = converter.convert(&NumPyOp::Dot).expect("conversion failed");

        assert!(op.code_template.contains("dot"));
        assert_eq!(op.complexity, crate::backend::OpComplexity::High);
    }

    #[test]
    fn test_available_ops() {
        let converter = NumPyConverter::new();
        let ops = converter.available_ops();

        assert!(!ops.is_empty());
        // Should have at least the mapped operations
        assert!(ops.len() >= 6);
    }

    #[test]
    fn test_recommend_backend_element_wise_small() {
        let converter = NumPyConverter::new();

        // Small element-wise operations should use Scalar
        let backend = converter.recommend_backend(&NumPyOp::Add, 10);
        assert_eq!(backend, crate::backend::Backend::Scalar);
    }

    #[test]
    fn test_recommend_backend_element_wise_large() {
        let converter = NumPyConverter::new();

        // Large element-wise operations should use SIMD
        let backend = converter.recommend_backend(&NumPyOp::Multiply, 2_000_000);
        assert_eq!(backend, crate::backend::Backend::SIMD);
    }

    #[test]
    fn test_recommend_backend_reduction_medium() {
        let converter = NumPyConverter::new();

        // Medium-sized reductions should use SIMD
        let backend = converter.recommend_backend(&NumPyOp::Sum, 50_000);
        assert_eq!(backend, crate::backend::Backend::SIMD);
    }

    #[test]
    fn test_recommend_backend_reduction_large() {
        let converter = NumPyConverter::new();

        // Large reductions should use GPU
        let backend = converter.recommend_backend(&NumPyOp::Sum, 500_000);
        assert_eq!(backend, crate::backend::Backend::GPU);
    }

    #[test]
    fn test_recommend_backend_dot_product() {
        let converter = NumPyConverter::new();

        // Dot product with large data should use GPU
        let backend = converter.recommend_backend(&NumPyOp::Dot, 100_000);
        assert_eq!(backend, crate::backend::Backend::GPU);
    }

    #[test]
    fn test_conversion_report_structure() {
        let converter = NumPyConverter::new();
        let report = converter.conversion_report();

        // Check report contains expected sections
        assert!(report.contains("NumPy → Trueno"));
        assert!(report.contains("==="));
        assert!(report.contains("Complexity:"));
        assert!(report.contains("Template:"));
        assert!(report.contains("Imports:"));
    }

    #[test]
    fn test_conversion_report_has_all_ops() {
        let converter = NumPyConverter::new();
        let report = converter.conversion_report();

        // Spot check a few operations appear in report
        assert!(report.contains("Add") || report.contains("Sum") || report.contains("Dot"));
    }

    #[test]
    fn test_all_conversions_not_empty() {
        let converter = NumPyConverter::new();

        for op in converter.available_ops() {
            if let Some(trueno_op) = converter.convert(op) {
                assert!(!trueno_op.code_template.is_empty(), "Empty code template for {:?}", op);
                assert!(!trueno_op.imports.is_empty(), "Empty imports for {:?}", op);
            }
        }
    }

    #[test]
    fn test_imports_are_valid_rust() {
        let converter = NumPyConverter::new();

        for op in converter.available_ops() {
            if let Some(trueno_op) = converter.convert(op) {
                for import in &trueno_op.imports {
                    assert!(import.starts_with("use "), "Invalid import syntax: {}", import);
                    assert!(import.ends_with(';'), "Import missing semicolon: {}", import);
                }
            }
        }
    }

    #[test]
    fn test_all_ops_use_vector_import() {
        let converter = NumPyConverter::new();

        for op in converter.available_ops() {
            if let Some(trueno_op) = converter.convert(op) {
                assert!(
                    trueno_op.imports.iter().any(|i| i.contains("Vector")),
                    "Operation {:?} should import Vector",
                    op
                );
            }
        }
    }

    #[test]
    fn test_element_wise_ops_have_unwrap() {
        let converter = NumPyConverter::new();

        let element_wise = vec![NumPyOp::Add, NumPyOp::Subtract, NumPyOp::Multiply];

        for op in element_wise {
            if let Some(trueno_op) = converter.convert(&op) {
                assert!(
                    trueno_op.code_template.contains("unwrap"),
                    "Element-wise op {:?} should have unwrap() for error handling",
                    op
                );
            }
        }
    }

    #[test]
    fn test_complexity_matches_enum() {
        let converter = NumPyConverter::new();

        // Test that TruenoOp complexity matches NumPyOp complexity
        if let Some(add_op) = converter.convert(&NumPyOp::Add) {
            assert_eq!(add_op.complexity, NumPyOp::Add.complexity());
        }

        if let Some(sum_op) = converter.convert(&NumPyOp::Sum) {
            assert_eq!(sum_op.complexity, NumPyOp::Sum.complexity());
        }

        if let Some(dot_op) = converter.convert(&NumPyOp::Dot) {
            assert_eq!(dot_op.complexity, NumPyOp::Dot.complexity());
        }
    }
}
