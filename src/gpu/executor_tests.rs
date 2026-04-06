//! Executor Extended Coverage Tests
//!
//! Additional tests for `gpu/executor.rs` to reach higher coverage.
//! Targets the ~12% uncovered code including:
//! - `MockExecutor::has_call()` method
//! - `MockExecutor::matmul_transpose_b()` with all code paths
//! - `CpuExecutor::matmul_transpose_b()` with various matrix sizes
//! - `ExecutorCall::MatmulTransposeB` variant
//! - Edge cases and error paths

use crate::error::Result;
use crate::gpu::executor::{CpuExecutor, ExecutorCall, GpuExecutorTrait, MockExecutor};

// ============================================================================
// MockExecutor::has_call() Tests
// ============================================================================

#[test]
fn test_mock_executor_has_call_found() {
    let mut mock = MockExecutor::new("test");
    let _ = mock.matmul(&[1.0; 4], &[1.0; 4], 2, 2, 2);

    let expected_call = ExecutorCall::Matmul {
        a_len: 4,
        b_len: 4,
        m: 2,
        k: 2,
        n: 2,
    };
    assert!(mock.has_call(&expected_call));
}

#[test]
fn test_mock_executor_has_call_not_found() {
    let mut mock = MockExecutor::new("test");
    let _ = mock.matmul(&[1.0; 4], &[1.0; 4], 2, 2, 2);

    // Look for a call with different dimensions
    let different_call = ExecutorCall::Matmul {
        a_len: 9,
        b_len: 9,
        m: 3,
        k: 3,
        n: 3,
    };
    assert!(!mock.has_call(&different_call));
}

#[test]
fn test_mock_executor_has_call_synchronize() {
    let mock = MockExecutor::new("test");
    // Synchronize doesn't record calls in current implementation
    let _ = mock.synchronize();
    assert!(!mock.has_call(&ExecutorCall::Synchronize));
}

#[test]
fn test_mock_executor_has_call_empty() {
    let mock = MockExecutor::new("test");
    let call = ExecutorCall::Matmul {
        a_len: 4,
        b_len: 4,
        m: 2,
        k: 2,
        n: 2,
    };
    assert!(!mock.has_call(&call));
}

// ============================================================================
// MockExecutor::matmul_transpose_b() Tests
// ============================================================================

#[test]
fn test_mock_executor_matmul_transpose_b_basic() {
    let mut mock = MockExecutor::new("test");
    // A is [m, k] = [2, 3], B is [n, k] = [4, 3] (transposed)
    let a = vec![1.0; 6]; // 2x3
    let b = vec![1.0; 12]; // 4x3

    let result = mock.matmul_transpose_b(&a, &b, 2, 3, 4).unwrap();

    // Default returns zeros of size m * n = 2 * 4 = 8
    assert_eq!(result.len(), 8);
    assert_eq!(result, vec![0.0; 8]);
    assert_eq!(mock.call_count(), 1);
}

#[test]
fn test_mock_executor_matmul_transpose_b_records_call() {
    let mut mock = MockExecutor::new("test");
    let a = vec![1.0; 6];
    let b = vec![1.0; 12];

    let _ = mock.matmul_transpose_b(&a, &b, 2, 3, 4);

    let last = mock.last_call().unwrap();
    assert!(matches!(
        last,
        ExecutorCall::MatmulTransposeB {
            a_len: 6,
            b_len: 12,
            m: 2,
            k: 3,
            n: 4,
        }
    ));
}

#[test]
fn test_mock_executor_matmul_transpose_b_with_custom_result() {
    let custom_result = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    let mut mock = MockExecutor::new("test").with_matmul_result(custom_result.clone());

    let a = vec![1.0; 6];
    let b = vec![1.0; 12];
    let result = mock.matmul_transpose_b(&a, &b, 2, 3, 4).unwrap();

    assert_eq!(result, custom_result);
}

#[test]
fn test_mock_executor_matmul_transpose_b_failure() {
    let mut mock = MockExecutor::new("test").with_matmul_failure();

    let a = vec![1.0; 6];
    let b = vec![1.0; 12];
    let result = mock.matmul_transpose_b(&a, &b, 2, 3, 4);

    assert!(result.is_err());
    let err_msg = format!("{:?}", result.unwrap_err());
    assert!(err_msg.contains("configured to fail"));
}

#[test]
fn test_mock_executor_matmul_transpose_b_call_counter() {
    let mut mock = MockExecutor::new("test");

    // Interleave matmul and matmul_transpose_b
    let _ = mock.matmul(&[1.0; 4], &[1.0; 4], 2, 2, 2);
    let _ = mock.matmul_transpose_b(&[1.0; 4], &[1.0; 4], 2, 2, 2);
    let _ = mock.matmul(&[1.0; 4], &[1.0; 4], 2, 2, 2);

    assert_eq!(mock.call_count(), 3);
    assert_eq!(mock.matmul_count(), 2); // Only counts Matmul, not MatmulTransposeB
}

// ============================================================================
// CpuExecutor::matmul_transpose_b() Tests
// ============================================================================

#[test]
fn test_cpu_executor_matmul_transpose_b_2x2() {
    let mut cpu = CpuExecutor::new();

    // A = [[1, 2], [3, 4]] (2x2)
    // B = [[5, 6], [7, 8]] (2x2, will be transposed to [[5, 7], [6, 8]])
    // C = A @ B^T
    // C[0,0] = 1*5 + 2*6 = 17
    // C[0,1] = 1*7 + 2*8 = 23
    // C[1,0] = 3*5 + 4*6 = 39
    // C[1,1] = 3*7 + 4*8 = 53
    let a = vec![1.0, 2.0, 3.0, 4.0];
    let b = vec![5.0, 6.0, 7.0, 8.0];

    let c = cpu.matmul_transpose_b(&a, &b, 2, 2, 2).unwrap();

    assert_eq!(c.len(), 4);
    assert!((c[0] - 17.0).abs() < 1e-5, "c[0] = {} expected 17", c[0]);
    assert!((c[1] - 23.0).abs() < 1e-5, "c[1] = {} expected 23", c[1]);
    assert!((c[2] - 39.0).abs() < 1e-5, "c[2] = {} expected 39", c[2]);
    assert!((c[3] - 53.0).abs() < 1e-5, "c[3] = {} expected 53", c[3]);
}

#[test]
fn test_cpu_executor_matmul_transpose_b_vector() {
    let mut cpu = CpuExecutor::new();

    // A = [1, 2, 3] (1x3)
    // B = [[1, 2, 3], [4, 5, 6]] (2x3)
    // C = A @ B^T = [1x3] @ [3x2] = [1x2]
    // C[0,0] = 1*1 + 2*2 + 3*3 = 14
    // C[0,1] = 1*4 + 2*5 + 3*6 = 32
    let a = vec![1.0, 2.0, 3.0];
    let b = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];

    let c = cpu.matmul_transpose_b(&a, &b, 1, 3, 2).unwrap();

    assert_eq!(c.len(), 2);
    assert!((c[0] - 14.0).abs() < 1e-5, "c[0] = {} expected 14", c[0]);
    assert!((c[1] - 32.0).abs() < 1e-5, "c[1] = {} expected 32", c[1]);
}

#[test]
fn test_cpu_executor_matmul_transpose_b_rectangular() {
    let mut cpu = CpuExecutor::new();

    // A = 2x3 matrix, B = 4x3 matrix (transposed to 3x4)
    // Result should be 2x4
    let a = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]; // 2x3
    let b = vec![1.0; 12]; // 4x3, all ones

    let c = cpu.matmul_transpose_b(&a, &b, 2, 3, 4).unwrap();

    assert_eq!(c.len(), 8);
    // First row: sum of [1,2,3] = 6 for each of 4 columns
    assert!((c[0] - 6.0).abs() < 1e-5);
    assert!((c[1] - 6.0).abs() < 1e-5);
    // Second row: sum of [4,5,6] = 15 for each of 4 columns
    assert!((c[4] - 15.0).abs() < 1e-5);
}

#[test]
fn test_cpu_executor_matmul_transpose_b_identity_like() {
    let mut cpu = CpuExecutor::new();

    // If B^T is identity-like, result should be close to A
    // B = [[1,0], [0,1]] transposed is still [[1,0], [0,1]]
    let a = vec![3.0, 7.0]; // 1x2
    let b = vec![1.0, 0.0, 0.0, 1.0]; // 2x2 identity

    let c = cpu.matmul_transpose_b(&a, &b, 1, 2, 2).unwrap();

    assert_eq!(c.len(), 2);
    assert!((c[0] - 3.0).abs() < 1e-5);
    assert!((c[1] - 7.0).abs() < 1e-5);
}

#[test]
fn test_cpu_executor_matmul_transpose_b_dimension_error_a() {
    let mut cpu = CpuExecutor::new();
    let a = vec![1.0; 5]; // Wrong: should be m*k = 2*3 = 6
    let b = vec![1.0; 12]; // Correct: n*k = 4*3 = 12

    let result = cpu.matmul_transpose_b(&a, &b, 2, 3, 4);
    assert!(result.is_err());
    let err_msg = format!("{:?}", result.unwrap_err());
    assert!(err_msg.contains("A size"));
}

#[test]
fn test_cpu_executor_matmul_transpose_b_dimension_error_b() {
    let mut cpu = CpuExecutor::new();
    let a = vec![1.0; 6]; // Correct: m*k = 2*3 = 6
    let b = vec![1.0; 10]; // Wrong: should be n*k = 4*3 = 12

    let result = cpu.matmul_transpose_b(&a, &b, 2, 3, 4);
    assert!(result.is_err());
    let err_msg = format!("{:?}", result.unwrap_err());
    assert!(err_msg.contains("B size"));
}

#[test]
fn test_cpu_executor_matmul_transpose_b_single_element() {
    let mut cpu = CpuExecutor::new();
    // 1x1 @ 1x1^T = 1x1
    let a = vec![3.0];
    let b = vec![4.0];

    let c = cpu.matmul_transpose_b(&a, &b, 1, 1, 1).unwrap();

    assert_eq!(c.len(), 1);
    assert!((c[0] - 12.0).abs() < 1e-5);
}

#[test]
fn test_cpu_executor_matmul_transpose_b_zeros() {
    let mut cpu = CpuExecutor::new();
    let a = vec![0.0; 6];
    let b = vec![1.0; 12];

    let c = cpu.matmul_transpose_b(&a, &b, 2, 3, 4).unwrap();

    assert!(c.iter().all(|&x| x == 0.0));
}

// ============================================================================
// ExecutorCall::MatmulTransposeB Tests
// ============================================================================

#[test]
fn test_executor_call_matmul_transpose_b_equality() {
    let call1 = ExecutorCall::MatmulTransposeB {
        a_len: 6,
        b_len: 12,
        m: 2,
        k: 3,
        n: 4,
    };
    let call2 = ExecutorCall::MatmulTransposeB {
        a_len: 6,
        b_len: 12,
        m: 2,
        k: 3,
        n: 4,
    };
    let call3 = ExecutorCall::MatmulTransposeB {
        a_len: 6,
        b_len: 12,
        m: 2,
        k: 3,
        n: 5, // Different n
    };

    assert_eq!(call1, call2);
    assert_ne!(call1, call3);
}

#[test]
fn test_executor_call_matmul_transpose_b_clone() {
    let call = ExecutorCall::MatmulTransposeB {
        a_len: 6,
        b_len: 12,
        m: 2,
        k: 3,
        n: 4,
    };
    let cloned = call.clone();
    assert_eq!(call, cloned);
}

#[test]
fn test_executor_call_matmul_transpose_b_debug() {
    let call = ExecutorCall::MatmulTransposeB {
        a_len: 6,
        b_len: 12,
        m: 2,
        k: 3,
        n: 4,
    };
    let debug_str = format!("{:?}", call);
    assert!(debug_str.contains("MatmulTransposeB"));
    assert!(debug_str.contains("a_len: 6"));
}

#[test]
fn test_executor_call_matmul_vs_transpose_b() {
    let matmul = ExecutorCall::Matmul {
        a_len: 6,
        b_len: 12,
        m: 2,
        k: 3,
        n: 4,
    };
    let transpose_b = ExecutorCall::MatmulTransposeB {
        a_len: 6,
        b_len: 12,
        m: 2,
        k: 3,
        n: 4,
    };

    // Same dimensions but different call types
    assert_ne!(matmul, transpose_b);
}

// ============================================================================
// Trait Object Tests for matmul_transpose_b
// ============================================================================

#[test]
fn test_executor_trait_matmul_transpose_b_polymorphism() {
    fn run_transpose_matmul(executor: &mut dyn GpuExecutorTrait) -> Result<Vec<f32>> {
        let a = vec![1.0, 2.0, 3.0, 4.0]; // 2x2
        let b = vec![1.0; 4]; // 2x2
        executor.matmul_transpose_b(&a, &b, 2, 2, 2)
    }

    let mut mock = MockExecutor::new("mock");
    let mut cpu = CpuExecutor::new();

    let mock_result = run_transpose_matmul(&mut mock).unwrap();
    let cpu_result = run_transpose_matmul(&mut cpu).unwrap();

    // Mock returns zeros
    assert_eq!(mock_result, vec![0.0; 4]);

    // CPU returns actual computation: A @ (all-ones)^T
    // With B = [[1,1], [1,1]], B^T = [[1,1], [1,1]]
    // C[0,0] = 1*1 + 2*1 = 3
    // C[0,1] = 1*1 + 2*1 = 3
    // C[1,0] = 3*1 + 4*1 = 7
    // C[1,1] = 3*1 + 4*1 = 7
    assert!((cpu_result[0] - 3.0).abs() < 1e-5);
    assert!((cpu_result[2] - 7.0).abs() < 1e-5);
}

// ============================================================================
// MockExecutor calls() Method Test
// ============================================================================

#[test]
fn test_mock_executor_calls_returns_all() {
    let mut mock = MockExecutor::new("test");

    let _ = mock.matmul(&[1.0; 4], &[1.0; 4], 2, 2, 2);
    let _ = mock.matmul_transpose_b(&[1.0; 6], &[1.0; 6], 2, 3, 2);
    let _ = mock.matmul(&[1.0; 9], &[1.0; 9], 3, 3, 3);

    let calls = mock.calls();
    assert_eq!(calls.len(), 3);

    assert!(matches!(calls[0], ExecutorCall::Matmul { m: 2, .. }));
    assert!(matches!(calls[1], ExecutorCall::MatmulTransposeB { m: 2, .. }));
    assert!(matches!(calls[2], ExecutorCall::Matmul { m: 3, .. }));
}

// ============================================================================
// Edge Cases and Stress Tests
// ============================================================================

#[test]
fn test_cpu_executor_matmul_transpose_b_large_k() {
    let mut cpu = CpuExecutor::new();

    // Large inner dimension
    let k = 128;
    let a = vec![1.0; k]; // 1 x k
    let b = vec![1.0; k]; // 1 x k

    let c = cpu.matmul_transpose_b(&a, &b, 1, k, 1).unwrap();

    assert_eq!(c.len(), 1);
    // Dot product of all ones = k
    assert!((c[0] - k as f32).abs() < 1e-3);
}

#[test]
fn test_mock_executor_has_call_matmul_transpose_b() {
    let mut mock = MockExecutor::new("test");
    let _ = mock.matmul_transpose_b(&[1.0; 6], &[1.0; 8], 2, 3, 4);

    let expected = ExecutorCall::MatmulTransposeB {
        a_len: 6,
        b_len: 8,
        m: 2,
        k: 3,
        n: 4,
    };
    assert!(mock.has_call(&expected));

    // Wrong type should not match
    let wrong_type = ExecutorCall::Matmul {
        a_len: 6,
        b_len: 8,
        m: 2,
        k: 3,
        n: 4,
    };
    assert!(!mock.has_call(&wrong_type));
}

#[test]
fn test_mock_executor_unavailable_matmul_transpose_b() {
    let mut mock = MockExecutor::unavailable("disabled");

    // Even when unavailable, operations still work (availability is just a flag)
    let result = mock.matmul_transpose_b(&[1.0; 4], &[1.0; 4], 2, 2, 2);
    assert!(result.is_ok());
    assert!(!mock.is_available());
}

#[test]
fn test_cpu_executor_matmul_transpose_b_negative_values() {
    let mut cpu = CpuExecutor::new();

    let a = vec![-1.0, 2.0, -3.0, 4.0]; // 2x2
    let b = vec![1.0, -1.0, 2.0, -2.0]; // 2x2

    // A @ B^T where B^T = [[1, 2], [-1, -2]]
    // C[0,0] = -1*1 + 2*(-1) = -3
    // C[0,1] = -1*2 + 2*(-2) = -6
    // C[1,0] = -3*1 + 4*(-1) = -7
    // C[1,1] = -3*2 + 4*(-2) = -14
    let c = cpu.matmul_transpose_b(&a, &b, 2, 2, 2).unwrap();

    assert!((c[0] - (-3.0)).abs() < 1e-5, "c[0] = {}", c[0]);
    assert!((c[1] - (-6.0)).abs() < 1e-5, "c[1] = {}", c[1]);
    assert!((c[2] - (-7.0)).abs() < 1e-5, "c[2] = {}", c[2]);
    assert!((c[3] - (-14.0)).abs() < 1e-5, "c[3] = {}", c[3]);
}
