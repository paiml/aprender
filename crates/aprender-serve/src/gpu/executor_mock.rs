
#[cfg(test)]
mod tests {
    use super::*;

    // =========================================================================
    // MockExecutor Tests
    // =========================================================================

    #[test]
    fn test_mock_executor_creation() {
        let mock = MockExecutor::new("test");
        assert_eq!(mock.name(), "test");
        assert!(mock.is_available());
        assert_eq!(mock.call_count(), 0);
    }

    #[test]
    fn test_mock_executor_unavailable() {
        let mock = MockExecutor::unavailable("disabled");
        assert!(!mock.is_available());
    }

    #[test]
    fn test_mock_executor_records_matmul() {
        let mut mock = MockExecutor::new("test");
        let a = vec![1.0, 2.0, 3.0, 4.0]; // 2x2
        let b = vec![5.0, 6.0, 7.0, 8.0]; // 2x2

        let result = mock.matmul(&a, &b, 2, 2, 2).expect("result");

        assert_eq!(result.len(), 4); // 2x2 output
        assert_eq!(mock.matmul_count(), 1);

        let call = mock.last_call().expect("call");
        assert!(matches!(
            call,
            ExecutorCall::Matmul {
                a_len: 4,
                b_len: 4,
                m: 2,
                k: 2,
                n: 2
            }
        ));
    }

    #[test]
    fn test_mock_executor_custom_result() {
        let mut mock = MockExecutor::new("test").with_matmul_result(vec![1.0, 2.0, 3.0, 4.0]);

        let a = vec![0.0; 4];
        let b = vec![0.0; 4];
        let result = mock.matmul(&a, &b, 2, 2, 2).expect("result");

        assert_eq!(result, vec![1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn test_mock_executor_failure() {
        let mut mock = MockExecutor::new("test").with_matmul_failure();

        let a = vec![1.0; 4];
        let b = vec![1.0; 4];
        let result = mock.matmul(&a, &b, 2, 2, 2);

        assert!(result.is_err());
    }

    #[test]
    fn test_mock_executor_clear_calls() {
        let mut mock = MockExecutor::new("test");
        let _ = mock.matmul(&[1.0; 4], &[1.0; 4], 2, 2, 2);
        let _ = mock.matmul(&[1.0; 4], &[1.0; 4], 2, 2, 2);

        assert_eq!(mock.call_count(), 2);

        mock.clear_calls();
        assert_eq!(mock.call_count(), 0);
    }

    #[test]
    fn test_mock_executor_debug() {
        let mock = MockExecutor::new("debug_test");
        let debug_str = format!("{:?}", mock);
        assert!(debug_str.contains("MockExecutor"));
        assert!(debug_str.contains("debug_test"));
    }

    #[test]
    fn test_mock_executor_synchronize() {
        let mock = MockExecutor::new("test");
        assert!(mock.synchronize().is_ok());
    }

    // =========================================================================
    // CpuExecutor Tests
    // =========================================================================

    #[test]
    fn test_cpu_executor_creation() {
        let cpu = CpuExecutor::new();
        assert_eq!(cpu.name(), "CpuExecutor");
        assert!(cpu.is_available());
    }

    #[test]
    fn test_cpu_executor_default() {
        let cpu = CpuExecutor::default();
        assert_eq!(cpu.name(), "CpuExecutor");
    }

    #[test]
    fn test_cpu_executor_matmul_2x2() {
        let mut cpu = CpuExecutor::new();

        // [1, 2]   [5, 6]   [1*5+2*7, 1*6+2*8]   [19, 22]
        // [3, 4] @ [7, 8] = [3*5+4*7, 3*6+4*8] = [43, 50]
        let a = vec![1.0, 2.0, 3.0, 4.0];
        let b = vec![5.0, 6.0, 7.0, 8.0];

        let c = cpu.matmul(&a, &b, 2, 2, 2).expect("c");

        assert_eq!(c.len(), 4);
        assert!((c[0] - 19.0).abs() < 1e-5);
        assert!((c[1] - 22.0).abs() < 1e-5);
        assert!((c[2] - 43.0).abs() < 1e-5);
        assert!((c[3] - 50.0).abs() < 1e-5);
    }

    #[test]
    fn test_cpu_executor_matmul_vector() {
        let mut cpu = CpuExecutor::new();

        // [1, 2, 3] @ [[1], [2], [3]] = [1*1 + 2*2 + 3*3] = [14]
        let a = vec![1.0, 2.0, 3.0]; // 1x3
        let b = vec![1.0, 2.0, 3.0]; // 3x1

        let c = cpu.matmul(&a, &b, 1, 3, 1).expect("c");

        assert_eq!(c.len(), 1);
        assert!((c[0] - 14.0).abs() < 1e-5);
    }

    #[test]
    fn test_cpu_executor_matmul_rectangular() {
        let mut cpu = CpuExecutor::new();

        // 2x3 @ 3x4 = 2x4
        let a = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]; // 2x3
        let b = vec![1.0; 12]; // 3x4

        let c = cpu.matmul(&a, &b, 2, 3, 4).expect("c");

        assert_eq!(c.len(), 8);
        // First row: [1+2+3, 1+2+3, 1+2+3, 1+2+3] = [6, 6, 6, 6]
        assert!((c[0] - 6.0).abs() < 1e-5);
        // Second row: [4+5+6, ...] = [15, 15, 15, 15]
        assert!((c[4] - 15.0).abs() < 1e-5);
    }

    #[test]
    fn test_cpu_executor_matmul_dimension_error_a() {
        let mut cpu = CpuExecutor::new();
        let a = vec![1.0; 5]; // Wrong size
        let b = vec![1.0; 4];

        let result = cpu.matmul(&a, &b, 2, 2, 2);
        assert!(result.is_err());
    }

    #[test]
    fn test_cpu_executor_matmul_dimension_error_b() {
        let mut cpu = CpuExecutor::new();
        let a = vec![1.0; 4];
        let b = vec![1.0; 5]; // Wrong size

        let result = cpu.matmul(&a, &b, 2, 2, 2);
        assert!(result.is_err());
    }

    #[test]
    fn test_cpu_executor_synchronize() {
        let cpu = CpuExecutor::new();
        assert!(cpu.synchronize().is_ok());
    }

    #[test]
    fn test_cpu_executor_debug() {
        let cpu = CpuExecutor::new();
        let debug_str = format!("{:?}", cpu);
        assert!(debug_str.contains("CpuExecutor"));
    }

    // =========================================================================
    // Trait Object Tests
    // =========================================================================

    #[test]
    fn test_executor_trait_object() {
        let mock: Box<dyn GpuExecutorTrait> = Box::new(MockExecutor::new("boxed"));
        assert_eq!(mock.name(), "boxed");
        assert!(mock.is_available());
    }

    #[test]
    fn test_executor_trait_polymorphism() {
        fn run_matmul(executor: &mut dyn GpuExecutorTrait) -> Result<Vec<f32>> {
            let a = vec![1.0, 2.0, 3.0, 4.0];
            let b = vec![1.0; 4];
            executor.matmul(&a, &b, 2, 2, 2)
        }

        let mut mock = MockExecutor::new("mock");
        let mut cpu = CpuExecutor::new();

        let mock_result = run_matmul(&mut mock).expect("mock_result");
        let cpu_result = run_matmul(&mut cpu).expect("cpu_result");

        // Mock returns zeros, CPU returns actual computation
        assert_eq!(mock_result, vec![0.0; 4]);
        assert!((cpu_result[0] - 3.0).abs() < 1e-5); // 1*1+2*1 = 3
    }

    // =========================================================================
    // ExecutorCall Tests
    // =========================================================================

    #[test]
    fn test_executor_call_equality() {
        let call1 = ExecutorCall::Matmul {
            a_len: 4,
            b_len: 4,
            m: 2,
            k: 2,
            n: 2,
        };
        let call2 = ExecutorCall::Matmul {
            a_len: 4,
            b_len: 4,
            m: 2,
            k: 2,
            n: 2,
        };
        let call3 = ExecutorCall::Synchronize;

        assert_eq!(call1, call2);
        assert_ne!(call1, call3);
    }

    #[test]
    fn test_executor_call_clone() {
        let call = ExecutorCall::Matmul {
            a_len: 8,
            b_len: 8,
            m: 4,
            k: 2,
            n: 4,
        };
        let cloned = call.clone();
        assert_eq!(call, cloned);
    }

    #[test]
    fn test_executor_call_debug() {
        let call = ExecutorCall::Matmul {
            a_len: 4,
            b_len: 4,
            m: 2,
            k: 2,
            n: 2,
        };
        let debug_str = format!("{:?}", call);
        assert!(debug_str.contains("Matmul"));
    }
}
