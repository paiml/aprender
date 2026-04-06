#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::disallowed_methods,
    clippy::float_cmp,
    clippy::cast_precision_loss,
    clippy::uninlined_format_args,
    clippy::unreadable_literal,
    clippy::panic
)]
//! Integration tests for repartir
//!
//! These tests verify end-to-end workflows combining multiple components.
//! Includes pepita-repartir integration testing.

use repartir::executor::cpu::CpuExecutor;
use repartir::executor::Executor;
use repartir::scheduler::Scheduler;
use repartir::task::{Backend, Priority, Task};
use std::time::Duration;

// ============================================================================
// PEPITA-REPARTIR INTEGRATION TESTS
// ============================================================================

#[cfg(feature = "simd")]
mod simd_integration {
    use repartir::executor::simd::{SimdExecutor, SimdTask};

    #[tokio::test]
    async fn test_pepita_simd_vector_addition() {
        let executor = SimdExecutor::new();

        // Test SIMD vector addition using pepita's SIMD primitives
        let a: Vec<f32> = (0..1024).map(|i| i as f32).collect();
        let b: Vec<f32> = (0..1024).map(|i| i as f32 * 2.0).collect();

        let task = SimdTask::vadd_f32(a.clone(), b.clone());
        let result = executor.execute_simd(task).await.unwrap();

        // Verify correctness
        for i in 0..1024 {
            assert_eq!(result.output_f32()[i], a[i] + b[i]);
        }

        // Verify metrics
        assert!(result.throughput() > 0.0);
        assert!(result.elements() == 1024);
    }

    #[tokio::test]
    async fn test_pepita_simd_dot_product() {
        let executor = SimdExecutor::new();

        // Test dot product: sum(a[i] * b[i])
        let a: Vec<f32> = vec![1.0; 1000];
        let b: Vec<f32> = vec![2.0; 1000];

        let task = SimdTask::dot_f32(a, b);
        let result = executor.execute_simd(task).await.unwrap();

        // 1.0 * 2.0 * 1000 = 2000.0
        assert_eq!(result.scalar_f32(), Some(2000.0));
    }

    #[tokio::test]
    async fn test_pepita_simd_matrix_multiply() {
        let executor = SimdExecutor::new();

        // 3x3 matrix multiplication
        // A = [[1,2,3],[4,5,6],[7,8,9]]
        // B = [[1,0,0],[0,1,0],[0,0,1]] (identity)
        // Result = A
        let a = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
        let b = vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];

        let task = SimdTask::matmul_f32(a.clone(), b, 3, 3, 3);
        let result = executor.execute_simd(task).await.unwrap();

        assert_eq!(result.output_f32(), &a);
    }

    #[tokio::test]
    async fn test_pepita_simd_large_computation() {
        let executor = SimdExecutor::new();

        // Large vector operations to test SIMD performance
        let size = 100_000;
        let a: Vec<f32> = (0..size).map(|i| (i % 100) as f32).collect();
        let b: Vec<f32> = (0..size).map(|i| ((i + 50) % 100) as f32).collect();

        let task = SimdTask::vmul_f32(a.clone(), b.clone());
        let result = executor.execute_simd(task).await.unwrap();

        // Verify a few samples
        assert_eq!(result.output_f32()[0], a[0] * b[0]);
        assert_eq!(result.output_f32()[1000], a[1000] * b[1000]);
        assert_eq!(result.output_f32()[99999], a[99999] * b[99999]);

        // Verify high throughput for large vectors
        assert!(result.throughput() > 1_000_000.0);
    }

    #[tokio::test]
    async fn test_pepita_simd_executor_metrics_accumulation() {
        let executor = SimdExecutor::new();

        // Execute multiple operations
        for _ in 0..10 {
            let task = SimdTask::vadd_f32(vec![1.0, 2.0, 3.0, 4.0], vec![5.0, 6.0, 7.0, 8.0]);
            executor.execute_simd(task).await.unwrap();
        }

        let metrics = executor.metrics();
        assert_eq!(metrics.operations(), 10);
        assert_eq!(metrics.elements_processed(), 40);
        assert!(metrics.avg_throughput() > 0.0);
    }

    #[tokio::test]
    async fn test_pepita_simd_capabilities() {
        let executor = SimdExecutor::new();

        // Test capability detection from pepita
        let caps = executor.capabilities();
        assert!(caps.best_vector_width() >= 64);

        // On x86_64 or aarch64, we should have some SIMD
        #[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
        assert!(executor.has_simd());
    }
}

#[cfg(feature = "microvm")]
mod microvm_integration {
    use repartir::executor::microvm::{MicroVmExecutor, MicroVmExecutorConfig};

    #[tokio::test]
    async fn test_pepita_microvm_executor_creation() {
        let config = MicroVmExecutorConfig::builder()
            .memory_mib(128)
            .vcpus(1)
            .warm_pool(true, 2)
            .build()
            .unwrap();

        let executor = MicroVmExecutor::new(config).unwrap();
        // MicroVmExecutor is successfully created — verify it has expected config
        assert_eq!(executor.config().memory_mib, 128);
    }

    #[tokio::test]
    async fn test_pepita_microvm_config_default() {
        let config = MicroVmExecutorConfig::default();
        assert_eq!(config.memory_mib, 128);
        assert_eq!(config.vcpus, 1);
        assert!(config.warm_pool_enabled);
    }

    #[tokio::test]
    async fn test_pepita_microvm_config_validation() {
        // Zero memory should fail
        let result = MicroVmExecutorConfig::builder().memory_mib(0).build();
        assert!(result.is_err());

        // Zero vcpus should fail
        let result = MicroVmExecutorConfig::builder().vcpus(0).build();
        assert!(result.is_err());
    }
}

#[cfg(feature = "serverless")]
mod serverless_integration {
    use repartir::serverless::{Function, FunctionService, Runtime, Trigger};
    use std::path::PathBuf;
    use std::time::Duration;

    #[tokio::test]
    async fn test_pepita_serverless_function_lifecycle() {
        let mut service = FunctionService::new();

        // Create a serverless function
        let function = Function::builder()
            .name("test-function")
            .runtime(Runtime::RustNative { binary: PathBuf::from("/bin/echo") })
            .memory_mib(128)
            .timeout(Duration::from_secs(30))
            .build()
            .unwrap();

        let function_name = function.name().to_string();

        // Register the function
        service.register(function).unwrap();

        // Verify registration
        assert!(service.get(&function_name).is_some());

        // List functions
        let functions = service.list();
        assert_eq!(functions.len(), 1);
    }

    #[tokio::test]
    async fn test_pepita_serverless_triggers() {
        use repartir::serverless::HttpMethod;

        let _http_trigger = Trigger::Http {
            path: "/api/test".to_string(),
            methods: vec![HttpMethod::Get, HttpMethod::Post],
        };

        let _schedule_trigger = Trigger::Schedule { cron: "0 * * * *".to_string() };

        let _queue_trigger = Trigger::Queue { queue_name: "test-queue".to_string() };
    }

    #[tokio::test]
    async fn test_pepita_serverless_multiple_functions() {
        let mut service = FunctionService::new();

        // Register multiple functions
        for i in 0..5 {
            let function = Function::builder()
                .name(format!("function-{i}"))
                .runtime(Runtime::RustNative { binary: PathBuf::from("/bin/echo") })
                .memory_mib(64)
                .build()
                .unwrap();
            service.register(function).unwrap();
        }

        let functions = service.list();
        assert_eq!(functions.len(), 5);
    }
}

// ============================================================================
// END-TO-END WORKFLOW TESTS
// ============================================================================

#[cfg(all(feature = "simd", feature = "microvm"))]
mod end_to_end {
    use super::*;
    use repartir::executor::simd::{SimdExecutor, SimdTask};

    #[tokio::test]
    async fn test_hybrid_simd_cpu_workflow() {
        // Test combining SIMD operations with CPU task execution
        let simd_executor = SimdExecutor::new();
        let cpu_executor = CpuExecutor::new();

        // SIMD computation
        let a: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0];
        let b: Vec<f32> = vec![5.0, 6.0, 7.0, 8.0];
        let simd_task = SimdTask::vadd_f32(a, b);
        let simd_result = simd_executor.execute_simd(simd_task).await.unwrap();

        // Verify SIMD result
        assert_eq!(simd_result.output_f32(), &[6.0, 8.0, 10.0, 12.0]);

        // CPU task
        #[cfg(unix)]
        {
            let cpu_task = Task::builder()
                .binary("/bin/echo")
                .arg("simd complete")
                .backend(Backend::Cpu)
                .build()
                .unwrap();

            let cpu_result = cpu_executor.execute(cpu_task).await.unwrap();
            assert!(cpu_result.is_success());
        }
    }

    #[tokio::test]
    async fn test_scheduler_with_simd_tasks() {
        let scheduler = Scheduler::new();
        let simd_executor = SimdExecutor::new();

        // Submit multiple SIMD-oriented tasks through scheduler
        for i in 0..5 {
            let task = Task::builder()
                .binary(format!("simd_task_{i}"))
                .backend(Backend::Cpu)
                .priority(Priority::High)
                .build()
                .unwrap();

            scheduler.submit(task).await.unwrap();
        }

        // Verify scheduling
        assert_eq!(scheduler.pending_count().await, 5);

        // Execute a parallel SIMD computation
        let task = SimdTask::vadd_f32(vec![1.0, 2.0, 3.0], vec![4.0, 5.0, 6.0]);
        let result = simd_executor.execute_simd(task).await.unwrap();
        assert_eq!(result.output_f32(), &[5.0, 7.0, 9.0]);
    }
}

// ============================================================================
// PEPITA PRIMITIVE INTEGRATION TESTS
// ============================================================================

#[cfg(feature = "simd")]
mod pepita_primitives {
    use pepita::simd::{SimdCapabilities, SimdOps};
    use pepita::virtio::{VirtQueue, VirtioVsock, VsockAddr};
    use pepita::vmm::VmState;
    use pepita::zram::{ZramCompressor, ZramConfig, ZramDevice};

    #[test]
    fn test_pepita_simd_capabilities() {
        let caps = SimdCapabilities::detect();
        assert!(caps.best_vector_width() >= 64);
    }

    #[test]
    fn test_pepita_simd_ops() {
        let ops = SimdOps::new();
        let a = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let b = vec![8.0f32, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0];
        let mut c = vec![0.0f32; 8];

        ops.vadd_f32(&a, &b, &mut c);

        // All elements should sum to 9.0
        for val in c {
            assert_eq!(val, 9.0);
        }
    }

    #[test]
    fn test_pepita_vm_state_transitions() {
        let state = VmState::default();
        assert_eq!(state, VmState::Created);
        assert!(state.can_start());

        let running = VmState::Running;
        assert!(!running.can_start());
        assert!(running.is_active());
    }

    #[test]
    fn test_pepita_zram_device() {
        let config = ZramConfig::default();
        let device = ZramDevice::new(config).unwrap();

        // Write and read a page
        let mut data = [0u8; 4096];
        data[0] = 0xAB;
        data[4095] = 0xCD;

        device.write_page(0, &data).unwrap();

        let mut read_data = [0u8; 4096];
        device.read_page(0, &mut read_data).unwrap();

        assert_eq!(read_data[0], 0xAB);
        assert_eq!(read_data[4095], 0xCD);
    }

    #[test]
    fn test_pepita_zram_compression_ratio() {
        let config = ZramConfig::with_size(256 * 4096); // 256 pages
        let device = ZramDevice::new(config).unwrap();

        // Write highly compressible data (all zeros)
        let zero_page = [0u8; 4096];
        for i in 0..100 {
            device.write_page(i, &zero_page).unwrap();
        }

        let stats = device.stats();
        // Zero pages should have excellent "compression"
        assert!(stats.zero_pages > 0 || stats.same_pages > 0);
    }

    #[test]
    fn test_pepita_virtqueue() {
        let queue = VirtQueue::with_size(64);
        assert_eq!(queue.size(), 64);
        assert!(!queue.is_ready());

        queue.set_ready(true);
        assert!(queue.is_ready());
    }

    #[test]
    fn test_pepita_virtio_vsock() {
        let vsock = VirtioVsock::new(3);
        assert_eq!(vsock.cid(), 3);
        assert!(!vsock.is_ready());

        vsock.activate();
        assert!(vsock.is_ready());
    }

    #[test]
    fn test_pepita_vsock_addr() {
        let addr = VsockAddr::new(3, 1234);
        assert_eq!(addr.cid, 3);
        assert_eq!(addr.port, 1234);
        assert!(!addr.is_host());

        let host = VsockAddr::host(8080);
        assert!(host.is_host());
        assert_eq!(format!("{}", host), "2:8080");
    }

    #[test]
    fn test_pepita_zram_compressor_types() {
        let lz4 = ZramCompressor::Lz4;
        assert_eq!(lz4.name(), "lz4");

        let none = ZramCompressor::None;
        assert_eq!(none.name(), "none");
    }
}

#[tokio::test]
async fn test_cpu_executor_basic_workflow() {
    let executor = CpuExecutor::new();

    #[cfg(unix)]
    {
        let task = Task::builder()
            .binary("/bin/echo")
            .arg("integration test")
            .backend(Backend::Cpu)
            .build()
            .unwrap();

        let result = executor.execute(task).await.unwrap();
        assert!(result.is_success());
        assert_eq!(result.stdout_str().unwrap().trim(), "integration test");
    }
}

#[tokio::test]
async fn test_scheduler_cpu_executor_workflow() {
    let scheduler = Scheduler::new();
    let executor = CpuExecutor::new();

    #[cfg(unix)]
    {
        // Submit multiple tasks
        let mut task_ids = Vec::new();
        for i in 0..5 {
            let task = Task::builder()
                .binary("/bin/echo")
                .arg(format!("task_{i}"))
                .backend(Backend::Cpu)
                .build()
                .unwrap();

            let task_id = scheduler.submit(task).await.unwrap();
            task_ids.push(task_id);
        }

        // Execute tasks
        for _ in 0..5 {
            if let Some(task) = scheduler.next_task().await {
                let result = executor.execute(task).await.unwrap();
                assert!(result.is_success());
                scheduler.store_result(result).await;
            }
        }

        // Retrieve results
        for task_id in task_ids {
            let result = scheduler.get_result(task_id).await;
            assert!(result.is_some());
            assert!(result.unwrap().is_success());
        }
    }
}

#[tokio::test]
async fn test_priority_scheduling_integration() {
    let scheduler = Scheduler::new();

    #[cfg(unix)]
    {
        // Submit tasks in reverse priority order
        let low_task = Task::builder()
            .binary("/bin/echo")
            .arg("low")
            .backend(Backend::Cpu)
            .priority(Priority::Low)
            .build()
            .unwrap();
        scheduler.submit(low_task).await.unwrap();

        let normal_task = Task::builder()
            .binary("/bin/echo")
            .arg("normal")
            .backend(Backend::Cpu)
            .priority(Priority::Normal)
            .build()
            .unwrap();
        scheduler.submit(normal_task).await.unwrap();

        let high_task = Task::builder()
            .binary("/bin/echo")
            .arg("high")
            .backend(Backend::Cpu)
            .priority(Priority::High)
            .build()
            .unwrap();
        scheduler.submit(high_task).await.unwrap();

        // Tasks should be retrieved in priority order
        let task1 = scheduler.next_task().await.unwrap();
        assert_eq!(task1.priority(), Priority::High);

        let task2 = scheduler.next_task().await.unwrap();
        assert_eq!(task2.priority(), Priority::Normal);

        let task3 = scheduler.next_task().await.unwrap();
        assert_eq!(task3.priority(), Priority::Low);
    }
}

#[tokio::test]
async fn test_task_timeout_integration() {
    let executor = CpuExecutor::new();

    #[cfg(unix)]
    {
        let task = Task::builder()
            .binary("/bin/sleep")
            .arg("5")
            .backend(Backend::Cpu)
            .timeout(Duration::from_millis(100))
            .build()
            .unwrap();

        let result = executor.execute(task).await;
        assert!(result.is_err());
    }
}

#[tokio::test]
async fn test_task_with_environment_variables() {
    let executor = CpuExecutor::new();

    #[cfg(unix)]
    {
        let task = Task::builder()
            .binary("/bin/sh")
            .arg("-c")
            .arg("echo $MY_VAR")
            .env_var("MY_VAR", "integration_test_value")
            .backend(Backend::Cpu)
            .build()
            .unwrap();

        let result = executor.execute(task).await.unwrap();
        assert!(result.is_success());
        assert_eq!(result.stdout_str().unwrap().trim(), "integration_test_value");
    }
}

#[tokio::test]
async fn test_concurrent_task_execution() {
    use std::sync::Arc;

    let scheduler = Arc::new(Scheduler::with_capacity(100));
    let _executor = CpuExecutor::new();

    #[cfg(unix)]
    {
        // Submit 10 tasks concurrently
        let mut handles = Vec::new();
        for i in 0..10 {
            let task = Task::builder()
                .binary("/bin/echo")
                .arg(format!("concurrent_{i}"))
                .backend(Backend::Cpu)
                .build()
                .unwrap();

            scheduler.submit(task).await.unwrap();
        }

        // Execute tasks concurrently
        for _ in 0..10 {
            let scheduler_clone = Arc::clone(&scheduler);
            let handle = tokio::spawn(async move {
                if let Some(task) = scheduler_clone.next_task().await {
                    let executor = CpuExecutor::new();
                    executor.execute(task).await
                } else {
                    Err(repartir::error::RepartirError::InvalidTask {
                        reason: "No task available".to_string(),
                    })
                }
            });
            handles.push(handle);
        }

        // Wait for all tasks
        for handle in handles {
            let result = handle.await.unwrap();
            assert!(result.is_ok());
        }
    }
}

#[tokio::test]
async fn test_scheduler_clear_integration() {
    let scheduler = Scheduler::new();

    #[cfg(unix)]
    {
        // Submit tasks
        for i in 0..5 {
            let task = Task::builder()
                .binary("/bin/echo")
                .arg(format!("task_{i}"))
                .backend(Backend::Cpu)
                .build()
                .unwrap();
            scheduler.submit(task).await.unwrap();
        }

        assert_eq!(scheduler.pending_count().await, 5);

        // Clear scheduler
        scheduler.clear().await;
        assert_eq!(scheduler.pending_count().await, 0);
    }
}

#[tokio::test]
async fn test_error_handling_integration() {
    let executor = CpuExecutor::new();

    // Test nonexistent binary
    let task =
        Task::builder().binary("/nonexistent/binary/path").backend(Backend::Cpu).build().unwrap();

    let result = executor.execute(task).await;
    assert!(result.is_err());
}

#[tokio::test]
async fn test_non_zero_exit_code_integration() {
    let executor = CpuExecutor::new();

    #[cfg(unix)]
    {
        let task = Task::builder()
            .binary("/bin/sh")
            .arg("-c")
            .arg("exit 123")
            .backend(Backend::Cpu)
            .build()
            .unwrap();

        let result = executor.execute(task).await.unwrap();
        assert!(!result.is_success());
        assert_eq!(result.exit_code(), 123);
    }
}
