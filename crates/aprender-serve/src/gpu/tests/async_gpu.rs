
#[test]
fn test_async_gpu_result_ready_ext_cov() {
    let result = AsyncGpuResult::ready(vec![1.0, 2.0, 3.0]);
    assert!(result.is_ready());
}

#[test]
fn test_async_gpu_result_try_get_ext_cov() {
    let result = AsyncGpuResult::ready(vec![1.0, 2.0, 3.0]);
    let data = result.try_get();
    assert!(data.is_some());
    assert_eq!(data.expect("index out of bounds"), &vec![1.0, 2.0, 3.0]);
}

#[test]
fn test_async_gpu_result_try_get_pending_ext_cov() {
    let result = AsyncGpuResult::pending();
    let data = result.try_get();
    assert!(data.is_none());
}

#[test]
fn test_async_gpu_result_wait_ext_cov() {
    let result = AsyncGpuResult::ready(vec![4.0, 5.0, 6.0]);
    let data = result.wait();
    assert_eq!(data, vec![4.0, 5.0, 6.0]);
}

#[test]
fn test_async_gpu_result_set_result_ext_cov() {
    let mut result = AsyncGpuResult::pending();
    assert!(!result.is_ready());
    result.set_result(vec![7.0, 8.0, 9.0]);
    assert!(result.is_ready());
    assert_eq!(
        result.try_get().expect("index out of bounds"),
        &vec![7.0, 8.0, 9.0]
    );
}

// --- GpuCompute Tests ---
#[test]
fn test_gpu_compute_auto_ext_cov() {
    let compute = GpuCompute::auto();
    assert!(compute.is_ok());
}

#[test]
fn test_gpu_compute_new_cpu_ext_cov() {
    let compute = GpuCompute::new(ComputeBackend::Cpu);
    assert!(compute.is_ok());
}

#[test]
fn test_gpu_compute_backend_ext_cov() {
    let compute = GpuCompute::new(ComputeBackend::Cpu).expect("GPU operation failed");
    let backend = compute.backend();
    assert!(matches!(backend, ComputeBackend::Cpu));
}
