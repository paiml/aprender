use super::*;

#[test]
fn test_loading_mode_default() {
    assert_eq!(LoadingMode::default(), LoadingMode::Eager);
}

#[test]
fn test_loading_mode_description() {
    assert!(!LoadingMode::Eager.description().is_empty());
    assert!(!LoadingMode::Streaming.description().is_empty());
}

#[test]
fn test_loading_mode_zero_copy() {
    assert!(LoadingMode::MappedDemand.supports_zero_copy());
    assert!(!LoadingMode::Eager.supports_zero_copy());
}

#[test]
fn test_loading_mode_deterministic() {
    assert!(LoadingMode::Eager.is_deterministic());
    assert!(LoadingMode::Streaming.is_deterministic());
    assert!(!LoadingMode::MappedDemand.is_deterministic());
}

#[test]
fn test_loading_mode_for_memory_budget() {
    assert_eq!(LoadingMode::for_memory_budget(200, 100), LoadingMode::Eager);
    assert_eq!(
        LoadingMode::for_memory_budget(100, 100),
        LoadingMode::MappedDemand
    );
    assert_eq!(
        LoadingMode::for_memory_budget(512 * 1024, 1024 * 1024),
        LoadingMode::Streaming
    );
}

#[test]
fn test_verification_level_default() {
    assert_eq!(VerificationLevel::default(), VerificationLevel::Standard);
}

#[test]
fn test_verification_level_checksums() {
    assert!(!VerificationLevel::UnsafeSkip.verifies_checksum());
    assert!(VerificationLevel::ChecksumOnly.verifies_checksum());
    assert!(VerificationLevel::Standard.verifies_checksum());
    assert!(VerificationLevel::Paranoid.verifies_checksum());
}

#[test]
fn test_verification_level_signatures() {
    assert!(!VerificationLevel::UnsafeSkip.verifies_signature());
    assert!(!VerificationLevel::ChecksumOnly.verifies_signature());
    assert!(VerificationLevel::Standard.verifies_signature());
    assert!(VerificationLevel::Paranoid.verifies_signature());
}

#[test]
fn test_verification_level_asil() {
    assert_eq!(VerificationLevel::Paranoid.asil_level(), "ASIL-D");
    assert_eq!(VerificationLevel::Standard.asil_level(), "ASIL-B");
}

#[test]
fn test_verification_level_dal() {
    assert_eq!(VerificationLevel::Paranoid.dal_level(), "DAL-A");
    assert_eq!(VerificationLevel::Standard.dal_level(), "DAL-C");
}

#[test]
fn test_buffer_pool_creation() {
    let pool = BufferPool::new(4, 1024);
    assert_eq!(pool.buffer_size(), 1024);
    assert_eq!(pool.free_count(), 4);
    assert_eq!(pool.total_count(), 4);
    assert_eq!(pool.total_memory(), 4096);
}

#[test]
fn test_backend_default() {
    assert_eq!(Backend::default(), Backend::CpuSimd);
}

#[test]
fn test_backend_simd_support() {
    assert!(Backend::CpuSimd.supports_simd());
    assert!(!Backend::Wasm.supports_simd());
}

#[test]
fn test_backend_std_requirement() {
    assert!(Backend::CpuSimd.requires_std());
    assert!(!Backend::Embedded.requires_std());
}

#[test]
fn test_load_config_default() {
    let config = LoadConfig::default();
    assert_eq!(config.mode, LoadingMode::Eager);
    assert_eq!(config.verification, VerificationLevel::Standard);
    assert_eq!(config.backend, Backend::CpuSimd);
    assert!(config.max_memory_bytes.is_none());
}

#[test]
fn test_load_config_embedded() {
    let config = LoadConfig::embedded(1024 * 1024);
    assert_eq!(config.mode, LoadingMode::Eager);
    assert_eq!(config.verification, VerificationLevel::Paranoid);
    assert_eq!(config.backend, Backend::Embedded);
    assert_eq!(config.max_memory_bytes, Some(1024 * 1024));
}

#[test]
fn test_load_config_server() {
    let config = LoadConfig::server();
    assert_eq!(config.mode, LoadingMode::MappedDemand);
    assert_eq!(config.verification, VerificationLevel::Standard);
    assert!(config.max_memory_bytes.is_none());
}

#[test]
fn test_load_config_wasm() {
    let config = LoadConfig::wasm();
    assert_eq!(config.mode, LoadingMode::Streaming);
    assert_eq!(config.backend, Backend::Wasm);
    assert!(config.streaming);
}

#[test]
fn test_load_config_builder() {
    let config = LoadConfig::new()
        .with_mode(LoadingMode::Streaming)
        .with_max_memory(1024)
        .with_verification(VerificationLevel::Paranoid)
        .with_backend(Backend::Gpu)
        .with_time_budget(Duration::from_millis(100))
        .with_streaming(512 * 1024);

    assert_eq!(config.mode, LoadingMode::Streaming);
    assert_eq!(config.max_memory_bytes, Some(1024));
    assert_eq!(config.verification, VerificationLevel::Paranoid);
    assert_eq!(config.backend, Backend::Gpu);
    assert!(config.streaming);
    assert_eq!(config.ring_buffer_size, 512 * 1024);
}

#[test]
fn test_load_result() {
    let result = LoadResult::new(Duration::from_millis(100), 10 * 1024 * 1024);
    assert_eq!(result.memory_used, 10 * 1024 * 1024);
    // 10 MB in 100ms = 100 MB/s
    let throughput = result.throughput_mbps();
    assert!((throughput - 100.0).abs() < 1.0);
}

#[test]
fn test_load_result_zero_time() {
    let result = LoadResult::new(Duration::ZERO, 1024);
    assert_eq!(result.throughput_mbps(), 0.0);
}

#[test]
fn test_backend_gpu_accelerated() {
    assert!(Backend::Gpu.is_gpu_accelerated());
    assert!(Backend::Cuda.is_gpu_accelerated());
    assert!(!Backend::CpuSimd.is_gpu_accelerated());
    assert!(!Backend::Wasm.is_gpu_accelerated());
    assert!(!Backend::Embedded.is_gpu_accelerated());
}

#[test]
fn test_backend_nvidia_driver_requirement() {
    assert!(Backend::Cuda.requires_nvidia_driver());
    assert!(!Backend::Gpu.requires_nvidia_driver());
    assert!(!Backend::CpuSimd.requires_nvidia_driver());
    assert!(!Backend::Wasm.requires_nvidia_driver());
    assert!(!Backend::Embedded.requires_nvidia_driver());
}

#[test]
fn test_load_config_cuda() {
    let config = LoadConfig::cuda();
    assert_eq!(config.mode, LoadingMode::MappedDemand);
    assert_eq!(config.backend, Backend::Cuda);
    assert_eq!(config.verification, VerificationLevel::Standard);
    assert!(config.max_memory_bytes.is_none());
    assert!(!config.streaming);
    assert!(config.backend.requires_nvidia_driver());
    assert!(config.backend.is_gpu_accelerated());
}

#[test]
fn test_load_config_gpu() {
    let config = LoadConfig::gpu();
    assert_eq!(config.mode, LoadingMode::MappedDemand);
    assert_eq!(config.backend, Backend::Gpu);
    assert_eq!(config.verification, VerificationLevel::Standard);
    assert!(config.max_memory_bytes.is_none());
    assert!(!config.streaming);
    assert!(!config.backend.requires_nvidia_driver());
    assert!(config.backend.is_gpu_accelerated());
}

#[test]
fn test_backend_cuda_properties() {
    let backend = Backend::Cuda;
    // CUDA requires std library
    assert!(backend.requires_std());
    // CUDA does not use CPU SIMD
    assert!(!backend.supports_simd());
    // CUDA is GPU accelerated
    assert!(backend.is_gpu_accelerated());
    // CUDA requires NVIDIA driver
    assert!(backend.requires_nvidia_driver());
}

#[test]
fn test_load_config_builder_with_cuda() {
    let config = LoadConfig::new()
        .with_backend(Backend::Cuda)
        .with_mode(LoadingMode::Eager)
        .with_verification(VerificationLevel::Paranoid);

    assert_eq!(config.backend, Backend::Cuda);
    assert_eq!(config.mode, LoadingMode::Eager);
    assert_eq!(config.verification, VerificationLevel::Paranoid);
}
