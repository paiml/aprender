use super::*;

#[test]
fn test_cuda_backend_name() {
    let backend = CudaBackend;
    assert_eq!(backend.name(), "CUDA");
}

#[test]
fn test_metal_backend_unavailable() {
    let backend = MetalBackend;
    assert!(!backend.is_available());
}

#[test]
fn test_detect_backend() {
    let backend = detect_backend();
    // Should return something
    assert!(!backend.name().is_empty());
}

#[test]
fn test_metal_backend_name() {
    let backend = MetalBackend;
    assert_eq!(backend.name(), "Metal");
}

#[test]
fn test_vulkan_backend_name() {
    let backend = VulkanBackend;
    assert_eq!(backend.name(), "Vulkan");
}

#[test]
fn test_vulkan_backend_unavailable() {
    let backend = VulkanBackend;
    assert!(!backend.is_available());
}

#[test]
fn test_cuda_backend_device_count() {
    let backend = CudaBackend;
    // Device count depends on hardware - just check it's non-negative
    let count = backend.device_count();
    // Count is 0 when CUDA unavailable, otherwise a positive number
    assert!(backend.is_available() || count == 0);
}

#[test]
fn test_metal_backend_device_count() {
    let backend = MetalBackend;
    assert_eq!(backend.device_count(), 0);
}

#[test]
fn test_vulkan_backend_device_count() {
    let backend = VulkanBackend;
    assert_eq!(backend.device_count(), 0);
}

#[test]
fn test_cuda_backend_default() {
    let backend = CudaBackend::default();
    assert_eq!(backend.name(), "CUDA");
}

#[test]
fn test_metal_backend_default() {
    let backend = MetalBackend::default();
    assert_eq!(backend.name(), "Metal");
}

#[test]
fn test_vulkan_backend_default() {
    let backend = VulkanBackend::default();
    assert_eq!(backend.name(), "Vulkan");
}

#[test]
fn test_wgpu_backend_name() {
    let backend = WgpuBackend;
    assert_eq!(backend.name(), "WGPU");
}

#[test]
fn test_wgpu_backend_default() {
    let backend = WgpuBackend::default();
    assert_eq!(backend.name(), "WGPU");
}

#[test]
fn test_wgpu_backend_device_count() {
    let backend = WgpuBackend;
    // Without wgpu feature, should be 0
    #[cfg(not(feature = "wgpu"))]
    assert_eq!(backend.device_count(), 0);
}

#[test]
fn test_wgpu_backend_is_available() {
    let backend = WgpuBackend;
    // Availability depends on wgpu feature flag
    #[cfg(not(feature = "wgpu"))]
    assert!(!backend.is_available());
    #[cfg(feature = "wgpu")]
    {
        // When feature is enabled, should be available
        let _ = backend.is_available(); // Just exercise the path
    }
}

#[test]
fn test_cuda_backend_is_available() {
    let backend = CudaBackend;
    // Without CUDA hardware, should return false
    // This exercises the is_available path
    let available = backend.is_available();
    // The result depends on hardware, but the call should succeed
    let _ = available;
}

#[test]
fn test_cuda_backend_debug() {
    let backend = CudaBackend;
    let debug_str = format!("{:?}", backend);
    assert!(debug_str.contains("CudaBackend"));
}

#[test]
fn test_metal_backend_debug() {
    let backend = MetalBackend;
    let debug_str = format!("{:?}", backend);
    assert!(debug_str.contains("MetalBackend"));
}

#[test]
fn test_vulkan_backend_debug() {
    let backend = VulkanBackend;
    let debug_str = format!("{:?}", backend);
    assert!(debug_str.contains("VulkanBackend"));
}

#[test]
fn test_wgpu_backend_debug() {
    let backend = WgpuBackend;
    let debug_str = format!("{:?}", backend);
    assert!(debug_str.contains("WgpuBackend"));
}

#[test]
fn test_detect_backend_returns_valid_name() {
    let backend = detect_backend();
    // Should always return a backend with a non-empty name
    let name = backend.name();
    assert!(!name.is_empty());
    // Name should be one of the known backends
    let valid_names = ["CUDA", "Metal", "Vulkan", "WGPU"];
    assert!(valid_names.contains(&name), "Unknown backend name");
}

#[test]
fn test_detect_backend_fallback_is_cuda() {
    // When no backends are available, detect_backend should return CUDA
    // as the fallback (for PTX generation)
    let backend = detect_backend();
    // In CI without GPU hardware, this should be CUDA
    // (it's the fallback at line 167)
    let any_available = CudaBackend.is_available()
        || WgpuBackend.is_available()
        || MetalBackend.is_available()
        || VulkanBackend.is_available();
    if !any_available {
        assert_eq!(backend.name(), "CUDA");
    }
}

#[test]
fn test_backend_trait_send_sync() {
    // Verify that all backends implement Send + Sync
    fn assert_send_sync<T: Send + Sync>() {}
    assert_send_sync::<CudaBackend>();
    assert_send_sync::<MetalBackend>();
    assert_send_sync::<VulkanBackend>();
    assert_send_sync::<WgpuBackend>();
}

#[test]
fn test_all_backends_device_count_consistent() {
    // Device count should be 0 when backend is not available
    // Test each backend: if unavailable, count must be 0
    let cuda = CudaBackend;
    let cuda_count = cuda.device_count();
    assert!(cuda.is_available() || cuda_count == 0);

    let metal = MetalBackend;
    let metal_count = metal.device_count();
    assert!(metal.is_available() || metal_count == 0);

    let vulkan = VulkanBackend;
    let vulkan_count = vulkan.device_count();
    assert!(vulkan.is_available() || vulkan_count == 0);

    let wgpu = WgpuBackend;
    let wgpu_count = wgpu.device_count();
    assert!(wgpu.is_available() || wgpu_count == 0);
}

#[test]
fn test_detect_backend_is_deterministic() {
    // Calling detect_backend multiple times should return the same backend
    let backend1 = detect_backend();
    let backend2 = detect_backend();
    assert_eq!(backend1.name(), backend2.name());
}

#[test]
fn test_boxed_backend_trait_object() {
    // Test that backends work correctly as trait objects
    let backends: Vec<Box<dyn Backend>> = vec![
        Box::new(CudaBackend),
        Box::new(MetalBackend),
        Box::new(VulkanBackend),
        Box::new(WgpuBackend),
    ];

    for backend in &backends {
        assert!(!backend.name().is_empty());
        let _ = backend.is_available();
        let _ = backend.device_count();
    }
}

#[test]
fn test_detect_backend_priority_cuda_first() {
    // CUDA has highest priority. If CUDA is available, detect_backend must return CUDA.
    let backend = detect_backend();
    if CudaBackend.is_available() {
        assert_eq!(
            backend.name(),
            "CUDA",
            "When CUDA is available, detect_backend must return CUDA (highest priority)"
        );
    }
}

#[test]
fn test_detect_backend_wgpu_priority_over_metal_and_vulkan() {
    // When CUDA is not available, WGPU has priority over Metal and Vulkan.
    // We cannot mock is_available, but we can verify the priority logic
    // by checking that if CUDA and WGPU are both unavailable,
    // we do NOT get WGPU.
    let backend = detect_backend();
    if !CudaBackend.is_available() && !WgpuBackend.is_available() {
        assert_ne!(
            backend.name(),
            "WGPU",
            "When WGPU is not available, detect_backend should not return WGPU"
        );
    }
}

#[test]
fn test_detect_backend_metal_not_returned_on_linux() {
    // MetalBackend has no dispatcher, so it never reports available and the
    // Metal branch in detect_backend is unreachable on every platform.
    let metal = MetalBackend;
    assert!(!metal.is_available());
}

#[test]
fn test_detect_backend_vulkan_never_available() {
    // Vulkan backend is a placeholder that always returns false
    let vulkan = VulkanBackend;
    assert!(!vulkan.is_available());
    assert_eq!(vulkan.device_count(), 0);
    // detect_backend should never return Vulkan since it's never available
    let backend = detect_backend();
    assert_ne!(
        backend.name(),
        "Vulkan",
        "Vulkan should never be returned by detect_backend (placeholder)"
    );
}

#[test]
fn test_detect_backend_default_fallback_is_cuda_for_ptx() {
    // Even when nothing is available, the fallback is CUDA
    // (for PTX generation which doesn't need hardware).
    // This verifies the final return at line 167.
    let cuda_avail = CudaBackend.is_available();
    let wgpu_avail = WgpuBackend.is_available();
    let metal_avail = MetalBackend.is_available();
    let vulkan_avail = VulkanBackend.is_available();

    let backend = detect_backend();

    if !cuda_avail && !wgpu_avail && !metal_avail && !vulkan_avail {
        assert_eq!(
            backend.name(),
            "CUDA",
            "Fallback should be CUDA for PTX generation"
        );
        // Fallback CUDA backend is still not available for execution
        // but its name is "CUDA" for PTX codegen purposes
    }
}

#[test]
fn test_backend_availability_mutual_exclusion() {
    // On any given platform, certain backends should be mutually exclusive.
    // Metal is only available on macOS, Vulkan is never available (placeholder).
    let vulkan = VulkanBackend;
    assert!(
        !vulkan.is_available(),
        "Vulkan placeholder is never available"
    );

    #[cfg(not(target_os = "macos"))]
    {
        let metal = MetalBackend;
        assert!(!metal.is_available(), "Metal is not available on non-macOS");
    }
}

#[test]
fn test_wgpu_device_count_matches_availability() {
    let wgpu = WgpuBackend;
    if wgpu.is_available() {
        assert_eq!(wgpu.device_count(), 1, "WGPU returns 1 when available");
    } else {
        assert_eq!(wgpu.device_count(), 0, "WGPU returns 0 when unavailable");
    }
}

#[test]
fn test_cuda_device_count_matches_availability() {
    let cuda = CudaBackend;
    if !cuda.is_available() {
        assert_eq!(
            cuda.device_count(),
            0,
            "CUDA device_count must be 0 when unavailable"
        );
    } else {
        assert!(
            cuda.device_count() > 0,
            "CUDA device_count must be >0 when available"
        );
    }
}
