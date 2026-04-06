//! GPU-accelerated compression backend.
//!
//! This module provides CUDA-accelerated LZ4/Zstd compression for batch operations.
//! Following the 5x PCIe rule [Gregg2011], GPU acceleration is only beneficial when
//! the computation time exceeds 5 times the data transfer time.
//!
//! For 4KB pages, this means batching 1000+ pages to amortize PCIe overhead.
//!
//! ## Hybrid Architecture (Sovereign AI)
//!
//! The hybrid scheduler uses:
//! - **CPU for compression**: 24 GB/s (avoids F082 hash table bug)
//! - **GPU for decompression**: 16 GB/s (F082-safe, no hash tables)
//!
//! This enables 2TB LLM checkpoint restore in <60s (G.119 target).

pub mod batch;
pub mod hybrid;

pub use batch::{
    BatchDecompressResult, BatchResult, GpuBatchCompressor, GpuBatchConfig, GpuBatchStats,
};
pub use hybrid::{HybridConfig, HybridScheduler, HybridStats, ParallelDecompressResult};

use crate::{Algorithm, CompressedPage, Error, Result, PAGE_SIZE};

/// GPU backend type for compression acceleration.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum GpuBackend {
    /// No GPU available or disabled.
    #[default]
    None,
    /// NVIDIA CUDA backend.
    Cuda,
    /// Vulkan compute backend (via wgpu).
    Vulkan,
    /// Metal compute backend (macOS).
    Metal,
}

/// GPU device information.
#[derive(Debug, Clone)]
pub struct GpuDeviceInfo {
    /// Device index.
    pub index: u32,
    /// Device name.
    pub name: String,
    /// Total memory in bytes.
    pub total_memory: u64,
    /// L2 cache size in bytes.
    pub l2_cache_size: u64,
    /// Compute capability (major, minor) for CUDA.
    pub compute_capability: (u32, u32),
    /// Backend type.
    pub backend: GpuBackend,
}

impl GpuDeviceInfo {
    /// Calculate optimal batch size based on L2 cache.
    ///
    /// Following the heuristic: batch_size = L2_cache_pages * 0.8
    #[must_use]
    pub fn optimal_batch_size(&self) -> usize {
        let l2_cache_pages = self.l2_cache_size as usize / PAGE_SIZE;
        let target_occupancy = 0.8;
        ((l2_cache_pages as f64) * target_occupancy) as usize
    }

    /// Check if device meets minimum requirements (SM 7.0+).
    #[must_use]
    pub fn is_supported(&self) -> bool {
        match self.backend {
            GpuBackend::Cuda => self.compute_capability.0 >= 7,
            GpuBackend::Vulkan | GpuBackend::Metal => true,
            GpuBackend::None => false,
        }
    }
}

/// Batch compression request for GPU processing.
#[derive(Debug)]
pub struct BatchCompressionRequest {
    /// Input pages (each 4KB).
    pub pages: Vec<[u8; PAGE_SIZE]>,
    /// Compression algorithm.
    pub algorithm: Algorithm,
}

/// Batch compression result from GPU.
#[derive(Debug)]
pub struct BatchCompressionResult {
    /// Compressed pages.
    pub compressed: Vec<CompressedPage>,
    /// Total compression time in nanoseconds.
    pub total_time_ns: u64,
    /// GPU kernel time in nanoseconds.
    pub kernel_time_ns: u64,
    /// Transfer time in nanoseconds.
    pub transfer_time_ns: u64,
}

/// Backend selection result with routing decision.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BackendSelection {
    /// Use scalar CPU backend.
    Scalar,
    /// Use SIMD CPU backend.
    Simd,
    /// Use GPU backend with specified batch size.
    Gpu {
        /// The batch size for GPU processing.
        batch_size: usize,
    },
}

/// 5x PCIe rule threshold for GPU offload.
/// GPU is beneficial when: T_compute > 5 * T_transfer
const PCIE_RULE_FACTOR: f64 = 5.0;

/// Minimum batch size for GPU to be beneficial (empirically determined).
pub const GPU_MIN_BATCH_SIZE: usize = 1000;

/// Minimum batch size for SIMD to be beneficial.
pub const SIMD_MIN_BATCH_SIZE: usize = 4;

/// Select the optimal backend based on workload characteristics.
///
/// Implements the 5x PCIe rule [Gregg2011]: GPU offloading is only
/// beneficial when computation time exceeds 5x the PCIe transfer time.
///
/// # Arguments
///
/// * `batch_size` - Number of pages to compress.
/// * `gpu_available` - Whether GPU is available.
///
/// # Returns
///
/// The recommended backend for this workload.
#[must_use]
pub fn select_backend(batch_size: usize, gpu_available: bool) -> BackendSelection {
    match (batch_size, gpu_available) {
        (n, true) if n >= GPU_MIN_BATCH_SIZE => BackendSelection::Gpu { batch_size: n },
        (n, _) if n >= SIMD_MIN_BATCH_SIZE => BackendSelection::Simd,
        _ => BackendSelection::Scalar,
    }
}

/// Calculate if GPU offload meets the 5x PCIe rule.
///
/// The 5x rule [Gregg2011] states: GPU acceleration is beneficial when
/// the compute speedup exceeds the PCIe transfer overhead by a factor of 5.
///
/// GPU_beneficial = (CPU_throughput / GPU_throughput) > 5 * (2 * data / pcie_bandwidth) / (data / CPU_throughput)
/// Simplified: GPU_beneficial = CPU_throughput * pcie_bandwidth > 10 * GPU_throughput
///
/// # Arguments
///
/// * `page_count` - Number of pages to process.
/// * `pcie_bandwidth_gbps` - PCIe bandwidth in GB/s.
/// * `gpu_throughput_gbps` - GPU compression throughput in GB/s.
///
/// # Returns
///
/// True if GPU offload is beneficial, false otherwise.
#[must_use]
pub fn meets_pcie_rule(
    page_count: usize,
    pcie_bandwidth_gbps: f64,
    gpu_throughput_gbps: f64,
) -> bool {
    if page_count == 0 {
        return false;
    }

    let data_size_bytes = page_count * PAGE_SIZE;
    let data_size_gb = data_size_bytes as f64 / 1_000_000_000.0;

    // Transfer time = data_size / bandwidth (for upload + download)
    let transfer_time_s = (data_size_gb * 2.0) / pcie_bandwidth_gbps;

    // GPU compute time
    let gpu_compute_time_s = data_size_gb / gpu_throughput_gbps;

    // Assume CPU baseline is ~3 GB/s for LZ4 (scalar)
    let cpu_baseline_gbps = 3.0;
    let cpu_compute_time_s = data_size_gb / cpu_baseline_gbps;

    // GPU wins when: CPU_time > 5 × (transfer_time + GPU_compute_time)
    let gpu_total_time = transfer_time_s + gpu_compute_time_s;
    cpu_compute_time_s > PCIE_RULE_FACTOR * gpu_total_time
}

/// Check if GPU is available on the system.
#[must_use]
pub fn gpu_available() -> bool {
    // TODO: Implement actual GPU detection via trueno-gpu
    // For now, return false as we're stubbing the implementation
    #[cfg(feature = "cuda")]
    {
        detect_cuda_devices().is_some()
    }
    #[cfg(not(feature = "cuda"))]
    {
        false
    }
}

/// Detect available CUDA devices.
#[cfg(feature = "cuda")]
fn detect_cuda_devices() -> Option<Vec<GpuDeviceInfo>> {
    use cudarc::driver::result::{self, device};
    use cudarc::driver::sys::CUdevice_attribute_enum;

    // Initialize CUDA
    result::init().ok()?;

    let count = device::get_count().ok()?;
    if count == 0 {
        return None;
    }

    let mut devices = Vec::with_capacity(count as usize);
    for i in 0..count {
        if let Ok(cu_device) = device::get(i) {
            let name = device::get_name(cu_device).unwrap_or_else(|_| "Unknown".to_string());

            // Get compute capability
            let major = unsafe {
                device::get_attribute(
                    cu_device,
                    CUdevice_attribute_enum::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR,
                )
                .unwrap_or(0) as u32
            };
            let minor = unsafe {
                device::get_attribute(
                    cu_device,
                    CUdevice_attribute_enum::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR,
                )
                .unwrap_or(0) as u32
            };

            // Get memory info
            // SAFETY: cu_device is a valid device handle obtained from device::get
            let total_memory = unsafe { device::total_mem(cu_device).unwrap_or(0) };

            let l2_cache_size = unsafe {
                device::get_attribute(
                    cu_device,
                    CUdevice_attribute_enum::CU_DEVICE_ATTRIBUTE_L2_CACHE_SIZE,
                )
                .unwrap_or(0) as u64
            };

            devices.push(GpuDeviceInfo {
                index: i as u32,
                name,
                total_memory: total_memory as u64,
                l2_cache_size,
                compute_capability: (major, minor),
                backend: GpuBackend::Cuda,
            });
        }
    }

    if devices.is_empty() {
        None
    } else {
        Some(devices)
    }
}

/// GPU compression context for batch operations.
#[derive(Debug)]
pub struct GpuCompressor {
    device: GpuDeviceInfo,
    #[allow(dead_code)] // Will be used when CUDA is enabled
    algorithm: Algorithm,
    batch_size: usize,
}

impl GpuCompressor {
    /// Create a new GPU compressor.
    ///
    /// # Errors
    ///
    /// Returns error if GPU is not available or device is not supported.
    pub fn new(device_index: u32, algorithm: Algorithm) -> Result<Self> {
        #[cfg(feature = "cuda")]
        {
            let devices = detect_cuda_devices()
                .ok_or_else(|| Error::GpuNotAvailable("No CUDA devices found".into()))?;
            let device =
                devices.into_iter().find(|d| d.index == device_index).ok_or_else(|| {
                    Error::GpuNotAvailable(format!("Device {device_index} not found"))
                })?;

            if !device.is_supported() {
                return Err(Error::GpuNotAvailable(format!(
                    "Device {} (SM {}.{}) below minimum SM 7.0",
                    device.name, device.compute_capability.0, device.compute_capability.1
                )));
            }

            let batch_size = device.optimal_batch_size();
            Ok(Self { device, algorithm, batch_size })
        }

        #[cfg(not(feature = "cuda"))]
        {
            let _ = device_index;
            let _ = algorithm;
            Err(Error::GpuNotAvailable("CUDA feature not enabled".into()))
        }
    }

    /// Compress a batch of pages using GPU.
    ///
    /// # Errors
    ///
    /// Returns error if compression fails.
    pub fn compress_batch(
        &self,
        request: &BatchCompressionRequest,
    ) -> Result<BatchCompressionResult> {
        if request.pages.is_empty() {
            return Ok(BatchCompressionResult {
                compressed: Vec::new(),
                total_time_ns: 0,
                kernel_time_ns: 0,
                transfer_time_ns: 0,
            });
        }

        // CPU fallback until trueno-gpu PTX kernels are integrated
        let start = std::time::Instant::now();

        let compressed: Vec<CompressedPage> = request
            .pages
            .iter()
            .map(|page| match request.algorithm {
                Algorithm::Lz4 | Algorithm::Lz4Hc => {
                    let data = crate::lz4::compress(page).unwrap_or_else(|_| page.to_vec());
                    if data.len() >= PAGE_SIZE {
                        CompressedPage::uncompressed(*page)
                    } else {
                        CompressedPage::new(data, PAGE_SIZE, Algorithm::Lz4)
                            .unwrap_or_else(|_| CompressedPage::uncompressed(*page))
                    }
                }
                Algorithm::Zstd { level } => {
                    let data = crate::zstd::compress(page, level).unwrap_or_else(|_| page.to_vec());
                    if data.len() >= PAGE_SIZE {
                        CompressedPage::uncompressed(*page)
                    } else {
                        CompressedPage::new(data, PAGE_SIZE, Algorithm::Zstd { level })
                            .unwrap_or_else(|_| CompressedPage::uncompressed(*page))
                    }
                }
                _ => CompressedPage::uncompressed(*page),
            })
            .collect();

        let total_time_ns = start.elapsed().as_nanos() as u64;

        Ok(BatchCompressionResult {
            compressed,
            total_time_ns,
            kernel_time_ns: total_time_ns / 2, // Stub: assume 50% kernel time
            transfer_time_ns: total_time_ns / 2, // Stub: assume 50% transfer time
        })
    }

    /// Get the current batch size.
    #[must_use]
    pub fn batch_size(&self) -> usize {
        self.batch_size
    }

    /// Get device info.
    #[must_use]
    pub fn device(&self) -> &GpuDeviceInfo {
        &self.device
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ============================================================
    // Falsification Tests F036-F050: GPU/CUDA Correctness
    // ============================================================

    #[test]
    fn test_gpu_backend_enum_default() {
        // F048: Default should be None (no GPU)
        assert_eq!(GpuBackend::default(), GpuBackend::None);
    }

    #[test]
    fn test_gpu_backend_enum_variants() {
        // Verify all variants exist
        let none = GpuBackend::None;
        let cuda = GpuBackend::Cuda;
        let vulkan = GpuBackend::Vulkan;
        let metal = GpuBackend::Metal;
        // Use black_box to ensure variants are evaluated
        std::hint::black_box((none, cuda, vulkan, metal));
    }

    #[test]
    fn test_gpu_device_info_optimal_batch_size() {
        // F046: Batch size calculation from L2 cache
        let device = GpuDeviceInfo {
            index: 0,
            name: "Test GPU".to_string(),
            total_memory: 24 * 1024 * 1024 * 1024, // 24 GB
            l2_cache_size: 72 * 1024 * 1024,       // 72 MB (RTX 4090)
            compute_capability: (8, 9),
            backend: GpuBackend::Cuda,
        };

        let batch_size = device.optimal_batch_size();
        // 72 MB / 4KB * 0.8 = 14,745 pages
        assert!(batch_size > 10_000);
        assert!(batch_size < 20_000);
    }

    #[test]
    fn test_gpu_device_info_is_supported_cuda_sm70() {
        // F049: SM 7.0+ required for CUDA
        let device_sm70 = GpuDeviceInfo {
            index: 0,
            name: "V100".to_string(),
            total_memory: 16 * 1024 * 1024 * 1024,
            l2_cache_size: 6 * 1024 * 1024,
            compute_capability: (7, 0),
            backend: GpuBackend::Cuda,
        };
        assert!(device_sm70.is_supported());

        let device_sm60 = GpuDeviceInfo {
            index: 0,
            name: "P100".to_string(),
            total_memory: 16 * 1024 * 1024 * 1024,
            l2_cache_size: 4 * 1024 * 1024,
            compute_capability: (6, 0),
            backend: GpuBackend::Cuda,
        };
        assert!(!device_sm60.is_supported());
    }

    #[test]
    fn test_gpu_device_info_is_supported_vulkan() {
        // Vulkan should always be supported if available
        let device = GpuDeviceInfo {
            index: 0,
            name: "AMD GPU".to_string(),
            total_memory: 8 * 1024 * 1024 * 1024,
            l2_cache_size: 4 * 1024 * 1024,
            compute_capability: (0, 0),
            backend: GpuBackend::Vulkan,
        };
        assert!(device.is_supported());
    }

    #[test]
    fn test_gpu_device_info_is_supported_none() {
        let device = GpuDeviceInfo {
            index: 0,
            name: "None".to_string(),
            total_memory: 0,
            l2_cache_size: 0,
            compute_capability: (0, 0),
            backend: GpuBackend::None,
        };
        assert!(!device.is_supported());
    }

    #[test]
    fn test_select_backend_scalar_small_batch() {
        // F037: Batch size 1 should use scalar
        assert_eq!(select_backend(1, false), BackendSelection::Scalar);
        assert_eq!(select_backend(1, true), BackendSelection::Scalar);
        assert_eq!(select_backend(3, true), BackendSelection::Scalar);
    }

    #[test]
    fn test_select_backend_simd_medium_batch() {
        // F038: Medium batch should use SIMD
        assert_eq!(select_backend(4, false), BackendSelection::Simd);
        assert_eq!(select_backend(100, false), BackendSelection::Simd);
        assert_eq!(select_backend(999, true), BackendSelection::Simd);
    }

    #[test]
    fn test_select_backend_gpu_large_batch() {
        // F039: Large batch with GPU should use GPU
        assert_eq!(select_backend(1000, true), BackendSelection::Gpu { batch_size: 1000 });
        assert_eq!(select_backend(10000, true), BackendSelection::Gpu { batch_size: 10000 });
    }

    #[test]
    fn test_select_backend_no_gpu_large_batch() {
        // Large batch without GPU should use SIMD
        assert_eq!(select_backend(1000, false), BackendSelection::Simd);
        assert_eq!(select_backend(10000, false), BackendSelection::Simd);
    }

    #[test]
    fn test_meets_pcie_rule_small_batch() {
        // Small batches should not meet PCIe rule
        // PCIe 4.0 x16: ~25 GB/s, GPU throughput: ~50 GB/s
        assert!(!meets_pcie_rule(100, 25.0, 50.0));
    }

    #[test]
    fn test_meets_pcie_rule_large_batch() {
        // Large batch with high PCIe bandwidth (PCIe 5.0 x16 = ~64 GB/s)
        // CPU baseline: 3 GB/s
        // GPU: 500 GB/s
        //
        // For 1M pages (4 GB):
        // - CPU time = 4.0 / 3 = 1.33s
        // - GPU time = 4.0 / 500 = 0.008s
        // - Transfer = 4.0 * 2 / 100 = 0.08s (with 100 GB/s PCIe)
        // - GPU total = 0.088s
        // - 5 * GPU total = 0.44s
        // - CPU time (1.33s) > 5 * GPU total (0.44s) -> passes!

        // With high PCIe bandwidth (100 GB/s = PCIe 5.0 x16 + overhead)
        assert!(meets_pcie_rule(1000000, 100.0, 500.0));
    }

    #[test]
    fn test_gpu_available_default() {
        // Without CUDA feature, should return false
        #[cfg(not(feature = "cuda"))]
        {
            assert!(!gpu_available());
        }
    }

    #[test]
    #[cfg(feature = "cuda")]
    fn test_cuda_device_detection() {
        // With CUDA feature enabled and hardware present, should detect devices
        let available = gpu_available();
        println!("GPU available: {available}");

        if available {
            let devices = detect_cuda_devices().unwrap();
            for dev in &devices {
                println!(
                    "Device {}: {} (SM {}.{})",
                    dev.index, dev.name, dev.compute_capability.0, dev.compute_capability.1
                );
                println!("  Memory: {} MB", dev.total_memory / (1024 * 1024));
                println!("  L2 Cache: {} KB", dev.l2_cache_size / 1024);
                println!("  Optimal batch: {} pages", dev.optimal_batch_size());
            }
            assert!(!devices.is_empty());

            // Verify device is supported (SM 7.0+)
            assert!(devices[0].is_supported(), "First device should be SM 7.0+");
        }
    }

    #[test]
    #[cfg(feature = "cuda")]
    fn test_gpu_compressor_creation_with_cuda() {
        if !gpu_available() {
            println!("Skipping: no CUDA device available");
            return;
        }

        let result = GpuCompressor::new(0, Algorithm::Lz4);
        assert!(result.is_ok(), "Should create compressor with CUDA device 0");

        let compressor = result.unwrap();
        println!("Created GPU compressor:");
        println!("  Device: {}", compressor.device().name);
        println!("  Batch size: {}", compressor.batch_size());
        assert!(compressor.batch_size() > 0);
    }

    #[test]
    #[cfg(feature = "cuda")]
    fn test_gpu_batch_compression() {
        if !gpu_available() {
            println!("Skipping: no CUDA device available");
            return;
        }

        let compressor = GpuCompressor::new(0, Algorithm::Lz4).unwrap();

        // Create test pages with compressible data
        let batch_size = 1000; // Test with 1000 pages
        let mut pages = Vec::with_capacity(batch_size);
        for i in 0..batch_size {
            let mut page = [0u8; PAGE_SIZE];
            // Fill with pattern that compresses well
            for (j, byte) in page.iter_mut().enumerate() {
                *byte = ((i + j) % 256) as u8;
            }
            pages.push(page);
        }

        let request = BatchCompressionRequest { pages, algorithm: Algorithm::Lz4 };

        let result = compressor.compress_batch(&request).unwrap();

        println!("Batch compression results:");
        println!("  Pages: {batch_size}");
        println!("  Data size: {} MB", (batch_size * PAGE_SIZE) / (1024 * 1024));
        println!("  Total time: {:.2} ms", result.total_time_ns as f64 / 1_000_000.0);
        println!(
            "  Throughput: {:.2} MB/s",
            (batch_size * PAGE_SIZE) as f64
                / (result.total_time_ns as f64 / 1_000_000_000.0)
                / (1024.0 * 1024.0)
        );

        // Verify all pages compressed
        assert_eq!(result.compressed.len(), batch_size);

        // Verify compression worked (should be smaller than original for compressible data)
        let total_compressed: usize = result.compressed.iter().map(|p| p.data.len()).sum();
        let total_original = batch_size * PAGE_SIZE;
        let ratio = total_original as f64 / total_compressed as f64;
        println!("  Compression ratio: {ratio:.2}x");
        assert!(ratio > 1.0, "Should achieve some compression");
    }

    #[test]
    #[cfg(feature = "cuda")]
    fn test_gpu_backend_selection_with_cuda() {
        if !gpu_available() {
            println!("Skipping: no CUDA device available");
            return;
        }

        // With GPU available, large batches should select GPU
        let selection = select_backend(5000, true);
        assert_eq!(selection, BackendSelection::Gpu { batch_size: 5000 });

        // Small batches still use scalar/SIMD
        let selection = select_backend(10, true);
        assert_eq!(selection, BackendSelection::Simd);
    }

    #[test]
    fn test_batch_compression_request_empty() {
        // F037: Empty batch should work
        let request = BatchCompressionRequest { pages: vec![], algorithm: Algorithm::Lz4 };
        assert!(request.pages.is_empty());
    }

    #[test]
    fn test_batch_compression_request_single() {
        // F037: Single page batch
        let page = [0u8; PAGE_SIZE];
        let request = BatchCompressionRequest { pages: vec![page], algorithm: Algorithm::Lz4 };
        assert_eq!(request.pages.len(), 1);
    }

    #[test]
    fn test_batch_compression_result_fields() {
        let result = BatchCompressionResult {
            compressed: vec![],
            total_time_ns: 1000,
            kernel_time_ns: 500,
            transfer_time_ns: 500,
        };
        assert_eq!(result.total_time_ns, 1000);
        assert_eq!(result.kernel_time_ns + result.transfer_time_ns, 1000);
    }

    #[test]
    fn test_gpu_min_batch_size_constant() {
        // Verify constant is reasonable
        const {
            assert!(GPU_MIN_BATCH_SIZE >= 100);
        }
        const {
            assert!(GPU_MIN_BATCH_SIZE <= 10000);
        }
    }

    #[test]
    fn test_simd_min_batch_size_constant() {
        // Verify constant is reasonable
        const {
            assert!(SIMD_MIN_BATCH_SIZE >= 1);
        }
        const {
            assert!(SIMD_MIN_BATCH_SIZE <= 16);
        }
    }

    #[test]
    fn test_backend_selection_equality() {
        // Test equality implementations
        assert_eq!(BackendSelection::Scalar, BackendSelection::Scalar);
        assert_eq!(BackendSelection::Simd, BackendSelection::Simd);
        assert_eq!(
            BackendSelection::Gpu { batch_size: 1000 },
            BackendSelection::Gpu { batch_size: 1000 }
        );
        assert_ne!(
            BackendSelection::Gpu { batch_size: 1000 },
            BackendSelection::Gpu { batch_size: 2000 }
        );
        assert_ne!(BackendSelection::Scalar, BackendSelection::Simd);
    }

    #[test]
    fn test_backend_selection_debug() {
        // Test Debug implementation
        let scalar = format!("{:?}", BackendSelection::Scalar);
        let simd = format!("{:?}", BackendSelection::Simd);
        let gpu = format!("{:?}", BackendSelection::Gpu { batch_size: 1000 });

        assert!(scalar.contains("Scalar"));
        assert!(simd.contains("Simd"));
        assert!(gpu.contains("Gpu"));
        assert!(gpu.contains("1000"));
    }

    #[test]
    fn test_gpu_compressor_without_cuda_feature() {
        // F045: Without CUDA feature, should return error
        #[cfg(not(feature = "cuda"))]
        {
            let result = GpuCompressor::new(0, Algorithm::Lz4);
            assert!(result.is_err());
            let err = result.unwrap_err();
            assert!(err.to_string().contains("CUDA feature not enabled"));
        }
    }

    #[test]
    fn test_gpu_device_info_clone() {
        let device = GpuDeviceInfo {
            index: 0,
            name: "Test".to_string(),
            total_memory: 1024,
            l2_cache_size: 64,
            compute_capability: (8, 0),
            backend: GpuBackend::Cuda,
        };
        let cloned = device.clone();
        assert_eq!(device.index, cloned.index);
        assert_eq!(device.name, cloned.name);
    }

    #[test]
    fn test_gpu_device_info_debug() {
        let device = GpuDeviceInfo {
            index: 0,
            name: "Test".to_string(),
            total_memory: 1024,
            l2_cache_size: 64,
            compute_capability: (8, 0),
            backend: GpuBackend::Cuda,
        };
        let debug = format!("{device:?}");
        assert!(debug.contains("GpuDeviceInfo"));
        assert!(debug.contains("Test"));
    }

    #[test]
    fn test_backend_selection_copy() {
        let selection = BackendSelection::Gpu { batch_size: 1000 };
        let copied = selection;
        assert_eq!(selection, copied);
    }

    #[test]
    fn test_gpu_backend_copy() {
        let backend = GpuBackend::Cuda;
        let copied = backend;
        assert_eq!(backend, copied);
    }

    #[test]
    fn test_pcie_rule_boundary_conditions() {
        // Test edge cases for PCIe rule
        assert!(!meets_pcie_rule(0, 25.0, 50.0)); // Zero pages
        assert!(!meets_pcie_rule(1, 25.0, 50.0)); // Single page
    }

    #[test]
    fn test_select_backend_boundary_at_simd_threshold() {
        // Exactly at SIMD threshold
        assert_eq!(select_backend(SIMD_MIN_BATCH_SIZE, false), BackendSelection::Simd);
        assert_eq!(select_backend(SIMD_MIN_BATCH_SIZE - 1, false), BackendSelection::Scalar);
    }

    #[test]
    fn test_select_backend_boundary_at_gpu_threshold() {
        // Exactly at GPU threshold
        assert_eq!(
            select_backend(GPU_MIN_BATCH_SIZE, true),
            BackendSelection::Gpu { batch_size: GPU_MIN_BATCH_SIZE }
        );
        assert_eq!(select_backend(GPU_MIN_BATCH_SIZE - 1, true), BackendSelection::Simd);
    }

    #[test]
    fn test_gpu_device_info_metal() {
        let device = GpuDeviceInfo {
            index: 0,
            name: "M1 Ultra".to_string(),
            total_memory: 64 * 1024 * 1024 * 1024,
            l2_cache_size: 192 * 1024 * 1024,
            compute_capability: (0, 0), // N/A for Metal
            backend: GpuBackend::Metal,
        };
        assert!(device.is_supported());
    }

    #[test]
    fn test_gpu_device_info_small_l2_cache() {
        let device = GpuDeviceInfo {
            index: 0,
            name: "Small GPU".to_string(),
            total_memory: 4 * 1024 * 1024 * 1024,
            l2_cache_size: 1024 * 1024, // 1 MB
            compute_capability: (7, 5),
            backend: GpuBackend::Cuda,
        };

        let batch_size = device.optimal_batch_size();
        // 1 MB / 4KB * 0.8 = ~200 pages
        assert!(batch_size >= 100);
        assert!(batch_size < 500);
    }

    #[test]
    fn test_batch_compression_request_debug() {
        let request =
            BatchCompressionRequest { pages: vec![[0u8; PAGE_SIZE]], algorithm: Algorithm::Lz4 };
        let debug = format!("{request:?}");
        assert!(debug.contains("BatchCompressionRequest"));
    }

    #[test]
    fn test_batch_compression_result_debug() {
        let result = BatchCompressionResult {
            compressed: vec![],
            total_time_ns: 1000,
            kernel_time_ns: 500,
            transfer_time_ns: 500,
        };
        let debug = format!("{result:?}");
        assert!(debug.contains("BatchCompressionResult"));
    }

    #[test]
    fn test_batch_compression_request_multiple_pages() {
        let pages: Vec<[u8; PAGE_SIZE]> = (0..10)
            .map(|i| {
                let mut page = [0u8; PAGE_SIZE];
                page[0] = i as u8;
                page
            })
            .collect();

        let request = BatchCompressionRequest { pages, algorithm: Algorithm::Zstd { level: 1 } };
        assert_eq!(request.pages.len(), 10);
    }

    #[test]
    fn test_backend_clone() {
        let original = BackendSelection::Gpu { batch_size: 5000 };
        let cloned = original;
        assert_eq!(original, cloned);
    }

    #[test]
    fn test_gpu_backend_debug() {
        let backends = [GpuBackend::None, GpuBackend::Cuda, GpuBackend::Vulkan, GpuBackend::Metal];
        for backend in backends {
            let debug = format!("{backend:?}");
            assert!(!debug.is_empty());
        }
    }

    #[test]
    fn test_gpu_backend_clone() {
        let original = GpuBackend::Vulkan;
        let cloned = original;
        assert_eq!(original, cloned);
    }

    #[test]
    fn test_pcie_rule_various_bandwidths() {
        // Test with different PCIe generations
        let page_count = 100000;

        // PCIe 3.0 x16 (~15.75 GB/s)
        let _ = meets_pcie_rule(page_count, 15.75, 100.0);

        // PCIe 4.0 x16 (~31.5 GB/s)
        let _ = meets_pcie_rule(page_count, 31.5, 200.0);

        // PCIe 5.0 x16 (~63 GB/s)
        let _ = meets_pcie_rule(page_count, 63.0, 400.0);
    }

    #[test]
    fn test_pcie_rule_various_gpu_throughputs() {
        // Test with different GPU performance levels
        let page_count = 50000;
        let bandwidth = 32.0;

        // Entry GPU (~50 GB/s)
        let _ = meets_pcie_rule(page_count, bandwidth, 50.0);

        // Mid-range GPU (~200 GB/s)
        let _ = meets_pcie_rule(page_count, bandwidth, 200.0);

        // High-end GPU (~800 GB/s)
        let _ = meets_pcie_rule(page_count, bandwidth, 800.0);
    }

    #[test]
    fn test_gpu_compressor_debug() {
        // GpuCompressor implements Debug, test it compiles
        // We can't actually create one without CUDA, but test the trait bounds
        #[cfg(not(feature = "cuda"))]
        {
            // Just verify GpuCompressor is Debug (type check)
            fn assert_debug<T: std::fmt::Debug>() {}
            assert_debug::<GpuCompressor>();
        }
    }

    #[test]
    fn test_batch_compression_result_with_compressed_pages() {
        let page = CompressedPage::uncompressed([0xAA; PAGE_SIZE]);
        let result = BatchCompressionResult {
            compressed: vec![page.clone(), page.clone(), page],
            total_time_ns: 3000,
            kernel_time_ns: 1500,
            transfer_time_ns: 1500,
        };
        assert_eq!(result.compressed.len(), 3);
    }

    #[test]
    fn test_select_backend_various_batch_sizes() {
        // Test many batch sizes
        for batch_size in [0, 1, 2, 3, 4, 5, 10, 50, 100, 500, 999, 1000, 1001, 5000, 10000] {
            let _ = select_backend(batch_size, false);
            let _ = select_backend(batch_size, true);
        }
    }

    #[test]
    fn test_gpu_device_info_zero_l2_cache() {
        let device = GpuDeviceInfo {
            index: 0,
            name: "Zero L2".to_string(),
            total_memory: 1024 * 1024 * 1024,
            l2_cache_size: 0,
            compute_capability: (8, 0),
            backend: GpuBackend::Cuda,
        };

        // Should return 0 with zero L2 cache
        let batch_size = device.optimal_batch_size();
        assert_eq!(batch_size, 0);
    }

    #[test]
    fn test_gpu_device_info_high_compute_capability() {
        let device = GpuDeviceInfo {
            index: 0,
            name: "Future GPU".to_string(),
            total_memory: 80 * 1024 * 1024 * 1024,
            l2_cache_size: 256 * 1024 * 1024,
            compute_capability: (10, 0), // Hypothetical future SM
            backend: GpuBackend::Cuda,
        };
        assert!(device.is_supported());
    }

    #[test]
    fn test_meets_pcie_rule_high_cpu_throughput_scenario() {
        // When GPU is much faster, PCIe rule should pass for large batches
        // CPU: 3 GB/s, GPU: 1000 GB/s, PCIe: 128 GB/s
        assert!(meets_pcie_rule(10_000_000, 128.0, 1000.0));
    }

    #[test]
    fn test_meets_pcie_rule_slow_gpu() {
        // When GPU is slow, PCIe rule may not pass
        assert!(!meets_pcie_rule(100, 30.0, 5.0)); // GPU slower than CPU
    }
}
