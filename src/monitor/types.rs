//! GPU vendor, backend, and device info types (TRUENO-SPEC-010 Section 3)

#[cfg(all(feature = "gpu", not(target_arch = "wasm32")))]
use super::backends::{enumerate_wgpu_devices, query_wgpu_device_info};
#[cfg(all(feature = "gpu", not(target_arch = "wasm32")))]
use super::MonitorError;

// ============================================================================
// GPU Vendor Identification (TRUENO-SPEC-010 Section 3.2)
// ============================================================================

/// GPU vendor identifier based on PCI vendor ID
///
/// Vendor IDs from PCI-SIG registry:
/// - NVIDIA: 0x10de
/// - AMD: 0x1002
/// - Intel: 0x8086
/// - Apple: 0x106b
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum GpuVendor {
    /// NVIDIA Corporation (0x10de)
    Nvidia,
    /// Advanced Micro Devices (0x1002)
    Amd,
    /// Intel Corporation (0x8086)
    Intel,
    /// Apple Inc. (0x106b)
    Apple,
    /// Unknown vendor with raw PCI vendor ID
    Unknown(u32),
}

impl GpuVendor {
    /// Create vendor from PCI vendor ID
    #[must_use]
    pub const fn from_vendor_id(id: u32) -> Self {
        match id {
            0x10de => Self::Nvidia,
            0x1002 => Self::Amd,
            0x8086 => Self::Intel,
            0x106b => Self::Apple,
            other => Self::Unknown(other),
        }
    }

    /// Get the display name for the vendor
    #[must_use]
    pub const fn name(&self) -> &'static str {
        match self {
            Self::Nvidia => "NVIDIA",
            Self::Amd => "AMD",
            Self::Intel => "Intel",
            Self::Apple => "Apple",
            Self::Unknown(_) => "Unknown",
        }
    }

    /// Check if this is an NVIDIA GPU (supports CUDA)
    #[must_use]
    pub const fn is_nvidia(&self) -> bool {
        matches!(self, Self::Nvidia)
    }
}

impl std::fmt::Display for GpuVendor {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Unknown(id) => write!(f, "Unknown (0x{id:04x})"),
            Self::Nvidia | Self::Amd | Self::Intel | Self::Apple => write!(f, "{}", self.name()),
        }
    }
}

// ============================================================================
// GPU Backend Selection (TRUENO-SPEC-010 Section 3.2)
// ============================================================================

/// GPU compute backend
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum GpuBackend {
    /// Vulkan (Linux, Windows, Android)
    Vulkan,
    /// Metal (macOS, iOS)
    Metal,
    /// DirectX 12 (Windows)
    Dx12,
    /// DirectX 11 (Windows, fallback)
    Dx11,
    /// WebGPU (WASM browser)
    WebGpu,
    /// Native CUDA (NVIDIA only, via trueno-gpu)
    Cuda,
    /// OpenGL (fallback)
    OpenGl,
    /// CPU fallback (no GPU)
    Cpu,
}

impl GpuBackend {
    /// Get the display name for the backend
    #[must_use]
    pub const fn name(&self) -> &'static str {
        match self {
            Self::Vulkan => "Vulkan",
            Self::Metal => "Metal",
            Self::Dx12 => "DirectX 12",
            Self::Dx11 => "DirectX 11",
            Self::WebGpu => "WebGPU",
            Self::Cuda => "CUDA",
            Self::OpenGl => "OpenGL",
            Self::Cpu => "CPU",
        }
    }

    /// Check if this is a GPU backend (not CPU fallback)
    #[must_use]
    pub const fn is_gpu(&self) -> bool {
        !matches!(self, Self::Cpu)
    }

    /// Check if this backend supports compute shaders
    #[must_use]
    pub const fn supports_compute(&self) -> bool {
        matches!(self, Self::Vulkan | Self::Metal | Self::Dx12 | Self::WebGpu | Self::Cuda)
    }
}

impl std::fmt::Display for GpuBackend {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.name())
    }
}

// ============================================================================
// GPU Device Information (TRUENO-SPEC-010 Section 3.1)
// ============================================================================

/// GPU device information (TRUENO-SPEC-010)
///
/// Contains static device properties that don't change during runtime.
#[derive(Debug, Clone)]
pub struct GpuDeviceInfo {
    /// Device index (0-based)
    pub index: u32,
    /// Device name (e.g., "NVIDIA GeForce RTX 4090")
    pub name: String,
    /// Vendor identifier
    pub vendor: GpuVendor,
    /// Total VRAM in bytes
    pub vram_total: u64,
    /// Compute capability (NVIDIA) or architecture info as (major, minor)
    pub compute_capability: Option<(u32, u32)>,
    /// Driver version string
    pub driver_version: Option<String>,
    /// PCI bus ID (e.g., "0000:01:00.0")
    pub pci_bus_id: Option<String>,
    /// wgpu/CUDA backend being used
    pub backend: GpuBackend,
}

impl GpuDeviceInfo {
    /// Create a new device info with required fields
    #[must_use]
    pub fn new(
        index: u32,
        name: impl Into<String>,
        vendor: GpuVendor,
        backend: GpuBackend,
    ) -> Self {
        Self {
            index,
            name: name.into(),
            vendor,
            vram_total: 0,
            compute_capability: None,
            driver_version: None,
            pci_bus_id: None,
            backend,
        }
    }

    /// Set VRAM total
    #[must_use]
    pub fn with_vram(mut self, bytes: u64) -> Self {
        self.vram_total = bytes;
        self
    }

    /// Set compute capability
    #[must_use]
    pub fn with_compute_capability(mut self, major: u32, minor: u32) -> Self {
        self.compute_capability = Some((major, minor));
        self
    }

    /// Set driver version
    #[must_use]
    pub fn with_driver_version(mut self, version: impl Into<String>) -> Self {
        self.driver_version = Some(version.into());
        self
    }

    /// Set PCI bus ID
    #[must_use]
    pub fn with_pci_bus_id(mut self, bus_id: impl Into<String>) -> Self {
        self.pci_bus_id = Some(bus_id.into());
        self
    }

    /// Query device info via wgpu (cross-platform, native only)
    ///
    /// On WASM, use async methods with `wasm_bindgen_futures`.
    ///
    /// # Errors
    ///
    /// Returns error if no GPU is available or query fails.
    #[cfg(all(feature = "gpu", not(target_arch = "wasm32")))]
    pub fn query() -> Result<Self, MonitorError> {
        query_wgpu_device_info(0)
    }

    /// Query device info via wgpu for a specific device index (native only)
    ///
    /// On WASM, use async methods with `wasm_bindgen_futures`.
    ///
    /// # Errors
    ///
    /// Returns error if device index is invalid or query fails.
    #[cfg(all(feature = "gpu", not(target_arch = "wasm32")))]
    pub fn query_device(index: u32) -> Result<Self, MonitorError> {
        query_wgpu_device_info(index)
    }

    /// Enumerate all available GPU devices (native only)
    ///
    /// On WASM, use async methods with `wasm_bindgen_futures`.
    ///
    /// # Errors
    ///
    /// Returns error if enumeration fails.
    #[cfg(all(feature = "gpu", not(target_arch = "wasm32")))]
    pub fn enumerate() -> Result<Vec<Self>, MonitorError> {
        enumerate_wgpu_devices()
    }

    /// Get VRAM in megabytes (convenience method)
    #[must_use]
    pub fn vram_mb(&self) -> u64 {
        self.vram_total / (1024 * 1024)
    }

    /// Get VRAM in gigabytes (convenience method)
    #[must_use]
    pub fn vram_gb(&self) -> f64 {
        self.vram_total as f64 / (1024.0 * 1024.0 * 1024.0)
    }

    /// Check if device supports CUDA (is NVIDIA)
    #[must_use]
    pub fn supports_cuda(&self) -> bool {
        self.vendor.is_nvidia()
    }
}

impl std::fmt::Display for GpuDeviceInfo {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "[{}] {} ({}) - {:.1} GB VRAM",
            self.index,
            self.name,
            self.backend,
            self.vram_gb()
        )
    }
}
