//! GPU Process Analyzer
//!
//! Queries GPU process VRAM usage from nvidia-smi, rocm-smi (AMD), or Apple IOKit.
//! Falls back gracefully if no GPU is available.
//!
//! ## Platform Support
//!
//! - **NVIDIA**: Uses nvidia-smi for detailed GPU/process stats
//! - **AMD**: Uses rocm-smi (PMAT-GAP-029) with sysfs fallback
//! - **Apple**: Uses IOKit/Metal for Apple Silicon GPUs (PMAT-GAP-030)
//! - **Intel**: Basic sysfs support

#![allow(clippy::uninlined_format_args)]

use std::path::Path;
use std::process::Command;
use std::time::Duration;

use super::{Analyzer, AnalyzerError};

/// GPU vendor type
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GpuVendor {
    Nvidia,
    Amd,
    Intel,
    Apple,
    Unknown,
}

impl GpuVendor {
    /// Get display name
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Nvidia => "NVIDIA",
            Self::Amd => "AMD",
            Self::Intel => "Intel",
            Self::Apple => "Apple",
            Self::Unknown => "Unknown",
        }
    }
}

/// GPU information
#[derive(Debug, Clone)]
pub struct GpuInfo {
    /// GPU index
    pub index: u32,
    /// GPU name
    pub name: String,
    /// GPU vendor
    pub vendor: GpuVendor,
    /// Total VRAM in bytes
    pub total_memory: u64,
    /// Used VRAM in bytes
    pub used_memory: u64,
    /// Free VRAM in bytes
    pub free_memory: u64,
    /// GPU utilization percentage (0-100)
    pub utilization: f32,
    /// Memory utilization percentage (0-100)
    pub memory_utilization: f32,
    /// Temperature in Celsius
    pub temperature: Option<f32>,
    /// Power draw in watts
    pub power_draw: Option<f32>,
    /// Power limit in watts
    pub power_limit: Option<f32>,
    /// Fan speed percentage
    pub fan_speed: Option<u32>,
    /// Driver version
    pub driver_version: Option<String>,
}

impl GpuInfo {
    /// Memory usage percentage
    pub fn memory_percent(&self) -> f32 {
        if self.total_memory > 0 {
            (self.used_memory as f64 / self.total_memory as f64 * 100.0) as f32
        } else {
            0.0
        }
    }

    /// Format memory for display
    pub fn display_memory(&self) -> String {
        format!(
            "{}/{} MB",
            self.used_memory / (1024 * 1024),
            self.total_memory / (1024 * 1024)
        )
    }
}

/// GPU process type (PMAT-GAP-041 - ttop parity)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum GpuProcType {
    /// Compute process (CUDA/OpenCL kernels)
    Compute,
    /// Graphics process (OpenGL/Vulkan contexts)
    Graphics,
    /// Unknown process type
    #[default]
    Unknown,
}

impl GpuProcType {
    /// Parse from pmon output character
    pub fn from_pmon(s: &str) -> Self {
        match s.trim() {
            "C" => Self::Compute,
            "G" => Self::Graphics,
            _ => Self::Unknown,
        }
    }

    /// Get display character
    pub fn as_char(&self) -> char {
        match self {
            Self::Compute => 'C',
            Self::Graphics => 'G',
            Self::Unknown => '?',
        }
    }

    /// Get display string
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Compute => "Compute",
            Self::Graphics => "Graphics",
            Self::Unknown => "Unknown",
        }
    }
}

impl std::fmt::Display for GpuProcType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.as_char())
    }
}

/// A process using GPU resources (PMAT-GAP-037-042 - ttop pmon parity)
#[derive(Debug, Clone)]
pub struct GpuProcess {
    /// Process ID
    pub pid: u32,
    /// Process name / command
    pub name: String,
    /// GPU index
    pub gpu_index: u32,
    /// Used VRAM in bytes
    pub used_memory: u64,
    /// Process type: Compute or Graphics (PMAT-GAP-041)
    pub proc_type: GpuProcType,
    /// SM (shader) utilization percentage 0-100 (PMAT-GAP-038)
    pub sm_util: u8,
    /// Memory utilization percentage 0-100
    pub mem_util: u8,
    /// Encoder (NVENC) utilization percentage 0-100 (PMAT-GAP-039)
    pub enc_util: u8,
    /// Decoder (NVDEC) utilization percentage 0-100 (PMAT-GAP-040)
    pub dec_util: u8,
}

impl Default for GpuProcess {
    fn default() -> Self {
        Self {
            pid: 0,
            name: String::new(),
            gpu_index: 0,
            used_memory: 0,
            proc_type: GpuProcType::Unknown,
            sm_util: 0,
            mem_util: 0,
            enc_util: 0,
            dec_util: 0,
        }
    }
}

impl GpuProcess {
    /// Format memory for display
    pub fn display_memory(&self) -> String {
        let mb = self.used_memory / (1024 * 1024);
        format!("{} MB", mb)
    }

    /// Get GPU utilization as Option<f32> for compatibility
    pub fn gpu_util(&self) -> Option<f32> {
        if self.sm_util > 0 {
            Some(self.sm_util as f32)
        } else {
            None
        }
    }
}

/// GPU processes data
#[derive(Debug, Clone, Default)]
pub struct GpuProcsData {
    /// GPU information for each GPU
    pub gpus: Vec<GpuInfo>,
    /// Processes using GPUs
    pub processes: Vec<GpuProcess>,
    /// Detected vendor
    pub vendor: Option<GpuVendor>,
    /// Total VRAM usage across all GPUs
    pub total_vram_used: u64,
    /// Total VRAM available across all GPUs
    pub total_vram: u64,
    /// Average GPU utilization
    pub avg_gpu_util: f32,
    /// Highest temperature across GPUs
    pub max_temperature: Option<f32>,
    /// Total power draw across GPUs
    pub total_power: Option<f32>,
}

impl GpuProcsData {
    /// Get total number of GPU processes
    pub fn process_count(&self) -> usize {
        self.processes.len()
    }

    /// Get total number of GPUs
    pub fn gpu_count(&self) -> usize {
        self.gpus.len()
    }

    /// Check if any GPU is available
    pub fn has_gpu(&self) -> bool {
        !self.gpus.is_empty()
    }
}

/// Analyzer for GPU process statistics
pub struct GpuProcsAnalyzer {
    data: GpuProcsData,
    interval: Duration,
    vendor: Option<GpuVendor>,
    nvidia_smi_path: Option<String>,
    /// Path to rocm-smi for AMD GPU stats (PMAT-GAP-029)
    rocm_smi_path: Option<String>,
    /// Last collection time — skip until interval elapses
    last_collect: Option<std::time::Instant>,
}

impl Default for GpuProcsAnalyzer {
    fn default() -> Self {
        Self::new()
    }
}

impl GpuProcsAnalyzer {
    /// Create a new GPU processes analyzer
    pub fn new() -> Self {
        // Detect available GPU
        let (vendor, nvidia_smi_path, rocm_smi_path) = Self::detect_gpu();

        Self {
            data: GpuProcsData::default(),
            interval: Duration::from_secs(2),
            vendor,
            nvidia_smi_path,
            rocm_smi_path,
            last_collect: None,
        }
    }

    /// Get the current GPU data
    pub fn data(&self) -> &GpuProcsData {
        &self.data
    }

    /// Detect available GPU and tools
    ///
    /// Uses same detection approach as ttop - directly invoke nvidia-smi
    /// rather than searching PATH with `which`.
    /// Enhanced for AMD with rocm-smi detection (PMAT-GAP-029).
    /// Enhanced for Apple Silicon GPU detection (PMAT-GAP-030).
    fn detect_gpu() -> (Option<GpuVendor>, Option<String>, Option<String>) {
        // Check for NVIDIA GPU by directly running nvidia-smi --version
        // This is more reliable than using `which` (matches ttop behavior)
        if let Ok(output) = Command::new("nvidia-smi").arg("--version").output() {
            if output.status.success() {
                // nvidia-smi is available, use it directly from PATH
                return (
                    Some(GpuVendor::Nvidia),
                    Some("nvidia-smi".to_string()),
                    None,
                );
            }
        }

        // Check for AMD GPU with rocm-smi (PMAT-GAP-029)
        let rocm_smi_path = Self::detect_rocm_smi();

        // Check for AMD GPU via sysfs - scan all cards, not just card0
        if let Ok(entries) = std::fs::read_dir("/sys/class/drm") {
            for entry in entries.flatten() {
                let name = entry.file_name();
                let name_str = name.to_string_lossy();
                if name_str.starts_with("card") && !name_str.contains('-') {
                    let vendor_path = entry.path().join("device/vendor");
                    if let Ok(vendor) = std::fs::read_to_string(&vendor_path) {
                        if vendor.trim() == "0x1002" {
                            return (Some(GpuVendor::Amd), None, rocm_smi_path);
                        }
                    }
                }
            }
        }

        // Check for Intel GPU
        if Path::new("/sys/class/drm/card0/gt/gt0").exists() {
            return (Some(GpuVendor::Intel), None, None);
        }

        // Check for Apple GPU (PMAT-GAP-030)
        #[cfg(target_os = "macos")]
        {
            if Self::detect_apple_gpu() {
                return (Some(GpuVendor::Apple), None, None);
            }
        }

        (None, None, None)
    }

    /// Detect rocm-smi availability (PMAT-GAP-029 - ttop parity)
    fn detect_rocm_smi() -> Option<String> {
        // Try rocm-smi directly
        if let Ok(output) = Command::new("rocm-smi").arg("--version").output() {
            if output.status.success() {
                return Some("rocm-smi".to_string());
            }
        }
        // Try common ROCm installation paths
        let common_paths = [
            "/opt/rocm/bin/rocm-smi",
            "/usr/bin/rocm-smi",
            "/usr/local/bin/rocm-smi",
        ];
        for path in common_paths {
            if Path::new(path).exists() {
                if let Ok(output) = Command::new(path).arg("--version").output() {
                    if output.status.success() {
                        return Some(path.to_string());
                    }
                }
            }
        }
        None
    }

    /// Detect Apple GPU (PMAT-GAP-030 - ttop parity)
    #[cfg(target_os = "macos")]
    fn detect_apple_gpu() -> bool {
        // Check for Apple Silicon by looking at CPU brand
        if let Ok(output) = Command::new("sysctl")
            .args(["-n", "machdep.cpu.brand_string"])
            .output()
        {
            if output.status.success() {
                let brand = String::from_utf8_lossy(&output.stdout);
                return brand.contains("Apple");
            }
        }
        false
    }

    /// Query NVIDIA GPU using nvidia-smi
    fn query_nvidia(&mut self) -> Result<(), AnalyzerError> {
        let nvidia_smi = self
            .nvidia_smi_path
            .as_ref()
            .ok_or_else(|| AnalyzerError::NotAvailable("nvidia-smi not found".to_string()))?;

        // Query GPU info
        let output = Command::new(nvidia_smi)
            .args([
                "--query-gpu=index,name,memory.total,memory.used,memory.free,utilization.gpu,utilization.memory,temperature.gpu,power.draw,power.limit,fan.speed,driver_version",
                "--format=csv,noheader,nounits",
            ])
            .output()
            .map_err(|e| AnalyzerError::IoError(format!("nvidia-smi failed: {}", e)))?;

        if !output.status.success() {
            return Err(AnalyzerError::IoError(
                "nvidia-smi returned error".to_string(),
            ));
        }

        let stdout = String::from_utf8_lossy(&output.stdout);
        let mut gpus = Vec::new();

        for line in stdout.lines() {
            if let Some(gpu) = self.parse_nvidia_gpu_line(line) {
                gpus.push(gpu);
            }
        }

        // Query per-process GPU metrics using pmon (PMAT-GAP-037 - ttop parity)
        // pmon provides SM/enc/dec utilization per process, which --query-compute-apps lacks
        let mut processes = self.query_nvidia_pmon(nvidia_smi);

        // Sort by SM utilization descending (PMAT-GAP-042)
        processes.sort_by(|a, b| b.sm_util.cmp(&a.sm_util));

        // Calculate aggregates
        let total_vram = gpus.iter().map(|g| g.total_memory).sum();
        let total_vram_used = gpus.iter().map(|g| g.used_memory).sum();
        let avg_gpu_util = if gpus.is_empty() {
            0.0
        } else {
            gpus.iter().map(|g| g.utilization).sum::<f32>() / gpus.len() as f32
        };
        let max_temperature = gpus.iter().filter_map(|g| g.temperature).reduce(f32::max);
        let total_power = {
            let powers: Vec<f32> = gpus.iter().filter_map(|g| g.power_draw).collect();
            if powers.is_empty() {
                None
            } else {
                Some(powers.iter().sum())
            }
        };

        self.data = GpuProcsData {
            gpus,
            processes,
            vendor: Some(GpuVendor::Nvidia),
            total_vram_used,
            total_vram,
            avg_gpu_util,
            max_temperature,
            total_power,
        };

        Ok(())
    }

    /// Parse a line of nvidia-smi GPU output
    fn parse_nvidia_gpu_line(&self, line: &str) -> Option<GpuInfo> {
        let parts: Vec<&str> = line.split(", ").collect();
        if parts.len() < 12 {
            return None;
        }

        let index: u32 = parts[0].parse().ok()?;
        let name = parts[1].to_string();
        let total_memory: u64 = parts[2].parse::<u64>().ok()? * 1024 * 1024; // MiB to bytes
        let used_memory: u64 = parts[3].parse::<u64>().ok()? * 1024 * 1024;
        let free_memory: u64 = parts[4].parse::<u64>().ok()? * 1024 * 1024;
        let utilization: f32 = parts[5].parse().unwrap_or(0.0);
        let memory_utilization: f32 = parts[6].parse().unwrap_or(0.0);
        let temperature: Option<f32> = parts[7].parse().ok();
        let power_draw: Option<f32> = parts[8].parse().ok();
        let power_limit: Option<f32> = parts[9].parse().ok();
        let fan_speed: Option<u32> = parts[10].parse().ok();
        let driver_version = if parts[11].is_empty() || parts[11] == "[N/A]" {
            None
        } else {
            Some(parts[11].to_string())
        };

        Some(GpuInfo {
            index,
            name,
            vendor: GpuVendor::Nvidia,
            total_memory,
            used_memory,
            free_memory,
            utilization,
            memory_utilization,
            temperature,
            power_draw,
            power_limit,
            fan_speed,
            driver_version,
        })
    }

    /// Query NVIDIA GPU processes using pmon (PMAT-GAP-037 - ttop parity)
    ///
    /// pmon provides per-process SM/enc/dec utilization which --query-compute-apps lacks.
    /// Format: gpu pid type sm mem enc dec jpg ofa command
    fn query_nvidia_pmon(&self, nvidia_smi: &str) -> Vec<GpuProcess> {
        // Run nvidia-smi pmon with single sample (-c 1)
        let output = match Command::new(nvidia_smi).args(["pmon", "-c", "1"]).output() {
            Ok(o) if o.status.success() => o,
            _ => return Vec::new(),
        };

        let stdout = String::from_utf8_lossy(&output.stdout);
        Self::parse_pmon_output(&stdout)
    }

    /// Parse nvidia-smi pmon output (PMAT-GAP-037)
    ///
    /// Format: gpu pid type sm mem enc dec jpg ofa command
    /// Header lines start with '#'
    pub fn parse_pmon_output(output: &str) -> Vec<GpuProcess> {
        let mut processes = Vec::new();

        for line in output.lines() {
            // Skip header lines (start with #) and empty lines
            if line.starts_with('#') || line.trim().is_empty() {
                continue;
            }

            let parts: Vec<&str> = line.split_whitespace().collect();
            if parts.len() < 10 {
                continue;
            }

            // Parse fields: gpu pid type sm mem enc dec jpg ofa command
            let gpu_index = match parts[0].parse::<u32>() {
                Ok(v) => v,
                Err(_) => continue,
            };

            let pid = match parts[1].parse::<u32>() {
                Ok(v) => v,
                Err(_) => continue,
            };

            let proc_type = GpuProcType::from_pmon(parts[2]);

            // SM and mem utilization (may be "-" if not available)
            let sm_util = parts[3].parse::<u8>().unwrap_or(0);
            let mem_util = parts[4].parse::<u8>().unwrap_or(0);
            // Encoder and decoder utilization (may be "-")
            let enc_util = parts[5].parse::<u8>().unwrap_or(0);
            let dec_util = parts[6].parse::<u8>().unwrap_or(0);

            // Command is the last field (index 9)
            let name = parts[9].to_string();

            processes.push(GpuProcess {
                pid,
                name,
                gpu_index,
                used_memory: 0, // pmon doesn't provide memory bytes
                proc_type,
                sm_util,
                mem_util,
                enc_util,
                dec_util,
            });
        }

        processes
    }

    /// Query AMD GPU using rocm-smi (PMAT-GAP-029 - ttop parity)
    ///
    /// rocm-smi provides more detailed AMD GPU stats including per-process memory,
    /// similar to nvidia-smi for NVIDIA GPUs.
    fn query_amd_rocm_smi(&mut self) -> Result<(), AnalyzerError> {
        let rocm_smi = self
            .rocm_smi_path
            .as_ref()
            .ok_or_else(|| AnalyzerError::NotAvailable("rocm-smi not found".to_string()))?;

        // Query GPU info using rocm-smi
        let output = Command::new(rocm_smi)
            .args([
                "--showid",
                "--showtemp",
                "--showuse",
                "--showmemuse",
                "--showpower",
                "--showfan",
                "--json",
            ])
            .output()
            .map_err(|e| AnalyzerError::IoError(format!("rocm-smi failed: {}", e)))?;

        if !output.status.success() {
            // Fall back to sysfs if rocm-smi fails
            return self.query_amd_sysfs();
        }

        let stdout = String::from_utf8_lossy(&output.stdout);
        let mut gpus = Vec::new();

        // Parse JSON output
        if let Ok(json) = serde_json::from_str::<serde_json::Value>(&stdout) {
            if let Some(card_obj) = json.as_object() {
                for (card_name, card_info) in card_obj {
                    if !card_name.starts_with("card") {
                        continue;
                    }

                    let index: u32 = card_name
                        .strip_prefix("card")
                        .and_then(|s| s.parse().ok())
                        .unwrap_or(0);

                    let get_f32 = |key: &str| -> f32 {
                        card_info
                            .get(key)
                            .and_then(|v| v.as_str())
                            .and_then(|s| {
                                s.trim_end_matches('%')
                                    .trim_end_matches('W')
                                    .trim_end_matches('C')
                                    .parse()
                                    .ok()
                            })
                            .unwrap_or(0.0)
                    };

                    let get_u64 = |key: &str| -> u64 {
                        card_info
                            .get(key)
                            .and_then(|v| v.as_str())
                            .and_then(|s| s.parse().ok())
                            .unwrap_or(0)
                    };

                    let gpu_name = card_info
                        .get("Card series")
                        .and_then(|v| v.as_str())
                        .unwrap_or("AMD GPU")
                        .to_string();

                    let utilization = get_f32("GPU use (%)");
                    let temperature = card_info
                        .get("Temperature (Sensor junction) (C)")
                        .or_else(|| card_info.get("Temperature (Sensor edge) (C)"))
                        .and_then(|v| v.as_str())
                        .and_then(|s| s.trim_end_matches('C').trim().parse().ok());
                    let power_draw = card_info
                        .get("Average Graphics Package Power (W)")
                        .and_then(|v| v.as_str())
                        .and_then(|s| s.trim_end_matches('W').trim().parse().ok());
                    let fan_speed = card_info
                        .get("Fan speed (%)")
                        .and_then(|v| v.as_str())
                        .and_then(|s| s.trim_end_matches('%').trim().parse().ok());

                    // Memory info
                    let total_memory = get_u64("VRAM Total Memory (B)");
                    let used_memory = get_u64("VRAM Total Used Memory (B)");
                    let free_memory = total_memory.saturating_sub(used_memory);
                    let memory_utilization = if total_memory > 0 {
                        (used_memory as f64 / total_memory as f64 * 100.0) as f32
                    } else {
                        0.0
                    };

                    gpus.push(GpuInfo {
                        index,
                        name: gpu_name,
                        vendor: GpuVendor::Amd,
                        total_memory,
                        used_memory,
                        free_memory,
                        utilization,
                        memory_utilization,
                        temperature,
                        power_draw,
                        power_limit: None,
                        fan_speed,
                        driver_version: None,
                    });
                }
            }
        }

        // If JSON parsing failed or no GPUs found, fall back to sysfs
        if gpus.is_empty() {
            return self.query_amd_sysfs();
        }

        // Query per-process GPU memory (PMAT-GAP-029)
        let mut processes = Vec::new();
        if let Ok(proc_output) = Command::new(rocm_smi)
            .args(["--showpidgpus", "--json"])
            .output()
        {
            if proc_output.status.success() {
                let proc_stdout = String::from_utf8_lossy(&proc_output.stdout);
                if let Ok(json) = serde_json::from_str::<serde_json::Value>(&proc_stdout) {
                    if let Some(obj) = json.as_object() {
                        for (pid_str, pid_info) in obj {
                            if let Ok(pid) = pid_str.parse::<u32>() {
                                let gpu_index =
                                    pid_info.get("GPU ID").and_then(|v| v.as_u64()).unwrap_or(0)
                                        as u32;
                                let name = pid_info
                                    .get("Process name")
                                    .and_then(|v| v.as_str())
                                    .unwrap_or("unknown")
                                    .to_string();
                                let used_memory = pid_info
                                    .get("VRAM used")
                                    .and_then(|v| v.as_u64())
                                    .unwrap_or(0);

                                processes.push(GpuProcess {
                                    pid,
                                    name,
                                    gpu_index,
                                    used_memory,
                                    proc_type: GpuProcType::Compute,
                                    sm_util: 0,
                                    mem_util: 0,
                                    enc_util: 0,
                                    dec_util: 0,
                                });
                            }
                        }
                    }
                }
            }
        }

        // Calculate aggregates
        let total_vram = gpus.iter().map(|g| g.total_memory).sum();
        let total_vram_used = gpus.iter().map(|g| g.used_memory).sum();
        let avg_gpu_util = if gpus.is_empty() {
            0.0
        } else {
            gpus.iter().map(|g| g.utilization).sum::<f32>() / gpus.len() as f32
        };
        let max_temperature = gpus.iter().filter_map(|g| g.temperature).reduce(f32::max);
        let total_power = {
            let powers: Vec<f32> = gpus.iter().filter_map(|g| g.power_draw).collect();
            if powers.is_empty() {
                None
            } else {
                Some(powers.iter().sum())
            }
        };

        self.data = GpuProcsData {
            gpus,
            processes,
            vendor: Some(GpuVendor::Amd),
            total_vram_used,
            total_vram,
            avg_gpu_util,
            max_temperature,
            total_power,
        };

        Ok(())
    }

    /// Query AMD GPU - tries rocm-smi first, then falls back to sysfs (PMAT-GAP-029)
    fn query_amd(&mut self) -> Result<(), AnalyzerError> {
        // Try rocm-smi first for more detailed stats
        if self.rocm_smi_path.is_some() && self.query_amd_rocm_smi().is_ok() {
            return Ok(());
        }
        // Fall back to sysfs
        self.query_amd_sysfs()
    }

    /// Query AMD GPU via sysfs (fallback when rocm-smi unavailable)
    fn query_amd_sysfs(&mut self) -> Result<(), AnalyzerError> {
        let mut gpus = Vec::new();
        let processes = Vec::new();

        // Scan for AMD GPUs in /sys/class/drm/
        let drm_path = Path::new("/sys/class/drm");
        if !drm_path.exists() {
            return Err(AnalyzerError::NotAvailable("DRM not available".to_string()));
        }

        for entry in std::fs::read_dir(drm_path)
            .map_err(|e| AnalyzerError::IoError(format!("Failed to read /sys/class/drm: {}", e)))?
        {
            let entry = entry.map_err(|e| AnalyzerError::IoError(e.to_string()))?;
            let name = entry.file_name();
            let name_str = name.to_string_lossy();

            // Only process card* directories (not render nodes)
            if !name_str.starts_with("card") || name_str.contains('-') {
                continue;
            }

            let card_path = entry.path();
            let device_path = card_path.join("device");

            // Check if it's an AMD GPU
            let vendor_path = device_path.join("vendor");
            if let Ok(vendor) = std::fs::read_to_string(&vendor_path) {
                if vendor.trim() != "0x1002" {
                    continue; // Not AMD
                }
            } else {
                continue;
            }

            let index: u32 = name_str
                .strip_prefix("card")
                .and_then(|s| s.parse().ok())
                .unwrap_or(0);

            // Read GPU info from hwmon
            let mut gpu = GpuInfo {
                index,
                name: Self::read_amd_gpu_name(&device_path),
                vendor: GpuVendor::Amd,
                total_memory: Self::read_amd_vram_total(&device_path),
                used_memory: Self::read_amd_vram_used(&device_path),
                free_memory: 0,
                utilization: Self::read_amd_gpu_busy(&device_path),
                memory_utilization: 0.0,
                temperature: Self::read_amd_temperature(&device_path),
                power_draw: Self::read_amd_power(&device_path),
                power_limit: None,
                fan_speed: Self::read_amd_fan_speed(&device_path),
                driver_version: None,
            };

            gpu.free_memory = gpu.total_memory.saturating_sub(gpu.used_memory);
            gpu.memory_utilization = gpu.memory_percent();

            gpus.push(gpu);
        }

        // AMD doesn't have easy per-process GPU memory tracking
        // We'd need to parse /proc/[pid]/fdinfo for DRM clients

        // Calculate aggregates
        let total_vram = gpus.iter().map(|g| g.total_memory).sum();
        let total_vram_used = gpus.iter().map(|g| g.used_memory).sum();
        let avg_gpu_util = if gpus.is_empty() {
            0.0
        } else {
            gpus.iter().map(|g| g.utilization).sum::<f32>() / gpus.len() as f32
        };
        let max_temperature = gpus.iter().filter_map(|g| g.temperature).reduce(f32::max);
        let total_power = {
            let powers: Vec<f32> = gpus.iter().filter_map(|g| g.power_draw).collect();
            if powers.is_empty() {
                None
            } else {
                Some(powers.iter().sum())
            }
        };

        self.data = GpuProcsData {
            gpus,
            processes,
            vendor: Some(GpuVendor::Amd),
            total_vram_used,
            total_vram,
            avg_gpu_util,
            max_temperature,
            total_power,
        };

        Ok(())
    }

    /// Read AMD GPU name
    fn read_amd_gpu_name(device_path: &Path) -> String {
        let product_path = device_path.join("product_name");
        if let Ok(name) = std::fs::read_to_string(&product_path) {
            return name.trim().to_string();
        }

        let uevent_path = device_path.join("uevent");
        if let Ok(uevent) = std::fs::read_to_string(&uevent_path) {
            for line in uevent.lines() {
                if let Some(name) = line.strip_prefix("PCI_SLOT_NAME=") {
                    return format!("AMD GPU {}", name);
                }
            }
        }

        "AMD GPU".to_string()
    }

    /// Read AMD VRAM total
    fn read_amd_vram_total(device_path: &Path) -> u64 {
        let path = device_path.join("mem_info_vram_total");
        std::fs::read_to_string(&path)
            .ok()
            .and_then(|s| s.trim().parse().ok())
            .unwrap_or(0)
    }

    /// Read AMD VRAM used
    fn read_amd_vram_used(device_path: &Path) -> u64 {
        let path = device_path.join("mem_info_vram_used");
        std::fs::read_to_string(&path)
            .ok()
            .and_then(|s| s.trim().parse().ok())
            .unwrap_or(0)
    }

    /// Read AMD GPU busy percentage
    fn read_amd_gpu_busy(device_path: &Path) -> f32 {
        let path = device_path.join("gpu_busy_percent");
        std::fs::read_to_string(&path)
            .ok()
            .and_then(|s| s.trim().parse().ok())
            .unwrap_or(0.0)
    }

    /// Read AMD temperature
    fn read_amd_temperature(device_path: &Path) -> Option<f32> {
        // Try hwmon
        let hwmon_path = device_path.join("hwmon");
        if let Ok(entries) = std::fs::read_dir(&hwmon_path) {
            for entry in entries.flatten() {
                let temp_path = entry.path().join("temp1_input");
                if let Ok(temp) = std::fs::read_to_string(&temp_path) {
                    if let Ok(millicelsius) = temp.trim().parse::<i32>() {
                        return Some(millicelsius as f32 / 1000.0);
                    }
                }
            }
        }
        None
    }

    /// Read AMD power draw
    fn read_amd_power(device_path: &Path) -> Option<f32> {
        let hwmon_path = device_path.join("hwmon");
        if let Ok(entries) = std::fs::read_dir(&hwmon_path) {
            for entry in entries.flatten() {
                let power_path = entry.path().join("power1_average");
                if let Ok(power) = std::fs::read_to_string(&power_path) {
                    if let Ok(microwatts) = power.trim().parse::<u64>() {
                        return Some(microwatts as f32 / 1_000_000.0);
                    }
                }
            }
        }
        None
    }

    /// Read AMD fan speed
    fn read_amd_fan_speed(device_path: &Path) -> Option<u32> {
        let hwmon_path = device_path.join("hwmon");
        if let Ok(entries) = std::fs::read_dir(&hwmon_path) {
            for entry in entries.flatten() {
                let pwm_path = entry.path().join("pwm1");
                let max_path = entry.path().join("pwm1_max");

                if let (Ok(pwm), Ok(max)) = (
                    std::fs::read_to_string(&pwm_path),
                    std::fs::read_to_string(&max_path),
                ) {
                    if let (Ok(pwm_val), Ok(max_val)) =
                        (pwm.trim().parse::<u32>(), max.trim().parse::<u32>())
                    {
                        if max_val > 0 {
                            return Some(pwm_val * 100 / max_val);
                        }
                    }
                }
            }
        }
        None
    }

    /// Query Apple GPU (PMAT-GAP-030 - ttop parity)
    ///
    /// Uses system_profiler on macOS to get Apple Silicon GPU information.
    /// Apple Silicon has unified memory architecture so GPU memory = system RAM.
    #[cfg(target_os = "macos")]
    fn query_apple(&mut self) -> Result<(), AnalyzerError> {
        // Use system_profiler to get GPU info
        let output = Command::new("system_profiler")
            .args(["SPDisplaysDataType", "-json"])
            .output()
            .map_err(|e| AnalyzerError::IoError(format!("system_profiler failed: {}", e)))?;

        if !output.status.success() {
            return Err(AnalyzerError::IoError(
                "system_profiler returned error".to_string(),
            ));
        }

        let stdout = String::from_utf8_lossy(&output.stdout);
        let mut gpus = Vec::new();

        // Parse JSON output
        if let Ok(json) = serde_json::from_str::<serde_json::Value>(&stdout) {
            if let Some(displays) = json.get("SPDisplaysDataType").and_then(|v| v.as_array()) {
                for (index, display) in displays.iter().enumerate() {
                    let name = display
                        .get("sppci_model")
                        .or_else(|| display.get("_name"))
                        .and_then(|v| v.as_str())
                        .unwrap_or("Apple GPU")
                        .to_string();

                    // Apple Silicon uses unified memory - get system RAM
                    let total_memory = Self::get_apple_memory_size();

                    // GPU utilization via powermetrics requires sudo, use ioreg instead
                    let utilization = Self::get_apple_gpu_utilization();

                    gpus.push(GpuInfo {
                        index: index as u32,
                        name,
                        vendor: GpuVendor::Apple,
                        total_memory,
                        used_memory: 0, // Not directly available without Metal API
                        free_memory: total_memory,
                        utilization,
                        memory_utilization: 0.0,
                        temperature: Self::get_apple_gpu_temperature(),
                        power_draw: Self::get_apple_gpu_power(),
                        power_limit: None,
                        fan_speed: None, // Apple Silicon typically fanless or uses shared cooling
                        driver_version: None,
                    });
                }
            }
        }

        if gpus.is_empty() {
            return Err(AnalyzerError::NotAvailable(
                "No Apple GPU found".to_string(),
            ));
        }

        // Calculate aggregates
        let total_vram = gpus.iter().map(|g| g.total_memory).sum();
        let total_vram_used = gpus.iter().map(|g| g.used_memory).sum();
        let avg_gpu_util = if gpus.is_empty() {
            0.0
        } else {
            gpus.iter().map(|g| g.utilization).sum::<f32>() / gpus.len() as f32
        };
        let max_temperature = gpus.iter().filter_map(|g| g.temperature).reduce(f32::max);
        let total_power = {
            let powers: Vec<f32> = gpus.iter().filter_map(|g| g.power_draw).collect();
            if powers.is_empty() {
                None
            } else {
                Some(powers.iter().sum())
            }
        };

        self.data = GpuProcsData {
            gpus,
            processes: Vec::new(), // Per-process GPU tracking requires Metal API
            vendor: Some(GpuVendor::Apple),
            total_vram_used,
            total_vram,
            avg_gpu_util,
            max_temperature,
            total_power,
        };

        Ok(())
    }

    /// Get Apple system memory size (PMAT-GAP-030)
    #[cfg(target_os = "macos")]
    fn get_apple_memory_size() -> u64 {
        if let Ok(output) = Command::new("sysctl").args(["-n", "hw.memsize"]).output() {
            if output.status.success() {
                let stdout = String::from_utf8_lossy(&output.stdout);
                return stdout.trim().parse().unwrap_or(0);
            }
        }
        0
    }

    /// Get Apple GPU utilization via ioreg (PMAT-GAP-030)
    #[cfg(target_os = "macos")]
    fn get_apple_gpu_utilization() -> f32 {
        // Try to get GPU utilization from ioreg
        if let Ok(output) = Command::new("ioreg")
            .args(["-r", "-c", "IOAccelerator", "-d", "1"])
            .output()
        {
            if output.status.success() {
                let stdout = String::from_utf8_lossy(&output.stdout);
                // Parse GPU utilization from ioreg output
                for line in stdout.lines() {
                    if line.contains("\"PerformanceStatistics\"") {
                        // Found performance stats section
                        continue;
                    }
                    if line.contains("\"Device Utilization %\"") {
                        if let Some(val) = line.split('=').nth(1) {
                            if let Ok(util) = val.trim().parse::<f32>() {
                                return util;
                            }
                        }
                    }
                }
            }
        }
        0.0
    }

    /// Get Apple GPU temperature (PMAT-GAP-030)
    #[cfg(target_os = "macos")]
    fn get_apple_gpu_temperature() -> Option<f32> {
        // Temperature requires SMC access which needs elevated privileges.
        None
    }

    /// Get Apple GPU power draw (PMAT-GAP-030)
    #[cfg(target_os = "macos")]
    fn get_apple_gpu_power() -> Option<f32> {
        // Power metrics require elevated privileges.
        None
    }
}

impl Analyzer for GpuProcsAnalyzer {
    fn name(&self) -> &'static str {
        "gpu_procs"
    }

    fn collect(&mut self) -> Result<(), AnalyzerError> {
        // Interval guard: nvidia-smi/rocm-smi are expensive subprocess calls
        if let Some(last) = self.last_collect {
            if last.elapsed() < self.interval {
                return Ok(());
            }
        }
        self.last_collect = Some(std::time::Instant::now());

        match self.vendor {
            Some(GpuVendor::Nvidia) => self.query_nvidia(),
            Some(GpuVendor::Amd) => self.query_amd(),
            Some(GpuVendor::Intel) => {
                // Intel GPU support is limited
                Err(AnalyzerError::NotAvailable(
                    "Intel GPU not fully supported".to_string(),
                ))
            }
            #[cfg(target_os = "macos")]
            Some(GpuVendor::Apple) => self.query_apple(),
            #[cfg(not(target_os = "macos"))]
            Some(GpuVendor::Apple) => Err(AnalyzerError::NotAvailable(
                "Apple GPU only supported on macOS".to_string(),
            )),
            _ => Err(AnalyzerError::NotAvailable("No GPU detected".to_string())),
        }
    }

    fn interval(&self) -> Duration {
        self.interval
    }

    fn available(&self) -> bool {
        self.vendor.is_some()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // GpuVendor tests
    #[test]
    fn test_gpu_vendor_display() {
        assert_eq!(GpuVendor::Nvidia.as_str(), "NVIDIA");
        assert_eq!(GpuVendor::Amd.as_str(), "AMD");
        assert_eq!(GpuVendor::Intel.as_str(), "Intel");
    }

    #[test]
    fn test_gpu_vendor_unknown() {
        assert_eq!(GpuVendor::Unknown.as_str(), "Unknown");
    }

    // GpuInfo tests
    #[test]
    fn test_gpu_info_memory_percent() {
        let gpu = GpuInfo {
            index: 0,
            name: "Test GPU".to_string(),
            vendor: GpuVendor::Nvidia,
            total_memory: 8 * 1024 * 1024 * 1024, // 8 GB
            used_memory: 4 * 1024 * 1024 * 1024,  // 4 GB
            free_memory: 4 * 1024 * 1024 * 1024,
            utilization: 50.0,
            memory_utilization: 50.0,
            temperature: Some(65.0),
            power_draw: Some(150.0),
            power_limit: Some(250.0),
            fan_speed: Some(40),
            driver_version: Some("535.154.05".to_string()),
        };

        assert!((gpu.memory_percent() - 50.0).abs() < 0.01);
        assert_eq!(gpu.display_memory(), "4096/8192 MB");
    }

    #[test]
    fn test_gpu_info_zero_memory() {
        let gpu = GpuInfo {
            index: 0,
            name: "Empty GPU".to_string(),
            vendor: GpuVendor::Unknown,
            total_memory: 0,
            used_memory: 0,
            free_memory: 0,
            utilization: 0.0,
            memory_utilization: 0.0,
            temperature: None,
            power_draw: None,
            power_limit: None,
            fan_speed: None,
            driver_version: None,
        };

        // Division by zero case
        assert_eq!(gpu.memory_percent(), 0.0);
        assert_eq!(gpu.display_memory(), "0/0 MB");
    }

    #[test]
    fn test_gpu_info_full_memory() {
        let gpu = GpuInfo {
            index: 1,
            name: "Full GPU".to_string(),
            vendor: GpuVendor::Amd,
            total_memory: 16 * 1024 * 1024 * 1024,
            used_memory: 16 * 1024 * 1024 * 1024,
            free_memory: 0,
            utilization: 100.0,
            memory_utilization: 100.0,
            temperature: Some(90.0),
            power_draw: Some(300.0),
            power_limit: Some(350.0),
            fan_speed: Some(100),
            driver_version: Some("latest".to_string()),
        };

        assert!((gpu.memory_percent() - 100.0).abs() < 0.01);
    }

    // GpuProcess tests
    #[test]
    fn test_gpu_process_display() {
        let proc = GpuProcess {
            pid: 1234,
            name: "python".to_string(),
            gpu_index: 0,
            used_memory: 2048 * 1024 * 1024, // 2 GB
            proc_type: GpuProcType::Compute,
            sm_util: 80,
            mem_util: 25,
            enc_util: 0,
            dec_util: 0,
        };

        assert_eq!(proc.display_memory(), "2048 MB");
    }

    #[test]
    fn test_gpu_process_small_memory() {
        let proc = GpuProcess {
            pid: 5678,
            name: "desktop".to_string(),
            gpu_index: 0,
            used_memory: 512 * 1024, // 512 KB
            proc_type: GpuProcType::Graphics,
            sm_util: 0,
            mem_util: 0,
            enc_util: 0,
            dec_util: 0,
        };

        assert_eq!(proc.display_memory(), "0 MB"); // Less than 1 MB
    }

    // GpuProcsData tests
    #[test]
    fn test_gpu_procs_data_empty() {
        let data = GpuProcsData::default();
        assert!(!data.has_gpu());
        assert_eq!(data.gpu_count(), 0);
        assert_eq!(data.process_count(), 0);
    }

    #[test]
    fn test_gpu_procs_data_with_gpus() {
        let data = GpuProcsData {
            gpus: vec![
                GpuInfo {
                    index: 0,
                    name: "GPU 0".to_string(),
                    vendor: GpuVendor::Nvidia,
                    total_memory: 8 * 1024 * 1024 * 1024,
                    used_memory: 4 * 1024 * 1024 * 1024,
                    free_memory: 4 * 1024 * 1024 * 1024,
                    utilization: 50.0,
                    memory_utilization: 50.0,
                    temperature: Some(65.0),
                    power_draw: Some(150.0),
                    power_limit: Some(250.0),
                    fan_speed: Some(40),
                    driver_version: Some("535".to_string()),
                },
                GpuInfo {
                    index: 1,
                    name: "GPU 1".to_string(),
                    vendor: GpuVendor::Nvidia,
                    total_memory: 8 * 1024 * 1024 * 1024,
                    used_memory: 2 * 1024 * 1024 * 1024,
                    free_memory: 6 * 1024 * 1024 * 1024,
                    utilization: 25.0,
                    memory_utilization: 25.0,
                    temperature: Some(55.0),
                    power_draw: Some(100.0),
                    power_limit: Some(250.0),
                    fan_speed: Some(30),
                    driver_version: Some("535".to_string()),
                },
            ],
            processes: vec![GpuProcess {
                pid: 1234,
                name: "test".to_string(),
                gpu_index: 0,
                used_memory: 1024 * 1024 * 1024,
                proc_type: GpuProcType::Compute,
                sm_util: 50,
                mem_util: 12,
                enc_util: 0,
                dec_util: 0,
            }],
            vendor: Some(GpuVendor::Nvidia),
            total_vram_used: 6 * 1024 * 1024 * 1024,
            total_vram: 16 * 1024 * 1024 * 1024,
            avg_gpu_util: 37.5,
            max_temperature: Some(65.0),
            total_power: Some(250.0),
        };

        assert!(data.has_gpu());
        assert_eq!(data.gpu_count(), 2);
        assert_eq!(data.process_count(), 1);
    }

    // GpuProcsAnalyzer tests
    #[test]
    fn test_analyzer_creation() {
        let analyzer = GpuProcsAnalyzer::new();
        // Just verify it doesn't panic
        let _ = analyzer.available();
    }

    #[test]
    fn test_analyzer_default() {
        let analyzer = GpuProcsAnalyzer::default();
        assert_eq!(analyzer.name(), "gpu_procs");
    }

    #[test]
    fn test_analyzer_name() {
        let analyzer = GpuProcsAnalyzer::new();
        assert_eq!(analyzer.name(), "gpu_procs");
    }

    #[test]
    fn test_analyzer_interval() {
        let analyzer = GpuProcsAnalyzer::new();
        assert_eq!(analyzer.interval(), Duration::from_secs(2));
    }

    #[test]
    fn test_analyzer_data() {
        let analyzer = GpuProcsAnalyzer::new();
        let data = analyzer.data();
        // Data should be default empty initially
        assert_eq!(data.gpu_count(), 0);
    }

    // Parsing tests
    #[test]
    fn test_parse_nvidia_gpu_line() {
        let analyzer = GpuProcsAnalyzer::new();
        let line = "0, NVIDIA GeForce RTX 3080, 10240, 4096, 6144, 45, 40, 65, 150.00, 320.00, 35, 535.154.05";

        let gpu = analyzer.parse_nvidia_gpu_line(line);
        assert!(gpu.is_some());

        let gpu = gpu.unwrap();
        assert_eq!(gpu.index, 0);
        assert!(gpu.name.contains("RTX 3080"));
        assert_eq!(gpu.total_memory, 10240 * 1024 * 1024);
        assert_eq!(gpu.used_memory, 4096 * 1024 * 1024);
        assert!((gpu.utilization - 45.0).abs() < 0.01);
        assert!((gpu.temperature.unwrap() - 65.0).abs() < 0.01);
    }

    #[test]
    fn test_parse_nvidia_gpu_line_with_na() {
        let analyzer = GpuProcsAnalyzer::new();
        let line = "0, Test GPU, 8192, 4096, 4096, 50, 50, 70, 100.00, 200.00, 50, [N/A]";

        let gpu = analyzer.parse_nvidia_gpu_line(line);
        assert!(gpu.is_some());
        let gpu = gpu.unwrap();
        assert!(gpu.driver_version.is_none());
    }

    #[test]
    fn test_parse_nvidia_gpu_line_empty_driver() {
        let analyzer = GpuProcsAnalyzer::new();
        let line = "0, Test GPU, 8192, 4096, 4096, 50, 50, 70, 100.00, 200.00, 50, ";

        let gpu = analyzer.parse_nvidia_gpu_line(line);
        assert!(gpu.is_some());
        let gpu = gpu.unwrap();
        assert!(gpu.driver_version.is_none());
    }

    #[test]
    fn test_parse_nvidia_gpu_line_invalid() {
        let analyzer = GpuProcsAnalyzer::new();

        // Too few parts
        assert!(analyzer.parse_nvidia_gpu_line("0, GPU").is_none());

        // Invalid numbers
        assert!(analyzer
            .parse_nvidia_gpu_line("invalid, GPU, x, y, z, a, b, c, d, e, f, g")
            .is_none());
    }

    #[test]
    fn test_parse_pmon_single_process() {
        let output = "# gpu        pid  type    sm   mem   enc   dec   jpg   ofa command\n# Idx          #   C/G     %     %     %     %     %     %   name\n    0       1234     C    85    45    10     5     0     0   python3";
        let procs = GpuProcsAnalyzer::parse_pmon_output(output);
        assert_eq!(procs.len(), 1);
        assert_eq!(procs[0].pid, 1234);
        assert_eq!(procs[0].proc_type, GpuProcType::Compute);
        assert_eq!(procs[0].sm_util, 85);
        assert_eq!(procs[0].mem_util, 45);
    }

    #[test]
    fn test_parse_pmon_invalid_lines() {
        // Empty output
        let procs = GpuProcsAnalyzer::parse_pmon_output("");
        assert!(procs.is_empty());

        // Only headers
        let procs = GpuProcsAnalyzer::parse_pmon_output("# header\n# header2");
        assert!(procs.is_empty());

        // Malformed line
        let procs = GpuProcsAnalyzer::parse_pmon_output("invalid data");
        assert!(procs.is_empty());
    }

    #[test]
    fn test_parse_pmon_graphics_process() {
        let output = "# gpu        pid  type    sm   mem   enc   dec   jpg   ofa command\n    0       5678     G    50    30     0     0     0     0   X";
        let procs = GpuProcsAnalyzer::parse_pmon_output(output);
        assert_eq!(procs.len(), 1);
        assert_eq!(procs[0].pid, 5678);
        assert_eq!(procs[0].proc_type, GpuProcType::Graphics);
        assert_eq!(procs[0].sm_util, 50);
    }

    // Analyzer trait tests
    #[test]
    fn test_analyzer_collect_no_gpu() {
        let mut analyzer = GpuProcsAnalyzer {
            data: GpuProcsData::default(),
            interval: Duration::from_secs(2),
            vendor: None,
            nvidia_smi_path: None,
            rocm_smi_path: None,
            last_collect: None,
        };

        let result = analyzer.collect();
        assert!(result.is_err());
        if let Err(AnalyzerError::NotAvailable(msg)) = result {
            assert!(msg.contains("No GPU"));
        }
    }

    #[test]
    fn test_analyzer_collect_intel_not_supported() {
        let mut analyzer = GpuProcsAnalyzer {
            data: GpuProcsData::default(),
            interval: Duration::from_secs(2),
            vendor: Some(GpuVendor::Intel),
            nvidia_smi_path: None,
            rocm_smi_path: None,
            last_collect: None,
        };

        let result = analyzer.collect();
        assert!(result.is_err());
        if let Err(AnalyzerError::NotAvailable(msg)) = result {
            assert!(msg.contains("Intel"));
        }
    }

    #[test]
    fn test_analyzer_available_with_vendor() {
        let analyzer = GpuProcsAnalyzer {
            data: GpuProcsData::default(),
            interval: Duration::from_secs(2),
            vendor: Some(GpuVendor::Nvidia),
            nvidia_smi_path: Some("/usr/bin/nvidia-smi".to_string()),
            rocm_smi_path: None,
            last_collect: None,
        };

        assert!(analyzer.available());
    }

    #[test]
    fn test_analyzer_available_without_vendor() {
        let analyzer = GpuProcsAnalyzer {
            data: GpuProcsData::default(),
            interval: Duration::from_secs(2),
            vendor: None,
            nvidia_smi_path: None,
            rocm_smi_path: None,
            last_collect: None,
        };

        assert!(!analyzer.available());
    }

    // PMAT-GAP-029 tests - AMD GPU rocm-smi
    #[test]
    fn test_analyzer_amd_with_rocm_smi() {
        let analyzer = GpuProcsAnalyzer {
            data: GpuProcsData::default(),
            interval: Duration::from_secs(2),
            vendor: Some(GpuVendor::Amd),
            nvidia_smi_path: None,
            rocm_smi_path: Some("/opt/rocm/bin/rocm-smi".to_string()),
            last_collect: None,
        };

        assert!(analyzer.available());
        assert!(analyzer.rocm_smi_path.is_some());
    }

    #[test]
    fn test_analyzer_amd_without_rocm_smi() {
        let analyzer = GpuProcsAnalyzer {
            data: GpuProcsData::default(),
            interval: Duration::from_secs(2),
            vendor: Some(GpuVendor::Amd),
            nvidia_smi_path: None,
            rocm_smi_path: None,
            last_collect: None,
        };

        assert!(analyzer.available());
        assert!(analyzer.rocm_smi_path.is_none());
    }

    #[test]
    fn test_detect_rocm_smi_nonexistent() {
        // Test that detect_rocm_smi doesn't panic with nonexistent paths
        let path = GpuProcsAnalyzer::detect_rocm_smi();
        // Result depends on system - just verify it doesn't panic
        let _ = path;
    }

    // PMAT-GAP-030 tests - Apple GPU
    #[test]
    fn test_gpu_vendor_apple() {
        assert_eq!(GpuVendor::Apple.as_str(), "Apple");
    }

    #[test]
    fn test_analyzer_apple_vendor() {
        let analyzer = GpuProcsAnalyzer {
            data: GpuProcsData::default(),
            interval: Duration::from_secs(2),
            vendor: Some(GpuVendor::Apple),
            nvidia_smi_path: None,
            rocm_smi_path: None,
            last_collect: None,
        };

        assert!(analyzer.available());
    }

    #[test]
    fn test_analyzer_collect_apple_non_macos() {
        let mut analyzer = GpuProcsAnalyzer {
            data: GpuProcsData::default(),
            interval: Duration::from_secs(2),
            vendor: Some(GpuVendor::Apple),
            nvidia_smi_path: None,
            rocm_smi_path: None,
            last_collect: None,
        };

        let result = analyzer.collect();
        // On non-macOS, Apple GPU should return NotAvailable
        #[cfg(not(target_os = "macos"))]
        assert!(result.is_err());
        // On macOS, result depends on actual hardware
        #[cfg(target_os = "macos")]
        let _ = result;
    }

    // Additional GpuVendor tests
    #[test]
    fn test_gpu_vendor_debug() {
        let vendor = GpuVendor::Nvidia;
        let debug = format!("{:?}", vendor);
        assert!(debug.contains("Nvidia"));
    }

    #[test]
    fn test_gpu_vendor_clone() {
        let vendor = GpuVendor::Amd;
        let cloned = vendor.clone();
        assert_eq!(vendor, cloned);
    }

    #[test]
    fn test_gpu_vendor_copy() {
        let vendor = GpuVendor::Intel;
        let copied: GpuVendor = vendor;
        assert_eq!(copied, GpuVendor::Intel);
    }

    // Additional GpuInfo tests
    #[test]
    fn test_gpu_info_debug() {
        let gpu = GpuInfo {
            index: 0,
            name: "Test".to_string(),
            vendor: GpuVendor::Nvidia,
            total_memory: 0,
            used_memory: 0,
            free_memory: 0,
            utilization: 0.0,
            memory_utilization: 0.0,
            temperature: None,
            power_draw: None,
            power_limit: None,
            fan_speed: None,
            driver_version: None,
        };
        let debug = format!("{:?}", gpu);
        assert!(debug.contains("GpuInfo"));
    }

    #[test]
    fn test_gpu_info_clone() {
        let gpu = GpuInfo {
            index: 0,
            name: "Clone Test".to_string(),
            vendor: GpuVendor::Nvidia,
            total_memory: 1024,
            used_memory: 512,
            free_memory: 512,
            utilization: 25.0,
            memory_utilization: 50.0,
            temperature: Some(60.0),
            power_draw: Some(100.0),
            power_limit: Some(200.0),
            fan_speed: Some(50),
            driver_version: Some("test".to_string()),
        };
        let cloned = gpu.clone();
        assert_eq!(cloned.name, "Clone Test");
        assert_eq!(cloned.index, 0);
    }

    // Additional GpuProcess tests
    #[test]
    fn test_gpu_process_debug() {
        let proc = GpuProcess {
            pid: 1,
            name: "test".to_string(),
            gpu_index: 0,
            used_memory: 0,
            proc_type: GpuProcType::Compute,
            sm_util: 0,
            mem_util: 0,
            enc_util: 0,
            dec_util: 0,
        };
        let debug = format!("{:?}", proc);
        assert!(debug.contains("GpuProcess"));
    }

    #[test]
    fn test_gpu_process_clone() {
        let proc = GpuProcess {
            pid: 1234,
            name: "clone_test".to_string(),
            gpu_index: 1,
            used_memory: 2048,
            proc_type: GpuProcType::Graphics,
            sm_util: 50,
            mem_util: 25,
            enc_util: 10,
            dec_util: 5,
        };
        let cloned = proc.clone();
        assert_eq!(cloned.pid, 1234);
        assert_eq!(cloned.name, "clone_test");
    }

    // Additional GpuProcsData tests
    #[test]
    fn test_gpu_procs_data_debug() {
        let data = GpuProcsData::default();
        let debug = format!("{:?}", data);
        assert!(debug.contains("GpuProcsData"));
    }

    #[test]
    fn test_gpu_procs_data_clone() {
        let data = GpuProcsData {
            gpus: vec![],
            processes: vec![],
            vendor: Some(GpuVendor::Amd),
            total_vram_used: 1000,
            total_vram: 2000,
            avg_gpu_util: 50.0,
            max_temperature: Some(75.0),
            total_power: Some(200.0),
        };
        let cloned = data.clone();
        assert_eq!(cloned.vendor, Some(GpuVendor::Amd));
        assert_eq!(cloned.total_vram_used, 1000);
    }

    // AMD sysfs helper function tests
    #[test]
    fn test_read_amd_gpu_name_nonexistent() {
        let path = Path::new("/nonexistent/path");
        let name = GpuProcsAnalyzer::read_amd_gpu_name(path);
        assert_eq!(name, "AMD GPU");
    }

    #[test]
    fn test_read_amd_vram_total_nonexistent() {
        let path = Path::new("/nonexistent/path");
        let total = GpuProcsAnalyzer::read_amd_vram_total(path);
        assert_eq!(total, 0);
    }

    #[test]
    fn test_read_amd_vram_used_nonexistent() {
        let path = Path::new("/nonexistent/path");
        let used = GpuProcsAnalyzer::read_amd_vram_used(path);
        assert_eq!(used, 0);
    }

    #[test]
    fn test_read_amd_gpu_busy_nonexistent() {
        let path = Path::new("/nonexistent/path");
        let busy = GpuProcsAnalyzer::read_amd_gpu_busy(path);
        assert!((busy - 0.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_read_amd_temperature_nonexistent() {
        let path = Path::new("/nonexistent/path");
        let temp = GpuProcsAnalyzer::read_amd_temperature(path);
        assert!(temp.is_none());
    }

    #[test]
    fn test_read_amd_power_nonexistent() {
        let path = Path::new("/nonexistent/path");
        let power = GpuProcsAnalyzer::read_amd_power(path);
        assert!(power.is_none());
    }

    #[test]
    fn test_read_amd_fan_speed_nonexistent() {
        let path = Path::new("/nonexistent/path");
        let speed = GpuProcsAnalyzer::read_amd_fan_speed(path);
        assert!(speed.is_none());
    }

    // Test detect_gpu function
    #[test]
    fn test_detect_gpu() {
        // This just tests that the function runs without panicking
        let (vendor, nvidia_path, rocm_path) = GpuProcsAnalyzer::detect_gpu();
        // Result depends on system hardware - just verify types
        let _ = vendor;
        let _ = nvidia_path;
        let _ = rocm_path;
    }

    // Test query methods with mock analyzer
    #[test]
    fn test_query_nvidia_no_nvidia_smi() {
        let mut analyzer = GpuProcsAnalyzer {
            data: GpuProcsData::default(),
            interval: Duration::from_secs(2),
            vendor: Some(GpuVendor::Nvidia),
            nvidia_smi_path: None,
            rocm_smi_path: None,
            last_collect: None,
        };

        let result = analyzer.query_nvidia();
        assert!(result.is_err());
    }

    #[test]
    fn test_query_amd_no_drm() {
        // Create analyzer that will fail because there's no AMD GPU
        let mut analyzer = GpuProcsAnalyzer {
            data: GpuProcsData::default(),
            interval: Duration::from_secs(2),
            vendor: Some(GpuVendor::Amd),
            nvidia_smi_path: None,
            rocm_smi_path: None,
            last_collect: None,
        };

        // This may succeed or fail depending on system
        let _ = analyzer.query_amd();
    }

    #[test]
    fn test_query_amd_rocm_smi_not_available() {
        let mut analyzer = GpuProcsAnalyzer {
            data: GpuProcsData::default(),
            interval: Duration::from_secs(2),
            vendor: Some(GpuVendor::Amd),
            nvidia_smi_path: None,
            rocm_smi_path: None,
            last_collect: None,
        };

        // Without rocm_smi_path, query_amd_rocm_smi should error
        let result = analyzer.query_amd_rocm_smi();
        assert!(result.is_err());
    }

    #[test]
    fn test_query_amd_sysfs() {
        let mut analyzer = GpuProcsAnalyzer {
            data: GpuProcsData::default(),
            interval: Duration::from_secs(2),
            vendor: Some(GpuVendor::Amd),
            nvidia_smi_path: None,
            rocm_smi_path: None,
            last_collect: None,
        };

        // May succeed or fail depending on system hardware
        let _ = analyzer.query_amd_sysfs();
    }

    // PMAT-GAP-037-042 pmon tests
    #[test]
    fn test_gpu_proc_type_from_pmon() {
        assert_eq!(GpuProcType::from_pmon("C"), GpuProcType::Compute);
        assert_eq!(GpuProcType::from_pmon("G"), GpuProcType::Graphics);
        assert_eq!(GpuProcType::from_pmon("X"), GpuProcType::Unknown);
        assert_eq!(GpuProcType::from_pmon(""), GpuProcType::Unknown);
    }

    #[test]
    fn test_gpu_proc_type_as_char() {
        assert_eq!(GpuProcType::Compute.as_char(), 'C');
        assert_eq!(GpuProcType::Graphics.as_char(), 'G');
        assert_eq!(GpuProcType::Unknown.as_char(), '?');
    }

    #[test]
    fn test_gpu_proc_type_as_str() {
        assert_eq!(GpuProcType::Compute.as_str(), "Compute");
        assert_eq!(GpuProcType::Graphics.as_str(), "Graphics");
        assert_eq!(GpuProcType::Unknown.as_str(), "Unknown");
    }

    #[test]
    fn test_gpu_proc_type_display() {
        assert_eq!(format!("{}", GpuProcType::Compute), "C");
        assert_eq!(format!("{}", GpuProcType::Graphics), "G");
        assert_eq!(format!("{}", GpuProcType::Unknown), "?");
    }

    #[test]
    fn test_parse_pmon_output_empty() {
        let output = "";
        let procs = GpuProcsAnalyzer::parse_pmon_output(output);
        assert!(procs.is_empty());
    }

    #[test]
    fn test_parse_pmon_output_headers_only() {
        let output = r#"# gpu         pid   type     sm    mem    enc    dec    jpg    ofa    command
# Idx           #    C/G      %      %      %      %      %      %    name
"#;
        let procs = GpuProcsAnalyzer::parse_pmon_output(output);
        assert!(procs.is_empty());
    }

    #[test]
    fn test_parse_pmon_output_single_process() {
        let output = r#"# gpu         pid   type     sm    mem    enc    dec    jpg    ofa    command
# Idx           #    C/G      %      %      %      %      %      %    name
    0        1234      C     50     25     10      5      -      -    python3
"#;
        let procs = GpuProcsAnalyzer::parse_pmon_output(output);
        assert_eq!(procs.len(), 1);
        assert_eq!(procs[0].pid, 1234);
        assert_eq!(procs[0].gpu_index, 0);
        assert_eq!(procs[0].proc_type, GpuProcType::Compute);
        assert_eq!(procs[0].sm_util, 50);
        assert_eq!(procs[0].mem_util, 25);
        assert_eq!(procs[0].enc_util, 10);
        assert_eq!(procs[0].dec_util, 5);
        assert_eq!(procs[0].name, "python3");
    }

    #[test]
    fn test_parse_pmon_output_graphics_process() {
        let output = "    0        5678      G     30     15      0      0      -      -    Xorg\n";
        let procs = GpuProcsAnalyzer::parse_pmon_output(output);
        assert_eq!(procs.len(), 1);
        assert_eq!(procs[0].proc_type, GpuProcType::Graphics);
    }

    #[test]
    fn test_parse_pmon_output_multiple_processes() {
        let output = r#"    0        1234      C     50     25      0      0      -      -    python3
    0        5678      G     30     15      0      0      -      -    Xorg
    1        9999      C     80     40     20     10      -      -    cuda_app
"#;
        let procs = GpuProcsAnalyzer::parse_pmon_output(output);
        assert_eq!(procs.len(), 3);
        assert_eq!(procs[0].sm_util, 50);
        assert_eq!(procs[1].sm_util, 30);
        assert_eq!(procs[2].sm_util, 80);
        assert_eq!(procs[2].gpu_index, 1);
    }

    #[test]
    fn test_parse_pmon_output_dash_values() {
        let output = "    0        1234      C      -      -      -      -      -      -    app\n";
        let procs = GpuProcsAnalyzer::parse_pmon_output(output);
        assert_eq!(procs.len(), 1);
        assert_eq!(procs[0].sm_util, 0);
        assert_eq!(procs[0].mem_util, 0);
        assert_eq!(procs[0].enc_util, 0);
        assert_eq!(procs[0].dec_util, 0);
    }

    #[test]
    fn test_parse_pmon_sort_by_sm_util() {
        // Verify processes sorted by SM util descending (PMAT-GAP-042)
        let output = r#"    0        1111      C     10      5      0      0      -      -    low
    0        2222      C     80     40      0      0      -      -    high
    0        3333      C     50     25      0      0      -      -    mid
"#;
        let mut procs = GpuProcsAnalyzer::parse_pmon_output(output);
        procs.sort_by(|a, b| b.sm_util.cmp(&a.sm_util));
        assert_eq!(procs[0].sm_util, 80);
        assert_eq!(procs[1].sm_util, 50);
        assert_eq!(procs[2].sm_util, 10);
    }

    #[test]
    fn test_gpu_process_gpu_util_compat() {
        let proc = GpuProcess {
            sm_util: 75,
            ..Default::default()
        };
        assert_eq!(proc.gpu_util(), Some(75.0));

        let proc_zero = GpuProcess::default();
        assert_eq!(proc_zero.gpu_util(), None);
    }

    #[test]
    fn test_gpu_process_default() {
        let proc = GpuProcess::default();
        assert_eq!(proc.pid, 0);
        assert_eq!(proc.proc_type, GpuProcType::Unknown);
        assert_eq!(proc.sm_util, 0);
        assert_eq!(proc.enc_util, 0);
        assert_eq!(proc.dec_util, 0);
    }
}
