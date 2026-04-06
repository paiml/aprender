//! Hardware detection and system metrics types.

/// Hardware information detected at startup
#[derive(Debug, Clone)]
pub struct HardwareInfo {
    /// CPU model name
    pub cpu_model: String,
    /// Number of CPU cores
    pub cpu_cores: usize,
    /// SIMD capability
    pub simd_type: &'static str,
    /// GPU name (if available)
    pub gpu_name: Option<String>,
    /// Total system memory in GB
    pub memory_gb: f64,
}

impl HardwareInfo {
    /// Detect hardware at startup
    pub fn detect() -> Self {
        let cpu_cores = std::thread::available_parallelism()
            .map(|p| p.get())
            .unwrap_or(1);

        // Detect SIMD capability
        let simd_type = Self::detect_simd();

        // Get CPU model via batuta-common
        let cpu_model = batuta_common::sys::get_cpu_info();

        // Try to get GPU name
        let gpu_name = Self::detect_gpu();

        // Get total memory
        let memory_gb = Self::read_memory_gb();

        Self {
            cpu_model,
            cpu_cores,
            simd_type,
            gpu_name,
            memory_gb,
        }
    }

    fn detect_simd() -> &'static str {
        #[cfg(target_arch = "x86_64")]
        {
            if std::arch::is_x86_feature_detected!("avx512f") {
                return "AVX-512";
            }
            if std::arch::is_x86_feature_detected!("avx2") {
                return "AVX2";
            }
            if std::arch::is_x86_feature_detected!("avx") {
                return "AVX";
            }
            if std::arch::is_x86_feature_detected!("sse4.2") {
                return "SSE4.2";
            }
            "SSE2"
        }
        #[cfg(target_arch = "aarch64")]
        {
            "NEON"
        }
        #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
        {
            "Scalar"
        }
    }

    fn detect_gpu() -> Option<String> {
        #[cfg(target_os = "linux")]
        {
            Self::detect_gpu_linux()
        }
        #[cfg(target_os = "macos")]
        {
            return Self::detect_gpu_macos();
        }
        #[cfg(not(any(target_os = "linux", target_os = "macos")))]
        {
            None
        }
    }

    #[cfg(target_os = "linux")]
    fn detect_gpu_linux() -> Option<String> {
        let output = std::process::Command::new("nvidia-smi")
            .args(["--query-gpu=name", "--format=csv,noheader"])
            .output()
            .ok()?;
        if !output.status.success() {
            return None;
        }
        String::from_utf8(output.stdout)
            .ok()
            .map(|s| s.lines().next().unwrap_or("").trim().to_string())
            .filter(|s| !s.is_empty())
    }

    #[cfg(target_os = "macos")]
    fn detect_gpu_macos() -> Option<String> {
        let output = std::process::Command::new("system_profiler")
            .args(["SPDisplaysDataType"])
            .output()
            .ok()?;
        if !output.status.success() {
            return None;
        }
        let text = String::from_utf8_lossy(&output.stdout);
        text.lines()
            .find(|line| line.contains("Chipset Model:"))
            .and_then(|line| line.split(':').nth(1))
            .map(|s| s.trim().to_string())
    }

    fn read_memory_gb() -> f64 {
        #[cfg(target_os = "linux")]
        {
            Self::read_memory_gb_linux()
        }
        #[cfg(target_os = "macos")]
        {
            return Self::read_memory_gb_macos();
        }
        #[cfg(not(any(target_os = "linux", target_os = "macos")))]
        {
            0.0
        }
    }

    #[cfg(target_os = "linux")]
    fn read_memory_gb_linux() -> f64 {
        let contents = match std::fs::read_to_string("/proc/meminfo") {
            Ok(c) => c,
            Err(_) => return 0.0,
        };
        contents
            .lines()
            .find(|line| line.starts_with("MemTotal:"))
            .and_then(|line| line.split_whitespace().nth(1))
            .and_then(|kb_str| kb_str.parse::<u64>().ok())
            .map_or(0.0, |kb| kb as f64 / 1_048_576.0)
    }

    #[cfg(target_os = "macos")]
    fn read_memory_gb_macos() -> f64 {
        let output = match std::process::Command::new("sysctl")
            .args(["-n", "hw.memsize"])
            .output()
        {
            Ok(o) => o,
            Err(_) => return 0.0,
        };
        String::from_utf8(output.stdout)
            .ok()
            .and_then(|s| s.trim().parse::<u64>().ok())
            .map_or(0.0, |bytes| bytes as f64 / 1_073_741_824.0)
    }
}

/// Memory breakdown metrics (PMAT-012 UI-04)
#[derive(Debug, Clone, Default)]
pub struct MemoryBreakdown {
    /// Total RAM in KB
    pub total_kb: u64,
    /// Used RAM in KB
    pub used_kb: u64,
    /// Cached RAM in KB
    pub cached_kb: u64,
    /// Buffers in KB
    pub buffers_kb: u64,
    /// Available RAM in KB
    pub available_kb: u64,
}

impl MemoryBreakdown {
    /// Usage percentage
    pub fn usage_percent(&self) -> f64 {
        if self.total_kb > 0 {
            ((self.total_kb - self.available_kb) as f64 / self.total_kb as f64) * 100.0
        } else {
            0.0
        }
    }

    /// Format KB as human-readable
    pub fn format_kb(kb: u64) -> String {
        if kb >= 1_048_576 {
            format!("{:.1}G", kb as f64 / 1_048_576.0)
        } else if kb >= 1024 {
            format!("{:.1}M", kb as f64 / 1024.0)
        } else {
            format!("{}K", kb)
        }
    }
}

/// Network metrics (PMAT-012 UI-07 P2)
#[derive(Debug, Clone, Default)]
pub struct NetworkMetrics {
    /// Total bytes received
    pub rx_bytes: u64,
    /// Total bytes transmitted
    pub tx_bytes: u64,
    /// Receive rate in bytes/sec
    pub rx_rate: f64,
    /// Transmit rate in bytes/sec
    pub tx_rate: f64,
}

impl NetworkMetrics {
    /// Format bytes as human-readable rate
    pub fn format_rate(bytes_per_sec: f64) -> String {
        if bytes_per_sec >= 1_073_741_824.0 {
            format!("{:.1} GB/s", bytes_per_sec / 1_073_741_824.0)
        } else if bytes_per_sec >= 1_048_576.0 {
            format!("{:.1} MB/s", bytes_per_sec / 1_048_576.0)
        } else if bytes_per_sec >= 1024.0 {
            format!("{:.1} KB/s", bytes_per_sec / 1024.0)
        } else {
            format!("{:.0} B/s", bytes_per_sec)
        }
    }
}

/// Disk metrics (PMAT-012 UI-08 P2)
#[derive(Debug, Clone, Default)]
pub struct DiskMetrics {
    /// Mount point
    pub mount: String,
    /// Total space in bytes
    pub total_bytes: u64,
    /// Used space in bytes
    pub used_bytes: u64,
    /// Usage percentage
    pub usage_percent: f64,
}

impl DiskMetrics {
    /// Format bytes as human-readable
    pub fn format_bytes(bytes: u64) -> String {
        batuta_common::fmt::format_bytes_compact(bytes)
    }
}

/// Real-time load metrics
#[derive(Debug, Clone, Default)]
pub struct LoadMetrics {
    /// Bricks executed per second
    pub bricks_per_second: f64,
    /// Total bricks executed
    pub total_bricks: u64,
    /// Average latency per brick in microseconds
    pub avg_latency_us: f64,
    /// Measured CPU usage from /proc/stat
    pub cpu_usage: f64,
    /// Per-core CPU usage (PMAT-012 UI-02)
    pub per_core_usage: Vec<f64>,
    /// Operations per second (FLOPS for GEMM)
    pub ops_per_second: f64,
    /// Bytes processed per second
    pub bytes_per_second: f64,
    /// Memory breakdown (PMAT-012 UI-04)
    pub memory: MemoryBreakdown,
    /// Network metrics (PMAT-012 UI-07 P2)
    pub network: NetworkMetrics,
    /// Disk metrics (PMAT-012 UI-08 P2)
    pub disks: Vec<DiskMetrics>,
}
