//! Device module - ublk device management
//!
//! Provides abstraction over libublk for managing trueno-ublk devices.
//! Also provides a pure Rust `BlockDevice` for testing without kernel dependencies.

#![allow(dead_code)]

use crate::daemon::PageStore;
use anyhow::{Context, Result};
use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, RwLock};
use trueno_zram_core::{CompressorBuilder, PageCompressor, PAGE_SIZE};

/// Device configuration
#[derive(Clone, Debug)]
pub struct DeviceConfig {
    /// Device ID to use (-1 for auto-assign)
    pub dev_id: i32,
    pub size: u64,
    pub algorithm: trueno_zram_core::Algorithm,
    pub streams: usize,
    pub gpu_enabled: bool,
    pub mem_limit: Option<u64>,
    pub backing_dev: Option<PathBuf>,
    pub writeback_limit: Option<u64>,
    pub entropy_skip_threshold: f64,
    pub gpu_batch_size: usize,
    pub foreground: bool,
    /// Enable batched compression mode for high throughput
    pub batched: bool,
    /// Batch threshold - pages before triggering batch compression
    pub batch_threshold: usize,
    /// Flush timeout in milliseconds
    pub flush_timeout_ms: u64,
    /// PERF-001: High-performance configuration (polling, affinity, NUMA)
    pub perf: Option<crate::perf::PerfConfig>,
    /// PERF-003: Number of hardware queues (1-8)
    /// Multiple queues enable parallel I/O with separate io_uring per queue.
    pub nr_hw_queues: u16,
    /// PERF-006: Enable ZERO_COPY mode (EXPERIMENTAL)
    /// Eliminates pwrite syscall per I/O by using mmap'd kernel buffers.
    pub zero_copy: bool,
    // =========================================================================
    // KERN-001/002/003: Kernel-Cooperative Tiered Storage
    // =========================================================================
    /// Storage backend type: memory, zram, tiered
    pub backend: crate::backend::BackendType,
    /// Enable entropy-based routing for tiered storage
    pub entropy_routing: bool,
    /// Kernel ZRAM device path (e.g., /dev/zram0)
    pub zram_device: Option<PathBuf>,
    /// NVMe cold tier directory path (KERN-003)
    pub cold_tier: Option<PathBuf>,
    /// Entropy threshold for kernel ZRAM routing (pages below this go to kernel)
    pub entropy_kernel_threshold: f64,
    // =========================================================================
    // VIZ-002: Renacer Visualization Integration
    // =========================================================================
    /// Enable real-time TUI visualization (renacer dashboard)
    pub visualize: bool,
    // =========================================================================
    // VIZ-004: OTLP Integration (OpenTelemetry)
    // =========================================================================
    /// OTLP endpoint for trace/metric export
    pub otlp_endpoint: Option<String>,
    /// Service name for OTLP traces
    pub otlp_service_name: String,
}

/// Device statistics (zram-compatible)
#[derive(Clone, Debug, Default)]
pub struct DeviceStats {
    // mm_stat fields (9 fields)
    pub orig_data_size: u64,
    pub compr_data_size: u64,
    pub mem_used_total: u64,
    pub mem_limit: u64,
    pub mem_used_max: u64,
    pub same_pages: u64,
    pub pages_compacted: u64,
    pub huge_pages: u64,
    pub huge_pages_since: u64,

    // io_stat fields (4 fields)
    pub failed_reads: u64,
    pub failed_writes: u64,
    pub invalid_io: u64,
    pub notify_free: u64,

    // bd_stat fields (3 fields)
    pub bd_count: u64,
    pub bd_reads: u64,
    pub bd_writes: u64,

    // Extended trueno-ublk stats
    pub gpu_pages: u64,
    pub simd_pages: u64,
    pub scalar_pages: u64,
    pub throughput_gbps: f64,
    pub avg_entropy: f64,
    pub simd_backend: String,
}

/// Runtime statistics tracking
struct RuntimeStats {
    orig_data_size: AtomicU64,
    compr_data_size: AtomicU64,
    mem_used_total: AtomicU64,
    mem_used_max: AtomicU64,
    same_pages: AtomicU64,
    pages_compacted: AtomicU64,
    huge_pages: AtomicU64,
    huge_pages_since: AtomicU64,
    failed_reads: AtomicU64,
    failed_writes: AtomicU64,
    invalid_io: AtomicU64,
    notify_free: AtomicU64,
    bd_count: AtomicU64,
    bd_reads: AtomicU64,
    bd_writes: AtomicU64,
    gpu_pages: AtomicU64,
    simd_pages: AtomicU64,
    scalar_pages: AtomicU64,
    bytes_processed: AtomicU64,
    entropy_sum: AtomicU64,
    entropy_count: AtomicU64,
}

impl Default for RuntimeStats {
    fn default() -> Self {
        Self {
            orig_data_size: AtomicU64::new(0),
            compr_data_size: AtomicU64::new(0),
            mem_used_total: AtomicU64::new(0),
            mem_used_max: AtomicU64::new(0),
            same_pages: AtomicU64::new(0),
            pages_compacted: AtomicU64::new(0),
            huge_pages: AtomicU64::new(0),
            huge_pages_since: AtomicU64::new(0),
            failed_reads: AtomicU64::new(0),
            failed_writes: AtomicU64::new(0),
            invalid_io: AtomicU64::new(0),
            notify_free: AtomicU64::new(0),
            bd_count: AtomicU64::new(0),
            bd_reads: AtomicU64::new(0),
            bd_writes: AtomicU64::new(0),
            gpu_pages: AtomicU64::new(0),
            simd_pages: AtomicU64::new(0),
            scalar_pages: AtomicU64::new(0),
            bytes_processed: AtomicU64::new(0),
            entropy_sum: AtomicU64::new(0),
            entropy_count: AtomicU64::new(0),
        }
    }
}

/// Global device registry
static DEVICES: std::sync::LazyLock<RwLock<HashMap<u32, Arc<DeviceInner>>>> =
    std::sync::LazyLock::new(|| RwLock::new(HashMap::new()));

struct DeviceInner {
    id: u32,
    config: RwLock<DeviceConfig>,
    stats: RuntimeStats,
    compressor: RwLock<Option<Box<dyn PageCompressor>>>,
}

/// A trueno-ublk device
pub struct UblkDevice {
    inner: Arc<DeviceInner>,
}

impl UblkDevice {
    /// DT-007: Lock daemon memory to prevent swap deadlock
    ///
    /// Must be called in the ACTUAL daemon process (after fork for background mode).
    /// Five Whys Root Cause: mlock only affects calling process, not forked children.
    #[cfg(not(test))]
    fn lock_daemon_memory() {
        use duende_mlock::{lock_all, MlockStatus};
        match lock_all() {
            Ok(MlockStatus::Locked { bytes_locked }) => {
                tracing::info!(
                    "DT-007: Memory locked ({} bytes) - swap deadlock prevention active",
                    bytes_locked
                );
            }
            Ok(MlockStatus::Failed { errno }) => {
                tracing::warn!(
                    "DT-007: mlock() failed (errno={}) - daemon may deadlock under swap pressure",
                    errno
                );
            }
            Ok(MlockStatus::Unsupported) => {
                tracing::debug!("DT-007: mlock() not supported on this platform");
            }
            Err(e) => {
                tracing::error!("DT-007: mlock() error: {:?}", e);
            }
        }
    }

    #[cfg(test)]
    fn lock_daemon_memory() {
        // No-op in tests
    }

    /// Create a new device with the given configuration
    #[cfg(not(test))]
    pub fn create(config: DeviceConfig) -> Result<Self> {
        // DT-007e: mlock moved to lock_daemon_memory() - called in daemon process
        // For foreground mode: called in start_ublk_daemon before run_with_mode
        // For background mode: called in child process after fork()

        let id = if config.dev_id >= 0 { config.dev_id as u32 } else { Self::next_free_id()? };

        // Initialize compressor using CompressorBuilder
        let compressor = CompressorBuilder::new().algorithm(config.algorithm).build()?;

        // Start ublk daemon (needs config before we move it)
        Self::start_ublk_daemon(id, &config)?;

        let inner = Arc::new(DeviceInner {
            id,
            config: RwLock::new(config),
            stats: RuntimeStats::default(),
            compressor: RwLock::new(Some(compressor)),
        });

        // Register device
        {
            let mut devices = DEVICES.write().expect("rwlock poisoned");
            devices.insert(id, inner.clone());
        }

        Ok(Self { inner })
    }

    /// Create a mock device for testing (doesn't start ublk daemon)
    #[cfg(test)]
    pub fn create(config: DeviceConfig) -> Result<Self> {
        let id = if config.dev_id >= 0 {
            config.dev_id as u32
        } else {
            0 // Default to device 0 in tests
        };

        // Initialize compressor using CompressorBuilder
        let compressor = CompressorBuilder::new().algorithm(config.algorithm).build()?;

        let inner = Arc::new(DeviceInner {
            id,
            config: RwLock::new(config),
            stats: RuntimeStats::default(),
            compressor: RwLock::new(Some(compressor)),
        });

        // Register device
        {
            let mut devices = DEVICES.write().expect("rwlock poisoned");
            devices.insert(id, inner.clone());
        }

        Ok(Self { inner })
    }

    /// Open an existing device by path
    pub fn open(path: &Path) -> Result<Self> {
        let id = Self::parse_device_id(path)?;

        let devices = DEVICES.read().expect("rwlock poisoned");
        let inner =
            devices.get(&id).cloned().context(format!("Device {} not found", path.display()))?;

        Ok(Self { inner })
    }

    /// List all trueno-ublk devices
    pub fn list_all() -> Result<Vec<Self>> {
        let devices = DEVICES.read().expect("rwlock poisoned");
        Ok(devices.values().map(|inner| Self { inner: inner.clone() }).collect())
    }

    /// Find the next free device ID
    pub fn next_free_id() -> Result<u32> {
        let devices = DEVICES.read().expect("rwlock poisoned");
        let mut id = 0;
        while devices.contains_key(&id) {
            id += 1;
        }
        Ok(id)
    }

    /// Get device name
    pub fn name(&self) -> String {
        format!("ublkb{}", self.inner.id)
    }

    /// Get device path
    pub fn path(&self) -> PathBuf {
        PathBuf::from(format!("/dev/ublkb{}", self.inner.id))
    }

    /// Get device configuration
    pub fn config(&self) -> DeviceConfig {
        self.inner.config.read().expect("rwlock poisoned").clone()
    }

    /// Get device statistics
    pub fn stats(&self) -> DeviceStats {
        let s = &self.inner.stats;
        let config = self.config();

        let bytes_processed = s.bytes_processed.load(Ordering::Relaxed);
        let entropy_count = s.entropy_count.load(Ordering::Relaxed);

        DeviceStats {
            orig_data_size: s.orig_data_size.load(Ordering::Relaxed),
            compr_data_size: s.compr_data_size.load(Ordering::Relaxed),
            mem_used_total: s.mem_used_total.load(Ordering::Relaxed),
            mem_limit: config.mem_limit.unwrap_or(0),
            mem_used_max: s.mem_used_max.load(Ordering::Relaxed),
            same_pages: s.same_pages.load(Ordering::Relaxed),
            pages_compacted: s.pages_compacted.load(Ordering::Relaxed),
            huge_pages: s.huge_pages.load(Ordering::Relaxed),
            huge_pages_since: s.huge_pages_since.load(Ordering::Relaxed),
            failed_reads: s.failed_reads.load(Ordering::Relaxed),
            failed_writes: s.failed_writes.load(Ordering::Relaxed),
            invalid_io: s.invalid_io.load(Ordering::Relaxed),
            notify_free: s.notify_free.load(Ordering::Relaxed),
            bd_count: s.bd_count.load(Ordering::Relaxed),
            bd_reads: s.bd_reads.load(Ordering::Relaxed),
            bd_writes: s.bd_writes.load(Ordering::Relaxed),
            gpu_pages: s.gpu_pages.load(Ordering::Relaxed),
            simd_pages: s.simd_pages.load(Ordering::Relaxed),
            scalar_pages: s.scalar_pages.load(Ordering::Relaxed),
            throughput_gbps: if bytes_processed > 0 {
                // Estimate based on recent throughput
                (bytes_processed as f64) / 1e9
            } else {
                0.0
            },
            avg_entropy: if entropy_count > 0 {
                (s.entropy_sum.load(Ordering::Relaxed) as f64) / (entropy_count as f64) / 100.0
            } else {
                0.0
            },
            simd_backend: detect_simd_backend(),
        }
    }

    /// Get mountpoint if device is mounted
    pub fn mountpoint(&self) -> Option<String> {
        let path = self.path();
        let mounts = fs::read_to_string("/proc/mounts").ok()?;

        for line in mounts.lines() {
            let parts: Vec<&str> = line.split_whitespace().collect();
            if parts.len() >= 2 && parts[0] == path.to_string_lossy() {
                return Some(parts[1].to_string());
            }
        }
        None
    }

    /// Remove the device
    pub fn remove(&self) -> Result<()> {
        Self::stop_ublk_daemon(self.inner.id)?;

        let mut devices = DEVICES.write().expect("rwlock poisoned");
        devices.remove(&self.inner.id);

        Ok(())
    }

    /// Trigger memory compaction
    pub fn compact(&self) -> Result<()> {
        // In a real implementation, this would trigger compaction
        self.inner.stats.pages_compacted.fetch_add(1, Ordering::Relaxed);
        Ok(())
    }

    /// Mark pages as idle
    pub fn mark_idle(&self) -> Result<()> {
        // In a real implementation, this would mark pages as idle
        Ok(())
    }

    /// Trigger writeback to backing device
    pub fn writeback(&self, idle: bool, huge: bool, all: bool) -> Result<()> {
        let config = self.config();
        if config.backing_dev.is_none() {
            anyhow::bail!("No backing device configured");
        }

        // In a real implementation, this would trigger writeback
        let _ = (idle, huge, all);
        Ok(())
    }

    /// Set memory limit
    pub fn set_mem_limit(&self, limit: u64) -> Result<()> {
        let mut config = self.inner.config.write().expect("rwlock poisoned");
        config.mem_limit = Some(limit);
        Ok(())
    }

    /// Set GPU enabled/disabled
    pub fn set_gpu_enabled(&self, enabled: bool) -> Result<()> {
        let mut config = self.inner.config.write().expect("rwlock poisoned");
        config.gpu_enabled = enabled;

        // Reinitialize compressor with new settings
        let compressor = CompressorBuilder::new().algorithm(config.algorithm).build()?;
        *self.inner.compressor.write().expect("rwlock poisoned") = Some(compressor);

        Ok(())
    }

    /// Set entropy skip threshold
    pub fn set_entropy_threshold(&self, threshold: f64) -> Result<()> {
        let mut config = self.inner.config.write().expect("rwlock poisoned");
        config.entropy_skip_threshold = threshold;
        Ok(())
    }

    /// Set writeback limit
    pub fn set_writeback_limit(&self, limit: u64) -> Result<()> {
        let mut config = self.inner.config.write().expect("rwlock poisoned");
        config.writeback_limit = Some(limit);
        Ok(())
    }

    /// Enable/disable writeback limit
    pub fn set_writeback_limit_enabled(&self, enabled: bool) -> Result<()> {
        let mut config = self.inner.config.write().expect("rwlock poisoned");
        if !enabled {
            config.writeback_limit = None;
        }
        Ok(())
    }

    /// Reset mem_used_max watermark
    pub fn reset_mem_used_max(&self) -> Result<()> {
        let current = self.inner.stats.mem_used_total.load(Ordering::Relaxed);
        self.inner.stats.mem_used_max.store(current, Ordering::Relaxed);
        Ok(())
    }

    // Private helpers

    fn parse_device_id(path: &Path) -> Result<u32> {
        let name = path.file_name().and_then(|n| n.to_str()).context("Invalid device path")?;

        if let Some(id_str) = name.strip_prefix("ublkb") {
            id_str.parse().context("Invalid device ID in path")
        } else {
            anyhow::bail!("Not a ublk device: {}", path.display())
        }
    }

    #[cfg(not(test))]
    fn start_ublk_daemon(id: u32, config: &DeviceConfig) -> Result<()> {
        use crate::ublk::{run_daemon, run_daemon_batched, BatchedDaemonConfig, DaemonError};
        use nix::libc;
        use std::sync::atomic::AtomicBool;

        let stop = Arc::new(AtomicBool::new(false));
        let stop_clone = stop.clone();

        // Setup signal handler for graceful shutdown
        ctrlc::set_handler(move || {
            stop_clone.store(true, std::sync::atomic::Ordering::Relaxed);
        })
        .ok();

        // Helper to run daemon with batched or non-batched mode
        let run_with_mode =
            |stop: Arc<AtomicBool>, ready_fd: Option<i32>| -> Result<(), DaemonError> {
                if config.batched {
                    // Batched mode: high-throughput compression (>10 GB/s)
                    tracing::info!(
                        "Starting daemon in batched mode (threshold={}, flush={}ms, perf={})",
                        config.batch_threshold,
                        config.flush_timeout_ms,
                        config.perf.is_some()
                    );
                    let batch_config = BatchedDaemonConfig {
                        dev_id: id as i32,
                        dev_size: config.size,
                        algorithm: config.algorithm,
                        batch_threshold: config.batch_threshold,
                        flush_timeout_ms: config.flush_timeout_ms,
                        gpu_batch_size: config.gpu_batch_size,
                        perf: config.perf.clone(), // PERF-001: Pass performance config
                        nr_hw_queues: config.nr_hw_queues, // PERF-003: Multi-queue
                        zero_copy: config.zero_copy, // PERF-006: Zero-copy mode
                        // KERN-001/002/003: Kernel-Cooperative Tiered Storage
                        backend: config.backend,
                        entropy_routing: config.entropy_routing,
                        zram_device: config.zram_device.clone(),
                        cold_tier: config.cold_tier.clone(), // KERN-003: NVMe cold tier
                        entropy_kernel_threshold: config.entropy_kernel_threshold,
                        entropy_skip_threshold: config.entropy_skip_threshold,
                    };
                    run_daemon_batched(batch_config, stop, ready_fd)
                } else {
                    // Standard mode: per-page compression
                    run_daemon(id as i32, config.size, config.algorithm, stop, ready_fd)
                }
            };

        // Foreground mode - run directly
        if config.foreground {
            tracing::info!("Running in foreground mode");
            // DT-007e: Lock memory in the daemon process (this IS the daemon)
            Self::lock_daemon_memory();
            return run_with_mode(stop, None).map_err(|e| anyhow::anyhow!("Daemon failed: {}", e));
        }

        // Create eventfd for synchronization
        // SAFETY: eventfd creates a file descriptor for event notification.
        // Return value is checked below for errors.
        let efd = unsafe { libc::eventfd(0, 0) };
        if efd < 0 {
            anyhow::bail!("Failed to create eventfd");
        }

        // Fork for daemon mode
        // SAFETY: fork() creates a new child process. Return value is checked
        // to handle error (-1), child (0), and parent (>0) cases.
        match unsafe { libc::fork() } {
            -1 => {
                // SAFETY: efd is a valid file descriptor from eventfd() above
                unsafe { libc::close(efd) };
                anyhow::bail!("Fork failed");
            }
            0 => {
                // Child process - run daemon
                // SAFETY: setsid() creates a new session with the calling process as leader.
                // Called in child process after fork to detach from controlling terminal.
                unsafe { libc::setsid() };

                // DT-007e: Lock memory in child process AFTER fork
                // Five Whys: mlock only affects calling process, not forked children
                Self::lock_daemon_memory();

                // Daemon will signal readiness via efd
                match run_with_mode(stop, Some(efd)) {
                    Ok(()) | Err(DaemonError::Stopped) => {
                        // SAFETY: efd is a valid file descriptor inherited from parent
                        unsafe { libc::close(efd) };
                        std::process::exit(0);
                    }
                    Err(e) => {
                        tracing::error!("Daemon failed: {}", e);
                        // SAFETY: efd is a valid file descriptor inherited from parent.
                        // Writing 0 signals failure to parent process waiting on read().
                        unsafe {
                            // Signal failure with 0
                            libc::write(efd, &0u64 as *const u64 as *const libc::c_void, 8);
                            libc::close(efd);
                        }
                        std::process::exit(1);
                    }
                }
            }
            _pid => {
                // Parent process - wait for device ID
                let mut val: u64 = 0;
                // SAFETY: efd is a valid eventfd, val is a properly aligned u64.
                // read() blocks until child writes success (device_id+1) or failure (0).
                let n = unsafe { libc::read(efd, &mut val as *mut u64 as *mut libc::c_void, 8) };
                // SAFETY: efd is a valid file descriptor, close releases it
                unsafe { libc::close(efd) };

                if n == 8 && val > 0 {
                    tracing::info!("Device created with ID: {}", (val - 1) as i64);
                    Ok(())
                } else {
                    anyhow::bail!("Failed to read device ID from daemon")
                }
            }
        }
    }

    fn stop_ublk_daemon(_id: u32) -> Result<()> {
        // Device cleanup is handled by UblkCtrl::drop
        Ok(())
    }
}

/// Detect the best available SIMD backend
fn detect_simd_backend() -> String {
    #[cfg(target_arch = "x86_64")]
    {
        if std::arch::is_x86_feature_detected!("avx512f") {
            return "avx512".to_string();
        }
        if std::arch::is_x86_feature_detected!("avx2") {
            return "avx2".to_string();
        }
        if std::arch::is_x86_feature_detected!("sse4.2") {
            return "sse4.2".to_string();
        }
    }

    #[cfg(target_arch = "aarch64")]
    {
        return "neon".to_string();
    }

    "scalar".to_string()
}

// ============================================================================
// BlockDevice - Pure Rust block device abstraction for testing
// ============================================================================

/// Statistics for BlockDevice
#[derive(Clone, Debug, Default)]
pub struct BlockDeviceStats {
    pub pages_stored: u64,
    pub bytes_written: u64,
    pub bytes_read: u64,
    pub bytes_compressed: u64,
    pub zero_pages: u64,
    pub gpu_pages: u64,
    pub simd_pages: u64,
    pub scalar_pages: u64,
}

impl BlockDeviceStats {
    /// Calculate compression ratio (original / compressed)
    pub fn compression_ratio(&self) -> f64 {
        if self.bytes_compressed > 0 {
            self.bytes_written as f64 / self.bytes_compressed as f64
        } else {
            1.0
        }
    }
}

/// Pure Rust block device abstraction for testing without kernel dependencies.
///
/// This provides a simple in-memory block device that compresses pages using
/// trueno-zram-core. It can be used for:
/// - Unit testing compression roundtrips
/// - Benchmarking throughput
/// - Validating correctness before kernel integration
///
/// # Example
///
/// ```ignore
/// use trueno_zram_core::{Algorithm, CompressorBuilder};
/// use trueno_ublk::device::BlockDevice;
///
/// let compressor = CompressorBuilder::new()
///     .algorithm(Algorithm::Lz4)
///     .build()
///     .unwrap();
///
/// let mut device = BlockDevice::new(1 << 30, compressor); // 1GB
///
/// // Write data
/// let data = vec![0xAB; 4096];
/// device.write(0, &data).unwrap();
///
/// // Read back
/// let mut buf = vec![0u8; 4096];
/// device.read(0, &mut buf).unwrap();
/// assert_eq!(data, buf);
/// ```
pub struct BlockDevice {
    store: PageStore,
    size: u64,
    block_size: u32,
    bytes_written: AtomicU64,
    bytes_read: AtomicU64,
}

impl BlockDevice {
    /// Create a new block device with the given size and compressor.
    ///
    /// # Arguments
    /// * `size` - Total device size in bytes
    /// * `compressor` - The compressor to use for page compression
    pub fn new(size: u64, compressor: Box<dyn PageCompressor>) -> Self {
        Self {
            store: PageStore::with_compressor(Arc::from(compressor), 7.5),
            size,
            block_size: PAGE_SIZE as u32,
            bytes_written: AtomicU64::new(0),
            bytes_read: AtomicU64::new(0),
        }
    }

    /// Create a new block device with custom entropy threshold.
    pub fn with_entropy_threshold(
        size: u64,
        compressor: Box<dyn PageCompressor>,
        entropy_threshold: f64,
    ) -> Self {
        Self {
            store: PageStore::with_compressor(Arc::from(compressor), entropy_threshold),
            size,
            block_size: PAGE_SIZE as u32,
            bytes_written: AtomicU64::new(0),
            bytes_read: AtomicU64::new(0),
        }
    }

    /// Get the device size in bytes.
    pub fn size(&self) -> u64 {
        self.size
    }

    /// Get the block size in bytes.
    pub fn block_size(&self) -> u32 {
        self.block_size
    }

    /// Read data from the device at the given byte offset.
    ///
    /// # Arguments
    /// * `offset` - Byte offset to read from (must be page-aligned)
    /// * `buf` - Buffer to read into (length must be multiple of PAGE_SIZE)
    pub fn read(&self, offset: u64, buf: &mut [u8]) -> Result<()> {
        self.validate_io(offset, buf.len())?;

        let pages = buf.len() / PAGE_SIZE;
        for i in 0..pages {
            let sector = self.offset_to_sector(offset + (i * PAGE_SIZE) as u64);
            let page_buf = &mut buf[i * PAGE_SIZE..(i + 1) * PAGE_SIZE];
            self.store.load(sector, page_buf)?;
        }

        self.bytes_read.fetch_add(buf.len() as u64, Ordering::Relaxed);
        Ok(())
    }

    /// Write data to the device at the given byte offset.
    ///
    /// # Arguments
    /// * `offset` - Byte offset to write to (must be page-aligned)
    /// * `data` - Data to write (length must be multiple of PAGE_SIZE)
    pub fn write(&mut self, offset: u64, data: &[u8]) -> Result<()> {
        self.validate_io(offset, data.len())?;

        let pages = data.len() / PAGE_SIZE;
        for i in 0..pages {
            let sector = self.offset_to_sector(offset + (i * PAGE_SIZE) as u64);
            let page_data = &data[i * PAGE_SIZE..(i + 1) * PAGE_SIZE];
            self.store.store(sector, page_data)?;
        }

        self.bytes_written.fetch_add(data.len() as u64, Ordering::Relaxed);
        Ok(())
    }

    /// Discard data at the given byte offset.
    ///
    /// # Arguments
    /// * `offset` - Byte offset to discard from (must be page-aligned)
    /// * `len` - Length to discard in bytes (must be multiple of PAGE_SIZE)
    pub fn discard(&mut self, offset: u64, len: u64) -> Result<()> {
        self.validate_io(offset, len as usize)?;

        let pages = len as usize / PAGE_SIZE;
        for i in 0..pages {
            let sector = self.offset_to_sector(offset + (i * PAGE_SIZE) as u64);
            self.store.remove(sector);
        }

        Ok(())
    }

    /// Sync all data to storage (no-op for in-memory device).
    pub fn sync(&self) -> Result<()> {
        Ok(())
    }

    /// Get device statistics.
    pub fn stats(&self) -> BlockDeviceStats {
        let store_stats = self.store.stats();
        BlockDeviceStats {
            pages_stored: store_stats.pages_stored,
            bytes_written: self.bytes_written.load(Ordering::Relaxed),
            bytes_read: self.bytes_read.load(Ordering::Relaxed),
            bytes_compressed: store_stats.bytes_compressed,
            zero_pages: store_stats.zero_pages,
            gpu_pages: store_stats.gpu_pages,
            simd_pages: store_stats.simd_pages,
            scalar_pages: store_stats.scalar_pages,
        }
    }

    // Private helpers

    fn validate_io(&self, offset: u64, len: usize) -> Result<()> {
        if offset % PAGE_SIZE as u64 != 0 {
            anyhow::bail!("Offset {} is not page-aligned (PAGE_SIZE={})", offset, PAGE_SIZE);
        }
        if len % PAGE_SIZE != 0 {
            anyhow::bail!("Length {} is not a multiple of PAGE_SIZE ({})", len, PAGE_SIZE);
        }
        if offset + len as u64 > self.size {
            anyhow::bail!(
                "I/O extends beyond device size: offset={}, len={}, size={}",
                offset,
                len,
                self.size
            );
        }
        Ok(())
    }

    fn offset_to_sector(&self, offset: u64) -> u64 {
        // Sector size is 512 bytes, so divide by 512 to get sector number
        offset / 512
    }
}

#[cfg(test)]
mod tests;
