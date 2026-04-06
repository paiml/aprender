//! PERF-003: Multi-Queue Daemon Implementation
//!
//! Enables parallel I/O processing with multiple hardware queues.
//! Each queue has its own io_uring instance and runs in a dedicated thread.
//!
//! ## Architecture
//!
//! ```text
//! [Block I/O] → [Queue 0] → [io_uring 0] → [Thread 0] → [Compress] → [Reply]
//!             → [Queue 1] → [io_uring 1] → [Thread 1] → [Compress] → [Reply]
//!             → [Queue N] → [io_uring N] → [Thread N] → [Compress] → [Reply]
//! ```
//!
//! ## Performance Target
//!
//! - 1 queue: ~162K IOPS (baseline)
//! - 4 queues: ~500K IOPS (3.5x)
//! - 8 queues: ~800K IOPS (6x)
//!
//! ## 10X Optimization Stack (PERF-005 through PERF-012)
//!
//! When TenXConfig is provided, enables:
//! - SQPOLL mode for zero-syscall submissions
//! - Registered buffers for reduced kernel mapping
//! - Fixed file descriptors for faster fd lookup

use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;
use std::thread::JoinHandle;

use libc::{mmap, munmap, MAP_ANONYMOUS, MAP_PRIVATE, MAP_SHARED, PROT_READ, PROT_WRITE};
use nix::libc;
use std::ptr::null_mut;

use crate::perf::tenx::{
    FixedFileRegistry, RegisteredBufferConfig, RegisteredBufferPool, SqpollConfig,
};

/// UBLK kernel constants for IOD buffer calculation
const UBLK_MAX_QUEUE_DEPTH: u32 = 4096;
const UBLKSRV_CMD_BUF_OFFSET: i64 = 0;
const IOD_ENTRY_SIZE: usize = std::mem::size_of::<crate::ublk::sys::UblkIoDesc>();

/// Calculate the page-aligned size for a command buffer
fn cmd_buf_sz(depth: u32) -> usize {
    let size = (depth as usize) * IOD_ENTRY_SIZE;
    let page_sz = unsafe { libc::sysconf(libc::_SC_PAGESIZE) } as usize;
    (size + page_sz - 1) & !(page_sz - 1)
}

/// Wrapper for raw pointers that can be sent across threads.
///
/// # Safety
/// The caller must ensure the underlying memory is valid for the lifetime of all threads.
#[derive(Clone, Copy)]
pub struct SendPtr(*mut u8);

unsafe impl Send for SendPtr {}
unsafe impl Sync for SendPtr {}

impl SendPtr {
    pub fn new(ptr: *mut u8) -> Self {
        Self(ptr)
    }

    pub fn as_ptr(self) -> *mut u8 {
        self.0
    }
}

/// Configuration for a single queue worker
#[derive(Debug, Clone)]
pub struct QueueWorkerConfig {
    /// Queue ID (0 to nr_hw_queues-1)
    pub queue_id: u16,
    /// Queue depth (tags per queue)
    pub queue_depth: u16,
    /// Maximum I/O buffer size
    pub max_io_size: u32,
    /// CPU core to pin this worker to (None for no pinning)
    pub cpu_core: Option<usize>,
    /// IOD buffer offset for this queue
    pub iod_offset: usize,
    /// Data buffer offset for this queue
    pub data_offset: usize,
}

impl QueueWorkerConfig {
    /// Calculate configs for all queues
    ///
    /// Note: iod_offset and data_offset are offsets into SEPARATE buffers
    /// (iod_buf and data_buf are allocated independently)
    pub fn for_all_queues(
        nr_hw_queues: u16,
        queue_depth: u16,
        max_io_size: u32,
        cpu_cores: &[usize],
    ) -> Vec<Self> {
        let iod_entry_size = std::mem::size_of::<crate::ublk::sys::UblkIoDesc>();
        let iod_per_queue = (queue_depth as usize) * iod_entry_size;
        let data_per_queue = (queue_depth as usize) * (max_io_size as usize);

        (0..nr_hw_queues)
            .map(|q| {
                let q_usize = q as usize;
                QueueWorkerConfig {
                    queue_id: q,
                    queue_depth,
                    max_io_size,
                    cpu_core: cpu_cores.get(q_usize).copied(),
                    // PERF-003 fix: offsets are into SEPARATE buffers
                    iod_offset: q_usize * iod_per_queue,
                    data_offset: q_usize * data_per_queue,
                }
            })
            .collect()
    }
}

/// Statistics for a single queue
#[derive(Debug, Default)]
pub struct QueueStats {
    /// Total I/Os processed
    pub ios_processed: AtomicU64,
    /// Read operations
    pub reads: AtomicU64,
    /// Write operations
    pub writes: AtomicU64,
    /// Errors encountered
    pub errors: AtomicU64,
}

impl QueueStats {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn record_read(&self) {
        self.ios_processed.fetch_add(1, Ordering::Relaxed);
        self.reads.fetch_add(1, Ordering::Relaxed);
    }

    pub fn record_write(&self) {
        self.ios_processed.fetch_add(1, Ordering::Relaxed);
        self.writes.fetch_add(1, Ordering::Relaxed);
    }

    pub fn record_error(&self) {
        self.errors.fetch_add(1, Ordering::Relaxed);
    }

    pub fn total_ios(&self) -> u64 {
        self.ios_processed.load(Ordering::Relaxed)
    }
}

/// Aggregated statistics across all queues
#[derive(Debug, Default)]
pub struct MultiQueueStats {
    /// Per-queue statistics
    pub queues: Vec<Arc<QueueStats>>,
}

impl MultiQueueStats {
    pub fn new(nr_queues: usize) -> Self {
        Self { queues: (0..nr_queues).map(|_| Arc::new(QueueStats::new())).collect() }
    }

    pub fn total_ios(&self) -> u64 {
        self.queues.iter().map(|q| q.total_ios()).sum()
    }

    pub fn total_reads(&self) -> u64 {
        self.queues.iter().map(|q| q.reads.load(Ordering::Relaxed)).sum()
    }

    pub fn total_writes(&self) -> u64 {
        self.queues.iter().map(|q| q.writes.load(Ordering::Relaxed)).sum()
    }

    pub fn total_errors(&self) -> u64 {
        self.queues.iter().map(|q| q.errors.load(Ordering::Relaxed)).sum()
    }
}

/// Handle for a running queue worker thread
pub struct QueueWorkerHandle {
    /// Queue ID
    pub queue_id: u16,
    /// Thread join handle
    pub thread: JoinHandle<()>,
    /// Statistics for this queue
    pub stats: Arc<QueueStats>,
}

/// Multi-queue daemon coordinator
///
/// Spawns and manages multiple queue worker threads, each with its own
/// io_uring instance for parallel I/O processing.
pub struct MultiQueueDaemon {
    /// Number of hardware queues
    pub nr_hw_queues: u16,
    /// Shutdown signal
    pub stop: Arc<AtomicBool>,
    /// Per-queue worker handles
    pub workers: Vec<QueueWorkerHandle>,
    /// Aggregated statistics
    pub stats: MultiQueueStats,
}

impl MultiQueueDaemon {
    /// Check if multi-queue mode should be used
    pub fn should_use_multi_queue(nr_hw_queues: u16) -> bool {
        nr_hw_queues > 1
    }

    /// Calculate total buffer size needed for all queues
    pub fn total_buffer_size(nr_hw_queues: u16, queue_depth: u16, max_io_size: u32) -> usize {
        let iod_entry_size = std::mem::size_of::<crate::ublk::sys::UblkIoDesc>();
        let per_queue = (queue_depth as usize) * iod_entry_size
            + (queue_depth as usize) * (max_io_size as usize);
        (nr_hw_queues as usize) * per_queue
    }

    /// Get optimal number of queues based on system
    pub fn optimal_queue_count() -> u16 {
        // Use number of CPUs, capped at 8
        let cpus = std::thread::available_parallelism().map(|p| p.get()).unwrap_or(1);
        (cpus as u16).clamp(1, 8)
    }
}

/// Pin current thread to specified CPU core
#[cfg(not(test))]
pub fn pin_to_cpu(core: usize) -> std::io::Result<()> {
    use nix::sched::{sched_setaffinity, CpuSet};
    use nix::unistd::Pid;

    let mut cpuset = CpuSet::new();
    cpuset.set(core).map_err(std::io::Error::other)?;
    sched_setaffinity(Pid::from_raw(0), &cpuset).map_err(std::io::Error::other)
}

// ============================================================================
// Per-Queue I/O Worker (PERF-003 Phase 3)
// ============================================================================

/// Per-queue I/O worker that handles a single ublk queue
///
/// Each worker has:
/// - Its own io_uring instance
/// - A view into its portion of the shared mmap'd buffers
/// - Shared access to the BatchedPageStore
/// - Optional registered buffer pool (PERF-005)
#[cfg(not(test))]
pub struct QueueIoWorker {
    /// Queue ID (0 to nr_hw_queues-1)
    pub queue_id: u16,
    /// Queue depth (tags per queue)
    pub queue_depth: u16,
    /// Maximum I/O size
    pub max_io_size: u32,
    /// Per-queue io_uring instance
    ring: io_uring::IoUring,
    /// Pointer to this queue's IOD buffer region (mmap'd by this worker)
    iod_buf: *mut u8,
    /// Size of the IOD buffer (for munmap on Drop)
    iod_buf_size: usize,
    /// Pointer to this queue's data buffer region
    data_buf: *mut u8,
    /// Character device fd (shared across all queues)
    char_fd: i32,
    /// Shutdown signal
    stop: Arc<AtomicBool>,
    /// Per-queue statistics
    pub stats: Arc<QueueStats>,
    /// Whether to use ioctl encoding
    use_ioctl_encode: bool,
    /// PERF-005: Registered buffer pool (owned, outlives ring registration)
    registered_buffers: Option<RegisteredBufferPool>,
    /// Whether buffers have been registered with io_uring
    buffers_registered: bool,
    /// PERF-008: Fixed file registry (owned, outlives ring registration)
    fixed_files: Option<FixedFileRegistry>,
    /// Whether files have been registered with io_uring
    files_registered: bool,
    /// PERF-007: Whether SQPOLL mode is enabled (for synchronization)
    sqpoll_enabled: bool,
    /// PERF-006: Whether ZERO_COPY mode is enabled
    zero_copy: bool,
    /// PERF-006: mmap'd IO buffer region (ZERO_COPY mode only)
    /// Points to the base of the mmap'd region at UBLKSRV_IO_BUF_OFFSET
    io_buf: *mut u8,
    /// Size of the IO buffer region (for munmap on Drop)
    io_buf_size: usize,
}

#[cfg(not(test))]
unsafe impl Send for QueueIoWorker {}

#[cfg(not(test))]
impl Drop for QueueIoWorker {
    fn drop(&mut self) {
        // Clean up the mmap'd IOD buffer
        if self.iod_buf_size > 0 && !self.iod_buf.is_null() {
            unsafe {
                munmap(self.iod_buf as *mut libc::c_void, self.iod_buf_size);
            }
            tracing::debug!(queue_id = self.queue_id, "IOD buffer unmapped");
        }
        // PERF-006: Clean up the mmap'd IO buffer (ZERO_COPY mode)
        if self.io_buf_size > 0 && !self.io_buf.is_null() {
            unsafe {
                munmap(self.io_buf as *mut libc::c_void, self.io_buf_size);
            }
            tracing::debug!(queue_id = self.queue_id, "IO buffer unmapped (ZERO_COPY)");
        }
    }
}

#[cfg(not(test))]
impl QueueIoWorker {
    /// Create a new queue worker
    ///
    /// # Safety
    /// The iod_buf and data_buf pointers must be valid for this queue's region
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn new(
        queue_id: u16,
        queue_depth: u16,
        max_io_size: u32,
        char_fd: i32,
        iod_buf: *mut u8,
        data_buf: *mut u8,
        stop: Arc<AtomicBool>,
        use_ioctl_encode: bool,
    ) -> Result<Self, super::daemon::DaemonError> {
        Self::new_with_sqpoll(
            queue_id,
            queue_depth,
            max_io_size,
            char_fd,
            iod_buf,
            data_buf,
            stop,
            use_ioctl_encode,
            None,
            false, // PERF-006: Default to USER_COPY mode
        )
    }

    /// Create a new queue worker with optional SQPOLL configuration (PERF-007)
    ///
    /// # Safety
    /// The data_buf pointer must be valid for this queue's region.
    /// The IOD buffer will be mmap'd internally by this worker with the correct per-queue offset.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn new_with_sqpoll(
        queue_id: u16,
        queue_depth: u16,
        max_io_size: u32,
        char_fd: i32,
        _iod_buf: *mut u8, // Ignored - we mmap our own
        data_buf: *mut u8,
        stop: Arc<AtomicBool>,
        use_ioctl_encode: bool,
        sqpoll_config: Option<&SqpollConfig>,
        zero_copy: bool, // PERF-006: Enable ZERO_COPY mode
    ) -> Result<Self, super::daemon::DaemonError> {
        // PERF-003 FIX: Each queue must mmap its own IOD buffer with correct offset
        // The kernel expects: offset = UBLKSRV_CMD_BUF_OFFSET + queue_id * max_cmd_buf_sz
        // where max_cmd_buf_sz is based on UBLK_MAX_QUEUE_DEPTH, not actual queue_depth
        let max_cmd_buf_sz = cmd_buf_sz(UBLK_MAX_QUEUE_DEPTH) as i64;
        let iod_buf_size = cmd_buf_sz(queue_depth as u32);
        let offset = UBLKSRV_CMD_BUF_OFFSET + (queue_id as i64) * max_cmd_buf_sz;

        tracing::debug!(queue_id, iod_buf_size, offset, "PERF-003: mmap'ing IOD buffer for queue");

        let iod_buf = mmap(null_mut(), iod_buf_size, PROT_READ, MAP_SHARED, char_fd, offset);

        if iod_buf == libc::MAP_FAILED {
            let err = std::io::Error::last_os_error();
            tracing::error!(queue_id, offset, size = iod_buf_size, "IOD mmap failed: {}", err);
            return Err(super::daemon::DaemonError::Mmap(err));
        }
        let iod_buf = iod_buf as *mut u8;
        tracing::debug!(queue_id, "IOD buffer mmap'd successfully");

        // Create per-queue io_uring with peak performance optimizations
        // PERF-004: io_uring tuning for maximum IOPS
        let mut builder = io_uring::IoUring::builder();

        // Check if SQPOLL is requested
        let sqpoll_requested = sqpoll_config.is_some_and(|cfg| cfg.enabled);

        // PERF-007: Apply SQPOLL configuration if enabled
        // NOTE: SINGLE_ISSUER and COOP_TASKRUN are incompatible with SQPOLL because:
        // - SINGLE_ISSUER requires all submissions from one task context
        // - SQPOLL uses a kernel thread which is a different task
        // - COOP_TASKRUN requires SINGLE_ISSUER
        if let Some(cfg) = sqpoll_config.filter(|c| c.enabled) {
            builder.setup_sqpoll(cfg.idle_timeout_ms);
            if cfg.cpu >= 0 {
                builder.setup_sqpoll_cpu(cfg.cpu as u32);
            }
            tracing::info!(
                queue_id,
                idle_ms = cfg.idle_timeout_ms,
                "PERF-007: SQPOLL mode enabled for queue"
            );
        } else {
            // SINGLE_ISSUER: Only one thread submits to this ring (per-queue threading)
            // Only valid when NOT using SQPOLL
            builder.setup_single_issuer();
            // COOP_TASKRUN: Cooperative task running reduces kernel overhead
            // NOTE: COOP_TASKRUN requires SINGLE_ISSUER, so only enable when not using SQPOLL
            builder.setup_coop_taskrun();
        }

        // Larger CQ to handle burst completions without overflow
        builder.setup_cqsize(queue_depth as u32 * 4);

        let ring = match builder.build(queue_depth as u32 * 2) {
            Ok(r) => r,
            Err(e) => {
                // Clean up IOD mmap on io_uring creation failure
                munmap(iod_buf as *mut libc::c_void, iod_buf_size);
                return Err(super::daemon::DaemonError::IoUringCreate(e));
            }
        };

        // Track if SQPOLL is enabled for synchronization (PERF-007 race fix)
        let sqpoll_enabled = sqpoll_requested;

        // PERF-006: mmap IO buffer region for ZERO_COPY mode
        // In ZERO_COPY mode, we access kernel buffers directly instead of using pread/pwrite
        let (io_buf, io_buf_size) = if zero_copy {
            // Calculate IO buffer size for this queue
            // Each queue needs: queue_depth * max_io_size bytes
            let buf_size = (queue_depth as usize) * (max_io_size as usize);
            // IO buffer offset for this queue
            // UBLKSRV_IO_BUF_OFFSET is the base, then each queue gets a slice
            let io_offset = crate::ublk::sys::UBLKSRV_IO_BUF_OFFSET as i64
                + (queue_id as i64) * (buf_size as i64);

            tracing::info!(
                queue_id,
                buf_size,
                io_offset,
                "PERF-006: mmap'ing IO buffer for ZERO_COPY mode"
            );

            let ptr = mmap(
                null_mut(),
                buf_size,
                PROT_READ | PROT_WRITE, // Need both read and write for ZERO_COPY
                MAP_SHARED,
                char_fd,
                io_offset,
            );

            if ptr == libc::MAP_FAILED {
                let err = std::io::Error::last_os_error();
                tracing::error!(
                    queue_id,
                    io_offset,
                    size = buf_size,
                    "IO buffer mmap failed (ZERO_COPY): {}",
                    err
                );
                // Clean up IOD mmap on failure
                munmap(iod_buf as *mut libc::c_void, iod_buf_size);
                return Err(super::daemon::DaemonError::Mmap(err));
            }
            tracing::info!(queue_id, "IO buffer mmap'd successfully (ZERO_COPY)");
            (ptr as *mut u8, buf_size)
        } else {
            (std::ptr::null_mut(), 0)
        };

        Ok(Self {
            queue_id,
            queue_depth,
            max_io_size,
            ring,
            iod_buf,
            iod_buf_size,
            data_buf,
            char_fd,
            stop,
            stats: Arc::new(QueueStats::new()),
            use_ioctl_encode,
            registered_buffers: None,
            buffers_registered: false,
            fixed_files: None,
            files_registered: false,
            sqpoll_enabled,
            zero_copy,
            io_buf,
            io_buf_size,
        })
    }

    /// Register buffers with io_uring for zero-copy I/O (PERF-005)
    ///
    /// This must be called after the worker is created but before processing I/O.
    /// The buffer pool will be owned by this worker and outlive the io_uring registration.
    ///
    /// # Returns
    /// - `Ok(true)` if buffers were registered successfully
    /// - `Ok(false)` if registration was skipped (already registered or config disabled)
    /// - `Err` if registration failed
    pub fn register_buffers(
        &mut self,
        config: &RegisteredBufferConfig,
    ) -> Result<bool, super::daemon::DaemonError> {
        if !config.enabled || self.buffers_registered {
            return Ok(false);
        }

        // Create buffer pool
        let pool = RegisteredBufferPool::new(config.clone()).map_err(|e| {
            super::daemon::DaemonError::IoUringCreate(std::io::Error::other(format!(
                "Failed to create buffer pool: {:?}",
                e
            )))
        })?;

        // Build iovec array for registration
        let io_slices = pool.build_io_slices();

        // Convert IoSliceMut to iovec for io_uring
        // SAFETY: IoSliceMut has the same layout as iovec
        let iovecs: Vec<libc::iovec> = io_slices
            .iter()
            .map(|s| libc::iovec { iov_base: s.as_ptr() as *mut libc::c_void, iov_len: s.len() })
            .collect();

        // Register with io_uring
        // SAFETY: The iovecs point to valid memory owned by the pool,
        // which will be stored in self.registered_buffers and outlive the registration.
        let result = unsafe { self.ring.submitter().register_buffers(&iovecs) };

        match result {
            Ok(()) => {
                tracing::info!(
                    queue_id = self.queue_id,
                    num_buffers = iovecs.len(),
                    buffer_size = config.buffer_size(),
                    "PERF-005: Registered {} buffers with io_uring",
                    iovecs.len()
                );
                self.registered_buffers = Some(pool);
                self.buffers_registered = true;
                Ok(true)
            }
            Err(e) => {
                tracing::warn!(
                    queue_id = self.queue_id,
                    error = %e,
                    "PERF-005: Buffer registration failed, continuing without"
                );
                // Don't fail - continue without registered buffers
                Ok(false)
            }
        }
    }

    /// Check if registered buffers are available
    pub fn has_registered_buffers(&self) -> bool {
        self.buffers_registered
    }

    /// Get a registered buffer by index (for use in SQEs)
    ///
    /// # Safety
    /// The returned pointer is only valid while the RegisteredBufferPool exists.
    pub fn get_registered_buffer(&self, index: usize) -> Option<*mut u8> {
        self.registered_buffers.as_ref().and_then(|pool| {
            // SAFETY: The pool is owned by this worker and will outlive any use of this pointer
            unsafe { pool.get_buffer_ptr(index).ok() }
        })
    }

    /// Register file descriptors with io_uring for faster fd lookup (PERF-008)
    ///
    /// This eliminates per-I/O fd table lookup overhead (~50ns per I/O).
    /// Must be called after worker creation but before processing I/O.
    ///
    /// # Arguments
    /// * `fds` - File descriptors to register (typically the ublk char device)
    ///
    /// # Returns
    /// - `Ok(true)` if files were registered successfully
    /// - `Ok(false)` if registration was skipped (already registered or empty)
    /// - `Err` if registration failed
    pub fn register_files(&mut self, fds: &[i32]) -> Result<bool, super::daemon::DaemonError> {
        if fds.is_empty() || self.files_registered {
            return Ok(false);
        }

        // Create fixed file registry
        let mut registry = FixedFileRegistry::new(fds.len());

        // Register each fd
        for &fd in fds {
            if registry.register(fd).is_err() {
                tracing::warn!(queue_id = self.queue_id, "PERF-008: Fixed file registry full");
                break;
            }
        }

        // Get fds with placeholders for io_uring registration
        let all_fds = registry.get_fds_with_placeholders();

        // Register with io_uring
        // Note: register_files is safe (unlike register_buffers)
        let result = self.ring.submitter().register_files(&all_fds);

        match result {
            Ok(()) => {
                registry.mark_registered();
                tracing::info!(
                    queue_id = self.queue_id,
                    num_files = fds.len(),
                    "PERF-008: Registered {} fixed files with io_uring",
                    fds.len()
                );
                self.fixed_files = Some(registry);
                self.files_registered = true;
                Ok(true)
            }
            Err(e) => {
                tracing::warn!(
                    queue_id = self.queue_id,
                    error = %e,
                    "PERF-008: Fixed file registration failed, continuing without"
                );
                // Don't fail - continue without fixed files
                Ok(false)
            }
        }
    }

    /// Check if fixed files are registered
    pub fn has_fixed_files(&self) -> bool {
        self.files_registered
    }

    /// Get fixed file index for the char device (always index 0 if registered)
    pub fn char_fd_fixed_index(&self) -> Option<u32> {
        if self.files_registered {
            Some(0) // char_fd is always registered first
        } else {
            None
        }
    }

    /// Get the fixed file registry for statistics
    pub fn fixed_files_stats(&self) -> Option<&FixedFileRegistry> {
        self.fixed_files.as_ref()
    }

    /// Get IOD for a tag in this queue
    fn get_iod(&self, tag: u16) -> &crate::ublk::sys::UblkIoDesc {
        let offset = (tag as usize) * std::mem::size_of::<crate::ublk::sys::UblkIoDesc>();
        // SAFETY: iod_buf is a valid pointer obtained from mmap during queue initialization.
        // - offset is within bounds: tag < queue_depth, and iod_buf has size queue_depth * sizeof(UblkIoDesc)
        // - The pointer is properly aligned for UblkIoDesc (guaranteed by mmap page alignment)
        // - The kernel populates the IOD structure before signaling completion
        unsafe { &*(self.iod_buf.add(offset) as *const crate::ublk::sys::UblkIoDesc) }
    }

    /// Get data buffer for a tag in this queue
    fn get_data_buf(&self, tag: u16) -> *mut u8 {
        let offset = (tag as usize) * (self.max_io_size as usize);
        // SAFETY: data_buf is a valid pointer obtained from mmap during queue initialization.
        // - offset is within bounds: tag < queue_depth, and data_buf has size queue_depth * max_io_size
        // - The kernel fills this buffer for reads and reads from it for writes
        unsafe { self.data_buf.add(offset) }
    }

    /// Submit FETCH command for a tag
    fn submit_fetch(&mut self, tag: u16) -> Result<(), super::daemon::DaemonError> {
        use crate::ublk::sys::*;
        use io_uring::{opcode, types};

        // Build UblkIoCmd struct like the daemon does
        let io_cmd = UblkIoCmd {
            q_id: self.queue_id,
            tag,
            result: -1, // libublk uses -1 for FETCH
            addr: 0,    // USER_COPY mode: kernel uses pread/pwrite for data
        };

        let cmd_op = if self.use_ioctl_encode { UBLK_U_IO_FETCH_REQ } else { UBLK_IO_FETCH_REQ };

        // Use UringCmd16 like the daemon - 16-byte cmd field for UblkIoCmd
        // SAFETY: UblkIoCmd is a repr(C) struct that fits in 16 bytes
        let io_cmd_bytes: [u8; 16] = unsafe { std::mem::transmute(io_cmd) };
        let sqe = opcode::UringCmd16::new(types::Fd(self.char_fd), cmd_op)
            .cmd(io_cmd_bytes)
            .build()
            .user_data(tag as u64);

        // SAFETY: The submission queue is properly initialized via IoUring::new().
        // The sqe contains valid data and the ring has capacity (checked by map_err).
        unsafe {
            self.ring.submission().push(&sqe).map_err(|_| super::daemon::DaemonError::QueueFull)?;
        }
        Ok(())
    }

    /// Submit COMMIT_AND_FETCH command
    fn submit_commit_and_fetch(
        &mut self,
        tag: u16,
        result: i32,
    ) -> Result<(), super::daemon::DaemonError> {
        use crate::ublk::sys::*;
        use io_uring::{opcode, types};

        // Build UblkIoCmd struct with result
        let io_cmd = UblkIoCmd {
            q_id: self.queue_id,
            tag,
            result,
            addr: 0, // USER_COPY mode
        };

        let cmd_op = if self.use_ioctl_encode {
            UBLK_U_IO_COMMIT_AND_FETCH_REQ
        } else {
            UBLK_IO_COMMIT_AND_FETCH_REQ
        };

        // Use UringCmd16 like the daemon
        // SAFETY: UblkIoCmd is a repr(C) struct that fits in 16 bytes
        let io_cmd_bytes: [u8; 16] = unsafe { std::mem::transmute(io_cmd) };
        let sqe = opcode::UringCmd16::new(types::Fd(self.char_fd), cmd_op)
            .cmd(io_cmd_bytes)
            .build()
            .user_data(tag as u64);

        // SAFETY: The submission queue is properly initialized via IoUring::new().
        // The sqe contains valid data and the ring has capacity (checked by map_err).
        unsafe {
            self.ring.submission().push(&sqe).map_err(|_| super::daemon::DaemonError::QueueFull)?;
        }
        Ok(())
    }

    /// Run the I/O loop for this queue
    ///
    /// This processes I/O requests until stop signal is received.
    pub fn run_io_loop(
        &mut self,
        store: &Arc<crate::daemon::BatchedPageStore>,
    ) -> Result<u64, super::daemon::DaemonError> {
        use crate::ublk::sys::{UBLK_IO_OP_READ, UBLK_IO_OP_WRITE};
        use tracing::{debug, info, warn};

        info!(queue_id = self.queue_id, "Queue worker starting I/O loop");

        // Submit initial FETCH commands for this queue
        for tag in 0..self.queue_depth {
            self.submit_fetch(tag)?;
        }
        self.ring.submission().sync();
        self.ring.submit().map_err(super::daemon::DaemonError::Submit)?;

        // PERF-007 FIX: With SQPOLL, wait until kernel has consumed all FETCH entries
        // This fixes the race where START_DEV is called before SQPOLL thread processes FETCHes
        if self.sqpoll_enabled {
            self.ring.submitter().squeue_wait().map_err(super::daemon::DaemonError::Submit)?;
            tracing::debug!(
                queue_id = self.queue_id,
                "PERF-007: SQPOLL sync complete - all FETCHes consumed by kernel"
            );
        }

        let mut total_ios = 0u64;

        loop {
            if self.stop.load(Ordering::Relaxed) {
                info!(queue_id = self.queue_id, total_ios, "Queue worker stopping");
                return Ok(total_ios);
            }

            // Wait for completions
            match self.ring.submit_and_wait(1) {
                Ok(_) => {}
                Err(e) if e.raw_os_error() == Some(nix::libc::EINTR) => continue,
                Err(e) => return Err(super::daemon::DaemonError::Submit(e)),
            }

            // Process completions
            let tags: Vec<(u16, i32)> = {
                let cq = self.ring.completion();
                cq.map(|cqe| (cqe.user_data() as u16, cqe.result())).collect()
            };

            for (tag, result) in tags {
                if result < 0 {
                    if result == -nix::libc::ENODEV {
                        info!(queue_id = self.queue_id, "Device stopped");
                        return Ok(total_ios);
                    }
                    self.stats.record_error();
                    self.submit_fetch(tag)?;
                    continue;
                }

                let iod = self.get_iod(tag);
                let op = (iod.op_flags & 0xff) as u8;
                let start_sector = iod.start_sector;
                let nr_sectors = iod.nr_sectors;
                let io_addr = iod.addr; // PERF-006: Buffer address for ZERO_COPY mode

                let io_result = match op {
                    UBLK_IO_OP_READ => {
                        self.stats.record_read();
                        if self.zero_copy {
                            self.handle_read_zerocopy(tag, start_sector, nr_sectors, io_addr, store)
                        } else {
                            self.handle_read(tag, start_sector, nr_sectors, store)
                        }
                    }
                    UBLK_IO_OP_WRITE => {
                        self.stats.record_write();
                        if self.zero_copy {
                            self.handle_write_zerocopy(
                                tag,
                                start_sector,
                                nr_sectors,
                                io_addr,
                                store,
                            )
                        } else {
                            self.handle_write(tag, start_sector, nr_sectors, store)
                        }
                    }
                    _ => {
                        debug!(queue_id = self.queue_id, tag, op, "Unknown op, returning success");
                        0
                    }
                };

                self.submit_commit_and_fetch(tag, io_result)?;
                total_ios += 1;
            }

            // Submit any pending commands
            if !self.ring.submission().is_empty() {
                self.ring.submit().map_err(super::daemon::DaemonError::Submit)?;
            }
        }
    }

    /// Handle read operation (USER_COPY mode)
    ///
    /// 1. Read data from store into buffer
    /// 2. pwrite buffer to char device for kernel to access
    fn handle_read(
        &self,
        tag: u16,
        start_sector: u64,
        nr_sectors: u32,
        store: &Arc<crate::daemon::BatchedPageStore>,
    ) -> i32 {
        use crate::ublk::sys::ublk_user_copy_offset;
        use nix::sys::uio::pwrite;
        use std::os::fd::BorrowedFd;

        let len = (nr_sectors as usize) * 512;
        let data_buf = self.get_data_buf(tag);
        let buf = unsafe { std::slice::from_raw_parts_mut(data_buf, len) };

        // Read from store into buffer
        match store.read(start_sector, buf) {
            Ok(n) => {
                // pwrite to char device for kernel to copy data
                let offset = ublk_user_copy_offset(self.queue_id, tag, 0);
                let fd = unsafe { BorrowedFd::borrow_raw(self.char_fd) };
                match pwrite(fd, buf, offset) {
                    Ok(_) => n as i32,
                    Err(e) => -(e as i32),
                }
            }
            Err(e) => {
                tracing::warn!(queue_id = self.queue_id, tag, "Read error: {}", e);
                -nix::libc::EIO
            }
        }
    }

    /// Handle write operation (USER_COPY mode)
    ///
    /// 1. pread data from char device into buffer
    /// 2. Write buffer to store
    fn handle_write(
        &self,
        tag: u16,
        start_sector: u64,
        nr_sectors: u32,
        store: &Arc<crate::daemon::BatchedPageStore>,
    ) -> i32 {
        use crate::ublk::sys::ublk_user_copy_offset;
        use nix::sys::uio::pread;
        use std::os::fd::BorrowedFd;

        let len = (nr_sectors as usize) * 512;
        let data_buf = self.get_data_buf(tag);
        let buf = unsafe { std::slice::from_raw_parts_mut(data_buf, len) };

        // pread from char device to get data from kernel
        let offset = ublk_user_copy_offset(self.queue_id, tag, 0);
        let fd = unsafe { BorrowedFd::borrow_raw(self.char_fd) };
        match pread(fd, buf, offset) {
            Ok(_) => {
                // Write to store
                match store.write(start_sector, buf) {
                    Ok(n) => n as i32,
                    Err(e) => {
                        tracing::warn!(queue_id = self.queue_id, tag, "Write error: {}", e);
                        -e.raw_os_error().unwrap_or(nix::libc::EIO)
                    }
                }
            }
            Err(e) => -(e as i32),
        }
    }

    /// Handle read operation (ZERO_COPY mode) - PERF-006
    ///
    /// In ZERO_COPY mode:
    /// 1. Read data from store directly into the mmap'd kernel buffer
    /// 2. No syscall needed - kernel sees data immediately
    ///
    /// # Arguments
    /// * `io_addr` - The buffer address from io_desc.addr (offset into mmap'd region)
    fn handle_read_zerocopy(
        &self,
        _tag: u16,
        start_sector: u64,
        nr_sectors: u32,
        io_addr: u64,
        store: &Arc<crate::daemon::BatchedPageStore>,
    ) -> i32 {
        let len = (nr_sectors as usize) * 512;

        // In ZERO_COPY mode, io_addr is the offset into the mmap'd IO buffer
        // We write directly to the kernel buffer - no syscall needed!
        let buf = unsafe {
            let ptr = self.io_buf.add(io_addr as usize);
            std::slice::from_raw_parts_mut(ptr, len)
        };

        // Read from store directly into kernel buffer
        match store.read(start_sector, buf) {
            Ok(n) => {
                // No pwrite needed! Kernel sees data immediately through mmap
                n as i32
            }
            Err(e) => {
                tracing::warn!(
                    queue_id = self.queue_id,
                    start_sector,
                    "ZERO_COPY read error: {}",
                    e
                );
                -nix::libc::EIO
            }
        }
    }

    /// Handle write operation (ZERO_COPY mode) - PERF-006
    ///
    /// In ZERO_COPY mode:
    /// 1. Read data directly from the mmap'd kernel buffer
    /// 2. Write to store
    /// 3. No syscall needed - data is already accessible
    ///
    /// # Arguments
    /// * `io_addr` - The buffer address from io_desc.addr (offset into mmap'd region)
    fn handle_write_zerocopy(
        &self,
        _tag: u16,
        start_sector: u64,
        nr_sectors: u32,
        io_addr: u64,
        store: &Arc<crate::daemon::BatchedPageStore>,
    ) -> i32 {
        let len = (nr_sectors as usize) * 512;

        // In ZERO_COPY mode, io_addr is the offset into the mmap'd IO buffer
        // We read directly from the kernel buffer - no syscall needed!
        let buf = unsafe {
            let ptr = self.io_buf.add(io_addr as usize);
            std::slice::from_raw_parts(ptr, len)
        };

        // Write from kernel buffer to store
        match store.write(start_sector, buf) {
            Ok(n) => n as i32,
            Err(e) => {
                tracing::warn!(
                    queue_id = self.queue_id,
                    start_sector,
                    "ZERO_COPY write error: {}",
                    e
                );
                -e.raw_os_error().unwrap_or(nix::libc::EIO)
            }
        }
    }
}

/// Spawn queue worker threads for multi-queue operation
///
/// Returns handles to all spawned workers
///
/// # Safety
/// The base_iod_buf and base_data_buf pointers must be valid for all queues
#[cfg(not(test))]
#[allow(clippy::too_many_arguments)]
pub unsafe fn spawn_queue_workers(
    nr_hw_queues: u16,
    queue_depth: u16,
    max_io_size: u32,
    char_fd: i32,
    base_iod_buf: *mut u8,
    base_data_buf: *mut u8,
    stop: Arc<AtomicBool>,
    store: Arc<crate::daemon::BatchedPageStore>,
    use_ioctl_encode: bool,
    cpu_cores: &[usize],
    zero_copy: bool, // PERF-006: Enable ZERO_COPY mode
) -> Vec<QueueWorkerHandle> {
    // Wire to new function with 10X optimizations enabled by default
    spawn_queue_workers_with_tenx(
        nr_hw_queues,
        queue_depth,
        max_io_size,
        char_fd,
        base_iod_buf,
        base_data_buf,
        stop,
        store,
        use_ioctl_encode,
        cpu_cores,
        // PERF-007 tuning: Use conservative config (SQPOLL disabled) for better baseline
        // SQPOLL adds overhead for ublk workloads due to extra kernel thread context switches
        Some(&crate::perf::TenXConfig::conservative()),
        zero_copy, // PERF-006: Pass through zero_copy setting
    )
}

/// Spawn queue workers with full 10X optimization stack (PERF-005 through PERF-012)
///
/// # Arguments
/// * `zero_copy` - PERF-006: Enable ZERO_COPY mode (explicit override, takes precedence over TenXConfig)
///
/// # Safety
/// The base_iod_buf and base_data_buf pointers must be valid for all queues
#[cfg(not(test))]
#[allow(clippy::too_many_arguments)]
pub unsafe fn spawn_queue_workers_with_tenx(
    nr_hw_queues: u16,
    queue_depth: u16,
    max_io_size: u32,
    char_fd: i32,
    base_iod_buf: *mut u8,
    base_data_buf: *mut u8,
    stop: Arc<AtomicBool>,
    store: Arc<crate::daemon::BatchedPageStore>,
    use_ioctl_encode: bool,
    cpu_cores: &[usize],
    tenx_config: Option<&crate::perf::TenXConfig>,
    zero_copy: bool, // PERF-006: Explicit ZERO_COPY override
) -> Vec<QueueWorkerHandle> {
    use tracing::info;

    let iod_entry_size = std::mem::size_of::<crate::ublk::sys::UblkIoDesc>();
    let iod_per_queue = (queue_depth as usize) * iod_entry_size;
    let data_per_queue = (queue_depth as usize) * (max_io_size as usize);

    let mut handles = Vec::with_capacity(nr_hw_queues as usize);

    for q in 0..nr_hw_queues {
        let q_usize = q as usize;

        // PERF-003 fix: offsets are into SEPARATE buffers (iod_buf and data_buf
        // are allocated independently in run_multi_queue_batched_internal)
        let iod_offset = q_usize * iod_per_queue;
        let data_offset = q_usize * data_per_queue;

        let iod_buf = SendPtr::new(unsafe { base_iod_buf.add(iod_offset) });
        let data_buf = SendPtr::new(unsafe { base_data_buf.add(data_offset) });

        let stop_clone = Arc::clone(&stop);
        let store_clone = Arc::clone(&store);
        let cpu_core = cpu_cores.get(q_usize).copied();

        let stats = Arc::new(QueueStats::new());
        let stats_clone = Arc::clone(&stats);

        // Clone TenXConfig for this thread (PERF-007)
        let sqpoll_config = tenx_config.map(|c| c.sqpoll.clone());
        let reg_buf_config = tenx_config.map(|c| c.registered_buffers.clone());
        // PERF-006: Use explicit zero_copy parameter (passed through from BatchedDaemonConfig)

        let thread = std::thread::spawn(move || {
            // Pin to CPU if specified
            if let Some(core) = cpu_core {
                if let Err(e) = pin_to_cpu(core) {
                    tracing::warn!(queue_id = q, core, "Failed to pin to CPU: {}", e);
                } else {
                    info!(queue_id = q, core, "Pinned to CPU");
                }
            }

            // Create worker with SQPOLL config (PERF-007 race fixed via squeue_wait)
            // PERF-006: Pass zero_copy flag to enable ZERO_COPY I/O path
            let worker_result = unsafe {
                QueueIoWorker::new_with_sqpoll(
                    q,
                    queue_depth,
                    max_io_size,
                    char_fd,
                    iod_buf.as_ptr(),
                    data_buf.as_ptr(),
                    stop_clone,
                    use_ioctl_encode,
                    sqpoll_config.as_ref(),
                    zero_copy,
                )
            };

            match worker_result {
                Ok(mut worker) => {
                    // Replace stats with our shared instance
                    worker.stats = stats_clone;

                    // PERF-005: Register buffers if enabled
                    if let Some(ref config) = reg_buf_config {
                        if config.enabled {
                            match worker.register_buffers(config) {
                                Ok(true) => info!(queue_id = q, "PERF-005: Registered buffers"),
                                Ok(false) => {} // Skipped
                                Err(e) => tracing::warn!(
                                    queue_id = q,
                                    "PERF-005: Buffer registration failed: {}",
                                    e
                                ),
                            }
                        }
                    }

                    match worker.run_io_loop(&store_clone) {
                        Ok(ios) => info!(queue_id = q, ios, "Queue worker completed"),
                        Err(e) => tracing::error!(queue_id = q, "Queue worker error: {}", e),
                    }
                }
                Err(e) => {
                    tracing::error!(queue_id = q, "Failed to create queue worker: {}", e);
                }
            }
        });

        handles.push(QueueWorkerHandle { queue_id: q, thread, stats });

        info!(queue_id = q, "Spawned queue worker thread");
    }

    handles
}

#[cfg(test)]
mod tests {
    use super::*;

    /// PERF-003.7: QueueWorkerConfig generates correct offsets for all queues
    #[test]
    fn test_perf003_queue_worker_config_offsets() {
        let nr_queues = 4u16;
        let queue_depth = 128u16;
        let max_io_size = 524288u32; // 512KB
        let cpu_cores = vec![0, 1, 2, 3];

        let configs =
            QueueWorkerConfig::for_all_queues(nr_queues, queue_depth, max_io_size, &cpu_cores);

        assert_eq!(configs.len(), 4);

        // Calculate expected sizes (buffers are SEPARATE, not contiguous)
        let iod_entry_size = std::mem::size_of::<crate::ublk::sys::UblkIoDesc>();
        let iod_per_queue = (queue_depth as usize) * iod_entry_size;
        let data_per_queue = (queue_depth as usize) * (max_io_size as usize);

        for (i, cfg) in configs.iter().enumerate() {
            assert_eq!(cfg.queue_id, i as u16);
            assert_eq!(cfg.cpu_core, Some(i));
            // PERF-003 fix: offsets are into SEPARATE buffers
            assert_eq!(cfg.iod_offset, i * iod_per_queue, "IOD offset for queue {}", i);
            assert_eq!(cfg.data_offset, i * data_per_queue, "Data offset for queue {}", i);

            // Verify no overlap with next queue in each SEPARATE buffer
            if i < 3 {
                let next = &configs[i + 1];
                assert!(
                    cfg.iod_offset + iod_per_queue <= next.iod_offset,
                    "Queue {} IOD overlaps with queue {} IOD",
                    i,
                    i + 1
                );
                assert!(
                    cfg.data_offset + data_per_queue <= next.data_offset,
                    "Queue {} data overlaps with queue {} data",
                    i,
                    i + 1
                );
            }
        }

        println!("PERF-003.7 VERIFIED: QueueWorkerConfig offsets correct (separate buffers)");
        println!("  IOD per queue: {} bytes", iod_per_queue);
        println!(
            "  Data per queue: {} bytes ({} MB)",
            data_per_queue,
            data_per_queue / (1024 * 1024)
        );
    }

    /// PERF-003.8: QueueStats tracks I/O operations correctly
    #[test]
    fn test_perf003_queue_stats_tracking() {
        let stats = QueueStats::new();

        // Simulate mixed workload
        for _ in 0..1000 {
            stats.record_read();
        }
        for _ in 0..500 {
            stats.record_write();
        }
        for _ in 0..10 {
            stats.record_error();
        }

        assert_eq!(stats.total_ios(), 1500);
        assert_eq!(stats.reads.load(Ordering::Relaxed), 1000);
        assert_eq!(stats.writes.load(Ordering::Relaxed), 500);
        assert_eq!(stats.errors.load(Ordering::Relaxed), 10);

        println!("PERF-003.8 VERIFIED: QueueStats tracking correct");
    }

    /// PERF-003.9: MultiQueueStats aggregates across queues
    #[test]
    fn test_perf003_multi_queue_stats_aggregation() {
        let stats = MultiQueueStats::new(4);

        // Simulate work on each queue
        for (i, q) in stats.queues.iter().enumerate() {
            for _ in 0..(i + 1) * 100 {
                q.record_read();
            }
            for _ in 0..(i + 1) * 50 {
                q.record_write();
            }
        }

        // Queue 0: 100 reads, 50 writes = 150
        // Queue 1: 200 reads, 100 writes = 300
        // Queue 2: 300 reads, 150 writes = 450
        // Queue 3: 400 reads, 200 writes = 600
        // Total: 1000 reads, 500 writes = 1500

        assert_eq!(stats.total_reads(), 1000);
        assert_eq!(stats.total_writes(), 500);
        assert_eq!(stats.total_ios(), 1500);

        println!("PERF-003.9 VERIFIED: MultiQueueStats aggregation correct");
    }

    /// PERF-003.10: MultiQueueDaemon buffer size calculation
    #[test]
    fn test_perf003_multi_queue_buffer_size_calculation() {
        let queue_depth = 128u16;
        let max_io_size = 524288u32;

        let size_1q = MultiQueueDaemon::total_buffer_size(1, queue_depth, max_io_size);
        let size_4q = MultiQueueDaemon::total_buffer_size(4, queue_depth, max_io_size);
        let size_8q = MultiQueueDaemon::total_buffer_size(8, queue_depth, max_io_size);

        assert_eq!(size_4q, 4 * size_1q);
        assert_eq!(size_8q, 8 * size_1q);

        println!("PERF-003.10 VERIFIED: Buffer size scales linearly");
        println!("  1 queue: {} MB", size_1q / (1024 * 1024));
        println!("  4 queues: {} MB", size_4q / (1024 * 1024));
        println!("  8 queues: {} MB", size_8q / (1024 * 1024));
    }

    /// PERF-003.11: should_use_multi_queue returns correct value
    #[test]
    fn test_perf003_should_use_multi_queue() {
        assert!(!MultiQueueDaemon::should_use_multi_queue(1));
        assert!(MultiQueueDaemon::should_use_multi_queue(2));
        assert!(MultiQueueDaemon::should_use_multi_queue(4));
        assert!(MultiQueueDaemon::should_use_multi_queue(8));

        println!("PERF-003.11 VERIFIED: should_use_multi_queue logic correct");
    }

    /// PERF-003.12: Concurrent queue processing simulation
    #[test]
    fn test_perf003_concurrent_queue_processing() {
        use std::sync::Arc;
        use std::thread;

        let nr_queues = 4;
        let stats = Arc::new(MultiQueueStats::new(nr_queues));
        let stop = Arc::new(AtomicBool::new(false));

        // Spawn worker threads
        let handles: Vec<_> = (0..nr_queues)
            .map(|q| {
                let stats = Arc::clone(&stats);
                let stop = Arc::clone(&stop);
                thread::spawn(move || {
                    let queue_stats = &stats.queues[q];
                    // Process 10000 IOs per queue
                    for i in 0..10000 {
                        if stop.load(Ordering::Relaxed) {
                            break;
                        }
                        if i % 2 == 0 {
                            queue_stats.record_read();
                        } else {
                            queue_stats.record_write();
                        }
                    }
                })
            })
            .collect();

        // Wait for all workers
        for h in handles {
            h.join().unwrap();
        }

        // Verify total
        assert_eq!(stats.total_ios(), 40000); // 4 queues × 10000 IOs
        assert_eq!(stats.total_reads(), 20000); // Half reads
        assert_eq!(stats.total_writes(), 20000); // Half writes

        println!("PERF-003.12 VERIFIED: Concurrent queue processing works");
        println!("  Total IOs: {}", stats.total_ios());
        println!("  Per queue: {} IOs", stats.total_ios() / nr_queues as u64);
    }

    /// PERF-003.13: Queue worker CPU affinity configuration
    #[test]
    fn test_perf003_queue_cpu_affinity_config() {
        // Test with full affinity list
        let configs = QueueWorkerConfig::for_all_queues(4, 128, 524288, &[0, 2, 4, 6]);
        assert_eq!(configs[0].cpu_core, Some(0));
        assert_eq!(configs[1].cpu_core, Some(2));
        assert_eq!(configs[2].cpu_core, Some(4));
        assert_eq!(configs[3].cpu_core, Some(6));

        // Test with partial affinity list (more queues than cores specified)
        let configs = QueueWorkerConfig::for_all_queues(4, 128, 524288, &[0, 1]);
        assert_eq!(configs[0].cpu_core, Some(0));
        assert_eq!(configs[1].cpu_core, Some(1));
        assert_eq!(configs[2].cpu_core, None); // No core specified
        assert_eq!(configs[3].cpu_core, None);

        // Test with empty affinity list
        let configs = QueueWorkerConfig::for_all_queues(4, 128, 524288, &[]);
        assert!(configs.iter().all(|c| c.cpu_core.is_none()));

        println!("PERF-003.13 VERIFIED: CPU affinity configuration correct");
    }

    /// PERF-003.14: optimal_queue_count returns valid value
    #[test]
    fn test_perf003_optimal_queue_count() {
        let optimal = MultiQueueDaemon::optimal_queue_count();

        // Must be between 1 and 8
        assert!(optimal >= 1, "optimal_queue_count must be >= 1");
        assert!(optimal <= 8, "optimal_queue_count must be <= 8");

        println!("PERF-003.14 VERIFIED: optimal_queue_count = {}", optimal);
    }
}
