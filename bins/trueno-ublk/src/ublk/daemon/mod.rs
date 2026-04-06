//! ublk daemon I/O loop
//!
//! Uses io_uring to handle block device I/O.
//!
//! ## Batched Mode
//!
//! For high-throughput scenarios, use `run_daemon_batched` which buffers pages
//! and compresses in batches using GPU when available.
//!
//! ## High-Performance Mode (PERF-001)
//!
//! Use `PerfConfig` with `run_daemon_batched` to enable:
//! - Polling mode (spin-wait on io_uring completions)
//! - CPU affinity (pin worker threads to dedicated cores)
//! - NUMA awareness (allocate buffers on local NUMA node)

#[cfg(not(test))]
use crate::daemon::PageStore;
#[cfg(not(test))]
use crate::daemon::{spawn_flush_thread, BatchConfig, BatchedPageStore, PageStoreTrait};
// KERN-001/002: Tiered storage imports
#[cfg(not(test))]
use crate::backend::BackendType;
#[cfg(not(test))]
use crate::daemon::{spawn_tiered_flush_thread, TieredConfig, TieredPageStore};
// PERF-001: Import PerfConfig unconditionally for BatchedDaemonConfig
#[cfg(not(test))]
use crate::perf::HiPerfContext;
use crate::perf::PerfConfig;
use crate::ublk::ctrl::{CtrlError, DeviceConfig};
use crate::ublk::sys::*;
#[cfg(not(test))]
use crate::ublk::UblkCtrl;
#[cfg(not(test))]
use io_uring::{opcode, types, IoUring};
use nix::libc;
#[cfg(not(test))]
use nix::libc::{mmap, munmap, MAP_ANONYMOUS, MAP_PRIVATE, PROT_READ, PROT_WRITE};
#[cfg(not(test))]
use nix::sys::uio::{pread, pwrite};
#[cfg(not(test))]
use std::os::fd::{AsRawFd, BorrowedFd};
#[cfg(not(test))]
use std::ptr::null_mut;
#[cfg(not(test))]
use std::sync::atomic::{AtomicBool, Ordering};
#[cfg(not(test))]
use std::sync::Arc;
#[cfg(not(test))]
use std::time::Duration;
use thiserror::Error;
#[cfg(not(test))]
use tracing::{debug, info, warn};

#[derive(Error, Debug)]
pub enum DaemonError {
    #[error("Control error: {0}")]
    Ctrl(#[from] CtrlError),

    #[error("io_uring creation failed: {0}")]
    IoUringCreate(std::io::Error),

    #[error("Failed to mmap shared buffer: {0}")]
    Mmap(std::io::Error),

    #[error("io_uring submission failed: {0}")]
    Submit(std::io::Error),

    #[error("Queue full, cannot submit")]
    QueueFull,

    #[error("Would block")]
    WouldBlock,

    #[error("Device stopped")]
    Stopped,
}

impl DaemonError {
    /// Returns true if this error is transient and the operation should be retried.
    /// Pattern from pepita error handling.
    #[inline]
    pub const fn is_retriable(&self) -> bool {
        matches!(self, Self::QueueFull | Self::WouldBlock)
    }

    /// Returns true if this is a resource exhaustion error.
    #[inline]
    pub const fn is_resource_error(&self) -> bool {
        matches!(self, Self::QueueFull | Self::Mmap(_) | Self::IoUringCreate(_))
    }

    /// Convert to negative errno for POSIX compatibility.
    /// Pattern from pepita error handling.
    pub fn to_errno(&self) -> i32 {
        match self {
            Self::Ctrl(_) => -libc::ENODEV,
            Self::IoUringCreate(e) => -e.raw_os_error().unwrap_or(libc::ENOMEM),
            Self::Mmap(e) => -e.raw_os_error().unwrap_or(libc::ENOMEM),
            Self::Submit(e) => -e.raw_os_error().unwrap_or(libc::EIO),
            Self::QueueFull => -libc::ENOSPC,
            Self::WouldBlock => -libc::EAGAIN,
            Self::Stopped => -libc::ENODEV,
        }
    }
}

/// Running daemon handle
#[cfg(not(test))]
pub struct UblkDaemon {
    ctrl: UblkCtrl,
    char_fd: i32,
    ring: IoUring,
    iod_buf: *mut u8, // IOD buffer (mmap'd from char device)
    iod_buf_size: usize,
    data_buf: *mut u8, // Data buffer (anonymous mmap for our use)
    data_buf_size: usize,
    queue_depth: u16,
    max_io_size: u32,
    stop: Arc<AtomicBool>,
    use_ioctl_encode: bool, // Whether kernel uses ioctl-encoded cmd_op
    ready_fd: Option<i32>,
}

// IOD buffer size must be page-aligned
pub(crate) fn iod_buf_size(queue_depth: u16) -> usize {
    let size = (queue_depth as usize) * std::mem::size_of::<UblkIoDesc>();
    (size + 4095) & !4095 // Round up to page size
}

#[cfg(not(test))]
impl UblkDaemon {
    pub fn new(
        config: DeviceConfig,
        stop: Arc<AtomicBool>,
        ready_fd: Option<i32>,
    ) -> Result<Self, DaemonError> {
        info!(dev_size = config.dev_size, "Creating ublk daemon");

        let ctrl = UblkCtrl::new(config)?;
        let dev_id = ctrl.dev_id();
        let queue_depth = ctrl.queue_depth();
        let max_io_size = ctrl.max_io_buf_bytes();

        info!(dev_id, queue_depth, max_io_size, "Device created");

        let char_fd_owned = ctrl.open_char_dev()?;
        let char_fd = char_fd_owned.as_raw_fd();
        std::mem::forget(char_fd_owned);

        // FIX B: Disable SQPOLL mode - use standard io_uring submission
        // SQPOLL creates a kernel polling thread but has race conditions with URING_CMD
        // operations used by ublk. Standard mode with explicit submit_and_wait is more reliable.
        //
        // The Five-Whys analysis (Section 8 of ublk-batched-gpu-compression.md) identified
        // that SQPOLL may not process FETCH commands before START_DEV is called, causing
        // the daemon to hang indefinitely.
        //
        // PERF-004: io_uring tuning for maximum IOPS (without SQPOLL)
        let ring: IoUring = IoUring::builder()
            // SINGLE_ISSUER: Only one thread submits to this ring
            .setup_single_issuer()
            // COOP_TASKRUN: Cooperative task running reduces kernel overhead
            .setup_coop_taskrun()
            // Larger CQ to handle burst completions
            .setup_cqsize(queue_depth as u32 * 4)
            .build(queue_depth as u32 * 2)
            .map_err(DaemonError::IoUringCreate)?;

        // mmap IOD buffer from char device (read-only, kernel writes I/O descriptors here)
        let iod_buf_size = iod_buf_size(queue_depth);
        // SAFETY: mmap is called with valid parameters:
        // - char_fd is a valid file descriptor from open_char_dev()
        // - iod_buf_size is computed based on queue_depth which comes from kernel
        // - MAP_SHARED|PROT_READ maps the kernel's IOD buffer read-only as required
        // - The returned pointer is checked for MAP_FAILED before use
        let iod_buf = unsafe {
            mmap(
                null_mut(),
                iod_buf_size,
                PROT_READ, // Read-only per kernel requirement
                libc::MAP_SHARED,
                char_fd,
                0, // UBLKSRV_CMD_BUF_OFFSET = 0
            )
        };
        if iod_buf == libc::MAP_FAILED {
            return Err(DaemonError::Mmap(std::io::Error::last_os_error()));
        }
        let iod_buf = iod_buf as *mut u8;
        debug!(iod_buf_size, "IOD buffer mmap'd from char device");

        // Allocate anonymous data buffer for our use (compress/decompress workspace)
        let data_buf_size = (queue_depth as usize) * (max_io_size as usize);
        // SAFETY: mmap is called with valid parameters for anonymous memory:
        // - MAP_ANONYMOUS allocates new zero-filled memory (no file backing)
        // - MAP_PRIVATE ensures this memory is not shared with other processes
        // - data_buf_size is computed from kernel-provided queue_depth and max_io_size
        // - The returned pointer is checked for MAP_FAILED before use
        let data_buf = unsafe {
            mmap(
                null_mut(),
                data_buf_size,
                PROT_READ | PROT_WRITE,
                MAP_PRIVATE | MAP_ANONYMOUS,
                -1,
                0,
            )
        };
        if data_buf == libc::MAP_FAILED {
            unsafe { munmap(iod_buf as *mut _, iod_buf_size) };
            return Err(DaemonError::Mmap(std::io::Error::last_os_error()));
        }
        let data_buf = data_buf as *mut u8;
        debug!(data_buf_size, "Data buffer allocated");

        // Check if kernel supports ioctl-encoded command opcodes
        let use_ioctl_encode = (ctrl.dev_info().flags & UBLK_F_CMD_IOCTL_ENCODE) != 0;
        eprintln!(
            "[TRACE] Device flags=0x{:X}, use_ioctl_encode={}",
            ctrl.dev_info().flags,
            use_ioctl_encode
        );

        Ok(Self {
            ctrl,
            char_fd,
            ring,
            iod_buf,
            iod_buf_size,
            data_buf,
            data_buf_size,
            queue_depth,
            max_io_size,
            stop,
            use_ioctl_encode,
            ready_fd,
        })
    }

    pub fn dev_id(&self) -> i32 {
        self.ctrl.dev_id()
    }

    pub fn block_dev_path(&self) -> String {
        self.ctrl.block_dev_path()
    }

    /// FIX F: Restructured run() to ensure io_uring is blocked in submit_and_wait
    /// BEFORE START_DEV is called.
    ///
    /// The ublk kernel driver requires FETCH commands to be "in flight" in io_uring
    /// before START_DEV can complete. The previous approach spawned START_DEV in a
    /// background thread but didn't guarantee io_uring was actively waiting.
    ///
    /// New approach:
    /// 1. Submit all FETCH commands
    /// 2. Spawn a thread that calls START_DEV after a delay
    /// 3. Main thread immediately enters submit_and_wait() loop (blocks in kernel)
    /// 4. START_DEV thread waits to ensure main thread is blocked, then calls START_DEV
    /// 5. Kernel sees io_uring waiting → START_DEV completes → block device appears
    pub fn run(&mut self, store: &mut PageStore) -> Result<(), DaemonError> {
        info!(queue_depth = self.queue_depth, "Starting ublk daemon (FIX B+F applied)");
        debug!(char_fd = self.char_fd, "Daemon initialized");

        // Step 1: Submit all FETCH commands
        info!(queue_depth = self.queue_depth, "Submitting initial FETCH commands");
        for tag in 0..self.queue_depth {
            self.submit_fetch(tag)?;
        }

        // Sync and submit to kernel
        self.ring.submission().sync();
        let submitted = self.ring.submit().map_err(DaemonError::Submit)?;
        info!(submitted, "FETCH commands submitted to io_uring");

        // Step 2: Spawn START_DEV thread with delay
        // This thread waits to ensure the main thread is blocked in submit_and_wait
        // before calling START_DEV
        let mut ctrl_clone = self.ctrl.clone_handle()?;
        let ready_fd = self.ready_fd;
        let dev_id = self.ctrl.dev_id();
        let block_dev_path = self.ctrl.block_dev_path();

        std::thread::spawn(move || {
            // Wait for main thread to enter submit_and_wait and block
            // 200ms should be more than enough for the main thread to enter the kernel
            debug!("START_DEV thread: waiting for io_uring to block...");
            std::thread::sleep(Duration::from_millis(200));

            debug!("START_DEV thread: calling START_DEV ioctl");
            match ctrl_clone.start() {
                Ok(()) => {
                    info!(block_dev = %block_dev_path, "START_DEV succeeded - block device ready!");
                    if let Some(fd) = ready_fd {
                        let val = (dev_id as u64) + 1;
                        unsafe {
                            libc::write(fd, &val as *const u64 as *const libc::c_void, 8);
                        }
                    }
                }
                Err(e) => {
                    warn!(error = %e, "START_DEV failed");
                }
            }
        });

        // Step 3: Main thread enters submit_and_wait loop IMMEDIATELY
        // This is critical - we must be blocked in the kernel when START_DEV is called
        info!(block_dev = %self.ctrl.block_dev_path(), "Entering I/O loop (waiting for START_DEV)");

        loop {
            if self.stop.load(Ordering::Relaxed) {
                info!("Stop signal received");
                return Err(DaemonError::Stopped);
            }

            // Block waiting for completions - this is where we must be when START_DEV is called
            match self.ring.submit_and_wait(1) {
                Ok(_) => {}
                Err(e) if e.raw_os_error() == Some(libc::EINTR) => continue,
                Err(e) => return Err(DaemonError::Submit(e)),
            }

            // Process completions
            let tags: Vec<(u16, i32)> = {
                let cq = self.ring.completion();
                cq.map(|cqe| (cqe.user_data() as u16, cqe.result())).collect()
            };

            for (tag, result) in tags {
                if result < 0 {
                    if result == -libc::ENODEV {
                        info!(tag, "Device stopped (ENODEV)");
                        return Ok(());
                    }
                    warn!(tag, result, "I/O completion error, resubmitting FETCH");
                    self.submit_fetch(tag)?;
                    continue;
                }

                let iod = self.get_iod(tag);
                let op = (iod.op_flags & 0xff) as u8;
                let start_sector = iod.start_sector;
                let nr_sectors = iod.nr_sectors;

                debug!(tag, op, start_sector, nr_sectors, "Processing I/O");

                let len = (nr_sectors as usize) * SECTOR_SIZE as usize;
                let char_fd = self.char_fd;
                let data_buf_ptr =
                    unsafe { self.data_buf.add((tag as usize) * (self.max_io_size as usize)) };

                let io_result = match op {
                    UBLK_IO_OP_READ => {
                        let buf = unsafe { std::slice::from_raw_parts_mut(data_buf_ptr, len) };
                        match store.read(start_sector, buf) {
                            Ok(n) => {
                                let offset = ublk_user_copy_offset(0, tag, 0);
                                let fd = unsafe { BorrowedFd::borrow_raw(char_fd) };
                                pwrite(fd, buf, offset)
                                    .map(|_| n)
                                    .map_err(|e| std::io::Error::from_raw_os_error(e as i32))
                            }
                            Err(e) => Err(e),
                        }
                    }
                    UBLK_IO_OP_WRITE => {
                        let buf = unsafe { std::slice::from_raw_parts_mut(data_buf_ptr, len) };
                        let offset = ublk_user_copy_offset(0, tag, 0);
                        let fd = unsafe { BorrowedFd::borrow_raw(char_fd) };
                        match pread(fd, buf, offset) {
                            Ok(_) => store.write(start_sector, buf),
                            Err(e) => Err(std::io::Error::from_raw_os_error(e as i32)),
                        }
                    }
                    UBLK_IO_OP_FLUSH => Ok(0),
                    UBLK_IO_OP_DISCARD => store.discard(start_sector, nr_sectors),
                    UBLK_IO_OP_WRITE_ZEROES => store.write_zeroes(start_sector, nr_sectors),
                    _ => {
                        warn!(op, "Unknown I/O operation");
                        Err(std::io::Error::from_raw_os_error(libc::ENOTSUP))
                    }
                };

                let result = match io_result {
                    Ok(n) => n as i32,
                    Err(e) => {
                        warn!(tag, error = %e, "I/O operation failed");
                        -e.raw_os_error().unwrap_or(libc::EIO)
                    }
                };

                self.submit_commit_and_fetch(tag, result)?;
            }

            self.ring.submit().map_err(DaemonError::Submit)?;
        }
    }

    /// Run daemon with BatchedPageStore for high-throughput batched compression.
    ///
    /// This method is similar to `run` but uses `BatchedPageStore` which:
    /// - Buffers writes until batch threshold (1000+ pages)
    /// - Uses GPU for large batches when available
    /// - Falls back to SIMD parallel compression for smaller batches
    ///
    /// ## High-Performance Mode (PERF-001)
    ///
    /// When `hiperf` is provided with polling enabled:
    /// - Uses spin-wait polling on io_uring completions instead of blocking
    /// - Tracks polling statistics for tuning
    /// - Target: 800K+ IOPS
    ///
    /// Uses FIX B+F: Disabled SQPOLL, main thread enters io_uring before START_DEV.
    #[cfg(not(test))]
    pub fn run_batched(
        &mut self,
        store: &Arc<crate::daemon::BatchedPageStore>,
        mut hiperf: Option<&mut HiPerfContext>,
    ) -> Result<(), DaemonError> {
        let polling_enabled = hiperf.as_ref().is_some_and(|ctx| ctx.is_polling_enabled());
        info!(
            queue_depth = self.queue_depth,
            polling = polling_enabled,
            "Starting batched I/O loop (FIX B+F applied)"
        );

        // Step 1: Submit initial FETCH commands
        for tag in 0..self.queue_depth {
            self.submit_fetch(tag)?;
        }
        self.ring.submission().sync();
        let submitted = self.ring.submit().map_err(DaemonError::Submit)?;
        info!(submitted, "Batched: FETCH commands submitted");

        // Step 2: Spawn START_DEV thread with delay (FIX F)
        let mut ctrl_clone = self.ctrl.clone_handle()?;
        let ready_fd = self.ready_fd;
        let dev_id = self.ctrl.dev_id();
        let block_dev_path = self.ctrl.block_dev_path();

        std::thread::spawn(move || {
            // Wait for main thread to enter submit_and_wait
            debug!("Batched START_DEV thread: waiting for io_uring to block...");
            std::thread::sleep(Duration::from_millis(200));

            debug!("Batched START_DEV thread: calling START_DEV ioctl");
            match ctrl_clone.start() {
                Ok(()) => {
                    info!(block_dev = %block_dev_path, "Batched START_DEV succeeded!");
                    if let Some(fd) = ready_fd {
                        let val = (dev_id as u64) + 1;
                        unsafe {
                            libc::write(fd, &val as *const u64 as *const libc::c_void, 8);
                        }
                    }
                }
                Err(e) => {
                    warn!(error = %e, "Batched START_DEV failed");
                }
            }
        });

        // Step 3: Enter I/O loop immediately (main thread blocks in kernel)
        info!(block_dev = %self.ctrl.block_dev_path(), "Batched: entering I/O loop");

        loop {
            if self.stop.load(Ordering::Relaxed) {
                info!("Stop signal received");
                return Err(DaemonError::Stopped);
            }

            // PERF-001: Use polling mode when enabled, otherwise block
            let _completion_count = if polling_enabled {
                self.poll_completions(hiperf.as_deref_mut())?
            } else {
                // Standard blocking mode
                match self.ring.submit_and_wait(1) {
                    Ok(_) => {}
                    Err(e) if e.raw_os_error() == Some(libc::EINTR) => continue,
                    Err(e) => return Err(DaemonError::Submit(e)),
                }
                0 // Signal to process all available completions
            };

            // Collect completions
            let tags: Vec<(u16, i32)> = {
                let cq = self.ring.completion();
                cq.map(|cqe| (cqe.user_data() as u16, cqe.result())).collect()
            };

            for (tag, result) in tags {
                if result < 0 {
                    if result == -libc::ENODEV {
                        info!(tag, "Device stopped");
                        return Ok(());
                    }
                    warn!(tag, result, "I/O completion error");
                    self.submit_fetch(tag)?;
                    continue;
                }

                let iod = self.get_iod(tag);
                let op = (iod.op_flags & 0xff) as u8;
                let start_sector = iod.start_sector;
                let nr_sectors = iod.nr_sectors;

                debug!(tag, op, start_sector, nr_sectors, "Processing batched I/O");

                let len = (nr_sectors as usize) * SECTOR_SIZE as usize;
                let char_fd = self.char_fd;
                let data_buf_ptr =
                    unsafe { self.data_buf.add((tag as usize) * (self.max_io_size as usize)) };

                // Process I/O using BatchedPageStore
                let io_result = match op {
                    UBLK_IO_OP_READ => {
                        let buf = unsafe { std::slice::from_raw_parts_mut(data_buf_ptr, len) };
                        match store.read(start_sector, buf) {
                            Ok(n) => {
                                let offset = ublk_user_copy_offset(0, tag, 0);
                                let fd = unsafe { BorrowedFd::borrow_raw(char_fd) };
                                pwrite(fd, buf, offset)
                                    .map(|_| n)
                                    .map_err(|e| std::io::Error::from_raw_os_error(e as i32))
                            }
                            Err(e) => Err(e),
                        }
                    }
                    UBLK_IO_OP_WRITE => {
                        let buf = unsafe { std::slice::from_raw_parts_mut(data_buf_ptr, len) };
                        let offset = ublk_user_copy_offset(0, tag, 0);
                        let fd = unsafe { BorrowedFd::borrow_raw(char_fd) };
                        match pread(fd, buf, offset) {
                            Ok(_) => store.write(start_sector, buf),
                            Err(e) => Err(std::io::Error::from_raw_os_error(e as i32)),
                        }
                    }
                    UBLK_IO_OP_FLUSH => {
                        // Flush pending batch on FLUSH request
                        if let Err(e) = store.flush_batch() {
                            warn!("Flush batch failed: {}", e);
                        }
                        Ok(0)
                    }
                    UBLK_IO_OP_DISCARD => store.discard(start_sector, nr_sectors),
                    UBLK_IO_OP_WRITE_ZEROES => store.write_zeroes(start_sector, nr_sectors),
                    _ => {
                        warn!(op, "Unknown I/O operation");
                        Err(std::io::Error::from_raw_os_error(libc::ENOTSUP))
                    }
                };

                let result = match io_result {
                    Ok(n) => n as i32,
                    Err(e) => {
                        warn!(tag, error = %e, "Batched I/O operation failed");
                        -e.raw_os_error().unwrap_or(libc::EIO)
                    }
                };

                self.submit_commit_and_fetch(tag, result)?;
            }

            self.ring.submit().map_err(DaemonError::Submit)?;
        }
    }

    /// KERN-001/002: Generic batched I/O loop for any PageStoreTrait implementation.
    ///
    /// This enables tiered storage by allowing TieredPageStore to be used
    /// in the same I/O loop as BatchedPageStore.
    pub fn run_batched_generic<S: crate::daemon::PageStoreTrait + ?Sized>(
        &mut self,
        store: &Arc<S>,
        mut hiperf: Option<&mut HiPerfContext>,
    ) -> Result<(), DaemonError> {
        let polling_enabled = hiperf.as_ref().is_some_and(|ctx| ctx.is_polling_enabled());
        info!(
            queue_depth = self.queue_depth,
            polling = polling_enabled,
            "Starting generic batched I/O loop (KERN-001/002)"
        );

        // Step 1: Submit initial FETCH commands
        for tag in 0..self.queue_depth {
            self.submit_fetch(tag)?;
        }
        self.ring.submission().sync();
        let submitted = self.ring.submit().map_err(DaemonError::Submit)?;
        info!(submitted, "Generic batched: FETCH commands submitted");

        // Step 2: Spawn START_DEV thread with delay
        let mut ctrl_clone = self.ctrl.clone_handle()?;
        let ready_fd = self.ready_fd;
        let dev_id = self.ctrl.dev_id();
        let block_dev_path = self.ctrl.block_dev_path();

        std::thread::spawn(move || {
            debug!("Generic batched START_DEV thread: waiting...");
            std::thread::sleep(Duration::from_millis(200));

            debug!("Generic batched START_DEV thread: calling START_DEV ioctl");
            match ctrl_clone.start() {
                Ok(()) => {
                    info!(block_dev = %block_dev_path, "Generic batched START_DEV succeeded!");
                    if let Some(fd) = ready_fd {
                        let val = (dev_id as u64) + 1;
                        unsafe {
                            libc::write(fd, &val as *const u64 as *const libc::c_void, 8);
                        }
                    }
                }
                Err(e) => {
                    warn!(error = %e, "Generic batched START_DEV failed");
                }
            }
        });

        // Step 3: Enter I/O loop
        info!(block_dev = %self.ctrl.block_dev_path(), "Generic batched: entering I/O loop");

        loop {
            if self.stop.load(Ordering::Relaxed) {
                info!("Generic batched: stop signal received, exiting");
                break Ok(());
            }

            // PERF-001: Use polling mode when enabled, otherwise block
            let _completion_count = if polling_enabled {
                self.poll_completions(hiperf.as_deref_mut())?
            } else {
                match self.ring.submit_and_wait(1) {
                    Ok(_) => {}
                    Err(e) if e.raw_os_error() == Some(libc::EINTR) => continue,
                    Err(e) => return Err(DaemonError::Submit(e)),
                }
                0
            };

            // Collect completions
            let tags: Vec<(u16, i32)> = {
                let cq = self.ring.completion();
                cq.map(|cqe| (cqe.user_data() as u16, cqe.result())).collect()
            };

            for (tag, result) in tags {
                if result < 0 {
                    if result == -libc::ENODEV {
                        info!(tag, "Generic batched: Device stopped");
                        return Ok(());
                    }
                    warn!(tag, result, "Generic batched: I/O completion error");
                    self.submit_fetch(tag)?;
                    continue;
                }

                let iod = self.get_iod(tag);
                let op = (iod.op_flags & 0xff) as u8;
                let start_sector = iod.start_sector;
                let nr_sectors = iod.nr_sectors;
                let len = (nr_sectors as usize) * SECTOR_SIZE as usize;
                let char_fd = self.char_fd;
                let data_buf_ptr =
                    unsafe { self.data_buf.add((tag as usize) * (self.max_io_size as usize)) };

                // Process I/O using generic PageStoreTrait
                let io_result = match op {
                    UBLK_IO_OP_READ => {
                        let buf = unsafe { std::slice::from_raw_parts_mut(data_buf_ptr, len) };
                        match store.read(start_sector, buf) {
                            Ok(n) => {
                                let offset = ublk_user_copy_offset(0, tag, 0);
                                let fd = unsafe { BorrowedFd::borrow_raw(char_fd) };
                                pwrite(fd, buf, offset)
                                    .map(|_| n)
                                    .map_err(|e| std::io::Error::from_raw_os_error(e as i32))
                            }
                            Err(e) => Err(e),
                        }
                    }
                    UBLK_IO_OP_WRITE => {
                        let buf = unsafe { std::slice::from_raw_parts_mut(data_buf_ptr, len) };
                        let offset = ublk_user_copy_offset(0, tag, 0);
                        let fd = unsafe { BorrowedFd::borrow_raw(char_fd) };
                        match pread(fd, buf, offset) {
                            Ok(_) => store.write(start_sector, buf),
                            Err(e) => Err(std::io::Error::from_raw_os_error(e as i32)),
                        }
                    }
                    UBLK_IO_OP_FLUSH => Ok(0),
                    UBLK_IO_OP_DISCARD => store.discard(start_sector, nr_sectors),
                    UBLK_IO_OP_WRITE_ZEROES => store.write_zeroes(start_sector, nr_sectors),
                    _ => {
                        warn!(op, "Generic batched: Unknown I/O operation");
                        Err(std::io::Error::from_raw_os_error(libc::ENOTSUP))
                    }
                };

                let result = match io_result {
                    Ok(n) => n as i32,
                    Err(e) => {
                        warn!(tag, error = %e, "Generic batched I/O operation failed");
                        -e.raw_os_error().unwrap_or(libc::EIO)
                    }
                };

                self.submit_commit_and_fetch(tag, result)?;
            }

            self.ring.submit().map_err(DaemonError::Submit)?;
        }
    }

    fn submit_fetch(&mut self, tag: u16) -> Result<(), DaemonError> {
        // With UBLK_F_USER_COPY, addr must be 0 (data via pread/pwrite)
        let io_cmd = UblkIoCmd {
            q_id: 0, // queue 0
            tag,
            result: -1, // libublk uses -1 for FETCH
            addr: 0,    // USER_COPY mode: kernel uses pread/pwrite for data
        };

        // Use ioctl-encoded opcode if kernel supports it, otherwise raw opcode
        let cmd_op = if self.use_ioctl_encode { UBLK_U_IO_FETCH_REQ } else { UBLK_IO_FETCH_REQ };

        eprintln!(
            "[TRACE] submit_fetch: tag={}, fd={}, cmd_op=0x{:08X}, ioctl_encode={}",
            tag, self.char_fd, cmd_op, self.use_ioctl_encode
        );

        // Use UringCmd16 like libublk - the 16-byte cmd field is sufficient for UblkIoCmd
        let io_cmd_bytes: [u8; 16] = unsafe { std::mem::transmute(io_cmd) };
        // Use Fd(char_fd) directly - simpler than fd registration
        let sqe = opcode::UringCmd16::new(types::Fd(self.char_fd), cmd_op)
            .cmd(io_cmd_bytes)
            .build()
            .user_data(tag as u64);

        unsafe {
            self.ring.submission().push(&sqe).map_err(|_| {
                DaemonError::Submit(std::io::Error::from_raw_os_error(libc::ENOSPC))
            })?;
        }
        eprintln!("[TRACE]   pushed to submission queue OK");
        Ok(())
    }

    fn submit_commit_and_fetch(&mut self, tag: u16, result: i32) -> Result<(), DaemonError> {
        // With UBLK_F_USER_COPY, addr must be 0
        let io_cmd = UblkIoCmd { q_id: 0, tag, result, addr: 0 };

        // Use ioctl-encoded opcode if kernel supports it, otherwise raw opcode
        let cmd_op = if self.use_ioctl_encode {
            UBLK_U_IO_COMMIT_AND_FETCH_REQ
        } else {
            UBLK_IO_COMMIT_AND_FETCH_REQ
        };

        // Use UringCmd16 like libublk
        let io_cmd_bytes: [u8; 16] = unsafe { std::mem::transmute(io_cmd) };
        // Use Fd(char_fd) directly - simpler than fd registration
        let sqe = opcode::UringCmd16::new(types::Fd(self.char_fd), cmd_op)
            .cmd(io_cmd_bytes)
            .build()
            .user_data(tag as u64);

        unsafe {
            self.ring.submission().push(&sqe).map_err(|_| {
                DaemonError::Submit(std::io::Error::from_raw_os_error(libc::ENOSPC))
            })?;
        }
        Ok(())
    }

    #[inline]
    fn get_iod(&self, tag: u16) -> &UblkIoDesc {
        unsafe {
            let ptr = self.iod_buf.add((tag as usize) * std::mem::size_of::<UblkIoDesc>())
                as *const UblkIoDesc;
            &*ptr
        }
    }

    #[inline]
    #[allow(dead_code)]
    fn get_data_buf(&mut self, tag: u16) -> &mut [u8] {
        unsafe {
            let ptr = self.data_buf.add((tag as usize) * (self.max_io_size as usize));
            std::slice::from_raw_parts_mut(ptr, self.max_io_size as usize)
        }
    }

    /// PERF-001: Poll for completions without blocking.
    ///
    /// Spin-waits on the completion queue, using HiPerfContext to track
    /// statistics and manage adaptive polling behavior.
    ///
    /// Returns the number of completions found after polling.
    fn poll_completions(
        &mut self,
        mut hiperf: Option<&mut HiPerfContext>,
    ) -> Result<u32, DaemonError> {
        use crate::perf::PollResult;

        // Submit any pending SQEs first
        self.ring.submit().map_err(DaemonError::Submit)?;

        let mut total_completions = 0u32;
        let mut empty_polls = 0u32;
        let max_empty_polls = 10000; // Configurable via HiPerfContext

        loop {
            // Check completion queue
            let cq_len = self.ring.completion().len();
            let has_completions = cq_len > 0;

            if has_completions {
                total_completions += cq_len as u32;

                // Track stats in HiPerfContext
                if let Some(ref mut ctx) = hiperf {
                    ctx.poll_once(true, cq_len as u32);
                }

                return Ok(total_completions);
            }

            // No completions - use HiPerfContext to decide next action
            if let Some(ref mut ctx) = hiperf {
                match ctx.poll_once(false, 0) {
                    PollResult::Ready(count) => {
                        // Should not happen since has_completions was false
                        return Ok(count);
                    }
                    PollResult::Empty => {
                        // Continue polling (spin-wait)
                        empty_polls += 1;
                        std::hint::spin_loop();
                    }
                    PollResult::SwitchToInterrupt => {
                        // Polling exhausted, fall back to blocking wait
                        debug!(
                            "PERF-001: Switching to interrupt mode after {} empty polls",
                            empty_polls
                        );
                        match self.ring.submit_and_wait(1) {
                            Ok(_) => {
                                let count = self.ring.completion().len() as u32;
                                ctx.record_interrupt_completions(count);
                                return Ok(count);
                            }
                            Err(e) if e.raw_os_error() == Some(libc::EINTR) => continue,
                            Err(e) => return Err(DaemonError::Submit(e)),
                        }
                    }
                }
            } else {
                // No HiPerfContext - simple spin with limit
                empty_polls += 1;
                if empty_polls >= max_empty_polls {
                    // Fall back to blocking
                    match self.ring.submit_and_wait(1) {
                        Ok(_) => return Ok(self.ring.completion().len() as u32),
                        Err(e) if e.raw_os_error() == Some(libc::EINTR) => continue,
                        Err(e) => return Err(DaemonError::Submit(e)),
                    }
                }
                std::hint::spin_loop();
            }
        }
    }
}

#[cfg(not(test))]
impl Drop for UblkDaemon {
    fn drop(&mut self) {
        unsafe {
            munmap(self.iod_buf as *mut _, self.iod_buf_size);
            munmap(self.data_buf as *mut _, self.data_buf_size);
            libc::close(self.char_fd);
        }
    }
}

/// Start daemon with a new PageStore (per-page compression)
#[cfg(not(test))]
pub fn run_daemon(
    dev_id: i32,
    dev_size: u64,
    algorithm: trueno_zram_core::Algorithm,
    stop: Arc<AtomicBool>,
    ready_fd: Option<i32>,
) -> Result<(), DaemonError> {
    let config = DeviceConfig { dev_id, dev_size, ..Default::default() };

    let mut daemon = UblkDaemon::new(config, stop, ready_fd)?;
    let mut store = PageStore::new(dev_size, algorithm);

    daemon.run(&mut store)
}

/// Batched daemon configuration
#[derive(Debug, Clone)]
pub struct BatchedDaemonConfig {
    /// Device ID
    pub dev_id: i32,
    /// Device size in bytes
    pub dev_size: u64,
    /// Compression algorithm
    pub algorithm: trueno_zram_core::Algorithm,
    /// Batch threshold (pages before triggering compression)
    pub batch_threshold: usize,
    /// Flush timeout for partial batches
    pub flush_timeout_ms: u64,
    /// GPU batch size for optimal throughput
    pub gpu_batch_size: usize,
    /// PERF-001: High-performance configuration (polling, affinity, NUMA)
    pub perf: Option<PerfConfig>,
    /// PERF-003: Number of hardware queues (1-8)
    pub nr_hw_queues: u16,
    /// PERF-006: Enable ZERO_COPY mode (EXPERIMENTAL)
    pub zero_copy: bool,
    // =========================================================================
    // KERN-001/002/003: Kernel-Cooperative Tiered Storage
    // =========================================================================
    /// Storage backend type: memory, zram, tiered
    pub backend: crate::backend::BackendType,
    /// Enable entropy-based routing for tiered storage
    pub entropy_routing: bool,
    /// Kernel ZRAM device path (e.g., /dev/zram0)
    pub zram_device: Option<std::path::PathBuf>,
    /// NVMe cold tier directory path (KERN-003)
    pub cold_tier: Option<std::path::PathBuf>,
    /// Entropy threshold for kernel ZRAM routing (pages below this go to kernel)
    pub entropy_kernel_threshold: f64,
    /// Entropy threshold for skipping compression
    pub entropy_skip_threshold: f64,
}

impl Default for BatchedDaemonConfig {
    fn default() -> Self {
        Self {
            dev_id: -1,
            dev_size: 1 << 30, // 1GB
            algorithm: trueno_zram_core::Algorithm::Lz4,
            batch_threshold: 1000,
            flush_timeout_ms: 10,
            gpu_batch_size: 4000,
            perf: None,       // Disabled by default
            nr_hw_queues: 1,  // PERF-003: Default single queue
            zero_copy: false, // PERF-006: Disabled by default
            // KERN-001/002/003: Kernel-Cooperative defaults
            backend: crate::backend::BackendType::Memory,
            entropy_routing: false,
            zram_device: None,
            cold_tier: None, // KERN-003: NVMe cold tier disabled by default
            entropy_kernel_threshold: 6.0,
            entropy_skip_threshold: 7.5,
        }
    }
}

impl BatchedDaemonConfig {
    /// Enable high-performance mode with moderate settings
    pub fn with_high_perf(mut self) -> Self {
        self.perf = Some(PerfConfig::high_performance());
        self
    }

    /// Enable maximum performance mode (high CPU usage)
    pub fn with_max_perf(mut self) -> Self {
        self.perf = Some(PerfConfig::maximum());
        self
    }

    /// Set custom performance configuration
    pub fn with_perf(mut self, config: PerfConfig) -> Self {
        self.perf = Some(config);
        self
    }

    /// PERF-003: Set number of hardware queues (1-8)
    pub fn with_queues(mut self, nr_hw_queues: u16) -> Self {
        self.nr_hw_queues = nr_hw_queues.clamp(1, 8);
        self
    }
}

// ============================================================================
// PERF-003: Multi-Queue Daemon Implementation
// ============================================================================

/// Internal function for multi-queue mode (nr_hw_queues > 1).
///
/// This spawns N worker threads, each with its own io_uring instance,
/// processing I/O requests in parallel for maximum IOPS.
#[cfg(not(test))]
fn run_multi_queue_batched_internal(
    batch_config: BatchedDaemonConfig,
    device_config: DeviceConfig,
    stop: Arc<AtomicBool>,
    ready_fd: Option<i32>,
    _hiperf: Option<crate::perf::HiPerfContext>,
) -> Result<(), DaemonError> {
    use crate::ublk::multi_queue::spawn_queue_workers;
    use std::os::fd::AsRawFd;
    use std::ptr::null_mut;

    let nr_hw_queues = batch_config.nr_hw_queues;
    info!("PERF-003: Starting multi-queue daemon with {} queues", nr_hw_queues);

    // Create the ublk control device with multi-queue support
    let ctrl = UblkCtrl::new(device_config)?;
    let queue_depth = ctrl.queue_depth();
    let max_io_size = ctrl.max_io_buf_bytes();

    info!(
        "PERF-003: Device created: dev_id={}, queue_depth={}, max_io_size={}, nr_queues={}",
        ctrl.dev_id(),
        queue_depth,
        max_io_size,
        nr_hw_queues
    );

    // Open char device and get fd
    tracing::debug!("PERF-003: About to open char device");
    let char_fd_owned = match ctrl.open_char_dev() {
        Ok(fd) => {
            tracing::info!("PERF-003: Char device opened successfully");
            fd
        }
        Err(e) => {
            tracing::error!(error = %e, "PERF-003: Failed to open char device - this will trigger Drop cleanup");
            return Err(e.into());
        }
    };
    let char_fd = char_fd_owned.as_raw_fd();
    std::mem::forget(char_fd_owned); // Keep fd alive
    tracing::debug!("PERF-003: char_fd={}", char_fd);

    // Calculate multi-queue buffer sizes
    // NOTE: IOD buffer is now mmap'd per-queue in QueueIoWorker with correct offsets
    let data_per_queue = (queue_depth as usize) * (max_io_size as usize);

    // IOD buffer pointer - each queue worker will mmap its own IOD buffer
    // Pass a null pointer; workers ignore this and create their own mmap
    let iod_buf = std::ptr::null_mut::<u8>();
    info!(
        "PERF-003: IOD buffers will be mmap'd per-queue (queue_depth={}, nr_queues={})",
        queue_depth, nr_hw_queues
    );

    // Allocate anonymous data buffer for all queues
    let total_data_size = (nr_hw_queues as usize) * data_per_queue;
    // SAFETY: mmap is called with valid parameters for anonymous memory:
    // - MAP_ANONYMOUS|MAP_PRIVATE allocates private zero-filled memory
    // - total_data_size computed from kernel-provided queue parameters
    // - Result checked for MAP_FAILED, and IOD buffer cleaned up on failure
    let data_buf = unsafe {
        mmap(
            null_mut(),
            total_data_size,
            PROT_READ | PROT_WRITE,
            MAP_PRIVATE | MAP_ANONYMOUS,
            -1,
            0,
        )
    };
    if data_buf == libc::MAP_FAILED {
        // Note: iod_buf is null here - each queue worker will handle its own IOD mmap
        return Err(DaemonError::Mmap(std::io::Error::last_os_error()));
    }
    let data_buf = data_buf as *mut u8;

    // PERF-001: Apply NUMA binding if configured
    let numa_node = batch_config.perf.as_ref().map(|p| p.numa_node).unwrap_or(-1);
    if numa_node >= 0 {
        use crate::perf::numa::NumaAllocator;
        let numa = NumaAllocator::new(numa_node);
        if let Err(e) = numa.bind_memory(data_buf, total_data_size) {
            warn!("PERF-001: NUMA bind failed (non-fatal): {}", e);
        } else {
            info!("PERF-001: Data buffer bound to NUMA node {}", numa_node);
        }
    }

    info!(
        "PERF-003: Data buffer allocated: {} MB for {} queues (NUMA node: {})",
        total_data_size / (1024 * 1024),
        nr_hw_queues,
        numa_node
    );

    // Check if kernel supports ioctl-encoded command opcodes
    let use_ioctl_encode = (ctrl.dev_info().flags & UBLK_F_CMD_IOCTL_ENCODE) != 0;
    info!("PERF-003: use_ioctl_encode={}", use_ioctl_encode);

    // Create batched page store (shared across all queues)
    let store_config = BatchConfig {
        batch_threshold: batch_config.batch_threshold,
        flush_timeout: Duration::from_millis(batch_config.flush_timeout_ms),
        gpu_batch_size: batch_config.gpu_batch_size,
    };
    let store = Arc::new(BatchedPageStore::with_config(batch_config.algorithm, store_config));

    // Spawn background flush thread
    let flush_handle = spawn_flush_thread(Arc::clone(&store));

    // Get CPU cores for affinity
    let cpu_cores: Vec<usize> =
        batch_config.perf.as_ref().map(|p| p.cpu_cores.clone()).unwrap_or_default();

    // Spawn queue worker threads
    info!(
        "PERF-003: Spawning {} queue worker threads (ZERO_COPY={})",
        nr_hw_queues, batch_config.zero_copy
    );
    // SAFETY: iod_buf and data_buf are valid mmap'd pointers for all queues
    let worker_handles = unsafe {
        spawn_queue_workers(
            nr_hw_queues,
            queue_depth,
            max_io_size,
            char_fd,
            iod_buf,
            data_buf,
            Arc::clone(&stop),
            Arc::clone(&store),
            use_ioctl_encode,
            &cpu_cores,
            batch_config.zero_copy, // PERF-006: Pass ZERO_COPY mode
        )
    };

    // Spawn START_DEV thread with delay (like single-queue mode)
    let mut ctrl_clone = ctrl.clone_handle()?;
    let dev_id = ctrl.dev_id();
    let block_dev_path = ctrl.block_dev_path();

    std::thread::spawn(move || {
        // Wait for workers to submit initial FETCH commands
        std::thread::sleep(Duration::from_millis(200));

        debug!("PERF-003: Calling START_DEV ioctl");
        match ctrl_clone.start() {
            Ok(()) => {
                info!(
                    block_dev = %block_dev_path,
                    "PERF-003: START_DEV succeeded - {} queues active!",
                    nr_hw_queues
                );
                if let Some(fd) = ready_fd {
                    let val = (dev_id as u64) + 1;
                    unsafe {
                        libc::write(fd, &val as *const u64 as *const libc::c_void, 8);
                    }
                }
            }
            Err(e) => {
                warn!(error = %e, "PERF-003: START_DEV failed");
            }
        }
    });

    // Wait for all worker threads to complete
    let mut total_ios = 0u64;
    for handle in worker_handles {
        let queue_ios = handle.stats.total_ios();
        info!("PERF-003: Queue {} processed {} IOs", handle.queue_id, queue_ios);
        total_ios += queue_ios;

        if let Err(e) = handle.thread.join() {
            warn!("PERF-003: Queue {} worker thread panicked: {:?}", handle.queue_id, e);
        }
    }

    // Signal shutdown to flush thread
    store.shutdown();

    // Wait for flush thread
    if let Err(e) = flush_handle.join() {
        warn!("Flush thread panicked: {:?}", e);
    }

    // Cleanup mmap'd data buffer (IOD buffers are cleaned up by QueueIoWorker::drop)
    unsafe {
        munmap(data_buf as *mut _, total_data_size);
    }

    // Log final stats
    let stats = store.stats();
    info!(
        "PERF-003: Multi-queue daemon complete: total_ios={}, pages={}, ratio={:.2}x",
        total_ios,
        stats.pages_stored,
        if stats.bytes_compressed > 0 {
            stats.bytes_stored as f64 / stats.bytes_compressed as f64
        } else {
            1.0
        }
    );

    Ok(())
}

/// Start daemon with BatchedPageStore for high-throughput compression.
///
/// This mode buffers pages and compresses them in batches:
/// - GPU for batches >= 1000 pages
/// - SIMD parallel for 100-999 pages
/// - SIMD sequential for < 100 pages
///
/// ## High-Performance Mode (PERF-001)
///
/// When `perf` configuration is provided:
/// - CPU affinity is applied before entering the I/O loop
/// - Polling mode enables spin-wait on io_uring completions
/// - NUMA-aware buffer allocation is used
///
/// Target: >10 GB/s sequential write throughput, 800K+ IOPS.
#[cfg(not(test))]
pub fn run_daemon_batched(
    batch_config: BatchedDaemonConfig,
    stop: Arc<AtomicBool>,
    ready_fd: Option<i32>,
) -> Result<(), DaemonError> {
    // PERF-006: Select flags based on zero_copy mode
    let flags = if batch_config.zero_copy {
        // ZERO_COPY mode: eliminates pwrite syscall per I/O
        // NOTE: UBLK_F_USER_COPY must NOT be set with ZERO_COPY
        info!("PERF-006: ZERO_COPY mode enabled (EXPERIMENTAL)");
        crate::ublk::sys::UBLK_F_SUPPORT_ZERO_COPY
            | crate::ublk::sys::UBLK_F_URING_CMD_COMP_IN_TASK
            | crate::ublk::sys::UBLK_F_CMD_IOCTL_ENCODE
    } else {
        // USER_COPY mode (default): requires pwrite syscall per I/O
        crate::ublk::sys::UBLK_F_USER_COPY | crate::ublk::sys::UBLK_F_CMD_IOCTL_ENCODE
    };

    let device_config = DeviceConfig {
        dev_id: batch_config.dev_id,
        dev_size: batch_config.dev_size,
        nr_hw_queues: batch_config.nr_hw_queues, // PERF-003
        flags,
        ..Default::default()
    };

    // PERF-001: Create high-performance context if configured
    let mut hiperf = batch_config.perf.as_ref().map(|cfg| {
        info!(
            "PERF-001: High-performance mode enabled: polling={}, batch_size={}, affinity={:?}, numa={}",
            cfg.polling_enabled,
            cfg.batch_size,
            cfg.cpu_cores,
            cfg.numa_node
        );
        HiPerfContext::new(cfg.clone())
    });

    // PERF-001: Apply CPU affinity before entering I/O loop
    if let Some(ref ctx) = hiperf {
        if let Err(e) = ctx.init() {
            warn!("Failed to initialize high-performance context: {}", e);
        }
    }

    // PERF-003: Multi-queue mode with parallel I/O processing
    if batch_config.nr_hw_queues > 1 {
        return run_multi_queue_batched_internal(
            batch_config,
            device_config,
            stop,
            ready_fd,
            hiperf,
        );
    }

    // Single queue mode - use existing UblkDaemon
    let mut daemon = UblkDaemon::new(device_config, stop.clone(), ready_fd)?;

    // Create batched page store (base store for all modes)
    let store_config = BatchConfig {
        batch_threshold: batch_config.batch_threshold,
        flush_timeout: Duration::from_millis(batch_config.flush_timeout_ms),
        gpu_batch_size: batch_config.gpu_batch_size,
    };
    let batched_store =
        Arc::new(BatchedPageStore::with_config(batch_config.algorithm, store_config));

    // KERN-001/002: Check if tiered storage is enabled
    let use_tiered =
        !matches!(batch_config.backend, BackendType::Memory) && batch_config.zram_device.is_some();

    if use_tiered {
        // KERN-001/002/003: Create tiered page store wrapping batched store
        let tiered_config = TieredConfig {
            backend: batch_config.backend,
            entropy_routing: batch_config.entropy_routing,
            zram_device: batch_config.zram_device.clone(),
            cold_tier: batch_config.cold_tier.clone(),
            kernel_threshold: batch_config.entropy_kernel_threshold,
            skip_threshold: batch_config.entropy_skip_threshold,
        };

        let tiered_store = match TieredPageStore::new(Arc::clone(&batched_store), tiered_config) {
            Ok(store) => Arc::new(store),
            Err(e) => {
                warn!("Failed to create tiered store, falling back to batched: {}", e);
                // Fall through to non-tiered path
                return run_batched_non_tiered(daemon, batched_store, batch_config, hiperf);
            }
        };

        // Spawn background flush thread for tiered store
        let flush_handle = spawn_tiered_flush_thread(Arc::clone(&tiered_store));

        info!(
            "KERN-001/002: Starting TIERED daemon: backend={}, entropy_routing={}, kernel_threshold={}, skip_threshold={}",
            batch_config.backend,
            batch_config.entropy_routing,
            batch_config.entropy_kernel_threshold,
            batch_config.entropy_skip_threshold
        );

        // Run daemon with tiered store
        let result = daemon.run_batched_generic(&tiered_store, hiperf.as_mut());

        // Signal shutdown
        tiered_store.shutdown();

        // Wait for flush thread
        if let Err(e) = flush_handle.join() {
            warn!("Tiered flush thread panicked: {:?}", e);
        }

        // Log tiered stats
        let stats = tiered_store.stats();
        info!(
            "KERN-001/002 stats: kernel_pages={}, trueno_pages={}, skipped_pages={}, samefill_pages={}",
            stats.kernel_pages,
            stats.trueno_pages,
            stats.skipped_pages,
            stats.samefill_pages
        );

        return result;
    }

    // Non-tiered path (original code)
    run_batched_non_tiered(daemon, batched_store, batch_config, hiperf)
}

/// Run batched daemon without tiered storage (original code path)
#[cfg(not(test))]
fn run_batched_non_tiered(
    mut daemon: UblkDaemon,
    store: Arc<BatchedPageStore>,
    batch_config: BatchedDaemonConfig,
    mut hiperf: Option<HiPerfContext>,
) -> Result<(), DaemonError> {
    // Spawn background flush thread
    let flush_handle = spawn_flush_thread(Arc::clone(&store));

    info!(
        "Starting batched daemon: batch_threshold={}, flush_timeout={}ms, gpu_batch_size={}, hiperf={}",
        batch_config.batch_threshold,
        batch_config.flush_timeout_ms,
        batch_config.gpu_batch_size,
        hiperf.is_some()
    );

    // Run daemon with batched store - pass HiPerfContext for polling mode (PERF-001)
    let result = daemon.run_batched(&store, hiperf.as_mut());

    // Signal shutdown to flush thread
    store.shutdown();

    // Wait for flush thread to complete final flush
    if let Err(e) = flush_handle.join() {
        warn!("Flush thread panicked: {:?}", e);
    }

    // Log final stats
    let stats = store.stats();
    info!(
        "Batched daemon stats: pages={}, gpu_pages={}, simd_pages={}, batch_flushes={}, ratio={:.2}x",
        stats.pages_stored,
        stats.gpu_pages,
        stats.simd_pages,
        stats.batch_flushes,
        if stats.bytes_compressed > 0 {
            stats.bytes_stored as f64 / stats.bytes_compressed as f64
        } else {
            1.0
        }
    );

    // Log high-performance stats if enabled
    if let Some(ctx) = hiperf {
        let hstats = ctx.stats().snapshot();
        if hstats.total_ios > 0 {
            info!(
                "PERF-001 stats: total_ios={}, polling_efficiency={:.1}%, avg_batch={:.1}",
                hstats.total_ios,
                hstats.polling_efficiency() * 100.0,
                hstats.avg_batch_size()
            );
        }
    }

    result
}

#[cfg(test)]
mod mock;
#[cfg(test)]
pub use mock::*;

#[cfg(test)]
mod tests;
