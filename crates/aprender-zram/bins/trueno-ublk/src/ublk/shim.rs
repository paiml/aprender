//! Kernel interaction shims for testability
//!
//! This module defines traits that abstract kernel operations, allowing
//! the business logic to be tested without actual kernel interaction.
//!
//! Architecture:
//! - `KernelShim`: Trait for kernel operations (open, mmap, ioctl via io_uring)
//! - `RealKernelShim`: Production implementation using actual syscalls
//! - `MockKernelShim`: Test implementation that simulates kernel behavior

use crate::ublk::sys::*;
use nix::libc;
use std::io;
#[cfg(not(test))]
use std::os::fd::AsRawFd;
use std::os::fd::{FromRawFd, OwnedFd};

// ============================================================================
// Kernel Shim Trait
// ============================================================================

/// Result of a control command submission
#[derive(Debug, Clone)]
pub struct CtrlCmdResult {
    pub retval: i32,
    pub dev_info: Option<UblkCtrlDevInfo>,
}

/// Result of mmap operation
#[derive(Debug)]
pub struct MmapResult {
    pub ptr: *mut u8,
    pub len: usize,
}

/// Trait abstracting kernel operations for control plane
pub trait CtrlShim: Send {
    /// Open the ublk control device
    fn open_ctrl_device(&self) -> io::Result<OwnedFd>;

    /// Submit a control command and wait for result
    fn submit_ctrl_cmd(
        &self,
        ctrl_fd: &OwnedFd,
        cmd_op: u32,
        cmd: &UblkCtrlCmd,
    ) -> io::Result<CtrlCmdResult>;

    /// Open a character device for a specific ublk device
    fn open_char_device(&self, dev_id: i32) -> io::Result<OwnedFd>;
}

/// Trait abstracting kernel operations for data plane (daemon)
pub trait DaemonShim {
    /// Create io_uring instance
    fn create_io_uring(&self, entries: u32) -> io::Result<Box<dyn IoUringOps>>;

    /// Memory map the IOD buffer from char device
    fn mmap_iod_buffer(&self, char_fd: &OwnedFd, size: usize) -> io::Result<MmapResult>;

    /// Allocate anonymous memory for data buffer
    fn mmap_anonymous(&self, size: usize) -> io::Result<MmapResult>;

    /// Unmap memory
    fn munmap(&self, ptr: *mut u8, len: usize) -> io::Result<()>;

    /// Read from char device using pread (USER_COPY mode)
    fn pread(&self, fd: &OwnedFd, buf: &mut [u8], offset: i64) -> io::Result<usize>;

    /// Write to char device using pwrite (USER_COPY mode)
    fn pwrite(&self, fd: &OwnedFd, buf: &[u8], offset: i64) -> io::Result<usize>;
}

/// Trait abstracting io_uring operations
pub trait IoUringOps {
    /// Submit a FETCH_REQ command
    fn submit_fetch(&mut self, char_fd: i32, tag: u16, queue_id: u16) -> io::Result<()>;

    /// Submit a COMMIT_AND_FETCH_REQ command
    fn submit_commit_fetch(
        &mut self,
        char_fd: i32,
        tag: u16,
        queue_id: u16,
        result: i32,
    ) -> io::Result<()>;

    /// Submit pending entries and wait for completions
    fn submit_and_wait(&mut self, wait_nr: u32) -> io::Result<u32>;

    /// Get completed entries
    fn get_completions(&mut self) -> Vec<IoCompletion>;
}

/// Completion entry from io_uring
#[derive(Debug, Clone)]
pub struct IoCompletion {
    pub user_data: u64,
    pub result: i32,
}

// ============================================================================
// Real Kernel Shim Implementation (excluded from test coverage)
// ============================================================================

/// Production implementation using actual kernel syscalls
#[cfg(not(test))]
pub struct RealKernelShim;

#[cfg(not(test))]
impl RealKernelShim {
    pub fn new() -> Self {
        Self
    }
}

#[cfg(not(test))]
impl Default for RealKernelShim {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(not(test))]
impl CtrlShim for RealKernelShim {
    fn open_ctrl_device(&self) -> io::Result<OwnedFd> {
        use std::ffi::CString;
        let path = CString::new(UBLK_CTRL_DEV).expect("UBLK_CTRL_DEV is a valid C string");
        let fd = unsafe { libc::open(path.as_ptr(), libc::O_RDWR) };
        if fd < 0 {
            return Err(io::Error::last_os_error());
        }
        Ok(unsafe { OwnedFd::from_raw_fd(fd) })
    }

    fn submit_ctrl_cmd(
        &self,
        ctrl_fd: &OwnedFd,
        cmd_op: u32,
        cmd: &UblkCtrlCmd,
    ) -> io::Result<CtrlCmdResult> {
        use io_uring::{opcode, types, IoUring};
        type IoUring128 = IoUring<io_uring::squeue::Entry128, io_uring::cqueue::Entry32>;

        // Create io_uring for this command
        let mut ring: IoUring128 = IoUring128::builder().build(4)?;

        // Wrap command in extended struct for 128-byte SQE
        let cmd_ext = UblkCtrlCmdExt { cmd: *cmd, padding: [0; 48] };
        let cmd_bytes: [u8; 80] = unsafe { std::mem::transmute(cmd_ext) };

        let sqe = opcode::UringCmd80::new(types::Fd(ctrl_fd.as_raw_fd()), cmd_op)
            .cmd(cmd_bytes)
            .build()
            .user_data(0x100);

        unsafe {
            ring.submission().push(&sqe).map_err(|_| io::Error::other("SQ full"))?;
        }

        ring.submit_and_wait(1)?;

        let cqe = ring.completion().next().ok_or_else(|| io::Error::other("No CQE"))?;

        let result = cqe.result();
        if result < 0 {
            return Err(io::Error::from_raw_os_error(-result));
        }

        // Extract dev_info if addr was provided
        let dev_info = if cmd.addr != 0 {
            Some(unsafe { *(cmd.addr as *const UblkCtrlDevInfo) })
        } else {
            None
        };

        Ok(CtrlCmdResult { retval: result, dev_info })
    }

    fn open_char_device(&self, dev_id: i32) -> io::Result<OwnedFd> {
        use std::ffi::CString;
        let path = format!("{}{}", UBLK_CHAR_DEV_FMT, dev_id);
        let path = CString::new(path).expect("char device path is a valid C string");
        let fd = unsafe { libc::open(path.as_ptr(), libc::O_RDWR) };
        if fd < 0 {
            return Err(io::Error::last_os_error());
        }
        Ok(unsafe { OwnedFd::from_raw_fd(fd) })
    }
}

#[cfg(not(test))]
impl DaemonShim for RealKernelShim {
    fn create_io_uring(&self, entries: u32) -> io::Result<Box<dyn IoUringOps>> {
        let ring = RealIoUring::new(entries)?;
        Ok(Box::new(ring))
    }

    fn mmap_iod_buffer(&self, char_fd: &OwnedFd, size: usize) -> io::Result<MmapResult> {
        let ptr = unsafe {
            libc::mmap(
                std::ptr::null_mut(),
                size,
                libc::PROT_READ,
                libc::MAP_SHARED | libc::MAP_POPULATE,
                char_fd.as_raw_fd(),
                0,
            )
        };
        if ptr == libc::MAP_FAILED {
            return Err(io::Error::last_os_error());
        }
        Ok(MmapResult { ptr: ptr as *mut u8, len: size })
    }

    fn mmap_anonymous(&self, size: usize) -> io::Result<MmapResult> {
        let ptr = unsafe {
            libc::mmap(
                std::ptr::null_mut(),
                size,
                libc::PROT_READ | libc::PROT_WRITE,
                libc::MAP_PRIVATE | libc::MAP_ANONYMOUS,
                -1,
                0,
            )
        };
        if ptr == libc::MAP_FAILED {
            return Err(io::Error::last_os_error());
        }
        Ok(MmapResult { ptr: ptr as *mut u8, len: size })
    }

    fn munmap(&self, ptr: *mut u8, len: usize) -> io::Result<()> {
        let result = unsafe { libc::munmap(ptr as *mut libc::c_void, len) };
        if result < 0 {
            return Err(io::Error::last_os_error());
        }
        Ok(())
    }

    fn pread(&self, fd: &OwnedFd, buf: &mut [u8], offset: i64) -> io::Result<usize> {
        let result = unsafe {
            libc::pread(fd.as_raw_fd(), buf.as_mut_ptr() as *mut libc::c_void, buf.len(), offset)
        };
        if result < 0 {
            return Err(io::Error::last_os_error());
        }
        Ok(result as usize)
    }

    fn pwrite(&self, fd: &OwnedFd, buf: &[u8], offset: i64) -> io::Result<usize> {
        let result = unsafe {
            libc::pwrite(fd.as_raw_fd(), buf.as_ptr() as *const libc::c_void, buf.len(), offset)
        };
        if result < 0 {
            return Err(io::Error::last_os_error());
        }
        Ok(result as usize)
    }
}

/// Real io_uring implementation
#[cfg(not(test))]
struct RealIoUring {
    ring: io_uring::IoUring,
}

#[cfg(not(test))]
impl RealIoUring {
    fn new(entries: u32) -> io::Result<Self> {
        let ring = io_uring::IoUring::new(entries)?;
        Ok(Self { ring })
    }
}

#[cfg(not(test))]
impl IoUringOps for RealIoUring {
    fn submit_fetch(&mut self, char_fd: i32, tag: u16, queue_id: u16) -> io::Result<()> {
        use io_uring::{opcode, types};

        let io_cmd = UblkIoCmd { q_id: queue_id, tag, result: 0, addr: 0 };
        let cmd_bytes: [u8; 16] = unsafe { std::mem::transmute(io_cmd) };

        let sqe = opcode::UringCmd16::new(types::Fd(char_fd), UBLK_U_IO_FETCH_REQ)
            .cmd(cmd_bytes)
            .build()
            .user_data((tag as u64) | ((queue_id as u64) << 16));

        unsafe {
            self.ring.submission().push(&sqe).map_err(|_| io::Error::other("SQ full"))?;
        }
        Ok(())
    }

    fn submit_commit_fetch(
        &mut self,
        char_fd: i32,
        tag: u16,
        queue_id: u16,
        result: i32,
    ) -> io::Result<()> {
        use io_uring::{opcode, types};

        let io_cmd = UblkIoCmd { q_id: queue_id, tag, result, addr: 0 };
        let cmd_bytes: [u8; 16] = unsafe { std::mem::transmute(io_cmd) };

        let sqe = opcode::UringCmd16::new(types::Fd(char_fd), UBLK_U_IO_COMMIT_AND_FETCH_REQ)
            .cmd(cmd_bytes)
            .build()
            .user_data((tag as u64) | ((queue_id as u64) << 16));

        unsafe {
            self.ring.submission().push(&sqe).map_err(|_| io::Error::other("SQ full"))?;
        }
        Ok(())
    }

    fn submit_and_wait(&mut self, wait_nr: u32) -> io::Result<u32> {
        self.ring.submit_and_wait(wait_nr as usize).map(|n| n as u32)
    }

    fn get_completions(&mut self) -> Vec<IoCompletion> {
        self.ring
            .completion()
            .map(|cqe| IoCompletion { user_data: cqe.user_data(), result: cqe.result() })
            .collect()
    }
}

// ============================================================================
// Mock Kernel Shim Implementation (for testing)
// ============================================================================

/// Mock implementation for testing without kernel
#[cfg(test)]
pub struct MockKernelShim {
    pub next_dev_id: std::sync::atomic::AtomicI32,
    pub fail_open_ctrl: std::sync::atomic::AtomicBool,
    pub fail_add_dev: std::sync::atomic::AtomicBool,
    pub commands: std::sync::Mutex<Vec<(u32, UblkCtrlCmd)>>,
}

#[cfg(test)]
impl MockKernelShim {
    pub fn new() -> Self {
        Self {
            next_dev_id: std::sync::atomic::AtomicI32::new(0),
            fail_open_ctrl: std::sync::atomic::AtomicBool::new(false),
            fail_add_dev: std::sync::atomic::AtomicBool::new(false),
            commands: std::sync::Mutex::new(Vec::new()),
        }
    }

    pub fn set_fail_open_ctrl(&self, fail: bool) {
        self.fail_open_ctrl.store(fail, std::sync::atomic::Ordering::SeqCst);
    }

    pub fn set_fail_add_dev(&self, fail: bool) {
        self.fail_add_dev.store(fail, std::sync::atomic::Ordering::SeqCst);
    }

    pub fn get_commands(&self) -> Vec<(u32, UblkCtrlCmd)> {
        self.commands.lock().unwrap().clone()
    }
}

#[cfg(test)]
impl Default for MockKernelShim {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
impl CtrlShim for MockKernelShim {
    fn open_ctrl_device(&self) -> io::Result<OwnedFd> {
        if self.fail_open_ctrl.load(std::sync::atomic::Ordering::SeqCst) {
            return Err(io::Error::from_raw_os_error(libc::ENOENT));
        }
        // Return a dummy fd (we won't actually use it)
        // Use /dev/null as a safe placeholder
        use std::ffi::CString;
        let path = CString::new("/dev/null").unwrap();
        let fd = unsafe { libc::open(path.as_ptr(), libc::O_RDWR) };
        if fd < 0 {
            return Err(io::Error::last_os_error());
        }
        Ok(unsafe { OwnedFd::from_raw_fd(fd) })
    }

    fn submit_ctrl_cmd(
        &self,
        _ctrl_fd: &OwnedFd,
        cmd_op: u32,
        cmd: &UblkCtrlCmd,
    ) -> io::Result<CtrlCmdResult> {
        self.commands.lock().unwrap().push((cmd_op, *cmd));

        match cmd_op {
            UBLK_U_CMD_ADD_DEV => {
                if self.fail_add_dev.load(std::sync::atomic::Ordering::SeqCst) {
                    return Err(io::Error::from_raw_os_error(libc::EBUSY));
                }
                let dev_id = self.next_dev_id.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
                let mut dev_info = if cmd.addr != 0 {
                    unsafe { *(cmd.addr as *const UblkCtrlDevInfo) }
                } else {
                    UblkCtrlDevInfo::default()
                };
                dev_info.dev_id = dev_id as u32;
                Ok(CtrlCmdResult { retval: 0, dev_info: Some(dev_info) })
            }
            UBLK_U_CMD_SET_PARAMS
            | UBLK_U_CMD_START_DEV
            | UBLK_U_CMD_STOP_DEV
            | UBLK_U_CMD_DEL_DEV => Ok(CtrlCmdResult { retval: 0, dev_info: None }),
            _ => Err(io::Error::from_raw_os_error(libc::EINVAL)),
        }
    }

    fn open_char_device(&self, _dev_id: i32) -> io::Result<OwnedFd> {
        // Return /dev/null as placeholder
        use std::ffi::CString;
        let path = CString::new("/dev/null").unwrap();
        let fd = unsafe { libc::open(path.as_ptr(), libc::O_RDWR) };
        if fd < 0 {
            return Err(io::Error::last_os_error());
        }
        Ok(unsafe { OwnedFd::from_raw_fd(fd) })
    }
}

#[cfg(test)]
pub struct MockDaemonShim {
    pub mmap_buffers: std::sync::Mutex<Vec<(*mut u8, usize)>>,
    pub fail_mmap: std::sync::atomic::AtomicBool,
}

#[cfg(test)]
impl MockDaemonShim {
    pub fn new() -> Self {
        Self {
            mmap_buffers: std::sync::Mutex::new(Vec::new()),
            fail_mmap: std::sync::atomic::AtomicBool::new(false),
        }
    }
}

#[cfg(test)]
impl Default for MockDaemonShim {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
impl DaemonShim for MockDaemonShim {
    fn create_io_uring(&self, entries: u32) -> io::Result<Box<dyn IoUringOps>> {
        Ok(Box::new(MockIoUring::new(entries)))
    }

    fn mmap_iod_buffer(&self, _char_fd: &OwnedFd, size: usize) -> io::Result<MmapResult> {
        if self.fail_mmap.load(std::sync::atomic::Ordering::SeqCst) {
            return Err(io::Error::from_raw_os_error(libc::ENOMEM));
        }
        // Allocate real memory for testing
        let layout = std::alloc::Layout::from_size_align(size, 4096).unwrap();
        let ptr = unsafe { std::alloc::alloc_zeroed(layout) };
        if ptr.is_null() {
            return Err(io::Error::from_raw_os_error(libc::ENOMEM));
        }
        self.mmap_buffers.lock().unwrap().push((ptr, size));
        Ok(MmapResult { ptr, len: size })
    }

    fn mmap_anonymous(&self, size: usize) -> io::Result<MmapResult> {
        if self.fail_mmap.load(std::sync::atomic::Ordering::SeqCst) {
            return Err(io::Error::from_raw_os_error(libc::ENOMEM));
        }
        let layout = std::alloc::Layout::from_size_align(size, 4096).unwrap();
        let ptr = unsafe { std::alloc::alloc_zeroed(layout) };
        if ptr.is_null() {
            return Err(io::Error::from_raw_os_error(libc::ENOMEM));
        }
        self.mmap_buffers.lock().unwrap().push((ptr, size));
        Ok(MmapResult { ptr, len: size })
    }

    fn munmap(&self, ptr: *mut u8, len: usize) -> io::Result<()> {
        let layout = std::alloc::Layout::from_size_align(len, 4096).unwrap();
        unsafe { std::alloc::dealloc(ptr, layout) };
        Ok(())
    }

    fn pread(&self, _fd: &OwnedFd, buf: &mut [u8], _offset: i64) -> io::Result<usize> {
        // Fill with test pattern
        for (i, byte) in buf.iter_mut().enumerate() {
            *byte = (i % 256) as u8;
        }
        Ok(buf.len())
    }

    fn pwrite(&self, _fd: &OwnedFd, buf: &[u8], _offset: i64) -> io::Result<usize> {
        Ok(buf.len())
    }
}

#[cfg(test)]
pub struct MockIoUring {
    pub fetch_count: std::sync::atomic::AtomicUsize,
    pub commit_count: std::sync::atomic::AtomicUsize,
    pub pending_completions: std::sync::Mutex<Vec<IoCompletion>>,
    entries: u32,
}

#[cfg(test)]
impl MockIoUring {
    pub fn new(entries: u32) -> Self {
        Self {
            fetch_count: std::sync::atomic::AtomicUsize::new(0),
            commit_count: std::sync::atomic::AtomicUsize::new(0),
            pending_completions: std::sync::Mutex::new(Vec::new()),
            entries,
        }
    }

    pub fn add_completion(&self, user_data: u64, result: i32) {
        self.pending_completions.lock().unwrap().push(IoCompletion { user_data, result });
    }
}

#[cfg(test)]
impl IoUringOps for MockIoUring {
    fn submit_fetch(&mut self, _char_fd: i32, tag: u16, queue_id: u16) -> io::Result<()> {
        self.fetch_count.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        // Add a mock completion
        let user_data = (tag as u64) | ((queue_id as u64) << 16);
        self.pending_completions.lock().unwrap().push(IoCompletion { user_data, result: 0 });
        Ok(())
    }

    fn submit_commit_fetch(
        &mut self,
        _char_fd: i32,
        tag: u16,
        queue_id: u16,
        _result: i32,
    ) -> io::Result<()> {
        self.commit_count.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        let user_data = (tag as u64) | ((queue_id as u64) << 16);
        self.pending_completions.lock().unwrap().push(IoCompletion { user_data, result: 0 });
        Ok(())
    }

    fn submit_and_wait(&mut self, _wait_nr: u32) -> io::Result<u32> {
        let count = self.pending_completions.lock().unwrap().len();
        Ok(count as u32)
    }

    fn get_completions(&mut self) -> Vec<IoCompletion> {
        std::mem::take(&mut *self.pending_completions.lock().unwrap())
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // ========================================================================
    // MockKernelShim Tests
    // ========================================================================

    #[test]
    fn test_mock_kernel_shim_open_ctrl() {
        let shim = MockKernelShim::new();
        let fd = shim.open_ctrl_device();
        assert!(fd.is_ok());
    }

    #[test]
    fn test_mock_kernel_shim_open_ctrl_fail() {
        let shim = MockKernelShim::new();
        shim.set_fail_open_ctrl(true);
        let fd = shim.open_ctrl_device();
        assert!(fd.is_err());
        assert_eq!(fd.unwrap_err().raw_os_error(), Some(libc::ENOENT));
    }

    #[test]
    fn test_mock_kernel_shim_add_dev() {
        let shim = MockKernelShim::new();
        let ctrl_fd = shim.open_ctrl_device().unwrap();

        let mut dev_info =
            UblkCtrlDevInfo { nr_hw_queues: 1, queue_depth: 128, ..Default::default() };

        let cmd = UblkCtrlCmd {
            dev_id: u32::MAX,
            addr: &mut dev_info as *mut _ as u64,
            len: std::mem::size_of::<UblkCtrlDevInfo>() as u16,
            ..Default::default()
        };

        let result = shim.submit_ctrl_cmd(&ctrl_fd, UBLK_U_CMD_ADD_DEV, &cmd);
        assert!(result.is_ok());
        let result = result.unwrap();
        assert_eq!(result.retval, 0);
        assert!(result.dev_info.is_some());
        assert_eq!(result.dev_info.unwrap().dev_id, 0);
    }

    #[test]
    fn test_mock_kernel_shim_add_dev_fail() {
        let shim = MockKernelShim::new();
        shim.set_fail_add_dev(true);
        let ctrl_fd = shim.open_ctrl_device().unwrap();

        let cmd = UblkCtrlCmd::default();
        let result = shim.submit_ctrl_cmd(&ctrl_fd, UBLK_U_CMD_ADD_DEV, &cmd);
        assert!(result.is_err());
        assert_eq!(result.unwrap_err().raw_os_error(), Some(libc::EBUSY));
    }

    #[test]
    fn test_mock_kernel_shim_set_params() {
        let shim = MockKernelShim::new();
        let ctrl_fd = shim.open_ctrl_device().unwrap();
        let cmd = UblkCtrlCmd::default();

        let result = shim.submit_ctrl_cmd(&ctrl_fd, UBLK_U_CMD_SET_PARAMS, &cmd);
        assert!(result.is_ok());
    }

    #[test]
    fn test_mock_kernel_shim_start_dev() {
        let shim = MockKernelShim::new();
        let ctrl_fd = shim.open_ctrl_device().unwrap();
        let cmd = UblkCtrlCmd::default();

        let result = shim.submit_ctrl_cmd(&ctrl_fd, UBLK_U_CMD_START_DEV, &cmd);
        assert!(result.is_ok());
    }

    #[test]
    fn test_mock_kernel_shim_stop_dev() {
        let shim = MockKernelShim::new();
        let ctrl_fd = shim.open_ctrl_device().unwrap();
        let cmd = UblkCtrlCmd::default();

        let result = shim.submit_ctrl_cmd(&ctrl_fd, UBLK_U_CMD_STOP_DEV, &cmd);
        assert!(result.is_ok());
    }

    #[test]
    fn test_mock_kernel_shim_del_dev() {
        let shim = MockKernelShim::new();
        let ctrl_fd = shim.open_ctrl_device().unwrap();
        let cmd = UblkCtrlCmd::default();

        let result = shim.submit_ctrl_cmd(&ctrl_fd, UBLK_U_CMD_DEL_DEV, &cmd);
        assert!(result.is_ok());
    }

    #[test]
    fn test_mock_kernel_shim_unknown_cmd() {
        let shim = MockKernelShim::new();
        let ctrl_fd = shim.open_ctrl_device().unwrap();
        let cmd = UblkCtrlCmd::default();

        let result = shim.submit_ctrl_cmd(&ctrl_fd, 0xFFFFFFFF, &cmd);
        assert!(result.is_err());
        assert_eq!(result.unwrap_err().raw_os_error(), Some(libc::EINVAL));
    }

    #[test]
    fn test_mock_kernel_shim_open_char_device() {
        let shim = MockKernelShim::new();
        let fd = shim.open_char_device(0);
        assert!(fd.is_ok());
    }

    #[test]
    fn test_mock_kernel_shim_command_tracking() {
        let shim = MockKernelShim::new();
        let ctrl_fd = shim.open_ctrl_device().unwrap();

        let cmd1 = UblkCtrlCmd { dev_id: 1, ..Default::default() };
        let cmd2 = UblkCtrlCmd { dev_id: 2, ..Default::default() };

        shim.submit_ctrl_cmd(&ctrl_fd, UBLK_U_CMD_START_DEV, &cmd1).unwrap();
        shim.submit_ctrl_cmd(&ctrl_fd, UBLK_U_CMD_STOP_DEV, &cmd2).unwrap();

        let commands = shim.get_commands();
        assert_eq!(commands.len(), 2);
        assert_eq!(commands[0].0, UBLK_U_CMD_START_DEV);
        assert_eq!(commands[0].1.dev_id, 1);
        assert_eq!(commands[1].0, UBLK_U_CMD_STOP_DEV);
        assert_eq!(commands[1].1.dev_id, 2);
    }

    #[test]
    fn test_mock_kernel_shim_auto_increment_dev_id() {
        let shim = MockKernelShim::new();
        let ctrl_fd = shim.open_ctrl_device().unwrap();

        let cmd = UblkCtrlCmd::default();

        let r1 = shim.submit_ctrl_cmd(&ctrl_fd, UBLK_U_CMD_ADD_DEV, &cmd).unwrap();
        let r2 = shim.submit_ctrl_cmd(&ctrl_fd, UBLK_U_CMD_ADD_DEV, &cmd).unwrap();
        let r3 = shim.submit_ctrl_cmd(&ctrl_fd, UBLK_U_CMD_ADD_DEV, &cmd).unwrap();

        assert_eq!(r1.dev_info.unwrap().dev_id, 0);
        assert_eq!(r2.dev_info.unwrap().dev_id, 1);
        assert_eq!(r3.dev_info.unwrap().dev_id, 2);
    }

    // ========================================================================
    // MockDaemonShim Tests
    // ========================================================================

    #[test]
    fn test_mock_daemon_shim_create_io_uring() {
        let shim = MockDaemonShim::new();
        let ring = shim.create_io_uring(128);
        assert!(ring.is_ok());
    }

    #[test]
    fn test_mock_daemon_shim_mmap_iod_buffer() {
        let shim = MockDaemonShim::new();
        let ctrl_shim = MockKernelShim::new();
        let char_fd = ctrl_shim.open_char_device(0).unwrap();

        let result = shim.mmap_iod_buffer(&char_fd, 4096);
        assert!(result.is_ok());
        let mmap = result.unwrap();
        assert!(!mmap.ptr.is_null());
        assert_eq!(mmap.len, 4096);

        // Cleanup
        shim.munmap(mmap.ptr, mmap.len).unwrap();
    }

    #[test]
    fn test_mock_daemon_shim_mmap_fail() {
        let shim = MockDaemonShim::new();
        shim.fail_mmap.store(true, std::sync::atomic::Ordering::SeqCst);

        let ctrl_shim = MockKernelShim::new();
        let char_fd = ctrl_shim.open_char_device(0).unwrap();

        let result = shim.mmap_iod_buffer(&char_fd, 4096);
        assert!(result.is_err());
        assert_eq!(result.unwrap_err().raw_os_error(), Some(libc::ENOMEM));
    }

    #[test]
    fn test_mock_daemon_shim_mmap_anonymous() {
        let shim = MockDaemonShim::new();
        let result = shim.mmap_anonymous(8192);
        assert!(result.is_ok());
        let mmap = result.unwrap();
        assert!(!mmap.ptr.is_null());
        assert_eq!(mmap.len, 8192);

        // Cleanup
        shim.munmap(mmap.ptr, mmap.len).unwrap();
    }

    #[test]
    fn test_mock_daemon_shim_pread() {
        let shim = MockDaemonShim::new();
        let ctrl_shim = MockKernelShim::new();
        let fd = ctrl_shim.open_char_device(0).unwrap();

        let mut buf = [0u8; 512];
        let result = shim.pread(&fd, &mut buf, 0);
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), 512);

        // Verify test pattern
        for (i, &byte) in buf.iter().enumerate() {
            assert_eq!(byte, (i % 256) as u8);
        }
    }

    #[test]
    fn test_mock_daemon_shim_pwrite() {
        let shim = MockDaemonShim::new();
        let ctrl_shim = MockKernelShim::new();
        let fd = ctrl_shim.open_char_device(0).unwrap();

        let buf = [0xABu8; 512];
        let result = shim.pwrite(&fd, &buf, 0);
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), 512);
    }

    // ========================================================================
    // MockIoUring Tests
    // ========================================================================

    #[test]
    fn test_mock_io_uring_submit_fetch() {
        let mut ring = MockIoUring::new(128);
        let result = ring.submit_fetch(3, 5, 0);
        assert!(result.is_ok());
        assert_eq!(ring.fetch_count.load(std::sync::atomic::Ordering::SeqCst), 1);
    }

    #[test]
    fn test_mock_io_uring_submit_commit_fetch() {
        let mut ring = MockIoUring::new(128);
        let result = ring.submit_commit_fetch(3, 7, 0, 4096);
        assert!(result.is_ok());
        assert_eq!(ring.commit_count.load(std::sync::atomic::Ordering::SeqCst), 1);
    }

    #[test]
    fn test_mock_io_uring_completions() {
        let mut ring = MockIoUring::new(128);

        // Submit some commands
        ring.submit_fetch(3, 0, 0).unwrap();
        ring.submit_fetch(3, 1, 0).unwrap();
        ring.submit_commit_fetch(3, 2, 0, 4096).unwrap();

        // Get completions
        let completions = ring.get_completions();
        assert_eq!(completions.len(), 3);

        // Completions should be cleared
        let completions2 = ring.get_completions();
        assert_eq!(completions2.len(), 0);
    }

    #[test]
    fn test_mock_io_uring_submit_and_wait() {
        let mut ring = MockIoUring::new(128);

        ring.submit_fetch(3, 0, 0).unwrap();
        ring.submit_fetch(3, 1, 0).unwrap();

        let count = ring.submit_and_wait(2).unwrap();
        assert_eq!(count, 2);
    }

    // ========================================================================
    // CtrlCmdResult Tests
    // ========================================================================

    #[test]
    fn test_ctrl_cmd_result_debug() {
        let result = CtrlCmdResult { retval: 0, dev_info: None };
        let debug = format!("{:?}", result);
        assert!(debug.contains("retval"));
    }

    #[test]
    fn test_ctrl_cmd_result_clone() {
        let result = CtrlCmdResult {
            retval: 42,
            dev_info: Some(UblkCtrlDevInfo { dev_id: 5, ..Default::default() }),
        };
        let cloned = result.clone();
        assert_eq!(cloned.retval, 42);
        assert_eq!(cloned.dev_info.unwrap().dev_id, 5);
    }

    // ========================================================================
    // MmapResult Tests
    // ========================================================================

    #[test]
    fn test_mmap_result_debug() {
        let result = MmapResult { ptr: std::ptr::null_mut(), len: 4096 };
        let debug = format!("{:?}", result);
        assert!(debug.contains("len"));
    }

    // ========================================================================
    // IoCompletion Tests
    // ========================================================================

    #[test]
    fn test_io_completion_debug() {
        let completion = IoCompletion { user_data: 0x12345678, result: -5 };
        let debug = format!("{:?}", completion);
        assert!(debug.contains("user_data"));
        assert!(debug.contains("result"));
    }

    #[test]
    fn test_io_completion_clone() {
        let completion = IoCompletion { user_data: 0xABCD, result: 4096 };
        let cloned = completion.clone();
        assert_eq!(cloned.user_data, 0xABCD);
        assert_eq!(cloned.result, 4096);
    }

    // ========================================================================
    // Default Impl Tests
    // ========================================================================

    #[test]
    fn test_mock_kernel_shim_default() {
        let shim = MockKernelShim::default();
        assert_eq!(shim.next_dev_id.load(std::sync::atomic::Ordering::SeqCst), 0);
    }

    #[test]
    fn test_mock_daemon_shim_default() {
        let shim = MockDaemonShim::default();
        assert!(!shim.fail_mmap.load(std::sync::atomic::Ordering::SeqCst));
    }

    #[test]
    fn test_mock_io_uring_add_completion() {
        let mut ring = MockIoUring::new(128);
        ring.add_completion(0x1234, 4096);

        let completions = ring.get_completions();
        assert_eq!(completions.len(), 1);
        assert_eq!(completions[0].user_data, 0x1234);
        assert_eq!(completions[0].result, 4096);
    }

    #[test]
    fn test_mock_daemon_shim_munmap() {
        let shim = MockDaemonShim::new();

        // Allocate some memory
        let mmap = shim.mmap_anonymous(4096).unwrap();
        assert!(!mmap.ptr.is_null());

        // Unmap it
        let result = shim.munmap(mmap.ptr, mmap.len);
        assert!(result.is_ok());
    }
}
