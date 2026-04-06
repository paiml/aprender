//! ublk control device operations via io_uring
//!
//! Linux 6.0+ requires io_uring URING_CMD for ublk control commands.
//! This module implements the control path using io_uring.

use crate::ublk::sys::*;
#[cfg(not(test))]
use io_uring::{opcode, squeue, types, IoUring};
use nix::libc;
#[cfg(not(test))]
use nix::libc::O_RDWR;
#[cfg(not(test))]
use std::ffi::CString;
#[cfg(not(test))]
use std::os::fd::AsRawFd;
use std::os::fd::{FromRawFd, OwnedFd};
use thiserror::Error;

/// IoUring type with 128-byte SQE support for URING_CMD
#[cfg(not(test))]
type IoUring128 = IoUring<squeue::Entry128>;

#[derive(Error, Debug)]
pub enum CtrlError {
    #[error("Failed to open control device: {0}")]
    OpenCtrl(std::io::Error),

    #[error("Failed to create io_uring: {0}")]
    IoUring(std::io::Error),

    #[error("Failed to add device {dev_id}: {source}")]
    AddDev { dev_id: i32, source: std::io::Error },

    #[error("Failed to set params for device {dev_id}: {source}")]
    SetParams { dev_id: i32, source: std::io::Error },

    #[error("Failed to start device {dev_id}: {source}")]
    StartDev { dev_id: i32, source: std::io::Error },

    #[error("Failed to stop device {dev_id}: {source}")]
    StopDev { dev_id: i32, source: std::io::Error },

    #[error("Failed to delete device {dev_id}: {source}")]
    DelDev { dev_id: i32, source: std::io::Error },

    #[error("Failed to open char device /dev/ublkc{dev_id}: {source}")]
    OpenChar { dev_id: i32, source: std::io::Error },
}

impl CtrlError {
    /// Returns true if this error is transient and the operation should be retried.
    /// Pattern from pepita error handling.
    #[inline]
    pub fn is_retriable(&self) -> bool {
        match self {
            Self::AddDev { source, .. }
            | Self::SetParams { source, .. }
            | Self::StartDev { source, .. }
            | Self::StopDev { source, .. }
            | Self::DelDev { source, .. } => {
                matches!(source.raw_os_error(), Some(libc::EAGAIN) | Some(libc::EBUSY))
            }
            _ => false,
        }
    }

    /// Returns true if this is a resource exhaustion error.
    #[inline]
    pub fn is_resource_error(&self) -> bool {
        match self {
            Self::IoUring(e) | Self::OpenCtrl(e) | Self::OpenChar { source: e, .. } => {
                matches!(
                    e.raw_os_error(),
                    Some(libc::ENOMEM) | Some(libc::EMFILE) | Some(libc::ENFILE)
                )
            }
            _ => false,
        }
    }

    /// Convert to negative errno for POSIX compatibility.
    /// Pattern from pepita error handling.
    pub fn to_errno(&self) -> i32 {
        match self {
            Self::OpenCtrl(e) | Self::IoUring(e) => -e.raw_os_error().unwrap_or(libc::ENODEV),
            Self::AddDev { source, .. } => -source.raw_os_error().unwrap_or(libc::ENODEV),
            Self::SetParams { source, .. } => -source.raw_os_error().unwrap_or(libc::EINVAL),
            Self::StartDev { source, .. } => -source.raw_os_error().unwrap_or(libc::EIO),
            Self::StopDev { source, .. } => -source.raw_os_error().unwrap_or(libc::EIO),
            Self::DelDev { source, .. } => -source.raw_os_error().unwrap_or(libc::EIO),
            Self::OpenChar { source, .. } => -source.raw_os_error().unwrap_or(libc::ENODEV),
        }
    }

    /// Get the device ID associated with this error, if any.
    pub fn dev_id(&self) -> Option<i32> {
        match self {
            Self::AddDev { dev_id, .. }
            | Self::SetParams { dev_id, .. }
            | Self::StartDev { dev_id, .. }
            | Self::StopDev { dev_id, .. }
            | Self::DelDev { dev_id, .. }
            | Self::OpenChar { dev_id, .. } => Some(*dev_id),
            _ => None,
        }
    }
}

/// Configuration for a new ublk device
#[derive(Debug, Clone)]
pub struct DeviceConfig {
    pub dev_id: i32,
    pub nr_hw_queues: u16,
    pub queue_depth: u16,
    pub dev_size: u64,
    pub flags: u64,
}

impl Default for DeviceConfig {
    fn default() -> Self {
        Self {
            dev_id: -1,
            nr_hw_queues: UBLK_DEF_NR_HW_QUEUES,
            queue_depth: UBLK_DEF_QUEUE_DEPTH,
            dev_size: 0,
            flags: UBLK_F_USER_COPY | UBLK_F_CMD_IOCTL_ENCODE,
        }
    }
}

/// ublk control handle using io_uring
#[cfg(not(test))]
pub struct UblkCtrl {
    ctrl_fd: OwnedFd,
    ring: IoUring128,
    dev_id: i32,
    dev_info: UblkCtrlDevInfo,
    params: UblkParams,
    /// True if this is a cloned handle - clones don't clean up on Drop
    is_clone: bool,
}

#[cfg(not(test))]
impl UblkCtrl {
    /// Open control device and add a new ublk device via io_uring
    pub fn new(config: DeviceConfig) -> Result<Self, CtrlError> {
        let ctrl_path = CString::new(UBLK_CTRL_DEV).expect("UBLK_CTRL_DEV is a valid C string");
        // SAFETY: libc::open is called with:
        // - ctrl_path is a valid null-terminated C string (verified by CString::new)
        // - O_RDWR is a valid flag for opening the control device
        // - Return value is checked for negative (error) before use
        let ctrl_fd = unsafe { libc::open(ctrl_path.as_ptr(), O_RDWR) };
        if ctrl_fd < 0 {
            return Err(CtrlError::OpenCtrl(std::io::Error::last_os_error()));
        }
        // SAFETY: ctrl_fd is a valid file descriptor (verified by check above)
        // and will be owned exclusively by this OwnedFd
        let ctrl_fd = unsafe { OwnedFd::from_raw_fd(ctrl_fd) };

        // Create io_uring with 128-byte SQE support for URING_CMD
        let ring: IoUring128 = IoUring128::builder().build(4).map_err(CtrlError::IoUring)?;

        let mut dev_info = UblkCtrlDevInfo {
            nr_hw_queues: config.nr_hw_queues,
            queue_depth: config.queue_depth,
            max_io_buf_bytes: UBLK_MAX_IO_BUF_BYTES,
            dev_id: if config.dev_id < 0 { u32::MAX } else { config.dev_id as u32 },
            ublksrv_pid: std::process::id() as i32,
            flags: config.flags,
            ..Default::default()
        };

        // Submit ADD_DEV command via io_uring
        // queue_id must be -1 (0xFFFF) for device-level commands
        let cmd = UblkCtrlCmd {
            dev_id: dev_info.dev_id,
            queue_id: u16::MAX, // -1 means not queue-specific
            addr: &mut dev_info as *mut _ as u64,
            len: std::mem::size_of::<UblkCtrlDevInfo>() as u16,
            ..Default::default()
        };

        let mut ctrl = Self {
            ctrl_fd,
            ring,
            dev_id: -1,
            dev_info,
            params: UblkParams::default(),
            is_clone: false, // Primary handle - will clean up on Drop
        };

        ctrl.submit_ctrl_cmd(UBLK_U_CMD_ADD_DEV, cmd)
            .map_err(|e| CtrlError::AddDev { dev_id: config.dev_id, source: e })?;

        ctrl.dev_id = ctrl.dev_info.dev_id as i32;

        tracing::debug!(
            dev_id = ctrl.dev_id,
            nr_hw_queues = ctrl.dev_info.nr_hw_queues,
            queue_depth = ctrl.dev_info.queue_depth,
            max_io_buf_bytes = ctrl.dev_info.max_io_buf_bytes,
            flags = ctrl.dev_info.flags,
            "Device info after ADD_DEV"
        );

        let dev_sectors = config.dev_size / SECTOR_SIZE;
        ctrl.params = UblkParams {
            len: std::mem::size_of::<UblkParams>() as u32,
            types: UBLK_PARAM_TYPE_BASIC,
            basic: UblkParamBasic {
                logical_bs_shift: 12,
                physical_bs_shift: 12,
                io_opt_shift: 12,
                io_min_shift: 12,
                max_sectors: UBLK_MAX_IO_BUF_BYTES / SECTOR_SIZE as u32,
                dev_sectors,
                ..Default::default()
            },
            ..Default::default()
        };

        ctrl.set_params()?;
        Ok(ctrl)
    }

    /// Submit a control command via io_uring URING_CMD
    fn submit_ctrl_cmd(&mut self, cmd_op: u32, cmd: UblkCtrlCmd) -> Result<i32, std::io::Error> {
        // Wrap in UblkCtrlCmdExt (80 bytes) for io_uring SQE cmd field
        let cmd_ext = UblkCtrlCmdExt { cmd, padding: [0; 48] };
        let cmd_bytes: [u8; 80] = unsafe { std::mem::transmute(cmd_ext) };

        let sqe = opcode::UringCmd80::new(types::Fd(self.ctrl_fd.as_raw_fd()), cmd_op)
            .cmd(cmd_bytes)
            .build()
            .user_data(0x100);

        unsafe {
            self.ring.submission().push(&sqe).map_err(|_| std::io::Error::other("SQ full"))?;
        }

        tracing::info!(cmd_op, "Submitting control command, waiting for completion...");
        self.ring.submit_and_wait(1)?;
        tracing::info!(cmd_op, "Control command completed");

        let cqe = self.ring.completion().next().ok_or_else(|| std::io::Error::other("No CQE"))?;

        let result = cqe.result();
        if result < 0 {
            return Err(std::io::Error::from_raw_os_error(-result));
        }

        Ok(result)
    }

    /// Clone the control handle for use in another thread.
    /// This clones the FD and creates a new io_uring for the new handle.
    /// Cloned handles do NOT clean up the device on Drop - only the primary handle does.
    pub fn clone_handle(&self) -> Result<Self, CtrlError> {
        let ctrl_fd = self.ctrl_fd.try_clone().map_err(CtrlError::OpenCtrl)?;
        let ring: IoUring128 = IoUring128::builder().build(4).map_err(CtrlError::IoUring)?;

        Ok(Self {
            ctrl_fd,
            ring,
            dev_id: self.dev_id,
            dev_info: self.dev_info,
            params: self.params,
            is_clone: true, // Cloned handle - does NOT clean up on Drop
        })
    }

    fn set_params(&mut self) -> Result<(), CtrlError> {
        let cmd = UblkCtrlCmd {
            dev_id: self.dev_id as u32,
            addr: &self.params as *const _ as u64,
            len: std::mem::size_of::<UblkParams>() as u16,
            ..Default::default()
        };

        self.submit_ctrl_cmd(UBLK_U_CMD_SET_PARAMS, cmd)
            .map_err(|e| CtrlError::SetParams { dev_id: self.dev_id, source: e })?;

        Ok(())
    }

    pub fn start(&mut self) -> Result<(), CtrlError> {
        let mut cmd = UblkCtrlCmd { dev_id: self.dev_id as u32, ..Default::default() };
        cmd.data[0] = self.dev_info.ublksrv_pid as u64;

        self.submit_ctrl_cmd(UBLK_U_CMD_START_DEV, cmd)
            .map_err(|e| CtrlError::StartDev { dev_id: self.dev_id, source: e })?;

        Ok(())
    }

    pub fn stop(&mut self) -> Result<(), CtrlError> {
        let cmd = UblkCtrlCmd { dev_id: self.dev_id as u32, ..Default::default() };

        self.submit_ctrl_cmd(UBLK_U_CMD_STOP_DEV, cmd)
            .map_err(|e| CtrlError::StopDev { dev_id: self.dev_id, source: e })?;

        Ok(())
    }

    fn delete(&mut self) -> Result<(), CtrlError> {
        let cmd = UblkCtrlCmd { dev_id: self.dev_id as u32, ..Default::default() };

        self.submit_ctrl_cmd(UBLK_U_CMD_DEL_DEV, cmd)
            .map_err(|e| CtrlError::DelDev { dev_id: self.dev_id, source: e })?;

        Ok(())
    }

    pub fn open_char_dev(&self) -> Result<OwnedFd, CtrlError> {
        let path = format!("{}{}", UBLK_CHAR_DEV_FMT, self.dev_id);
        let cpath = CString::new(path).expect("char device path is a valid C string");
        let fd = unsafe { libc::open(cpath.as_ptr(), O_RDWR) };
        if fd < 0 {
            return Err(CtrlError::OpenChar {
                dev_id: self.dev_id,
                source: std::io::Error::last_os_error(),
            });
        }
        Ok(unsafe { OwnedFd::from_raw_fd(fd) })
    }

    #[inline]
    pub fn dev_id(&self) -> i32 {
        self.dev_id
    }

    #[inline]
    pub fn dev_info(&self) -> &UblkCtrlDevInfo {
        &self.dev_info
    }

    #[inline]
    pub fn queue_depth(&self) -> u16 {
        self.dev_info.queue_depth
    }

    #[inline]
    pub fn max_io_buf_bytes(&self) -> u32 {
        self.dev_info.max_io_buf_bytes
    }

    pub fn block_dev_path(&self) -> String {
        format!("{}{}", UBLK_BLOCK_DEV_FMT, self.dev_id)
    }
}

#[cfg(not(test))]
impl Drop for UblkCtrl {
    fn drop(&mut self) {
        // Only the primary handle (not clones) cleans up the device
        if !self.is_clone {
            tracing::warn!(
                dev_id = self.dev_id,
                "UblkCtrl::drop called - stopping and deleting device"
            );
            let _ = self.stop();
            let _ = self.delete();
        } else {
            tracing::debug!(dev_id = self.dev_id, "UblkCtrl::drop skipped (clone)");
        }
    }
}

// ============================================================================
// Mock infrastructure for testing without kernel access
// ============================================================================

/// Mock control device for testing
#[cfg(test)]
pub struct MockUblkCtrl {
    pub dev_id: i32,
    pub dev_info: UblkCtrlDevInfo,
    pub params: UblkParams,
    pub started: bool,
    pub stopped: bool,
    pub deleted: bool,
    pub commands_issued: Vec<(u32, UblkCtrlCmd)>,
}

#[cfg(test)]
impl MockUblkCtrl {
    pub fn new(config: DeviceConfig) -> Result<Self, CtrlError> {
        // Simulate device creation
        let dev_id = if config.dev_id < 0 { 0 } else { config.dev_id };

        let dev_info = UblkCtrlDevInfo {
            nr_hw_queues: config.nr_hw_queues,
            queue_depth: config.queue_depth,
            max_io_buf_bytes: UBLK_MAX_IO_BUF_BYTES,
            dev_id: dev_id as u32,
            ublksrv_pid: std::process::id() as i32,
            flags: config.flags,
            ..Default::default()
        };

        let dev_sectors = config.dev_size / SECTOR_SIZE;
        let params = UblkParams {
            len: std::mem::size_of::<UblkParams>() as u32,
            types: UBLK_PARAM_TYPE_BASIC,
            basic: UblkParamBasic {
                logical_bs_shift: 12,
                physical_bs_shift: 12,
                io_opt_shift: 12,
                io_min_shift: 12,
                max_sectors: UBLK_MAX_IO_BUF_BYTES / SECTOR_SIZE as u32,
                dev_sectors,
                ..Default::default()
            },
            ..Default::default()
        };

        Ok(Self {
            dev_id,
            dev_info,
            params,
            started: false,
            stopped: false,
            deleted: false,
            commands_issued: Vec::new(),
        })
    }

    pub fn submit_ctrl_cmd(
        &mut self,
        cmd_op: u32,
        cmd: UblkCtrlCmd,
    ) -> Result<i32, std::io::Error> {
        // Record the command
        self.commands_issued.push((cmd_op, cmd));

        // Simulate command processing
        match cmd_op {
            UBLK_U_CMD_ADD_DEV => Ok(0),
            UBLK_U_CMD_SET_PARAMS => Ok(0),
            UBLK_U_CMD_START_DEV => {
                self.started = true;
                Ok(0)
            }
            UBLK_U_CMD_STOP_DEV => {
                self.stopped = true;
                Ok(0)
            }
            UBLK_U_CMD_DEL_DEV => {
                self.deleted = true;
                Ok(0)
            }
            _ => Err(std::io::Error::from_raw_os_error(libc::EINVAL)),
        }
    }

    pub fn set_params(&mut self) -> Result<(), CtrlError> {
        let cmd = UblkCtrlCmd {
            dev_id: self.dev_id as u32,
            addr: &self.params as *const _ as u64,
            len: std::mem::size_of::<UblkParams>() as u16,
            ..Default::default()
        };

        self.submit_ctrl_cmd(UBLK_U_CMD_SET_PARAMS, cmd)
            .map_err(|e| CtrlError::SetParams { dev_id: self.dev_id, source: e })?;

        Ok(())
    }

    pub fn start(&mut self) -> Result<(), CtrlError> {
        let mut cmd = UblkCtrlCmd { dev_id: self.dev_id as u32, ..Default::default() };
        cmd.data[0] = self.dev_info.ublksrv_pid as u64;

        self.submit_ctrl_cmd(UBLK_U_CMD_START_DEV, cmd)
            .map_err(|e| CtrlError::StartDev { dev_id: self.dev_id, source: e })?;

        Ok(())
    }

    pub fn stop(&mut self) -> Result<(), CtrlError> {
        let cmd = UblkCtrlCmd { dev_id: self.dev_id as u32, ..Default::default() };

        self.submit_ctrl_cmd(UBLK_U_CMD_STOP_DEV, cmd)
            .map_err(|e| CtrlError::StopDev { dev_id: self.dev_id, source: e })?;

        Ok(())
    }

    pub fn delete(&mut self) -> Result<(), CtrlError> {
        let cmd = UblkCtrlCmd { dev_id: self.dev_id as u32, ..Default::default() };

        self.submit_ctrl_cmd(UBLK_U_CMD_DEL_DEV, cmd)
            .map_err(|e| CtrlError::DelDev { dev_id: self.dev_id, source: e })?;

        Ok(())
    }

    pub fn open_char_dev(&self) -> Result<OwnedFd, CtrlError> {
        // Create a dummy fd using memfd for testing
        // This allows tests to have a valid fd without kernel access
        let fd = unsafe { libc::memfd_create(c"mock_char".as_ptr(), 0) };
        if fd < 0 {
            return Err(CtrlError::OpenChar {
                dev_id: self.dev_id,
                source: std::io::Error::last_os_error(),
            });
        }
        Ok(unsafe { OwnedFd::from_raw_fd(fd) })
    }

    #[inline]
    pub fn dev_id(&self) -> i32 {
        self.dev_id
    }

    #[inline]
    pub fn dev_info(&self) -> &UblkCtrlDevInfo {
        &self.dev_info
    }

    #[inline]
    pub fn queue_depth(&self) -> u16 {
        self.dev_info.queue_depth
    }

    #[inline]
    pub fn max_io_buf_bytes(&self) -> u32 {
        self.dev_info.max_io_buf_bytes
    }

    pub fn block_dev_path(&self) -> String {
        format!("{}{}", UBLK_BLOCK_DEV_FMT, self.dev_id)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ========================================================================
    // MockUblkCtrl Tests - Full lifecycle without kernel
    // ========================================================================

    #[test]
    fn test_mock_ctrl_new() {
        let config = DeviceConfig { dev_size: 1 << 30, ..Default::default() };
        let ctrl = MockUblkCtrl::new(config).unwrap();
        assert_eq!(ctrl.dev_id(), 0);
        assert_eq!(ctrl.queue_depth(), UBLK_DEF_QUEUE_DEPTH);
    }

    #[test]
    fn test_mock_ctrl_with_specific_dev_id() {
        let config = DeviceConfig { dev_id: 5, dev_size: 1 << 30, ..Default::default() };
        let ctrl = MockUblkCtrl::new(config).unwrap();
        assert_eq!(ctrl.dev_id(), 5);
    }

    #[test]
    fn test_mock_ctrl_start_stop() {
        let config = DeviceConfig { dev_size: 1 << 30, ..Default::default() };
        let mut ctrl = MockUblkCtrl::new(config).unwrap();

        assert!(!ctrl.started);
        ctrl.start().unwrap();
        assert!(ctrl.started);

        assert!(!ctrl.stopped);
        ctrl.stop().unwrap();
        assert!(ctrl.stopped);
    }

    #[test]
    fn test_mock_ctrl_delete() {
        let config = DeviceConfig { dev_size: 1 << 30, ..Default::default() };
        let mut ctrl = MockUblkCtrl::new(config).unwrap();

        assert!(!ctrl.deleted);
        ctrl.delete().unwrap();
        assert!(ctrl.deleted);
    }

    #[test]
    fn test_mock_ctrl_set_params() {
        let config = DeviceConfig { dev_size: 1 << 30, ..Default::default() };
        let mut ctrl = MockUblkCtrl::new(config).unwrap();
        ctrl.set_params().unwrap();

        // Verify SET_PARAMS command was issued
        assert!(ctrl.commands_issued.iter().any(|(op, _)| *op == UBLK_U_CMD_SET_PARAMS));
    }

    #[test]
    fn test_mock_ctrl_block_dev_path() {
        let config = DeviceConfig { dev_id: 42, dev_size: 1 << 30, ..Default::default() };
        let ctrl = MockUblkCtrl::new(config).unwrap();
        assert_eq!(ctrl.block_dev_path(), "/dev/ublkb42");
    }

    #[test]
    fn test_mock_ctrl_dev_info() {
        let config = DeviceConfig {
            nr_hw_queues: 4,
            queue_depth: 256,
            dev_size: 1 << 30,
            flags: UBLK_F_USER_COPY | UBLK_F_NEED_GET_DATA,
            ..Default::default()
        };
        let ctrl = MockUblkCtrl::new(config).unwrap();

        let info = ctrl.dev_info();
        assert_eq!(info.nr_hw_queues, 4);
        assert_eq!(info.queue_depth, 256);
        assert!(info.flags & UBLK_F_USER_COPY != 0);
        assert!(info.flags & UBLK_F_NEED_GET_DATA != 0);
    }

    #[test]
    fn test_mock_ctrl_max_io_buf_bytes() {
        let config = DeviceConfig::default();
        let ctrl = MockUblkCtrl::new(config).unwrap();
        assert_eq!(ctrl.max_io_buf_bytes(), UBLK_MAX_IO_BUF_BYTES);
    }

    #[test]
    fn test_mock_ctrl_params_dev_sectors() {
        let dev_size = 1 << 30; // 1GB
        let config = DeviceConfig { dev_size, ..Default::default() };
        let ctrl = MockUblkCtrl::new(config).unwrap();

        let expected_sectors = dev_size / SECTOR_SIZE;
        assert_eq!(ctrl.params.basic.dev_sectors, expected_sectors);
    }

    #[test]
    fn test_mock_ctrl_full_lifecycle() {
        let config = DeviceConfig {
            dev_id: 10,
            dev_size: 2 << 30, // 2GB
            nr_hw_queues: 2,
            queue_depth: 64,
            flags: UBLK_F_USER_COPY,
        };
        let mut ctrl = MockUblkCtrl::new(config).unwrap();

        // Set params
        ctrl.set_params().unwrap();

        // Start
        ctrl.start().unwrap();
        assert!(ctrl.started);
        assert_eq!(ctrl.block_dev_path(), "/dev/ublkb10");

        // Stop
        ctrl.stop().unwrap();
        assert!(ctrl.stopped);

        // Delete
        ctrl.delete().unwrap();
        assert!(ctrl.deleted);

        // Verify all commands were issued in order
        let ops: Vec<u32> = ctrl.commands_issued.iter().map(|(op, _)| *op).collect();
        assert!(ops.contains(&UBLK_U_CMD_SET_PARAMS));
        assert!(ops.contains(&UBLK_U_CMD_START_DEV));
        assert!(ops.contains(&UBLK_U_CMD_STOP_DEV));
        assert!(ops.contains(&UBLK_U_CMD_DEL_DEV));
    }

    #[test]
    fn test_mock_ctrl_command_dev_id_consistency() {
        let config = DeviceConfig { dev_id: 7, dev_size: 1 << 30, ..Default::default() };
        let mut ctrl = MockUblkCtrl::new(config).unwrap();

        ctrl.start().unwrap();
        ctrl.stop().unwrap();

        // All commands should have correct dev_id
        for (_, cmd) in &ctrl.commands_issued {
            assert_eq!(cmd.dev_id, 7);
        }
    }

    #[test]
    fn test_mock_ctrl_submit_unknown_command() {
        let config = DeviceConfig::default();
        let mut ctrl = MockUblkCtrl::new(config).unwrap();

        let cmd = UblkCtrlCmd::default();
        let result = ctrl.submit_ctrl_cmd(0xFFFFFFFF, cmd);

        assert!(result.is_err());
    }

    #[test]
    fn test_mock_ctrl_submit_add_dev() {
        let config = DeviceConfig::default();
        let mut ctrl = MockUblkCtrl::new(config).unwrap();

        let cmd = UblkCtrlCmd::default();
        let result = ctrl.submit_ctrl_cmd(UBLK_U_CMD_ADD_DEV, cmd);

        assert!(result.is_ok());
        assert_eq!(result.unwrap(), 0);
        assert!(ctrl.commands_issued.iter().any(|(op, _)| *op == UBLK_U_CMD_ADD_DEV));
    }

    // ========================================================================
    // DeviceConfig Tests
    // ========================================================================

    /// PERF-004: Default queue depth increased to 256 for higher IOPS
    #[test]
    fn test_device_config_default() {
        let config = DeviceConfig::default();
        assert_eq!(config.dev_id, -1);
        assert_eq!(config.nr_hw_queues, 1);
        assert_eq!(config.queue_depth, 256); // PERF-004: 128 -> 256
        assert!(config.flags & UBLK_F_USER_COPY != 0);
    }

    #[test]
    fn test_device_config_custom() {
        let config = DeviceConfig {
            dev_id: 5,
            nr_hw_queues: 4,
            queue_depth: 256,
            dev_size: 1 << 30,
            flags: UBLK_F_USER_COPY | UBLK_F_NEED_GET_DATA,
        };
        assert_eq!(config.dev_id, 5);
        assert_eq!(config.nr_hw_queues, 4);
        assert_eq!(config.queue_depth, 256);
        assert_eq!(config.dev_size, 1 << 30);
        assert!(config.flags & UBLK_F_USER_COPY != 0);
        assert!(config.flags & UBLK_F_NEED_GET_DATA != 0);
    }

    #[test]
    fn test_device_config_clone() {
        let config = DeviceConfig { dev_id: 10, dev_size: 1 << 20, ..Default::default() };
        let cloned = config.clone();
        assert_eq!(cloned.dev_id, 10);
        assert_eq!(cloned.dev_size, 1 << 20);
    }

    #[test]
    fn test_device_config_debug() {
        let config = DeviceConfig::default();
        let debug_str = format!("{:?}", config);
        assert!(debug_str.contains("DeviceConfig"));
        assert!(debug_str.contains("dev_id"));
    }

    // ========================================================================
    // Path Tests
    // ========================================================================

    #[test]
    fn test_device_paths() {
        let dev_id = 42;
        let block_path = format!("{}{}", UBLK_BLOCK_DEV_FMT, dev_id);
        assert_eq!(block_path, "/dev/ublkb42");
    }

    #[test]
    fn test_char_device_path() {
        let dev_id = 0;
        let char_path = format!("{}{}", UBLK_CHAR_DEV_FMT, dev_id);
        assert_eq!(char_path, "/dev/ublkc0");
    }

    #[test]
    fn test_ctrl_device_path() {
        assert_eq!(UBLK_CTRL_DEV, "/dev/ublk-control");
    }

    // ========================================================================
    // UblkParams Tests
    // ========================================================================

    #[test]
    fn test_params_layout() {
        let params = UblkParams {
            len: std::mem::size_of::<UblkParams>() as u32,
            types: UBLK_PARAM_TYPE_BASIC,
            basic: UblkParamBasic {
                logical_bs_shift: 12,
                dev_sectors: 1024 * 1024,
                ..Default::default()
            },
            ..Default::default()
        };
        assert_eq!(params.basic.logical_bs_shift, 12);
    }

    #[test]
    fn test_params_discard() {
        let params = UblkParams {
            len: std::mem::size_of::<UblkParams>() as u32,
            types: UBLK_PARAM_TYPE_BASIC | UBLK_PARAM_TYPE_DISCARD,
            basic: UblkParamBasic::default(),
            discard: UblkParamDiscard {
                discard_alignment: 4096,
                discard_granularity: 4096,
                max_discard_sectors: 1 << 20,
                max_write_zeroes_sectors: 1 << 20,
                max_discard_segments: 1,
                reserved0: 0,
            },
            ..Default::default()
        };
        assert!(params.types & UBLK_PARAM_TYPE_DISCARD != 0);
        assert_eq!(params.discard.discard_granularity, 4096);
    }

    #[test]
    fn test_params_basic_sectors() {
        let dev_size = 1 << 30; // 1GB
        let dev_sectors = dev_size / SECTOR_SIZE;
        let params = UblkParams {
            len: std::mem::size_of::<UblkParams>() as u32,
            types: UBLK_PARAM_TYPE_BASIC,
            basic: UblkParamBasic {
                logical_bs_shift: 12,
                physical_bs_shift: 12,
                io_opt_shift: 12,
                io_min_shift: 12,
                max_sectors: UBLK_MAX_IO_BUF_BYTES / SECTOR_SIZE as u32,
                dev_sectors,
                ..Default::default()
            },
            ..Default::default()
        };
        assert_eq!(params.basic.dev_sectors, dev_sectors);
        assert_eq!(params.basic.logical_bs_shift, 12);
    }

    // ========================================================================
    // CtrlError Tests
    // ========================================================================

    #[test]
    fn test_ctrl_error_display_open_ctrl() {
        let err = CtrlError::OpenCtrl(std::io::Error::from_raw_os_error(2));
        let msg = format!("{}", err);
        assert!(msg.contains("Failed to open control device"));
    }

    #[test]
    fn test_ctrl_error_display_io_uring() {
        let err = CtrlError::IoUring(std::io::Error::from_raw_os_error(12));
        let msg = format!("{}", err);
        assert!(msg.contains("Failed to create io_uring"));
    }

    #[test]
    fn test_ctrl_error_display_add_dev() {
        let err = CtrlError::AddDev { dev_id: 5, source: std::io::Error::from_raw_os_error(1) };
        let msg = format!("{}", err);
        assert!(msg.contains("Failed to add device 5"));
    }

    #[test]
    fn test_ctrl_error_display_set_params() {
        let err = CtrlError::SetParams { dev_id: 3, source: std::io::Error::from_raw_os_error(22) };
        let msg = format!("{}", err);
        assert!(msg.contains("Failed to set params for device 3"));
    }

    #[test]
    fn test_ctrl_error_display_start_dev() {
        let err = CtrlError::StartDev { dev_id: 7, source: std::io::Error::from_raw_os_error(16) };
        let msg = format!("{}", err);
        assert!(msg.contains("Failed to start device 7"));
    }

    #[test]
    fn test_ctrl_error_display_stop_dev() {
        let err = CtrlError::StopDev { dev_id: 2, source: std::io::Error::from_raw_os_error(19) };
        let msg = format!("{}", err);
        assert!(msg.contains("Failed to stop device 2"));
    }

    #[test]
    fn test_ctrl_error_display_del_dev() {
        let err = CtrlError::DelDev { dev_id: 9, source: std::io::Error::from_raw_os_error(6) };
        let msg = format!("{}", err);
        assert!(msg.contains("Failed to delete device 9"));
    }

    #[test]
    fn test_ctrl_error_display_open_char() {
        let err = CtrlError::OpenChar { dev_id: 1, source: std::io::Error::from_raw_os_error(13) };
        let msg = format!("{}", err);
        assert!(msg.contains("Failed to open char device /dev/ublkc1"));
    }

    #[test]
    fn test_ctrl_error_debug() {
        let err = CtrlError::AddDev { dev_id: 0, source: std::io::Error::from_raw_os_error(1) };
        let debug = format!("{:?}", err);
        assert!(debug.contains("AddDev"));
    }

    // ========================================================================
    // UblkCtrlCmd Tests
    // ========================================================================

    #[test]
    fn test_ctrl_cmd_default() {
        let cmd = UblkCtrlCmd::default();
        // dev_id = u32::MAX means "auto-assign"
        assert_eq!(cmd.dev_id, u32::MAX);
        // queue_id = u16::MAX means "not queue-specific" (device-level command)
        assert_eq!(cmd.queue_id, u16::MAX);
        assert_eq!(cmd.len, 0);
        assert_eq!(cmd.addr, 0);
    }

    #[test]
    fn test_ctrl_cmd_with_data() {
        let cmd = UblkCtrlCmd {
            dev_id: 42,
            queue_id: u16::MAX,
            len: 128,
            addr: 0x12345678,
            data: [0xDEADBEEF],
            ..Default::default()
        };
        assert_eq!(cmd.dev_id, 42);
        assert_eq!(cmd.queue_id, u16::MAX);
        assert_eq!(cmd.data[0], 0xDEADBEEF);
    }

    // ========================================================================
    // UblkCtrlCmdExt Tests
    // ========================================================================

    #[test]
    fn test_ctrl_cmd_ext_size() {
        // UblkCtrlCmdExt must be exactly 80 bytes for io_uring SQE cmd field
        assert_eq!(std::mem::size_of::<UblkCtrlCmdExt>(), 80);
    }

    #[test]
    fn test_ctrl_cmd_ext_transmute_roundtrip() {
        let cmd = UblkCtrlCmd {
            dev_id: 123,
            queue_id: 456,
            len: 789,
            addr: 0xDEADBEEF,
            data: [0x11223344],
            ..Default::default()
        };
        let cmd_ext = UblkCtrlCmdExt { cmd, padding: [0; 48] };
        let bytes: [u8; 80] = unsafe { std::mem::transmute(cmd_ext) };
        let recovered: UblkCtrlCmdExt = unsafe { std::mem::transmute(bytes) };
        assert_eq!(recovered.cmd.dev_id, 123);
        assert_eq!(recovered.cmd.addr, 0xDEADBEEF);
    }

    // ========================================================================
    // Command Building Tests (extracted logic)
    // ========================================================================

    #[test]
    fn test_add_dev_cmd_building() {
        let mut dev_info = UblkCtrlDevInfo {
            nr_hw_queues: 2,
            queue_depth: 128,
            max_io_buf_bytes: UBLK_MAX_IO_BUF_BYTES,
            dev_id: u32::MAX,
            ublksrv_pid: 12345,
            flags: UBLK_F_USER_COPY,
            ..Default::default()
        };

        let cmd = UblkCtrlCmd {
            dev_id: dev_info.dev_id,
            queue_id: u16::MAX,
            addr: &mut dev_info as *mut _ as u64,
            len: std::mem::size_of::<UblkCtrlDevInfo>() as u16,
            ..Default::default()
        };

        assert_eq!(cmd.dev_id, u32::MAX);
        assert_eq!(cmd.queue_id, u16::MAX);
        assert!(cmd.addr != 0);
        assert_eq!(cmd.len as usize, std::mem::size_of::<UblkCtrlDevInfo>());
    }

    #[test]
    fn test_set_params_cmd_building() {
        let dev_id = 5i32;
        let params = UblkParams {
            len: std::mem::size_of::<UblkParams>() as u32,
            types: UBLK_PARAM_TYPE_BASIC,
            basic: UblkParamBasic {
                logical_bs_shift: 12,
                dev_sectors: 1 << 20,
                ..Default::default()
            },
            ..Default::default()
        };

        let cmd = UblkCtrlCmd {
            dev_id: dev_id as u32,
            addr: &params as *const _ as u64,
            len: std::mem::size_of::<UblkParams>() as u16,
            ..Default::default()
        };

        assert_eq!(cmd.dev_id, 5);
        assert!(cmd.addr != 0);
        assert_eq!(cmd.len as usize, std::mem::size_of::<UblkParams>());
    }

    #[test]
    fn test_start_cmd_building() {
        let dev_id = 3i32;
        let pid = std::process::id() as i32;

        let mut cmd = UblkCtrlCmd { dev_id: dev_id as u32, ..Default::default() };
        cmd.data[0] = pid as u64;

        assert_eq!(cmd.dev_id, 3);
        assert_eq!(cmd.data[0], pid as u64);
    }

    #[test]
    fn test_stop_cmd_building() {
        let dev_id = 7i32;

        let cmd = UblkCtrlCmd { dev_id: dev_id as u32, ..Default::default() };

        assert_eq!(cmd.dev_id, 7);
    }

    #[test]
    fn test_dev_info_initialization() {
        let config = DeviceConfig {
            nr_hw_queues: 4,
            queue_depth: 256,
            dev_id: -1,
            dev_size: 1 << 30,
            flags: UBLK_F_USER_COPY,
        };

        let dev_info = UblkCtrlDevInfo {
            nr_hw_queues: config.nr_hw_queues,
            queue_depth: config.queue_depth,
            max_io_buf_bytes: UBLK_MAX_IO_BUF_BYTES,
            dev_id: if config.dev_id < 0 { u32::MAX } else { config.dev_id as u32 },
            ublksrv_pid: std::process::id() as i32,
            flags: config.flags,
            ..Default::default()
        };

        assert_eq!(dev_info.nr_hw_queues, 4);
        assert_eq!(dev_info.queue_depth, 256);
        assert_eq!(dev_info.dev_id, u32::MAX);
        assert!(dev_info.flags & UBLK_F_USER_COPY != 0);
    }

    #[test]
    fn test_params_initialization() {
        let dev_size = 2u64 << 30; // 2GB
        let dev_sectors = dev_size / SECTOR_SIZE;

        let params = UblkParams {
            len: std::mem::size_of::<UblkParams>() as u32,
            types: UBLK_PARAM_TYPE_BASIC,
            basic: UblkParamBasic {
                logical_bs_shift: 12,
                physical_bs_shift: 12,
                io_opt_shift: 12,
                io_min_shift: 12,
                max_sectors: UBLK_MAX_IO_BUF_BYTES / SECTOR_SIZE as u32,
                dev_sectors,
                ..Default::default()
            },
            ..Default::default()
        };

        assert_eq!(params.len as usize, std::mem::size_of::<UblkParams>());
        assert_eq!(params.types, UBLK_PARAM_TYPE_BASIC);
        assert_eq!(params.basic.dev_sectors, dev_sectors);
        assert_eq!(params.basic.logical_bs_shift, 12);
    }

    #[test]
    fn test_ctrl_cmd_ext_byte_conversion() {
        let cmd = UblkCtrlCmd {
            dev_id: 42,
            queue_id: u16::MAX,
            len: 128,
            addr: 0xDEADBEEF,
            data: [0x12345678],
            ..Default::default()
        };
        let cmd_ext = UblkCtrlCmdExt { cmd, padding: [0; 48] };
        let cmd_bytes: [u8; 80] = unsafe { std::mem::transmute(cmd_ext) };

        // Verify first 4 bytes are dev_id (little-endian)
        let dev_id = u32::from_le_bytes([cmd_bytes[0], cmd_bytes[1], cmd_bytes[2], cmd_bytes[3]]);
        assert_eq!(dev_id, 42);
    }

    #[test]
    fn test_error_mapping() {
        // Test that we can create errors from raw os errors
        let io_err = std::io::Error::from_raw_os_error(libc::ENOENT);
        let ctrl_err = CtrlError::OpenCtrl(io_err);

        let msg = format!("{}", ctrl_err);
        assert!(msg.contains("Failed to open control device"));
    }

    #[test]
    fn test_result_to_error_conversion() {
        // Test negative result to error conversion
        let result = -libc::EPERM;
        let err = std::io::Error::from_raw_os_error(-result);
        assert_eq!(err.raw_os_error(), Some(libc::EPERM));
    }

    // ========================================================================
    // CtrlError Method Tests (pepita patterns)
    // ========================================================================

    #[test]
    fn test_ctrl_error_is_retriable_eagain() {
        let err = CtrlError::StartDev {
            dev_id: 0,
            source: std::io::Error::from_raw_os_error(libc::EAGAIN),
        };
        assert!(err.is_retriable());
    }

    #[test]
    fn test_ctrl_error_is_retriable_ebusy() {
        let err = CtrlError::StopDev {
            dev_id: 0,
            source: std::io::Error::from_raw_os_error(libc::EBUSY),
        };
        assert!(err.is_retriable());
    }

    #[test]
    fn test_ctrl_error_not_retriable() {
        let err = CtrlError::OpenCtrl(std::io::Error::from_raw_os_error(libc::ENOENT));
        assert!(!err.is_retriable());
    }

    #[test]
    fn test_ctrl_error_is_resource_error_enomem() {
        let err = CtrlError::IoUring(std::io::Error::from_raw_os_error(libc::ENOMEM));
        assert!(err.is_resource_error());
    }

    #[test]
    fn test_ctrl_error_is_resource_error_emfile() {
        let err = CtrlError::OpenCtrl(std::io::Error::from_raw_os_error(libc::EMFILE));
        assert!(err.is_resource_error());
    }

    #[test]
    fn test_ctrl_error_not_resource_error() {
        let err = CtrlError::AddDev {
            dev_id: 0,
            source: std::io::Error::from_raw_os_error(libc::EINVAL),
        };
        assert!(!err.is_resource_error());
    }

    #[test]
    fn test_ctrl_error_to_errno() {
        let err = CtrlError::AddDev {
            dev_id: 5,
            source: std::io::Error::from_raw_os_error(libc::EEXIST),
        };
        assert_eq!(err.to_errno(), -libc::EEXIST);
    }

    #[test]
    fn test_ctrl_error_to_errno_default() {
        let err = CtrlError::SetParams {
            dev_id: 0,
            source: std::io::Error::new(std::io::ErrorKind::Other, "custom"),
        };
        assert_eq!(err.to_errno(), -libc::EINVAL);
    }

    #[test]
    fn test_ctrl_error_dev_id_some() {
        let err = CtrlError::StartDev { dev_id: 42, source: std::io::Error::from_raw_os_error(1) };
        assert_eq!(err.dev_id(), Some(42));
    }

    #[test]
    fn test_ctrl_error_dev_id_none() {
        let err = CtrlError::OpenCtrl(std::io::Error::from_raw_os_error(1));
        assert_eq!(err.dev_id(), None);
    }

    #[test]
    fn test_ctrl_error_all_variants_have_errno() {
        // Ensure all error variants produce valid negative errno
        let errors = vec![
            CtrlError::OpenCtrl(std::io::Error::from_raw_os_error(1)),
            CtrlError::IoUring(std::io::Error::from_raw_os_error(2)),
            CtrlError::AddDev { dev_id: 0, source: std::io::Error::from_raw_os_error(3) },
            CtrlError::SetParams { dev_id: 0, source: std::io::Error::from_raw_os_error(4) },
            CtrlError::StartDev { dev_id: 0, source: std::io::Error::from_raw_os_error(5) },
            CtrlError::StopDev { dev_id: 0, source: std::io::Error::from_raw_os_error(6) },
            CtrlError::DelDev { dev_id: 0, source: std::io::Error::from_raw_os_error(7) },
            CtrlError::OpenChar { dev_id: 0, source: std::io::Error::from_raw_os_error(8) },
        ];

        for err in errors {
            let errno = err.to_errno();
            assert!(errno < 0, "errno should be negative: {}", errno);
            assert!(errno >= -4095, "errno out of range: {}", errno);
        }
    }

    // ========================================================================
    // Comprehensive is_retriable() tests for ALL device operation variants
    // ========================================================================

    #[test]
    fn test_is_retriable_add_dev_eagain() {
        let err = CtrlError::AddDev {
            dev_id: 1,
            source: std::io::Error::from_raw_os_error(libc::EAGAIN),
        };
        assert!(err.is_retriable());
    }

    #[test]
    fn test_is_retriable_add_dev_ebusy() {
        let err =
            CtrlError::AddDev { dev_id: 1, source: std::io::Error::from_raw_os_error(libc::EBUSY) };
        assert!(err.is_retriable());
    }

    #[test]
    fn test_is_retriable_set_params_eagain() {
        let err = CtrlError::SetParams {
            dev_id: 2,
            source: std::io::Error::from_raw_os_error(libc::EAGAIN),
        };
        assert!(err.is_retriable());
    }

    #[test]
    fn test_is_retriable_del_dev_ebusy() {
        let err =
            CtrlError::DelDev { dev_id: 3, source: std::io::Error::from_raw_os_error(libc::EBUSY) };
        assert!(err.is_retriable());
    }

    #[test]
    fn test_is_retriable_open_char_not_retriable() {
        // OpenChar is NOT in the retriable list
        let err = CtrlError::OpenChar {
            dev_id: 4,
            source: std::io::Error::from_raw_os_error(libc::EAGAIN),
        };
        assert!(!err.is_retriable());
    }

    // ========================================================================
    // Comprehensive dev_id() tests for ALL variants with dev_id
    // ========================================================================

    #[test]
    fn test_dev_id_add_dev() {
        let err = CtrlError::AddDev { dev_id: 10, source: std::io::Error::from_raw_os_error(1) };
        assert_eq!(err.dev_id(), Some(10));
    }

    #[test]
    fn test_dev_id_set_params() {
        let err = CtrlError::SetParams { dev_id: 20, source: std::io::Error::from_raw_os_error(1) };
        assert_eq!(err.dev_id(), Some(20));
    }

    #[test]
    fn test_dev_id_stop_dev() {
        let err = CtrlError::StopDev { dev_id: 30, source: std::io::Error::from_raw_os_error(1) };
        assert_eq!(err.dev_id(), Some(30));
    }

    #[test]
    fn test_dev_id_del_dev() {
        let err = CtrlError::DelDev { dev_id: 40, source: std::io::Error::from_raw_os_error(1) };
        assert_eq!(err.dev_id(), Some(40));
    }

    #[test]
    fn test_dev_id_open_char() {
        let err = CtrlError::OpenChar { dev_id: 50, source: std::io::Error::from_raw_os_error(1) };
        assert_eq!(err.dev_id(), Some(50));
    }

    #[test]
    fn test_dev_id_io_uring_none() {
        let err = CtrlError::IoUring(std::io::Error::from_raw_os_error(1));
        assert_eq!(err.dev_id(), None);
    }

    // ========================================================================
    // Comprehensive to_errno() tests for ALL variants
    // ========================================================================

    #[test]
    fn test_to_errno_open_ctrl() {
        let err = CtrlError::OpenCtrl(std::io::Error::from_raw_os_error(libc::ENOENT));
        assert_eq!(err.to_errno(), -libc::ENOENT);
    }

    #[test]
    fn test_to_errno_io_uring() {
        let err = CtrlError::IoUring(std::io::Error::from_raw_os_error(libc::ENOMEM));
        assert_eq!(err.to_errno(), -libc::ENOMEM);
    }

    #[test]
    fn test_to_errno_start_dev() {
        let err =
            CtrlError::StartDev { dev_id: 1, source: std::io::Error::from_raw_os_error(libc::EIO) };
        assert_eq!(err.to_errno(), -libc::EIO);
    }

    #[test]
    fn test_to_errno_stop_dev() {
        let err = CtrlError::StopDev {
            dev_id: 1,
            source: std::io::Error::from_raw_os_error(libc::EBUSY),
        };
        assert_eq!(err.to_errno(), -libc::EBUSY);
    }

    #[test]
    fn test_to_errno_del_dev() {
        let err =
            CtrlError::DelDev { dev_id: 1, source: std::io::Error::from_raw_os_error(libc::EPERM) };
        assert_eq!(err.to_errno(), -libc::EPERM);
    }

    #[test]
    fn test_to_errno_open_char() {
        let err = CtrlError::OpenChar {
            dev_id: 1,
            source: std::io::Error::from_raw_os_error(libc::EACCES),
        };
        assert_eq!(err.to_errno(), -libc::EACCES);
    }

    // ========================================================================
    // is_resource_error() tests for OpenChar variant
    // ========================================================================

    #[test]
    fn test_is_resource_error_open_char_enfile() {
        let err = CtrlError::OpenChar {
            dev_id: 1,
            source: std::io::Error::from_raw_os_error(libc::ENFILE),
        };
        assert!(err.is_resource_error());
    }

    #[test]
    fn test_is_resource_error_open_char_not_resource() {
        let err = CtrlError::OpenChar {
            dev_id: 1,
            source: std::io::Error::from_raw_os_error(libc::ENOENT),
        };
        assert!(!err.is_resource_error());
    }
}
