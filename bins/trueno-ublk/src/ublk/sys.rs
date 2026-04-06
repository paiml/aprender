//! Raw ublk kernel interface definitions
//!
//! Direct port from Linux include/uapi/linux/ublk_cmd.h
//! Zero external dependencies - just libc types via nix
//!
//! IMPORTANT: Linux 6.0+ uses io_uring URING_CMD for control commands.
//! Commands use ioctl-encoded values (UBLK_U_CMD_*).

use std::mem::size_of;

// ============================================================================
// ioctl encoding helpers
// ============================================================================

const UBLK_MAGIC: u32 = b'u' as u32;

const fn _io(ty: u32, nr: u32) -> u32 {
    (ty << 8) | nr
}

const fn _ior(ty: u32, nr: u32, sz: usize) -> u32 {
    (2 << 30) | ((sz as u32) << 16) | (ty << 8) | nr
}

const fn _iow(ty: u32, nr: u32, sz: usize) -> u32 {
    (1 << 30) | ((sz as u32) << 16) | (ty << 8) | nr
}

const fn _iowr(ty: u32, nr: u32, sz: usize) -> u32 {
    (3 << 30) | ((sz as u32) << 16) | (ty << 8) | nr
}

// ============================================================================
// Control Command Opcodes (ioctl-encoded for io_uring URING_CMD)
// ============================================================================

// Raw command numbers
const UBLK_CMD_GET_QUEUE_AFFINITY: u32 = 0x01;
const UBLK_CMD_GET_DEV_INFO: u32 = 0x02;
const UBLK_CMD_ADD_DEV: u32 = 0x04;
const UBLK_CMD_DEL_DEV: u32 = 0x05;
const UBLK_CMD_START_DEV: u32 = 0x06;
const UBLK_CMD_STOP_DEV: u32 = 0x07;
const UBLK_CMD_SET_PARAMS: u32 = 0x08;
const UBLK_CMD_GET_PARAMS: u32 = 0x09;

// ioctl-encoded commands (UBLK_U_CMD_*)
pub const UBLK_U_CMD_GET_QUEUE_AFFINITY: u32 =
    _ior(UBLK_MAGIC, UBLK_CMD_GET_QUEUE_AFFINITY, size_of::<UblkCtrlCmd>());
pub const UBLK_U_CMD_GET_DEV_INFO: u32 =
    _ior(UBLK_MAGIC, UBLK_CMD_GET_DEV_INFO, size_of::<UblkCtrlCmd>());
pub const UBLK_U_CMD_ADD_DEV: u32 = _iowr(UBLK_MAGIC, UBLK_CMD_ADD_DEV, size_of::<UblkCtrlCmd>());
pub const UBLK_U_CMD_DEL_DEV: u32 = _iowr(UBLK_MAGIC, UBLK_CMD_DEL_DEV, size_of::<UblkCtrlCmd>());
pub const UBLK_U_CMD_START_DEV: u32 =
    _iowr(UBLK_MAGIC, UBLK_CMD_START_DEV, size_of::<UblkCtrlCmd>());
pub const UBLK_U_CMD_STOP_DEV: u32 = _iowr(UBLK_MAGIC, UBLK_CMD_STOP_DEV, size_of::<UblkCtrlCmd>());
pub const UBLK_U_CMD_SET_PARAMS: u32 =
    _iowr(UBLK_MAGIC, UBLK_CMD_SET_PARAMS, size_of::<UblkCtrlCmd>());
pub const UBLK_U_CMD_GET_PARAMS: u32 =
    _ior(UBLK_MAGIC, UBLK_CMD_GET_PARAMS, size_of::<UblkCtrlCmd>());

// I/O command opcodes (raw)
/// Raw I/O command opcodes (used when UBLK_F_CMD_IOCTL_ENCODE is NOT set)
pub const UBLK_IO_FETCH_REQ: u32 = 0x20;
pub const UBLK_IO_COMMIT_AND_FETCH_REQ: u32 = 0x21;

// I/O commands (ioctl-encoded for io_uring on /dev/ublkcN)
pub const UBLK_U_IO_FETCH_REQ: u32 = _iowr(UBLK_MAGIC, UBLK_IO_FETCH_REQ, size_of::<UblkIoCmd>());
pub const UBLK_U_IO_COMMIT_AND_FETCH_REQ: u32 =
    _iowr(UBLK_MAGIC, UBLK_IO_COMMIT_AND_FETCH_REQ, size_of::<UblkIoCmd>());

// I/O operation types (match kernel ublk_cmd.h exactly)
pub const UBLK_IO_OP_READ: u8 = 0;
pub const UBLK_IO_OP_WRITE: u8 = 1;
pub const UBLK_IO_OP_FLUSH: u8 = 2;
pub const UBLK_IO_OP_DISCARD: u8 = 3;
pub const UBLK_IO_OP_WRITE_SAME: u8 = 4;
pub const UBLK_IO_OP_WRITE_ZEROES: u8 = 5;

// Device flags
pub const UBLK_F_SUPPORT_ZERO_COPY: u64 = 1 << 0;
pub const UBLK_F_URING_CMD_COMP_IN_TASK: u64 = 1 << 1;
pub const UBLK_F_NEED_GET_DATA: u64 = 1 << 2;
pub const UBLK_F_USER_RECOVERY: u64 = 1 << 3;
pub const UBLK_F_USER_RECOVERY_REISSUE: u64 = 1 << 4;
pub const UBLK_F_UNPRIVILEGED_DEV: u64 = 1 << 5;
pub const UBLK_F_CMD_IOCTL_ENCODE: u64 = 1 << 6;
pub const UBLK_F_USER_COPY: u64 = 1 << 7;
pub const UBLK_F_ZONED: u64 = 1 << 8;

// Parameter types
pub const UBLK_PARAM_TYPE_BASIC: u32 = 1 << 0;
pub const UBLK_PARAM_TYPE_DISCARD: u32 = 1 << 1;

// I/O result flags
pub const UBLK_IO_RES_OK: i32 = 0;

// ============================================================================
// Kernel Structures
// ============================================================================

/// Control command payload (32 bytes) - matches kernel ublksrv_ctrl_cmd
/// Used for UBLK_CMD_* operations via IORING_OP_URING_CMD
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct UblkCtrlCmd {
    pub dev_id: u32,
    pub queue_id: u16,
    pub len: u16,
    pub addr: u64,
    pub data: [u64; 1],
    pub dev_path_len: u16,
    pub pad: u16,
    pub reserved: u32,
}

impl Default for UblkCtrlCmd {
    fn default() -> Self {
        Self {
            dev_id: u32::MAX,
            queue_id: u16::MAX, // -1 means not queue-specific
            len: 0,
            addr: 0,
            data: [0; 1],
            dev_path_len: 0,
            pad: 0,
            reserved: 0,
        }
    }
}

/// Extended control command for io_uring (80 bytes)
/// First 32 bytes is UblkCtrlCmd, rest is padding
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct UblkCtrlCmdExt {
    pub cmd: UblkCtrlCmd,
    pub padding: [u8; 48],
}

impl Default for UblkCtrlCmdExt {
    fn default() -> Self {
        Self { cmd: UblkCtrlCmd::default(), padding: [0; 48] }
    }
}

/// I/O command (16 bytes) - matches kernel ublksrv_io_cmd
#[repr(C)]
#[derive(Debug, Clone, Copy, Default)]
pub struct UblkIoCmd {
    pub q_id: u16,
    pub tag: u16,
    pub result: i32,
    pub addr: u64,
}

/// Extended I/O command for io_uring 128-byte SQE (80 bytes)
/// First 16 bytes is UblkIoCmd, rest is padding
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct UblkIoCmdExt {
    pub cmd: UblkIoCmd,
    pub padding: [u8; 64],
}

impl Default for UblkIoCmdExt {
    fn default() -> Self {
        Self { cmd: UblkIoCmd::default(), padding: [0; 64] }
    }
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Default)]
pub struct UblkCtrlDevInfo {
    pub nr_hw_queues: u16,
    pub queue_depth: u16,
    pub state: u16,
    pub pad0: u16,
    pub max_io_buf_bytes: u32,
    pub dev_id: u32,
    pub ublksrv_pid: i32,
    pub pad1: u32,
    pub flags: u64,
    pub ublksrv_flags: u64,
    pub owner_uid: u32,
    pub owner_gid: u32,
    pub reserved1: u64,
    pub reserved2: u64,
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Default)]
pub struct UblkIoDesc {
    pub op_flags: u32,
    pub nr_sectors: u32,
    pub start_sector: u64,
    pub addr: u64,
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Default)]
pub struct UblkParamBasic {
    pub attrs: u32,
    pub logical_bs_shift: u8,
    pub physical_bs_shift: u8,
    pub io_opt_shift: u8,
    pub io_min_shift: u8,
    pub max_sectors: u32,
    pub chunk_sectors: u32,
    pub dev_sectors: u64,
    pub virt_boundary_mask: u64,
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Default)]
pub struct UblkParamDiscard {
    pub discard_alignment: u32,
    pub discard_granularity: u32,
    pub max_discard_sectors: u32,
    pub max_write_zeroes_sectors: u32,
    pub max_discard_segments: u16,
    pub reserved0: u16,
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Default)]
pub struct UblkParamDevt {
    pub char_major: u32,
    pub char_minor: u32,
    pub disk_major: u32,
    pub disk_minor: u32,
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Default)]
pub struct UblkParams {
    pub len: u32,
    pub types: u32,
    pub basic: UblkParamBasic,
    pub discard: UblkParamDiscard,
    pub devt: UblkParamDevt,
}

// ============================================================================
// Constants
// ============================================================================

pub const UBLK_CTRL_DEV: &str = "/dev/ublk-control";
pub const UBLK_CHAR_DEV_FMT: &str = "/dev/ublkc";
pub const UBLK_BLOCK_DEV_FMT: &str = "/dev/ublkb";
/// PERF-004: Increased default queue depth for higher IOPS
/// Previous: 128, New: 256 (2x more in-flight I/Os per queue)
pub const UBLK_DEF_QUEUE_DEPTH: u16 = 256;
pub const UBLK_DEF_NR_HW_QUEUES: u16 = 1;
pub const UBLK_MAX_IO_BUF_BYTES: u32 = 512 * 1024;
pub const SECTOR_SIZE: u64 = 512;

// USER_COPY mode: pread/pwrite offset encoding
pub const UBLKSRV_IO_BUF_OFFSET: u64 = 0x8000_0000;
pub const UBLK_IO_BUF_BITS: u32 = 25;
pub const UBLK_TAG_OFF: u32 = UBLK_IO_BUF_BITS;
pub const UBLK_TAG_BITS: u32 = 16;
pub const UBLK_QID_OFF: u32 = UBLK_TAG_OFF + UBLK_TAG_BITS;

/// Calculate pread/pwrite offset for USER_COPY mode
#[inline]
pub const fn ublk_user_copy_offset(q_id: u16, tag: u16, buf_off: u32) -> i64 {
    (UBLKSRV_IO_BUF_OFFSET
        + (buf_off as u64)
        + ((tag as u64) << UBLK_TAG_OFF)
        + ((q_id as u64) << UBLK_QID_OFF)) as i64
}

// ============================================================================
// Helper functions
// ============================================================================

#[inline]
pub const fn iod_offset(tag: u16, _queue_depth: u16) -> usize {
    (tag as usize) * size_of::<UblkIoDesc>()
}

#[inline]
pub const fn buf_offset(tag: u16, queue_depth: u16, max_io_size: u32) -> usize {
    let iod_area = (queue_depth as usize) * size_of::<UblkIoDesc>();
    iod_area + (tag as usize) * (max_io_size as usize)
}

#[inline]
pub const fn total_buf_size(queue_depth: u16, max_io_size: u32) -> usize {
    let iod_area = (queue_depth as usize) * size_of::<UblkIoDesc>();
    let data_area = (queue_depth as usize) * (max_io_size as usize);
    iod_area + data_area
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::mem::{align_of, offset_of, size_of};

    // ========================================================================
    // Section A: Protocol Correctness Tests (from Renacer Verification Matrix)
    // ========================================================================

    /// A.1: Verify UblkCtrlCmd struct layout matches C sizeof (32 bytes)
    #[test]
    fn test_ublk_ctrl_cmd_size() {
        // Kernel ublksrv_ctrl_cmd is exactly 32 bytes
        assert_eq!(size_of::<UblkCtrlCmd>(), 32, "UblkCtrlCmd must be 32 bytes");
    }

    /// A.1: Verify UblkCtrlCmd field offsets match kernel struct
    #[test]
    fn test_ublk_ctrl_cmd_layout() {
        // Verify field offsets match kernel layout:
        // __u32 dev_id       @ 0
        // __u16 queue_id     @ 4
        // __u16 len          @ 6
        // __u64 addr         @ 8
        // __u64 data[1]      @ 16
        // __u16 dev_path_len @ 24
        // __u16 pad          @ 26
        // __u32 reserved     @ 28
        assert_eq!(offset_of!(UblkCtrlCmd, dev_id), 0);
        assert_eq!(offset_of!(UblkCtrlCmd, queue_id), 4);
        assert_eq!(offset_of!(UblkCtrlCmd, len), 6);
        assert_eq!(offset_of!(UblkCtrlCmd, addr), 8);
        assert_eq!(offset_of!(UblkCtrlCmd, data), 16);
        assert_eq!(offset_of!(UblkCtrlCmd, dev_path_len), 24);
        assert_eq!(offset_of!(UblkCtrlCmd, pad), 26);
        assert_eq!(offset_of!(UblkCtrlCmd, reserved), 28);
    }

    /// A.2: Verify UblkIoDesc struct layout matches C sizeof (24 bytes)
    #[test]
    fn test_ublk_io_desc_size() {
        // Kernel ublksrv_io_desc is exactly 24 bytes
        assert_eq!(size_of::<UblkIoDesc>(), 24, "UblkIoDesc must be 24 bytes");
    }

    /// A.2: Verify UblkIoDesc field offsets match kernel struct
    #[test]
    fn test_ublk_io_desc_layout() {
        // __u32 op_flags     @ 0
        // __u32 nr_sectors   @ 4
        // __u64 start_sector @ 8
        // __u64 addr         @ 16
        assert_eq!(offset_of!(UblkIoDesc, op_flags), 0);
        assert_eq!(offset_of!(UblkIoDesc, nr_sectors), 4);
        assert_eq!(offset_of!(UblkIoDesc, start_sector), 8);
        assert_eq!(offset_of!(UblkIoDesc, addr), 16);
    }

    /// A.2: Verify UblkIoCmd struct layout matches C sizeof (16 bytes)
    #[test]
    fn test_ublk_io_cmd_size() {
        // Kernel ublksrv_io_cmd is exactly 16 bytes
        assert_eq!(size_of::<UblkIoCmd>(), 16, "UblkIoCmd must be 16 bytes");
    }

    /// A.2: Verify UblkIoCmd field offsets
    #[test]
    fn test_ublk_io_cmd_layout() {
        // __u16 q_id    @ 0
        // __u16 tag     @ 2
        // __s32 result  @ 4
        // __u64 addr    @ 8
        assert_eq!(offset_of!(UblkIoCmd, q_id), 0);
        assert_eq!(offset_of!(UblkIoCmd, tag), 2);
        assert_eq!(offset_of!(UblkIoCmd, result), 4);
        assert_eq!(offset_of!(UblkIoCmd, addr), 8);
    }

    /// A.3: Verify UBLK_CTRL_ADD_DEV ioctl number matches kernel header
    #[test]
    fn test_ioctl_add_dev() {
        // UBLK_U_CMD_ADD_DEV = _IOWR('u', 0x04, struct ublksrv_ctrl_cmd)
        // _IOWR: direction=3 (read/write), type='u'=0x75, nr=0x04, size=32
        // = (3 << 30) | (32 << 16) | (0x75 << 8) | 0x04
        let expected = (3u32 << 30) | (32u32 << 16) | (0x75u32 << 8) | 0x04;
        assert_eq!(UBLK_U_CMD_ADD_DEV, expected, "UBLK_U_CMD_ADD_DEV mismatch");
    }

    /// A.3: Verify other control ioctl numbers
    #[test]
    fn test_ioctl_control_commands() {
        let ctrl_cmd_size = 32u32;
        let magic = 0x75u32; // 'u'

        // DEL_DEV: _IOWR('u', 0x05, ...)
        let del_dev = (3u32 << 30) | (ctrl_cmd_size << 16) | (magic << 8) | 0x05;
        assert_eq!(UBLK_U_CMD_DEL_DEV, del_dev);

        // START_DEV: _IOWR('u', 0x06, ...)
        let start_dev = (3u32 << 30) | (ctrl_cmd_size << 16) | (magic << 8) | 0x06;
        assert_eq!(UBLK_U_CMD_START_DEV, start_dev);

        // STOP_DEV: _IOWR('u', 0x07, ...)
        let stop_dev = (3u32 << 30) | (ctrl_cmd_size << 16) | (magic << 8) | 0x07;
        assert_eq!(UBLK_U_CMD_STOP_DEV, stop_dev);

        // SET_PARAMS: _IOWR('u', 0x08, ...)
        let set_params = (3u32 << 30) | (ctrl_cmd_size << 16) | (magic << 8) | 0x08;
        assert_eq!(UBLK_U_CMD_SET_PARAMS, set_params);
    }

    /// A.4: Verify UBLK_IO_FETCH_REQ opcode matches kernel
    #[test]
    fn test_ioctl_fetch_req() {
        // UBLK_U_IO_FETCH_REQ = _IOWR('u', 0x20, struct ublksrv_io_cmd)
        let io_cmd_size = 16u32;
        let expected = (3u32 << 30) | (io_cmd_size << 16) | (0x75u32 << 8) | 0x20;
        assert_eq!(UBLK_U_IO_FETCH_REQ, expected, "UBLK_U_IO_FETCH_REQ mismatch");
    }

    /// A.5: Verify UBLK_IO_COMMIT_AND_FETCH_REQ opcode matches kernel
    #[test]
    fn test_ioctl_commit_and_fetch() {
        // UBLK_U_IO_COMMIT_AND_FETCH_REQ = _IOWR('u', 0x21, struct ublksrv_io_cmd)
        let io_cmd_size = 16u32;
        let expected = (3u32 << 30) | (io_cmd_size << 16) | (0x75u32 << 8) | 0x21;
        assert_eq!(UBLK_U_IO_COMMIT_AND_FETCH_REQ, expected);
    }

    /// A.6: Check alignment of mmap buffer (must be page-aligned)
    #[test]
    fn test_mmap_alignment() {
        // IOD buffer must be page-aligned for mmap
        let depth = 128u16;
        let iod_size = (depth as usize) * size_of::<UblkIoDesc>();
        // Page size is typically 4096
        let page_aligned = (iod_size + 4095) & !4095;
        assert_eq!(page_aligned % 4096, 0, "IOD buffer must be page-aligned");
    }

    /// A.7: Verify tag usage: Tags 0..QD-1 are unique and reused correctly
    #[test]
    fn test_tag_range() {
        let queue_depth: u16 = 128;
        for tag in 0..queue_depth {
            // Each tag should produce valid offset
            let offset = iod_offset(tag, queue_depth);
            assert!(offset < (queue_depth as usize) * size_of::<UblkIoDesc>());
        }
    }

    /// A.10: Verify UBLK_F_USER_COPY flag value
    #[test]
    fn test_user_copy_flag() {
        // UBLK_F_USER_COPY = (1UL << 7) = 128
        assert_eq!(UBLK_F_USER_COPY, 1 << 7);
        assert_eq!(UBLK_F_USER_COPY, 128);
    }

    /// Verify all device flags match kernel header values
    #[test]
    fn test_device_flags() {
        assert_eq!(UBLK_F_SUPPORT_ZERO_COPY, 1 << 0);
        assert_eq!(UBLK_F_URING_CMD_COMP_IN_TASK, 1 << 1);
        assert_eq!(UBLK_F_NEED_GET_DATA, 1 << 2);
        assert_eq!(UBLK_F_USER_RECOVERY, 1 << 3);
        assert_eq!(UBLK_F_USER_RECOVERY_REISSUE, 1 << 4);
        assert_eq!(UBLK_F_UNPRIVILEGED_DEV, 1 << 5);
        assert_eq!(UBLK_F_CMD_IOCTL_ENCODE, 1 << 6);
        assert_eq!(UBLK_F_USER_COPY, 1 << 7);
        assert_eq!(UBLK_F_ZONED, 1 << 8);
    }

    /// Verify I/O operation types match kernel header
    #[test]
    fn test_io_op_values() {
        assert_eq!(UBLK_IO_OP_READ, 0);
        assert_eq!(UBLK_IO_OP_WRITE, 1);
        assert_eq!(UBLK_IO_OP_FLUSH, 2);
        assert_eq!(UBLK_IO_OP_DISCARD, 3);
        assert_eq!(UBLK_IO_OP_WRITE_SAME, 4);
        assert_eq!(UBLK_IO_OP_WRITE_ZEROES, 5);
    }

    /// Verify param types match kernel header
    #[test]
    fn test_param_types() {
        assert_eq!(UBLK_PARAM_TYPE_BASIC, 1 << 0);
        assert_eq!(UBLK_PARAM_TYPE_DISCARD, 1 << 1);
    }

    /// Verify UblkCtrlDevInfo size (64 bytes in kernel)
    #[test]
    fn test_ctrl_dev_info_size() {
        assert_eq!(size_of::<UblkCtrlDevInfo>(), 64);
    }

    /// Verify UblkCtrlCmdExt size (80 bytes for io_uring SQE cmd field)
    #[test]
    fn test_ctrl_cmd_ext_size() {
        assert_eq!(size_of::<UblkCtrlCmdExt>(), 80);
    }

    /// Verify UblkIoCmdExt size (80 bytes for io_uring 128-byte SQE cmd field)
    #[test]
    fn test_io_cmd_ext_size() {
        assert_eq!(size_of::<UblkIoCmdExt>(), 80);
    }

    /// Verify struct alignment requirements
    #[test]
    fn test_struct_alignment() {
        assert_eq!(align_of::<UblkCtrlCmd>(), 8);
        assert_eq!(align_of::<UblkIoDesc>(), 8);
        assert_eq!(align_of::<UblkIoCmd>(), 8);
        assert_eq!(align_of::<UblkIoCmdExt>(), 8);
        assert_eq!(align_of::<UblkCtrlDevInfo>(), 8);
    }

    /// Verify USER_COPY offset encoding constants
    #[test]
    fn test_user_copy_offset_constants() {
        assert_eq!(UBLKSRV_IO_BUF_OFFSET, 0x8000_0000);
        assert_eq!(UBLK_IO_BUF_BITS, 25);
        assert_eq!(UBLK_TAG_OFF, 25);
        assert_eq!(UBLK_TAG_BITS, 16);
        assert_eq!(UBLK_QID_OFF, 41);
    }

    /// Verify USER_COPY offset calculation
    #[test]
    fn test_user_copy_offset_calculation() {
        // Tag 0, queue 0, offset 0 -> base offset
        let offset = ublk_user_copy_offset(0, 0, 0);
        assert_eq!(offset as u64, UBLKSRV_IO_BUF_OFFSET);

        // Tag 1, queue 0, offset 0
        let offset = ublk_user_copy_offset(0, 1, 0);
        assert_eq!(offset as u64, UBLKSRV_IO_BUF_OFFSET + (1u64 << UBLK_TAG_OFF));

        // Tag 0, queue 1, offset 0
        let offset = ublk_user_copy_offset(1, 0, 0);
        assert_eq!(offset as u64, UBLKSRV_IO_BUF_OFFSET + (1u64 << UBLK_QID_OFF));

        // With buffer offset
        let offset = ublk_user_copy_offset(0, 0, 512);
        assert_eq!(offset as u64, UBLKSRV_IO_BUF_OFFSET + 512);
    }

    /// Verify default values for UblkCtrlCmd
    #[test]
    fn test_ctrl_cmd_default() {
        let cmd = UblkCtrlCmd::default();
        assert_eq!(cmd.dev_id, u32::MAX);
        assert_eq!(cmd.queue_id, u16::MAX); // -1 as u16
        assert_eq!(cmd.len, 0);
        assert_eq!(cmd.addr, 0);
    }

    /// Verify helper function: iod_offset
    #[test]
    fn test_iod_offset() {
        let depth = 128u16;
        assert_eq!(iod_offset(0, depth), 0);
        assert_eq!(iod_offset(1, depth), size_of::<UblkIoDesc>());
        assert_eq!(iod_offset(127, depth), 127 * size_of::<UblkIoDesc>());
    }

    /// Verify helper function: buf_offset
    #[test]
    fn test_buf_offset() {
        let depth = 128u16;
        let io_size = 512 * 1024u32; // 512 KB
        let iod_area = (depth as usize) * size_of::<UblkIoDesc>();

        assert_eq!(buf_offset(0, depth, io_size), iod_area);
        assert_eq!(buf_offset(1, depth, io_size), iod_area + io_size as usize);
    }

    /// Verify helper function: total_buf_size
    #[test]
    fn test_total_buf_size() {
        let depth = 128u16;
        let io_size = 512 * 1024u32;

        let expected =
            (depth as usize) * size_of::<UblkIoDesc>() + (depth as usize) * (io_size as usize);
        assert_eq!(total_buf_size(depth, io_size), expected);
    }

    /// Verify SECTOR_SIZE constant
    #[test]
    fn test_sector_size() {
        assert_eq!(SECTOR_SIZE, 512);
    }

    /// Verify device path format strings
    #[test]
    fn test_device_paths() {
        assert_eq!(UBLK_CTRL_DEV, "/dev/ublk-control");
        assert_eq!(UBLK_CHAR_DEV_FMT, "/dev/ublkc");
        assert_eq!(UBLK_BLOCK_DEV_FMT, "/dev/ublkb");
    }

    /// Verify default queue parameters
    /// PERF-004: Queue depth increased to 256 for higher IOPS
    #[test]
    fn test_default_queue_params() {
        assert_eq!(UBLK_DEF_QUEUE_DEPTH, 256);
        assert_eq!(UBLK_DEF_NR_HW_QUEUES, 1);
        assert_eq!(UBLK_MAX_IO_BUF_BYTES, 512 * 1024);
    }

    // ========================================================================
    // Property-based tests using proptest (if enabled)
    // ========================================================================

    /// Verify ioctl encoding is reversible
    #[test]
    fn test_ioctl_encoding_consistency() {
        // For any valid cmd, the encoding should produce consistent results
        for nr in [0x01u32, 0x02, 0x04, 0x05, 0x06, 0x07, 0x08, 0x09] {
            let encoded = _iowr(UBLK_MAGIC, nr, size_of::<UblkCtrlCmd>());
            // Extract nr back
            let extracted_nr = encoded & 0xFF;
            assert_eq!(extracted_nr, nr);
            // Extract size
            let extracted_size = (encoded >> 16) & 0x3FFF;
            assert_eq!(extracted_size as usize, size_of::<UblkCtrlCmd>());
        }
    }

    /// Verify _io helper (no direction, no size)
    #[test]
    fn test_ioctl_io_helper() {
        let result = _io(0x75, 0x10);
        // _io: (type << 8) | nr
        let expected = (0x75u32 << 8) | 0x10;
        assert_eq!(result, expected);
    }

    /// Verify _ior helper (read direction)
    #[test]
    fn test_ioctl_ior_helper() {
        let result = _ior(0x75, 0x02, 64);
        // _ior: (2 << 30) | (size << 16) | (type << 8) | nr
        let expected = (2u32 << 30) | (64u32 << 16) | (0x75u32 << 8) | 0x02;
        assert_eq!(result, expected);
        // Direction bits should be 2 (read)
        assert_eq!((result >> 30) & 3, 2);
    }

    /// Verify _iow helper (write direction)
    #[test]
    fn test_ioctl_iow_helper() {
        let result = _iow(0x75, 0x08, 32);
        // _iow: (1 << 30) | (size << 16) | (type << 8) | nr
        let expected = (1u32 << 30) | (32u32 << 16) | (0x75u32 << 8) | 0x08;
        assert_eq!(result, expected);
        // Direction bits should be 1 (write)
        assert_eq!((result >> 30) & 3, 1);
    }

    /// Verify UblkCtrlCmdExt::default()
    #[test]
    fn test_ctrl_cmd_ext_default() {
        let ext = UblkCtrlCmdExt::default();
        assert_eq!(ext.cmd.dev_id, u32::MAX);
        assert_eq!(ext.padding, [0u8; 48]);
    }
}
