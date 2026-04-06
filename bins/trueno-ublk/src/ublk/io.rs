//! ublk Data Plane I/O operations
//!
//! Handles io_uring-based I/O processing for ublk block devices.
//! This module implements the "fetch/commit" cycle for block I/O.

use crate::ublk::sys::*;
use nix::libc;
use std::io;

/// I/O operation result
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IoResult {
    /// Operation completed successfully with byte count
    Success(usize),
    /// Operation would block (retriable)
    WouldBlock,
    /// End of device
    EndOfDevice,
    /// Error with errno
    Error(i32),
}

impl IoResult {
    /// Convert to i32 result for kernel (negative errno on error)
    #[inline]
    pub fn to_kernel_result(self) -> i32 {
        match self {
            IoResult::Success(n) => n as i32,
            IoResult::WouldBlock => -libc::EAGAIN,
            IoResult::EndOfDevice => 0,
            IoResult::Error(e) => -e,
        }
    }

    /// Check if this is a success result
    #[inline]
    pub fn is_success(&self) -> bool {
        matches!(self, IoResult::Success(_))
    }

    /// Check if this is retriable
    #[inline]
    pub fn is_retriable(&self) -> bool {
        matches!(self, IoResult::WouldBlock)
    }
}

/// I/O request descriptor (parsed from UblkIoDesc)
#[derive(Debug, Clone, Copy)]
pub struct IoRequest {
    /// Operation type
    pub op: IoOp,
    /// Starting sector
    pub start_sector: u64,
    /// Number of sectors
    pub nr_sectors: u32,
    /// Tag for this request
    pub tag: u16,
    /// Queue ID
    pub queue_id: u16,
}

/// I/O operation type
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IoOp {
    Read,
    Write,
    Flush,
    Discard,
    WriteZeroes,
    WriteSame,
    Unknown(u8),
}

impl From<u8> for IoOp {
    fn from(op: u8) -> Self {
        match op {
            UBLK_IO_OP_READ => IoOp::Read,
            UBLK_IO_OP_WRITE => IoOp::Write,
            UBLK_IO_OP_FLUSH => IoOp::Flush,
            UBLK_IO_OP_DISCARD => IoOp::Discard,
            UBLK_IO_OP_WRITE_ZEROES => IoOp::WriteZeroes,
            UBLK_IO_OP_WRITE_SAME => IoOp::WriteSame,
            _ => IoOp::Unknown(op),
        }
    }
}

impl IoOp {
    /// Check if this operation requires data transfer
    #[inline]
    pub fn needs_data(&self) -> bool {
        matches!(self, IoOp::Read | IoOp::Write | IoOp::WriteSame)
    }

    /// Check if this is a write operation
    #[inline]
    pub fn is_write(&self) -> bool {
        matches!(self, IoOp::Write | IoOp::WriteZeroes | IoOp::WriteSame | IoOp::Discard)
    }

    /// Convert to u8 opcode
    #[inline]
    pub fn to_opcode(self) -> u8 {
        match self {
            IoOp::Read => UBLK_IO_OP_READ,
            IoOp::Write => UBLK_IO_OP_WRITE,
            IoOp::Flush => UBLK_IO_OP_FLUSH,
            IoOp::Discard => UBLK_IO_OP_DISCARD,
            IoOp::WriteZeroes => UBLK_IO_OP_WRITE_ZEROES,
            IoOp::WriteSame => UBLK_IO_OP_WRITE_SAME,
            IoOp::Unknown(op) => op,
        }
    }
}

impl IoRequest {
    /// Parse from UblkIoDesc
    pub fn from_desc(desc: &UblkIoDesc, tag: u16, queue_id: u16) -> Self {
        let op = IoOp::from((desc.op_flags & 0xff) as u8);
        Self { op, start_sector: desc.start_sector, nr_sectors: desc.nr_sectors, tag, queue_id }
    }

    /// Calculate byte length for this request
    #[inline]
    pub fn byte_len(&self) -> usize {
        (self.nr_sectors as usize) * (SECTOR_SIZE as usize)
    }

    /// Calculate byte offset for this request
    #[inline]
    pub fn byte_offset(&self) -> u64 {
        self.start_sector * SECTOR_SIZE
    }

    /// Validate the request against device size
    pub fn validate(&self, dev_sectors: u64) -> Result<(), io::Error> {
        // Check for zero sectors (except flush)
        if self.nr_sectors == 0 && self.op != IoOp::Flush {
            return Err(io::Error::from_raw_os_error(libc::EINVAL));
        }

        // Check bounds
        let end_sector = self
            .start_sector
            .checked_add(self.nr_sectors as u64)
            .ok_or_else(|| io::Error::from_raw_os_error(libc::EOVERFLOW))?;

        if end_sector > dev_sectors {
            return Err(io::Error::from_raw_os_error(libc::ENOSPC));
        }

        Ok(())
    }
}

/// Build UblkIoCmd for FETCH_REQ
#[inline]
pub fn build_fetch_cmd(queue_id: u16, tag: u16) -> UblkIoCmd {
    UblkIoCmd {
        q_id: queue_id,
        tag,
        result: 0,
        addr: 0, // USER_COPY mode: kernel uses pread/pwrite
    }
}

/// Build UblkIoCmd for COMMIT_AND_FETCH_REQ
#[inline]
pub fn build_commit_fetch_cmd(queue_id: u16, tag: u16, result: i32) -> UblkIoCmd {
    UblkIoCmd {
        q_id: queue_id,
        tag,
        result,
        addr: 0, // USER_COPY mode
    }
}

/// Calculate pread/pwrite offset for USER_COPY mode
#[inline]
pub fn user_copy_offset(queue_id: u16, tag: u16, byte_offset: u32) -> i64 {
    ublk_user_copy_offset(queue_id, tag, byte_offset)
}

#[cfg(test)]
mod tests {
    use super::*;

    // ========================================================================
    // IoResult Tests
    // ========================================================================

    #[test]
    fn test_io_result_success() {
        let result = IoResult::Success(4096);
        assert!(result.is_success());
        assert!(!result.is_retriable());
        assert_eq!(result.to_kernel_result(), 4096);
    }

    #[test]
    fn test_io_result_would_block() {
        let result = IoResult::WouldBlock;
        assert!(!result.is_success());
        assert!(result.is_retriable());
        assert_eq!(result.to_kernel_result(), -libc::EAGAIN);
    }

    #[test]
    fn test_io_result_end_of_device() {
        let result = IoResult::EndOfDevice;
        assert!(!result.is_success());
        assert!(!result.is_retriable());
        assert_eq!(result.to_kernel_result(), 0);
    }

    #[test]
    fn test_io_result_error() {
        let result = IoResult::Error(libc::EIO);
        assert!(!result.is_success());
        assert!(!result.is_retriable());
        assert_eq!(result.to_kernel_result(), -libc::EIO);
    }

    // ========================================================================
    // IoOp Tests
    // ========================================================================

    #[test]
    fn test_io_op_from_u8() {
        assert_eq!(IoOp::from(UBLK_IO_OP_READ), IoOp::Read);
        assert_eq!(IoOp::from(UBLK_IO_OP_WRITE), IoOp::Write);
        assert_eq!(IoOp::from(UBLK_IO_OP_FLUSH), IoOp::Flush);
        assert_eq!(IoOp::from(UBLK_IO_OP_DISCARD), IoOp::Discard);
        assert_eq!(IoOp::from(UBLK_IO_OP_WRITE_ZEROES), IoOp::WriteZeroes);
        assert_eq!(IoOp::from(UBLK_IO_OP_WRITE_SAME), IoOp::WriteSame);
        assert_eq!(IoOp::from(255), IoOp::Unknown(255));
    }

    #[test]
    fn test_io_op_needs_data() {
        assert!(IoOp::Read.needs_data());
        assert!(IoOp::Write.needs_data());
        assert!(IoOp::WriteSame.needs_data());
        assert!(!IoOp::Flush.needs_data());
        assert!(!IoOp::Discard.needs_data());
        assert!(!IoOp::WriteZeroes.needs_data());
    }

    #[test]
    fn test_io_op_is_write() {
        assert!(!IoOp::Read.is_write());
        assert!(IoOp::Write.is_write());
        assert!(IoOp::WriteZeroes.is_write());
        assert!(IoOp::WriteSame.is_write());
        assert!(IoOp::Discard.is_write());
        assert!(!IoOp::Flush.is_write());
    }

    #[test]
    fn test_io_op_to_opcode() {
        assert_eq!(IoOp::Read.to_opcode(), UBLK_IO_OP_READ);
        assert_eq!(IoOp::Write.to_opcode(), UBLK_IO_OP_WRITE);
        assert_eq!(IoOp::Unknown(42).to_opcode(), 42);
    }

    // ========================================================================
    // IoRequest Tests
    // ========================================================================

    #[test]
    fn test_io_request_from_desc() {
        let desc = UblkIoDesc {
            op_flags: UBLK_IO_OP_READ as u32,
            nr_sectors: 8,
            start_sector: 100,
            addr: 0,
        };
        let req = IoRequest::from_desc(&desc, 5, 0);

        assert_eq!(req.op, IoOp::Read);
        assert_eq!(req.start_sector, 100);
        assert_eq!(req.nr_sectors, 8);
        assert_eq!(req.tag, 5);
        assert_eq!(req.queue_id, 0);
    }

    #[test]
    fn test_io_request_byte_len() {
        let req = IoRequest { op: IoOp::Read, start_sector: 0, nr_sectors: 8, tag: 0, queue_id: 0 };
        assert_eq!(req.byte_len(), 8 * 512); // 4KB
    }

    #[test]
    fn test_io_request_byte_offset() {
        let req =
            IoRequest { op: IoOp::Read, start_sector: 16, nr_sectors: 8, tag: 0, queue_id: 0 };
        assert_eq!(req.byte_offset(), 16 * 512); // 8KB
    }

    #[test]
    fn test_io_request_validate_success() {
        let req = IoRequest { op: IoOp::Read, start_sector: 0, nr_sectors: 8, tag: 0, queue_id: 0 };
        assert!(req.validate(1024).is_ok());
    }

    #[test]
    fn test_io_request_validate_zero_sectors() {
        let req = IoRequest {
            op: IoOp::Read,
            start_sector: 0,
            nr_sectors: 0, // Invalid
            tag: 0,
            queue_id: 0,
        };
        let err = req.validate(1024).unwrap_err();
        assert_eq!(err.raw_os_error(), Some(libc::EINVAL));
    }

    #[test]
    fn test_io_request_validate_flush_zero_sectors() {
        let req = IoRequest {
            op: IoOp::Flush,
            start_sector: 0,
            nr_sectors: 0, // Valid for flush
            tag: 0,
            queue_id: 0,
        };
        assert!(req.validate(1024).is_ok());
    }

    #[test]
    fn test_io_request_validate_out_of_bounds() {
        let req = IoRequest {
            op: IoOp::Read,
            start_sector: 1000,
            nr_sectors: 100, // Would exceed 1024 sectors
            tag: 0,
            queue_id: 0,
        };
        let err = req.validate(1024).unwrap_err();
        assert_eq!(err.raw_os_error(), Some(libc::ENOSPC));
    }

    #[test]
    fn test_io_request_validate_overflow() {
        let req = IoRequest {
            op: IoOp::Read,
            start_sector: u64::MAX,
            nr_sectors: 1,
            tag: 0,
            queue_id: 0,
        };
        let err = req.validate(u64::MAX).unwrap_err();
        assert_eq!(err.raw_os_error(), Some(libc::EOVERFLOW));
    }

    // ========================================================================
    // Command Building Tests
    // ========================================================================

    #[test]
    fn test_build_fetch_cmd() {
        let cmd = build_fetch_cmd(0, 5);
        assert_eq!(cmd.q_id, 0);
        assert_eq!(cmd.tag, 5);
        assert_eq!(cmd.result, 0);
        assert_eq!(cmd.addr, 0); // USER_COPY mode
    }

    #[test]
    fn test_build_commit_fetch_cmd() {
        let cmd = build_commit_fetch_cmd(0, 7, 4096);
        assert_eq!(cmd.q_id, 0);
        assert_eq!(cmd.tag, 7);
        assert_eq!(cmd.result, 4096);
        assert_eq!(cmd.addr, 0); // USER_COPY mode
    }

    #[test]
    fn test_build_commit_fetch_cmd_error() {
        let cmd = build_commit_fetch_cmd(0, 3, -libc::EIO);
        assert_eq!(cmd.q_id, 0);
        assert_eq!(cmd.tag, 3);
        assert_eq!(cmd.result, -libc::EIO);
    }

    // ========================================================================
    // User Copy Offset Tests
    // ========================================================================

    #[test]
    fn test_user_copy_offset_base() {
        let offset = user_copy_offset(0, 0, 0);
        assert_eq!(offset as u64, UBLKSRV_IO_BUF_OFFSET);
    }

    #[test]
    fn test_user_copy_offset_with_tag() {
        let offset = user_copy_offset(0, 1, 0);
        assert_eq!(offset as u64, UBLKSRV_IO_BUF_OFFSET + (1u64 << UBLK_TAG_OFF));
    }

    #[test]
    fn test_user_copy_offset_with_queue() {
        let offset = user_copy_offset(1, 0, 0);
        assert_eq!(offset as u64, UBLKSRV_IO_BUF_OFFSET + (1u64 << UBLK_QID_OFF));
    }

    #[test]
    fn test_user_copy_offset_combined() {
        let offset = user_copy_offset(2, 5, 1024u32);
        let expected =
            UBLKSRV_IO_BUF_OFFSET + (2u64 << UBLK_QID_OFF) + (5u64 << UBLK_TAG_OFF) + 1024;
        assert_eq!(offset as u64, expected);
    }

    // ========================================================================
    // Renacer Verification Matrix Tests (Section A items 4-10)
    // ========================================================================

    // A.4: Verify UBLK_IO_FETCH_REQ opcode matches kernel
    #[test]
    fn test_fetch_req_opcode() {
        // UBLK_U_IO_FETCH_REQ = _IOWR('u', UBLK_IO_FETCH_REQ, struct ublksrv_io_cmd)
        // = _IOWR(0x75, 0x20, 16)
        assert_eq!(UBLK_U_IO_FETCH_REQ & 0xFF, 0x20);
    }

    // A.5: Verify UBLK_IO_COMMIT_AND_FETCH_REQ logic
    #[test]
    fn test_commit_and_fetch_chaining() {
        // Commit completes previous I/O and fetches next in one operation
        let cmd = build_commit_fetch_cmd(0, 5, 4096);
        assert_eq!(cmd.result, 4096); // Previous I/O result
        assert_eq!(cmd.tag, 5); // Same tag for fetch
    }

    // A.7: Verify tag usage (tags 0..QD-1 are unique)
    #[test]
    fn test_tag_range() {
        let qd = UBLK_DEF_QUEUE_DEPTH;
        for tag in 0..qd {
            let cmd = build_fetch_cmd(0, tag);
            assert_eq!(cmd.tag, tag);
        }
    }

    // A.8: Falsify: nr_sectors=0 handling
    #[test]
    fn test_falsify_zero_sectors() {
        let req = IoRequest { op: IoOp::Read, start_sector: 0, nr_sectors: 0, tag: 0, queue_id: 0 };
        // Non-flush with zero sectors should fail
        assert!(req.validate(1024).is_err());
    }

    // A.9: Falsify: start_sector out of bounds
    #[test]
    fn test_falsify_sector_out_of_bounds() {
        let dev_sectors = 1024u64;
        let req = IoRequest {
            op: IoOp::Read,
            start_sector: dev_sectors, // At boundary (invalid)
            nr_sectors: 1,
            tag: 0,
            queue_id: 0,
        };
        assert!(req.validate(dev_sectors).is_err());
    }

    // A.10: Verify UBLK_F_USER_COPY flag usage (addr=0 in commands)
    #[test]
    fn test_user_copy_addr_zero() {
        let fetch = build_fetch_cmd(0, 0);
        assert_eq!(fetch.addr, 0, "USER_COPY mode requires addr=0");

        let commit = build_commit_fetch_cmd(0, 0, 0);
        assert_eq!(commit.addr, 0, "USER_COPY mode requires addr=0");
    }
}
