//! Mock infrastructure for testing without kernel access.

use super::*;

/// Mock I/O descriptor for testing
#[derive(Debug, Clone)]
pub struct MockIoDesc {
    pub op: u8,
    pub nr_sectors: u32,
    pub start_sector: u64,
}

/// Mock daemon for testing I/O processing logic
pub struct MockUblkDaemon {
    pub dev_id: i32,
    pub queue_depth: u16,
    pub max_io_size: u32,
    pub pending_ios: Vec<MockIoDesc>,
    pub completed_ios: Vec<(u16, i32)>, // (tag, result)
    pub fetch_count: usize,
    pub commit_count: usize,
}

impl MockUblkDaemon {
    pub fn new(dev_id: i32, queue_depth: u16) -> Self {
        Self {
            dev_id,
            queue_depth,
            max_io_size: UBLK_MAX_IO_BUF_BYTES,
            pending_ios: Vec::new(),
            completed_ios: Vec::new(),
            fetch_count: 0,
            commit_count: 0,
        }
    }

    pub fn submit_fetch(&mut self, tag: u16) -> Result<(), DaemonError> {
        if tag >= self.queue_depth {
            return Err(DaemonError::Submit(std::io::Error::from_raw_os_error(libc::EINVAL)));
        }
        self.fetch_count += 1;
        Ok(())
    }

    pub fn submit_commit_and_fetch(&mut self, tag: u16, result: i32) -> Result<(), DaemonError> {
        if tag >= self.queue_depth {
            return Err(DaemonError::Submit(std::io::Error::from_raw_os_error(libc::EINVAL)));
        }
        self.completed_ios.push((tag, result));
        self.commit_count += 1;
        Ok(())
    }

    pub fn process_io(&mut self, io: MockIoDesc, store: &mut crate::daemon::PageStore) -> i32 {
        let start_sector = io.start_sector;
        let nr_sectors = io.nr_sectors;
        let len = (nr_sectors as usize) * SECTOR_SIZE as usize;

        let result = match io.op {
            UBLK_IO_OP_READ => {
                let mut buf = vec![0u8; len];
                match store.read(start_sector, &mut buf) {
                    Ok(n) => n as i32,
                    Err(e) => -e.raw_os_error().unwrap_or(libc::EIO),
                }
            }
            UBLK_IO_OP_WRITE => {
                let buf = vec![0xABu8; len]; // Mock data
                match store.write(start_sector, &buf) {
                    Ok(n) => n as i32,
                    Err(e) => -e.raw_os_error().unwrap_or(libc::EIO),
                }
            }
            UBLK_IO_OP_FLUSH => 0,
            UBLK_IO_OP_DISCARD => match store.discard(start_sector, nr_sectors) {
                Ok(n) => n as i32,
                Err(e) => -e.raw_os_error().unwrap_or(libc::EIO),
            },
            UBLK_IO_OP_WRITE_ZEROES => match store.write_zeroes(start_sector, nr_sectors) {
                Ok(n) => n as i32,
                Err(e) => -e.raw_os_error().unwrap_or(libc::EIO),
            },
            _ => -libc::ENOTSUP,
        };

        result
    }

    pub fn dev_id(&self) -> i32 {
        self.dev_id
    }

    pub fn block_dev_path(&self) -> String {
        format!("{}{}", UBLK_BLOCK_DEV_FMT, self.dev_id)
    }
}
