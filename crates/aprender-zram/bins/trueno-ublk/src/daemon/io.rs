//! Block I/O request types and processing for the ublk daemon.

use anyhow::Result;
use trueno_zram_core::PAGE_SIZE;

use super::page_store::PageStore;

/// I/O request type
#[derive(Debug, Clone, Copy)]
pub enum IoType {
    Read,
    Write,
    Discard,
}

/// Block I/O request
pub struct IoRequest {
    pub io_type: IoType,
    pub sector: u64,
    pub len: u32,
    pub buffer: *mut u8,
}

unsafe impl Send for IoRequest {}
unsafe impl Sync for IoRequest {}

/// Process a block I/O request
pub fn process_io(store: &mut PageStore, req: &IoRequest) -> Result<()> {
    let pages = (req.len as usize).div_ceil(PAGE_SIZE);

    match req.io_type {
        IoType::Read => {
            for i in 0..pages {
                let sector = req.sector + (i * (PAGE_SIZE / 512)) as u64;
                let offset = i * PAGE_SIZE;
                // SAFETY: req.buffer is a valid pointer to the ublk data buffer obtained via mmap.
                // - The buffer was mapped with sufficient size (max_io_size * queue_depth)
                // - offset is within bounds (i < pages, where pages = req.len / PAGE_SIZE)
                // - PAGE_SIZE alignment is maintained by the sector-based calculation
                let buffer =
                    unsafe { std::slice::from_raw_parts_mut(req.buffer.add(offset), PAGE_SIZE) };
                store.load(sector, buffer)?;
            }
        }
        IoType::Write => {
            for i in 0..pages {
                let sector = req.sector + (i * (PAGE_SIZE / 512)) as u64;
                let offset = i * PAGE_SIZE;
                // SAFETY: req.buffer is a valid pointer to the ublk data buffer obtained via mmap.
                // - The buffer was mapped with sufficient size (max_io_size * queue_depth)
                // - offset is within bounds (i < pages, where pages = req.len / PAGE_SIZE)
                // - PAGE_SIZE alignment is maintained by the sector-based calculation
                let buffer =
                    unsafe { std::slice::from_raw_parts(req.buffer.add(offset), PAGE_SIZE) };
                store.store(sector, buffer)?;
            }
        }
        IoType::Discard => {
            for i in 0..pages {
                let sector = req.sector + (i * (PAGE_SIZE / 512)) as u64;
                store.remove(sector);
            }
        }
    }

    Ok(())
}
