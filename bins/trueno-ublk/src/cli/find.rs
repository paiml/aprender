//! Find command - finds a free device slot

use crate::device::UblkDevice;
use anyhow::Result;

pub fn run() -> Result<()> {
    let next_id = UblkDevice::next_free_id()?;
    println!("/dev/ublkb{}", next_id);
    Ok(())
}
