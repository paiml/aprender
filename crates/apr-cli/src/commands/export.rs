use crate::error::Result;
use std::path::Path;

pub(crate) fn run(file: &Path, format: &str, output: &Path) -> Result<()> {
    println!(
        "Exporting {} to {} format at {}",
        file.display(),
        format,
        output.display()
    );
    Ok(())
}
