use crate::error::Result;
use std::path::{Path, PathBuf};

pub(crate) fn run(files: &[PathBuf], strategy: &str, output: &Path) -> Result<()> {
    println!(
        "Merging {} files using strategy '{}'",
        files.len(),
        strategy
    );
    println!("Output: {}", output.display());
    Ok(())
}
