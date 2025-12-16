use crate::error::Result;
use std::path::Path;

pub(crate) fn run(source: &str, output: &Path) -> Result<()> {
    println!("Importing from {} to {}", source, output.display());
    Ok(())
}
