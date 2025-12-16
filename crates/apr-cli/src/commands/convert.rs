use crate::error::Result;
use std::path::Path;

pub(crate) fn run(file: &Path, quantize: Option<String>, output: &Path) -> Result<()> {
    println!("Converting model: {}", file.display());
    if let Some(q) = quantize {
        println!("Quantization: {}", q);
    }
    println!("Output: {}", output.display());
    Ok(())
}
