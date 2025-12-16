use crate::error::Result;
use std::path::PathBuf;

pub(crate) fn run(file: Option<PathBuf>) -> Result<()> {
    if let Some(f) = file {
        println!("Starting TUI for file: {}", f.display());
    } else {
        println!("Starting TUI (no file loaded)");
    }
    println!("(TUI implementation stub - requires ratatui)");
    Ok(())
}
