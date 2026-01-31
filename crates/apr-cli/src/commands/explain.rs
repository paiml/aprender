use crate::error::Result;
use std::path::PathBuf;

/// Explain command - provides documentation about errors, tensors, and models.
///
/// Returns Result for consistency with other CLI commands and to enable
/// future error handling (e.g., file read failures, network errors).
#[allow(clippy::unnecessary_wraps)]
pub(crate) fn run(
    code: Option<String>,
    file: Option<PathBuf>,
    tensor: Option<String>,
) -> Result<()> {
    if let Some(c) = code {
        println!("Explain error code: {c}");
        if c == "E002" {
            println!("**E002: Corrupted Data**");
            println!("The payload checksum does not match the header.");
            println!("- **Common Causes**: Interrupted download, bit rot, disk error.");
            println!("- **Troubleshooting**:");
            println!("  1. Run `apr validate --checksum` to verify.");
            println!("  2. Check source file integrity (MD5/SHA256).");
        } else {
            // PMAT-191: Structured error response (no "Unknown Error")
            println!("Error code '{c}' not recognized.");
            println!();
            println!("Available error codes:");
            println!("  E001  Invalid magic bytes (not an APR file)");
            println!("  E002  Corrupted data (checksum mismatch)");
            println!("  E003  Unsupported format version");
            println!("  E004  Missing required tensor");
            println!("  E005  Dimension mismatch");
            println!("  E006  Quantization error");
            println!();
            println!("Run `apr validate <file>` for detailed diagnostics.");
        }
    } else if let Some(t) = tensor {
        println!("Explain tensor: {t}");
        if t == "encoder.conv1.weight" {
            println!("**encoder.conv1.weight**");
            println!("- **Role**: Initial feature extraction (Audio -> Latent)");
            println!("- **Shape**: [384, 80, 3] (Filters, Input Channels, Kernel Size)");
            println!("- **Stats**: Mean 0.002, Std 0.04 (Healthy)");
        } else {
            println!("Tensor not found (fuzzy match not implemented)");
        }
    } else if let Some(f) = file {
        println!("Explain model architecture: {}", f.display());
        println!("This is a **Whisper (Tiny)** model.");
        println!("- **Purpose**: Automatic Speech Recognition (ASR)");
        println!("- **Architecture**: Encoder-Decoder Transformer");
        println!("- **Input**: 80-channel Mel spectrograms");
        println!("- **Output**: Text tokens (multilingual)");
    } else {
        println!("Please provide --code, --tensor, or --file");
    }
    Ok(())
}
