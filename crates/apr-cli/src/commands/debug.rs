//! Debug command implementation
//!
//! Toyota Way: Visualization - Make problems visible.
//! Simple debugging with optional "drama" theatrical mode.

use crate::error::CliError;
use crate::output;
use aprender::format::rosetta::{FormatType, RosettaStone};
use aprender::format::HEADER_SIZE;
use colored::Colorize;
use std::ffi::OsStr;
use std::fs::File;
use std::io::{BufReader, Read};
use std::path::Path;

/// Parsed header information for debug output.
///
/// These flags represent independent header properties that are naturally
/// expressed as booleans. A state machine would over-complicate this simple
/// debug data structure.
#[allow(clippy::struct_excessive_bools)]
struct HeaderInfo {
    magic_valid: bool,
    magic_str: String,
    version: (u8, u8),
    model_type: u16,
    compressed: bool,
    signed: bool,
    encrypted: bool,
}

/// Run the debug command
pub(crate) fn run(
    path: &Path,
    drama: bool,
    hex: bool,
    strings: bool,
    limit: usize,
) -> Result<(), CliError> {
    validate_path(path)?;

    // Dispatch to appropriate mode
    if hex {
        return run_hex_mode(path, limit);
    }
    if strings {
        return run_strings_mode(path, limit);
    }

    // Rosetta Stone dispatch: detect format first
    let detected = FormatType::from_magic(path)
        .or_else(|_| FormatType::from_extension(path));

    if let Ok(FormatType::Gguf | FormatType::SafeTensors) = detected {
        return run_rosetta_debug(path, drama);
    }

    // APR path: read and parse header
    let (header_bytes, file_size) = read_header(path)?;
    let info = parse_header(&header_bytes);

    if drama {
        run_drama_mode(path, &header_bytes, file_size, info.magic_valid);
    } else {
        run_basic_mode(path, file_size, &info);
    }

    Ok(())
}

/// Debug output for GGUF/SafeTensors via Rosetta Stone
fn run_rosetta_debug(path: &Path, drama: bool) -> Result<(), CliError> {
    let rosetta = RosettaStone::new();
    let report = rosetta
        .inspect(path)
        .map_err(|e| CliError::InvalidFormat(format!("Failed to inspect: {e}")))?;

    let filename = path
        .file_name()
        .unwrap_or(OsStr::new("unknown"))
        .to_string_lossy();

    let format_name = format!("{}", report.format);
    let tensor_count = report.tensors.len();
    let arch_str = report.architecture.as_deref().unwrap_or("unknown");

    if drama {
        println!();
        println!("{}", "====[ DRAMA: ".yellow().bold());
        println!("{}{}", filename.cyan().bold(), " ]====".yellow().bold());
        println!();
        println!("{}", "ACT I: THE FORMAT".magenta().bold());
        print!("  Scene 1: Format detection... ");
        println!(
            "{} {}",
            format_name.green().bold(),
            "(standing ovation!)".green()
        );
        print!("  Scene 2: Architecture... ");
        println!("{} {}", arch_str.green().bold(), "(bravo!)".green());
        print!("  Scene 3: Tensor count... ");
        println!(
            "{} {}",
            tensor_count.to_string().cyan().bold(),
            "(impressive!)".cyan()
        );
        println!();
        println!("{}", "CURTAIN FALLS".yellow().bold());
    } else {
        output::header(&format!(
            "{}: {} {} ({})",
            filename, format_name, arch_str, tensor_count
        ));

        let file_size = path.metadata().map(|m| m.len()).unwrap_or(0);

        println!(
            "{}",
            output::kv_table(&[
                (
                    "Size",
                    humansize::format_size(file_size, humansize::BINARY)
                ),
                ("Format", format!("{} {}", format_name, output::badge_pass("valid"))),
                (
                    "Architecture",
                    report
                        .architecture
                        .clone()
                        .unwrap_or_else(|| "unknown".to_string())
                ),
                ("Tensors", tensor_count.to_string()),
                (
                    "Parameters",
                    report.total_params.to_string()
                ),
                ("Health", output::badge_pass("OK")),
            ])
        );
    }

    Ok(())
}

/// Validate the input path.
fn validate_path(path: &Path) -> Result<(), CliError> {
    if !path.exists() {
        return Err(CliError::FileNotFound(path.to_path_buf()));
    }
    if !path.is_file() {
        return Err(CliError::NotAFile(path.to_path_buf()));
    }
    Ok(())
}

/// Read header bytes from file.
fn read_header(path: &Path) -> Result<([u8; HEADER_SIZE], u64), CliError> {
    let file = File::open(path)?;
    let file_size = file.metadata()?.len();
    let mut reader = BufReader::new(file);

    let mut header_bytes = [0u8; HEADER_SIZE];
    reader
        .read_exact(&mut header_bytes)
        .map_err(|_| CliError::InvalidFormat("File too small".to_string()))?;

    Ok((header_bytes, file_size))
}

/// Parse header bytes into structured info.
fn parse_header(header: &[u8; HEADER_SIZE]) -> HeaderInfo {
    let flags = header[21];
    HeaderInfo {
        magic_valid: output::is_valid_magic(&header[0..4]),
        magic_str: String::from_utf8_lossy(&header[0..4]).to_string(),
        version: (header[4], header[5]),
        model_type: u16::from_le_bytes([header[6], header[7]]),
        compressed: flags & 0x01 != 0 || header[20] != 0,
        signed: flags & 0x02 != 0,
        encrypted: flags & 0x04 != 0,
    }
}

/// Run basic debug output mode.
fn run_basic_mode(path: &Path, file_size: u64, info: &HeaderInfo) {
    let filename = path
        .file_name()
        .unwrap_or(OsStr::new("unknown"))
        .to_string_lossy();

    output::header(&format!(
        "{}: APR v{}.{} {}",
        filename,
        info.version.0,
        info.version.1,
        format_model_type(info.model_type)
    ));

    let magic_status = if info.magic_valid {
        output::badge_pass("valid")
    } else {
        output::badge_fail("INVALID")
    };
    let health = if info.magic_valid {
        output::badge_pass("OK")
    } else {
        output::badge_fail("CORRUPTED")
    };
    let flag_list = collect_flags(info);
    let flags_str = if flag_list.is_empty() {
        "none".to_string()
    } else {
        flag_list.join(", ")
    };

    println!(
        "{}",
        output::kv_table(&[
            ("Size", humansize::format_size(file_size, humansize::BINARY)),
            ("Magic", format!("{} {}", info.magic_str, magic_status)),
            ("Flags", flags_str),
            ("Health", health),
        ])
    );
}

/// Collect active flags into a list.
fn collect_flags(info: &HeaderInfo) -> Vec<&'static str> {
    let mut flags = Vec::new();
    if info.compressed {
        flags.push("compressed");
    }
    if info.signed {
        flags.push("signed");
    }
    if info.encrypted {
        flags.push("encrypted");
    }
    flags
}

/// Drama mode - theatrical debugging output
fn run_drama_mode(path: &Path, header: &[u8; HEADER_SIZE], file_size: u64, magic_valid: bool) {
    let filename = path
        .file_name()
        .unwrap_or(OsStr::new("unknown"))
        .to_string_lossy();
    let magic_str = String::from_utf8_lossy(&header[0..4]);
    let version = (header[4], header[5]);
    let model_type = u16::from_le_bytes([header[6], header[7]]);
    let flags = header[21];

    println!();
    println!("{}", "====[ DRAMA: ".yellow().bold());
    println!("{}{}", filename.cyan().bold(), " ]====".yellow().bold());
    println!();

    // ACT I: THE HEADER
    println!("{}", "ACT I: THE HEADER".magenta().bold());

    print!("  Scene 1: Magic bytes... ");
    if magic_valid {
        println!("{} {}", magic_str.green().bold(), "(applause!)".green());
    } else {
        println!("{} {}", magic_str.red().bold(), "(gasp! the horror!)".red());
    }

    print!("  Scene 2: Version check... ");
    let version_str = format!("{}.{}", version.0, version.1);
    if version.0 == 1 {
        println!(
            "{} {}",
            version_str.green().bold(),
            "(standing ovation!)".green()
        );
    } else {
        println!(
            "{} {}",
            version_str.yellow(),
            "(murmurs of concern)".yellow()
        );
    }

    print!("  Scene 3: Model type... ");
    let type_name = format_model_type(model_type);
    println!(
        "{} {}",
        type_name.cyan().bold(),
        "(the protagonist!)".cyan()
    );

    println!();

    // ACT II: THE METADATA
    println!("{}", "ACT II: THE METADATA".magenta().bold());

    print!("  Scene 1: File size... ");
    let size_str = humansize::format_size(file_size, humansize::BINARY);
    println!("{}", size_str.white().bold());

    print!("  Scene 2: Flags... ");
    let mut flag_drama = Vec::new();
    if flags & 0x01 != 0 || header[20] != 0 {
        flag_drama.push("COMPRESSED");
    }
    if flags & 0x02 != 0 {
        flag_drama.push("SIGNED");
    }
    if flags & 0x04 != 0 {
        flag_drama.push("ENCRYPTED");
    }
    if flags & 0x20 != 0 {
        flag_drama.push("QUANTIZED");
    }

    if flag_drama.is_empty() {
        println!("{}", "(bare, unadorned)".white());
    } else {
        println!("{}", flag_drama.join(" | ").yellow().bold());
    }

    println!();

    // ACT III: THE VERDICT
    println!("{}", "ACT III: THE VERDICT".magenta().bold());

    if magic_valid {
        println!();
        println!(
            "  {} {}",
            "CURTAIN CALL:".green().bold(),
            "Model is READY!".green().bold()
        );
    } else {
        println!();
        println!(
            "  {} {}",
            "TRAGEDY:".red().bold(),
            "Model is CORRUPTED!".red().bold()
        );
    }

    println!();
    println!("{}", "====[ END DRAMA ]====".yellow().bold());
    println!();
}

/// Hex dump mode
fn run_hex_mode(path: &Path, limit: usize) -> Result<(), CliError> {
    let mut file = File::open(path)?;
    let mut buffer = vec![0u8; limit.min(4096)];
    let bytes_read = file.read(&mut buffer)?;
    buffer.truncate(bytes_read);

    println!("Hex dump of {} (first {bytes_read} bytes):", path.display());
    println!();

    for (i, chunk) in buffer.chunks(16).enumerate() {
        // Offset
        print!("{:08x}: ", i * 16);

        // Hex bytes
        for (j, byte) in chunk.iter().enumerate() {
            if j == 8 {
                print!(" ");
            }
            print!("{byte:02x} ");
        }

        // Padding if less than 16 bytes
        for j in chunk.len()..16 {
            if j == 8 {
                print!(" ");
            }
            print!("   ");
        }

        // ASCII representation
        print!(" |");
        for byte in chunk {
            if *byte >= 0x20 && *byte < 0x7f {
                print!("{}", *byte as char);
            } else {
                print!(".");
            }
        }
        println!("|");
    }

    Ok(())
}

/// Strings extraction mode
fn run_strings_mode(path: &Path, limit: usize) -> Result<(), CliError> {
    let mut file = File::open(path)?;
    let mut buffer = Vec::new();
    file.read_to_end(&mut buffer)?;

    println!("Extracted strings from {} (min length 4):", path.display());
    println!();

    let mut current_string = String::new();
    let mut count = 0;

    for byte in &buffer {
        if *byte >= 0x20 && *byte < 0x7f {
            current_string.push(*byte as char);
        } else {
            if current_string.len() >= 4 {
                println!("  {current_string}");
                count += 1;
                if count >= limit {
                    println!("  ... (truncated at {limit} strings)");
                    break;
                }
            }
            current_string.clear();
        }
    }

    // Don't forget the last string
    if current_string.len() >= 4 && count < limit {
        println!("  {current_string}");
    }

    Ok(())
}

/// Format model type as human-readable string
fn format_model_type(type_id: u16) -> String {
    match type_id {
        0x0001 => "LinearRegression".to_string(),
        0x0002 => "LogisticRegression".to_string(),
        0x0003 => "DecisionTree".to_string(),
        0x0004 => "RandomForest".to_string(),
        0x0005 => "GradientBoosting".to_string(),
        0x0006 => "KMeans".to_string(),
        0x0007 => "PCA".to_string(),
        0x0008 => "NaiveBayes".to_string(),
        0x0009 => "KNN".to_string(),
        0x000A => "SVM".to_string(),
        0x0010 => "NgramLM".to_string(),
        0x0011 => "TfIdf".to_string(),
        0x0012 => "CountVectorizer".to_string(),
        0x0020 => "NeuralSequential".to_string(),
        0x0021 => "NeuralCustom".to_string(),
        0x0030 => "ContentRecommender".to_string(),
        0x0040 => "MixtureOfExperts".to_string(),
        0x00FF => "Custom".to_string(),
        _ => format!("Unknown(0x{type_id:04X})"),
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
#[path = "debug_tests.rs"]
mod tests;
