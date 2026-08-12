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
// GH-685: added verbose param
#[provable_contracts_macros::contract(
    "apr-cli-operations-v1",
    equation = "side_effect_classification"
)]
pub(crate) fn run(
    path: &Path,
    drama: bool,
    hex: bool,
    strings: bool,
    limit: usize,
    json: bool,
    verbose: bool,
) -> Result<(), CliError> {
    contract_pre_flag_integrity!();
    validate_path(path)?;

    // GH-248: JSON output mode
    if json {
        return run_json_mode(path);
    }

    // Dispatch to appropriate mode
    if hex {
        return run_hex_mode(path, limit);
    }
    if strings {
        return run_strings_mode(path, limit);
    }

    // Rosetta Stone dispatch: detect format first
    let detected = FormatType::from_magic(path).or_else(|_| FormatType::from_extension(path));

    if let Ok(FormatType::Gguf | FormatType::SafeTensors) = detected {
        let result = run_rosetta_debug(path, drama);
        if verbose {
            log_verbose_metadata(path);
        }
        return result;
    }

    run_apr_mode(path, drama)?;
    contract_post_flag_integrity!(&());
    Ok(())
}

fn log_verbose_metadata(path: &Path) {
    let Ok(rosetta) = aprender::format::rosetta::RosettaStone::new().inspect(path) else {
        return;
    };
    eprintln!();
    eprintln!(
        "  [verbose] {} metadata keys, {} tensors, {} bytes",
        rosetta.metadata.len(),
        rosetta.tensors.len(),
        rosetta.file_size
    );
    for (k, v) in &rosetta.metadata {
        let display_v = if v.len() > 80 {
            format!("{}...", &v[..80])
        } else {
            v.clone()
        };
        eprintln!("  [verbose] {k} = {display_v}");
    }
}

/// The health verdict of a debug run.
///
/// dogfood-0.63.0, issue #2394 finding 1: `apr debug garbage.bin` printed
/// `Magic ✗ INVALID` and `Health ✗ CORRUPTED` and exited 0, so
/// `apr debug f && use f` proceeded on a file the tool had just called
/// corrupt. The badge and the exit code were computed independently — one from
/// `info.magic_valid`, the other not at all. They are now one value: the badge
/// is rendered from it and the exit code is derived from it, so they cannot
/// disagree, and it is `#[must_use]` so a caller cannot print the report and
/// drop the verdict.
#[must_use]
#[derive(Debug, Clone, PartialEq, Eq)]
enum Health {
    Ok,
    Corrupt(String),
}

impl Health {
    /// The badge printed next to "Health".
    fn badge(&self) -> String {
        match self {
            Self::Ok => output::badge_pass("OK"),
            Self::Corrupt(_) => output::badge_fail("CORRUPTED"),
        }
    }

    /// The exit status this verdict implies.
    fn into_result(self) -> Result<(), CliError> {
        match self {
            Self::Ok => Ok(()),
            Self::Corrupt(reason) => Err(CliError::InvalidFormat(reason)),
        }
    }
}

fn run_apr_mode(path: &Path, drama: bool) -> Result<(), CliError> {
    let (header_bytes, file_size) = read_header(path)?;
    let info = parse_header(&header_bytes);
    let health = header_health(&info);

    if drama {
        run_drama_mode(path, &header_bytes, file_size, info.magic_valid);
    } else {
        run_basic_mode(path, file_size, &info, &health);
    }
    health.into_result()
}

/// The verdict the printed report is rendered from.
fn header_health(info: &HeaderInfo) -> Health {
    if info.magic_valid {
        Health::Ok
    } else {
        Health::Corrupt(format!(
            "not an APR file: magic bytes are {:?}, expected \"APR\\0\" — the file is corrupt or is another format",
            info.magic_str
        ))
    }
}

/// GH-248: JSON debug output via Rosetta Stone
// serde_json::json!() macro uses infallible unwrap internally
#[allow(clippy::disallowed_methods)]
fn run_json_mode(path: &Path) -> Result<(), CliError> {
    let rosetta = RosettaStone::new();
    let report = rosetta
        .inspect(path)
        .map_err(|e| CliError::InvalidFormat(format!("Failed to inspect: {e}")))?;

    let file_size = path.metadata().map(|m| m.len()).unwrap_or(0);
    let output = serde_json::json!({
        "model": path.display().to_string(),
        "format": format!("{}", report.format),
        "architecture": report.architecture.as_deref().unwrap_or("unknown"),
        "tensors": report.tensors.len(),
        "parameters": report.total_params,
        "size_bytes": file_size,
        "health": "OK",
    });
    println!(
        "{}",
        serde_json::to_string_pretty(&output).unwrap_or_default()
    );
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
                ("Size", humansize::format_size(file_size, humansize::BINARY)),
                (
                    "Format",
                    format!("{} {}", format_name, output::badge_pass("valid"))
                ),
                (
                    "Architecture",
                    report
                        .architecture
                        .clone()
                        .unwrap_or_else(|| "unknown".to_string())
                ),
                ("Tensors", tensor_count.to_string()),
                ("Parameters", report.total_params.to_string()),
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
    // GH-653: Use the canonical Header::from_bytes parser, not ad-hoc byte reading.
    // The old code mis-read byte 20 (compression enum) as a boolean flag,
    // showing "compressed" when compression=None has a non-zero enum discriminant,
    // and "encrypted" from byte 21 which is actually a Flags bitfield.
    let flags_byte = header[21];
    let compression_byte = header[20];
    HeaderInfo {
        magic_valid: output::is_valid_magic(&header[0..4]),
        magic_str: String::from_utf8_lossy(&header[0..4]).to_string(),
        version: (header[4], header[5]),
        model_type: u16::from_le_bytes([header[6], header[7]]),
        // Compression: byte 20 is an enum (0=None, 1=ZstdDefault, 2=ZstdMax, 3=Lz4)
        // Values outside 0-3 indicate the header layout doesn't match v2 spec
        compressed: matches!(compression_byte, 1..=3),
        // Flags: only bits 0-2 are defined. If higher bits are set, flags are garbage
        // (likely reading metadata/payload bytes as flags on non-v2 files)
        signed: flags_byte & 0x02 != 0 && flags_byte < 0x08,
        encrypted: flags_byte & 0x04 != 0 && flags_byte < 0x08,
    }
}

/// Run basic debug output mode.
fn run_basic_mode(path: &Path, file_size: u64, info: &HeaderInfo, health: &Health) {
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
            ("Health", health.badge()),
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
        print_hex_row(i * 16, chunk);
    }

    Ok(())
}

fn print_hex_row(offset: usize, chunk: &[u8]) {
    print!("{offset:08x}: ");
    print_hex_bytes(chunk);
    print_hex_padding(chunk.len());
    print_ascii(chunk);
}

fn print_hex_bytes(chunk: &[u8]) {
    for (j, byte) in chunk.iter().enumerate() {
        if j == 8 {
            print!(" ");
        }
        print!("{byte:02x} ");
    }
}

fn print_hex_padding(len: usize) {
    for j in len..16 {
        if j == 8 {
            print!(" ");
        }
        print!("   ");
    }
}

fn print_ascii(chunk: &[u8]) {
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
