//! Debug command implementation
//!
//! Toyota Way: Visualization - Make problems visible.
//! Simple debugging with optional "drama" theatrical mode.

use crate::error::CliError;
use crate::output;
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

    // Read and parse header for drama/basic modes
    let (header_bytes, file_size) = read_header(path)?;
    let info = parse_header(&header_bytes);

    if drama {
        run_drama_mode(path, &header_bytes, file_size, info.magic_valid);
    } else {
        run_basic_mode(path, file_size, &info);
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

    println!(
        "{}: APR v{}.{} {} ({})",
        filename.cyan().bold(),
        info.version.0,
        info.version.1,
        format_model_type(info.model_type),
        humansize::format_size(file_size, humansize::BINARY)
    );

    println!(
        "  magic: {} ({})",
        info.magic_str,
        if info.magic_valid {
            "valid".green()
        } else {
            "INVALID".red().bold()
        }
    );

    // Flags
    let flag_list = collect_flags(info);
    if !flag_list.is_empty() {
        println!("  flags: {}", flag_list.join(", "));
    }

    println!(
        "  health: {}",
        if info.magic_valid {
            "OK".green().bold()
        } else {
            "CORRUPTED".red().bold()
        }
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
mod tests {
    use super::*;
    use std::io::Write;
    use tempfile::{tempdir, NamedTempFile};

    // ========================================================================
    // Path Validation Tests
    // ========================================================================

    #[test]
    fn test_validate_path_not_found() {
        let result = validate_path(Path::new("/nonexistent/model.apr"));
        assert!(result.is_err());
        match result {
            Err(CliError::FileNotFound(_)) => {}
            _ => panic!("Expected FileNotFound error"),
        }
    }

    #[test]
    fn test_validate_path_is_directory() {
        let dir = tempdir().expect("create dir");
        let result = validate_path(dir.path());
        assert!(result.is_err());
        match result {
            Err(CliError::NotAFile(_)) => {}
            _ => panic!("Expected NotAFile error"),
        }
    }

    #[test]
    fn test_validate_path_valid() {
        let file = NamedTempFile::new().expect("create file");
        let result = validate_path(file.path());
        assert!(result.is_ok());
    }

    // ========================================================================
    // Model Type Formatting Tests
    // ========================================================================

    #[test]
    fn test_format_model_type_linear_regression() {
        assert_eq!(format_model_type(0x0001), "LinearRegression");
    }

    #[test]
    fn test_format_model_type_logistic_regression() {
        assert_eq!(format_model_type(0x0002), "LogisticRegression");
    }

    #[test]
    fn test_format_model_type_decision_tree() {
        assert_eq!(format_model_type(0x0003), "DecisionTree");
    }

    #[test]
    fn test_format_model_type_random_forest() {
        assert_eq!(format_model_type(0x0004), "RandomForest");
    }

    #[test]
    fn test_format_model_type_gradient_boosting() {
        assert_eq!(format_model_type(0x0005), "GradientBoosting");
    }

    #[test]
    fn test_format_model_type_kmeans() {
        assert_eq!(format_model_type(0x0006), "KMeans");
    }

    #[test]
    fn test_format_model_type_pca() {
        assert_eq!(format_model_type(0x0007), "PCA");
    }

    #[test]
    fn test_format_model_type_naive_bayes() {
        assert_eq!(format_model_type(0x0008), "NaiveBayes");
    }

    #[test]
    fn test_format_model_type_knn() {
        assert_eq!(format_model_type(0x0009), "KNN");
    }

    #[test]
    fn test_format_model_type_svm() {
        assert_eq!(format_model_type(0x000A), "SVM");
    }

    #[test]
    fn test_format_model_type_ngram_lm() {
        assert_eq!(format_model_type(0x0010), "NgramLM");
    }

    #[test]
    fn test_format_model_type_tfidf() {
        assert_eq!(format_model_type(0x0011), "TfIdf");
    }

    #[test]
    fn test_format_model_type_count_vectorizer() {
        assert_eq!(format_model_type(0x0012), "CountVectorizer");
    }

    #[test]
    fn test_format_model_type_neural_sequential() {
        assert_eq!(format_model_type(0x0020), "NeuralSequential");
    }

    #[test]
    fn test_format_model_type_neural_custom() {
        assert_eq!(format_model_type(0x0021), "NeuralCustom");
    }

    #[test]
    fn test_format_model_type_content_recommender() {
        assert_eq!(format_model_type(0x0030), "ContentRecommender");
    }

    #[test]
    fn test_format_model_type_mixture_of_experts() {
        assert_eq!(format_model_type(0x0040), "MixtureOfExperts");
    }

    #[test]
    fn test_format_model_type_custom() {
        assert_eq!(format_model_type(0x00FF), "Custom");
    }

    #[test]
    fn test_format_model_type_unknown() {
        assert_eq!(format_model_type(0xDEAD), "Unknown(0xDEAD)");
    }

    // ========================================================================
    // Run Command Tests
    // ========================================================================

    #[test]
    fn test_run_file_not_found() {
        let result = run(
            Path::new("/nonexistent/model.apr"),
            false,
            false,
            false,
            100,
        );
        assert!(result.is_err());
    }

    #[test]
    fn test_run_with_small_file() {
        let mut file = NamedTempFile::with_suffix(".apr").expect("create file");
        // Write something smaller than header size
        file.write_all(b"short").expect("write");

        // Should fail because file is too small for header
        let result = run(file.path(), false, false, false, 100);
        assert!(result.is_err());
    }

    // ========================================================================
    // Header Parsing Tests
    // ========================================================================

    #[test]
    fn test_parse_header_invalid_magic() {
        let mut header = [0u8; HEADER_SIZE];
        header[0..4].copy_from_slice(b"XXXX");
        let info = parse_header(&header);
        assert!(!info.magic_valid);
        assert_eq!(info.magic_str, "XXXX");
    }

    #[test]
    fn test_parse_header_valid_aprn_magic() {
        let mut header = [0u8; HEADER_SIZE];
        header[0..4].copy_from_slice(b"APRN");
        header[4] = 1; // version major
        header[5] = 0; // version minor
        let info = parse_header(&header);
        assert!(info.magic_valid);
        assert_eq!(info.magic_str, "APRN");
        assert_eq!(info.version, (1, 0));
    }

    #[test]
    fn test_parse_header_valid_apr2_magic() {
        let mut header = [0u8; HEADER_SIZE];
        header[0..4].copy_from_slice(b"APR2");
        header[4] = 2; // version major
        header[5] = 0; // version minor
        let info = parse_header(&header);
        assert!(info.magic_valid);
        assert_eq!(info.magic_str, "APR2");
        assert_eq!(info.version, (2, 0));
    }

    #[test]
    fn test_parse_header_flags() {
        let mut header = [0u8; HEADER_SIZE];
        header[0..4].copy_from_slice(b"APRN");
        header[21] = 0b00000111; // compressed, signed, encrypted
        let info = parse_header(&header);
        assert!(info.compressed);
        assert!(info.signed);
        assert!(info.encrypted);
    }

    #[test]
    fn test_parse_header_model_type() {
        let mut header = [0u8; HEADER_SIZE];
        header[0..4].copy_from_slice(b"APRN");
        header[6] = 0x06; // KMeans (low byte)
        header[7] = 0x00; // (high byte)
        let info = parse_header(&header);
        assert_eq!(info.model_type, 0x0006);
    }
}
