//! APR Format Falsification Test Suite
//!
//! **Purpose**: Attempt to BREAK the APR format's safety claims through adversarial inputs.
//!
//! ## Hypotheses Under Attack
//!
//! 1. **Type Safety**: "The `.apr` format is type-safe and validated."
//! 2. **Zero-Copy Efficiency**: "The `.apr` format is zero-copy efficient."
//! 3. **Metadata Robustness**: "The `.apr` format supports robust metadata."
//!
//! ## Success Criteria
//!
//! - **Falsification succeeds** if any test causes `panic!`, `segfault`, or OOM
//! - **Falsification fails** (format is safe) if library returns `Result::Err`
//!
//! ## Methodology
//!
//! - Raw bytes written to tempfile (no helper library)
//! - Fuzzing-style adversarial payloads
//! - Boundary condition exploitation

#![allow(clippy::cast_possible_truncation, clippy::match_wild_err_arm)]

use std::io::Write;
use std::time::{Duration, Instant};
use tempfile::NamedTempFile;

// APR format constants (from spec)
const MAGIC: [u8; 4] = [0x41, 0x50, 0x52, 0x00]; // "APR\0"
const HEADER_SIZE: usize = 64;
const ALIGNMENT: usize = 64;

// ============================================================================
// Helper: Raw APR File Builder (No Library Dependencies)
// ============================================================================

/// Build a raw APR header from scratch
fn build_apr_header(
    tensor_count: u32,
    metadata_offset: u64,
    metadata_size: u32,
    tensor_index_offset: u64,
    data_offset: u64,
) -> [u8; HEADER_SIZE] {
    let mut header = [0u8; HEADER_SIZE];

    // Magic (4 bytes)
    header[0..4].copy_from_slice(&MAGIC);

    // Version major.minor (2 bytes)
    header[4] = 2; // major
    header[5] = 0; // minor

    // Flags (2 bytes) - uncompressed
    header[6] = 0;
    header[7] = 0;

    // Tensor count (4 bytes LE)
    header[8..12].copy_from_slice(&tensor_count.to_le_bytes());

    // Metadata offset (8 bytes LE)
    header[12..20].copy_from_slice(&metadata_offset.to_le_bytes());

    // Metadata size (4 bytes LE)
    header[20..24].copy_from_slice(&metadata_size.to_le_bytes());

    // Tensor index offset (8 bytes LE)
    header[24..32].copy_from_slice(&tensor_index_offset.to_le_bytes());

    // Data offset (8 bytes LE)
    header[32..40].copy_from_slice(&data_offset.to_le_bytes());

    // Checksum (4 bytes) - set to 0 (unchecked)
    header[40..44].copy_from_slice(&0u32.to_le_bytes());

    // Reserved (20 bytes) - already zeroed

    header
}

/// Build a tensor entry for the tensor index
fn build_tensor_entry(name: &str, dtype: u8, shape: &[u64], offset: u64, size: u64) -> Vec<u8> {
    let mut entry = Vec::new();

    // Name length (1 byte)
    entry.push(name.len() as u8);
    // Name bytes
    entry.extend_from_slice(name.as_bytes());
    // dtype (1 byte)
    entry.push(dtype);
    // ndim (1 byte)
    entry.push(shape.len() as u8);
    // dims (8 bytes each)
    for &dim in shape {
        entry.extend_from_slice(&dim.to_le_bytes());
    }
    // offset (8 bytes)
    entry.extend_from_slice(&offset.to_le_bytes());
    // size (8 bytes)
    entry.extend_from_slice(&size.to_le_bytes());

    entry
}

/// Pad data to 64-byte alignment
fn pad_to_alignment(data: &[u8]) -> Vec<u8> {
    let padded_size = data.len().div_ceil(ALIGNMENT) * ALIGNMENT;
    let mut result = data.to_vec();
    result.resize(padded_size, 0);
    result
}

// ============================================================================
// ATTACK 1: Type Safety - Invalid Enum Values
// ============================================================================

/// Attack: dtype byte = 0xFF (invalid enum value)
/// Expected: Parser returns Err, not panic
#[test]
fn test_falsify_type_safety_invalid_dtype_0xff() {
    // Build a minimal APR file with invalid dtype
    let metadata = r#"{"architecture":"test"}"#;
    let metadata_bytes = pad_to_alignment(metadata.as_bytes());

    // Tensor entry with dtype = 0xFF (invalid)
    let tensor_entry = build_tensor_entry("invalid.weight", 0xFF, &[4], 0, 16);

    // Calculate offsets
    let metadata_offset = HEADER_SIZE as u64;
    let tensor_index_offset = metadata_offset + metadata_bytes.len() as u64;
    let data_offset = tensor_index_offset + tensor_entry.len() as u64;
    let data_offset_aligned = data_offset.div_ceil(64) * 64;

    // Build header
    let header = build_apr_header(
        1,
        metadata_offset,
        metadata.len() as u32,
        tensor_index_offset,
        data_offset_aligned,
    );

    // Assemble file
    let mut file_data = Vec::new();
    file_data.extend_from_slice(&header);
    file_data.extend_from_slice(&metadata_bytes);
    file_data.extend_from_slice(&tensor_entry);
    file_data.resize(data_offset_aligned as usize, 0); // Pad to data offset
    file_data.extend_from_slice(&[0u8; 16]); // Tensor data (4 F32s)

    // Write to tempfile
    let mut tmpfile = NamedTempFile::new().expect("tempfile creation failed");
    tmpfile.write_all(&file_data).expect("write failed");
    tmpfile.flush().expect("flush failed");

    // Attempt to load - should NOT panic
    let result = std::panic::catch_unwind(|| realizar::apr::AprV2Model::load(tmpfile.path()));

    // Falsification check: panic = format is unsafe
    match result {
        Ok(Ok(_model)) => {
            // Loaded successfully - dtype 0xFF defaults to F32 (acceptable)
            println!("[PASS] Invalid dtype 0xFF handled gracefully (defaults to F32)");
        },
        Ok(Err(e)) => {
            println!("[PASS] Invalid dtype 0xFF returned Err: {e}");
        },
        Err(_panic) => {
            panic!("[FALSIFIED] Invalid dtype 0xFF caused PANIC - format is UNSAFE!");
        },
    }
}

/// Attack: dtype byte values 11-254 (undefined enum range)
#[test]
fn test_falsify_type_safety_undefined_dtype_range() {
    for dtype in [11u8, 50, 100, 200, 254] {
        let metadata = r#"{"architecture":"test"}"#;
        let metadata_bytes = pad_to_alignment(metadata.as_bytes());
        let tensor_entry = build_tensor_entry("test.weight", dtype, &[4], 0, 16);

        let metadata_offset = HEADER_SIZE as u64;
        let tensor_index_offset = metadata_offset + metadata_bytes.len() as u64;
        let data_offset = (tensor_index_offset + tensor_entry.len() as u64).div_ceil(64) * 64;

        let header = build_apr_header(
            1,
            metadata_offset,
            metadata.len() as u32,
            tensor_index_offset,
            data_offset,
        );

        let mut file_data = Vec::new();
        file_data.extend_from_slice(&header);
        file_data.extend_from_slice(&metadata_bytes);
        file_data.extend_from_slice(&tensor_entry);
        file_data.resize(data_offset as usize, 0);
        file_data.extend_from_slice(&[0u8; 16]);

        let mut tmpfile = NamedTempFile::new().expect("tempfile creation failed");
        tmpfile.write_all(&file_data).expect("write failed");
        tmpfile.flush().expect("flush failed");

        let result = std::panic::catch_unwind(|| realizar::apr::AprV2Model::load(tmpfile.path()));

        match result {
            Ok(_) => {
                println!("[PASS] dtype={dtype} handled without panic");
            },
            Err(_) => {
                panic!("[FALSIFIED] dtype={dtype} caused PANIC!");
            },
        }
    }
}

// ============================================================================
// ATTACK 2: Type Safety - Shape/Data Mismatch
// ============================================================================

/// Attack: shape=[100] but only 50 bytes of data
/// Expected: Parser returns Err, not panic
#[test]
fn test_falsify_type_safety_shape_data_mismatch() {
    let metadata = r#"{"architecture":"test"}"#;
    let metadata_bytes = pad_to_alignment(metadata.as_bytes());

    // Claim shape [100] (100 F32s = 400 bytes) but size is only 50 bytes
    let tensor_entry = build_tensor_entry("mismatched.weight", 0, &[100], 0, 50);

    let metadata_offset = HEADER_SIZE as u64;
    let tensor_index_offset = metadata_offset + metadata_bytes.len() as u64;
    let data_offset = (tensor_index_offset + tensor_entry.len() as u64).div_ceil(64) * 64;

    let header = build_apr_header(
        1,
        metadata_offset,
        metadata.len() as u32,
        tensor_index_offset,
        data_offset,
    );

    let mut file_data = Vec::new();
    file_data.extend_from_slice(&header);
    file_data.extend_from_slice(&metadata_bytes);
    file_data.extend_from_slice(&tensor_entry);
    file_data.resize(data_offset as usize, 0);
    file_data.extend_from_slice(&[0u8; 50]); // Only 50 bytes, not 400

    let mut tmpfile = NamedTempFile::new().expect("tempfile creation failed");
    tmpfile.write_all(&file_data).expect("write failed");
    tmpfile.flush().expect("flush failed");

    let result = std::panic::catch_unwind(|| {
        let model = realizar::apr::AprV2Model::load(tmpfile.path())?;
        // Try to read the tensor - this should trigger the mismatch
        model.get_tensor_f32("mismatched.weight")
    });

    match result {
        Ok(Ok(_)) => {
            println!("[WARN] Shape/data mismatch not detected during read");
        },
        Ok(Err(e)) => {
            println!("[PASS] Shape/data mismatch returned Err: {e}");
        },
        Err(_) => {
            panic!("[FALSIFIED] Shape/data mismatch caused PANIC!");
        },
    }
}

/// Attack: Negative-equivalent shape dimensions (via overflow)
#[test]
fn test_falsify_type_safety_overflow_shape() {
    let metadata = r#"{"architecture":"test"}"#;
    let metadata_bytes = pad_to_alignment(metadata.as_bytes());

    // Shape with maximum u64 value (would overflow any multiplication)
    let tensor_entry = build_tensor_entry("overflow.weight", 0, &[u64::MAX, 2], 0, 16);

    let metadata_offset = HEADER_SIZE as u64;
    let tensor_index_offset = metadata_offset + metadata_bytes.len() as u64;
    let data_offset = (tensor_index_offset + tensor_entry.len() as u64).div_ceil(64) * 64;

    let header = build_apr_header(
        1,
        metadata_offset,
        metadata.len() as u32,
        tensor_index_offset,
        data_offset,
    );

    let mut file_data = Vec::new();
    file_data.extend_from_slice(&header);
    file_data.extend_from_slice(&metadata_bytes);
    file_data.extend_from_slice(&tensor_entry);
    file_data.resize(data_offset as usize, 0);
    file_data.extend_from_slice(&[0u8; 16]);

    let mut tmpfile = NamedTempFile::new().expect("tempfile creation failed");
    tmpfile.write_all(&file_data).expect("write failed");
    tmpfile.flush().expect("flush failed");

    let result = std::panic::catch_unwind(|| realizar::apr::AprV2Model::load(tmpfile.path()));

    match result {
        Ok(Ok(_)) => {
            println!("[WARN] Overflow shape loaded successfully (may be deferred validation)");
        },
        Ok(Err(e)) => {
            println!("[PASS] Overflow shape returned Err: {e}");
        },
        Err(_) => {
            panic!("[FALSIFIED] Overflow shape caused PANIC!");
        },
    }
}

// ============================================================================
// ATTACK 3: Zero-Copy Efficiency - Many Tiny Tensors
// ============================================================================

/// Attack: 10,000 tiny tensors (1 byte each)
/// Falsification: If load time scales linearly with count, implies heap allocation per entry
#[test]
fn test_falsify_efficiency_many_tiny_tensors() {
    let tensor_counts = [100, 1000, 10_000];
    let mut load_times = Vec::new();

    for &count in &tensor_counts {
        let metadata = format!(r#"{{"architecture":"test","tensor_count":{count}}}"#);
        let metadata_bytes = pad_to_alignment(metadata.as_bytes());

        // Build tensor entries
        let mut tensor_index = Vec::new();
        for i in 0..count {
            let name = format!("t{i}");
            let entry = build_tensor_entry(&name, 7, &[1], i as u64, 1); // U8, 1 byte each
            tensor_index.extend_from_slice(&entry);
        }

        let metadata_offset = HEADER_SIZE as u64;
        let tensor_index_offset = metadata_offset + metadata_bytes.len() as u64;
        let data_offset = (tensor_index_offset + tensor_index.len() as u64).div_ceil(64) * 64;

        let header = build_apr_header(
            count as u32,
            metadata_offset,
            metadata.len() as u32,
            tensor_index_offset,
            data_offset,
        );

        let mut file_data = Vec::new();
        file_data.extend_from_slice(&header);
        file_data.extend_from_slice(&metadata_bytes);
        file_data.extend_from_slice(&tensor_index);
        file_data.resize(data_offset as usize, 0);
        file_data.extend_from_slice(&vec![0u8; count]); // 1 byte per tensor

        let mut tmpfile = NamedTempFile::new().expect("tempfile creation failed");
        tmpfile.write_all(&file_data).expect("write failed");
        tmpfile.flush().expect("flush failed");

        // Time the load
        let start = Instant::now();
        let result = std::panic::catch_unwind(|| realizar::apr::AprV2Model::load(tmpfile.path()));
        let elapsed = start.elapsed();

        match result {
            Ok(Ok(_model)) => {
                load_times.push((count, elapsed));
                println!("[INFO] {count} tensors loaded in {:?}", elapsed);
            },
            Ok(Err(e)) => {
                println!("[PASS] {count} tensors returned Err: {e}");
                return; // Early return, test passes
            },
            Err(_) => {
                panic!("[FALSIFIED] {count} tensors caused PANIC!");
            },
        }
    }

    // Analyze scaling: if O(n) with high constant, efficiency claim is weakened
    if load_times.len() >= 2 {
        let (c1, t1) = load_times[0];
        let (c2, t2) = load_times[load_times.len() - 1];
        let ratio = c2 as f64 / c1 as f64;
        let time_ratio = t2.as_secs_f64() / t1.as_secs_f64();

        println!("\n[ANALYSIS] Scaling factor:");
        println!("  Tensor count ratio: {ratio:.1}x");
        println!("  Load time ratio: {time_ratio:.1}x");

        // If time scales worse than O(n log n), flag it
        let expected_ratio = ratio * (ratio.ln() / (c1 as f64).ln()).max(1.0);
        if time_ratio > expected_ratio * 2.0 {
            println!("[WARN] Load time scaling worse than O(n log n)");
        } else {
            println!("[PASS] Load time scaling is acceptable");
        }
    }
}

/// Attack: Unaligned tensor offsets (offset=63 instead of 64)
#[test]
fn test_falsify_efficiency_unaligned_offset() {
    let metadata = r#"{"architecture":"test"}"#;
    let metadata_bytes = pad_to_alignment(metadata.as_bytes());

    // Tensor at offset 63 (misaligned by 1 byte)
    let tensor_entry = build_tensor_entry("unaligned.weight", 0, &[4], 63, 16);

    let metadata_offset = HEADER_SIZE as u64;
    let tensor_index_offset = metadata_offset + metadata_bytes.len() as u64;
    let data_offset = (tensor_index_offset + tensor_entry.len() as u64).div_ceil(64) * 64;

    let header = build_apr_header(
        1,
        metadata_offset,
        metadata.len() as u32,
        tensor_index_offset,
        data_offset,
    );

    let mut file_data = Vec::new();
    file_data.extend_from_slice(&header);
    file_data.extend_from_slice(&metadata_bytes);
    file_data.extend_from_slice(&tensor_entry);
    file_data.resize(data_offset as usize + 63 + 16, 0); // Ensure enough data

    let mut tmpfile = NamedTempFile::new().expect("tempfile creation failed");
    tmpfile.write_all(&file_data).expect("write failed");
    tmpfile.flush().expect("flush failed");

    let result = std::panic::catch_unwind(|| {
        let model = realizar::apr::AprV2Model::load(tmpfile.path())?;
        model.get_tensor_f32("unaligned.weight")
    });

    match result {
        Ok(Ok(data)) => {
            println!("[PASS] Unaligned offset handled (data len={})", data.len());
        },
        Ok(Err(e)) => {
            println!("[PASS] Unaligned offset returned Err: {e}");
        },
        Err(_) => {
            panic!("[FALSIFIED] Unaligned offset caused PANIC!");
        },
    }
}

// ============================================================================
// ATTACK 4: Metadata Robustness - Malformed JSON
// ============================================================================

/// Attack: Deeply nested JSON (stack overflow attempt)
#[test]
fn test_falsify_metadata_deep_nesting() {
    // Create JSON with 1000 levels of nesting
    let depth = 1000;
    let mut json = String::new();
    for _ in 0..depth {
        json.push_str(r#"{"a":"#);
    }
    json.push_str(r#""deep""#);
    for _ in 0..depth {
        json.push('}');
    }

    let metadata_bytes = pad_to_alignment(json.as_bytes());
    let tensor_entry = build_tensor_entry("test.weight", 0, &[4], 0, 16);

    let metadata_offset = HEADER_SIZE as u64;
    let tensor_index_offset = metadata_offset + metadata_bytes.len() as u64;
    let data_offset = (tensor_index_offset + tensor_entry.len() as u64).div_ceil(64) * 64;

    let header = build_apr_header(
        1,
        metadata_offset,
        json.len() as u32,
        tensor_index_offset,
        data_offset,
    );

    let mut file_data = Vec::new();
    file_data.extend_from_slice(&header);
    file_data.extend_from_slice(&metadata_bytes);
    file_data.extend_from_slice(&tensor_entry);
    file_data.resize(data_offset as usize, 0);
    file_data.extend_from_slice(&[0u8; 16]);

    let mut tmpfile = NamedTempFile::new().expect("tempfile creation failed");
    tmpfile.write_all(&file_data).expect("write failed");
    tmpfile.flush().expect("flush failed");

    let start = Instant::now();
    let result = std::panic::catch_unwind(|| realizar::apr::AprV2Model::load(tmpfile.path()));
    let elapsed = start.elapsed();

    // Check for hang (>5 seconds)
    assert!(
        elapsed <= Duration::from_secs(5),
        "[FALSIFIED] Deep nesting caused hang ({}s)",
        elapsed.as_secs()
    );

    match result {
        Ok(Ok(_)) => {
            println!("[PASS] Deep nesting handled in {:?}", elapsed);
        },
        Ok(Err(e)) => {
            println!("[PASS] Deep nesting returned Err: {e}");
        },
        Err(_) => {
            panic!("[FALSIFIED] Deep nesting caused PANIC!");
        },
    }
}

/// Attack: JSON with special floats (NaN, Infinity)
#[test]
fn test_falsify_metadata_special_floats() {
    let test_cases = [
        r#"{"architecture":"test","value":NaN}"#,
        r#"{"architecture":"test","value":Infinity}"#,
        r#"{"architecture":"test","value":-Infinity}"#,
    ];

    for json in &test_cases {
        let metadata_bytes = pad_to_alignment(json.as_bytes());
        let tensor_entry = build_tensor_entry("test.weight", 0, &[4], 0, 16);

        let metadata_offset = HEADER_SIZE as u64;
        let tensor_index_offset = metadata_offset + metadata_bytes.len() as u64;
        let data_offset = (tensor_index_offset + tensor_entry.len() as u64).div_ceil(64) * 64;

        let header = build_apr_header(
            1,
            metadata_offset,
            json.len() as u32,
            tensor_index_offset,
            data_offset,
        );

        let mut file_data = Vec::new();
        file_data.extend_from_slice(&header);
        file_data.extend_from_slice(&metadata_bytes);
        file_data.extend_from_slice(&tensor_entry);
        file_data.resize(data_offset as usize, 0);
        file_data.extend_from_slice(&[0u8; 16]);

        let mut tmpfile = NamedTempFile::new().expect("tempfile creation failed");
        tmpfile.write_all(&file_data).expect("write failed");
        tmpfile.flush().expect("flush failed");

        let result = std::panic::catch_unwind(|| realizar::apr::AprV2Model::load(tmpfile.path()));

        match result {
            Ok(Ok(_)) => {
                println!(
                    "[INFO] Special float JSON accepted: {}",
                    &json[..50.min(json.len())]
                );
            },
            Ok(Err(_)) => {
                println!("[PASS] Special float JSON rejected");
            },
            Err(_) => {
                panic!("[FALSIFIED] Special float JSON caused PANIC!");
            },
        }
    }
}

/// Attack: Extremely long string in metadata
#[test]
fn test_falsify_metadata_huge_string() {
    // 10MB string
    let huge_string = "x".repeat(10_000_000);
    let json = format!(r#"{{"architecture":"test","huge":"{}"}}"#, huge_string);

    let metadata_bytes = pad_to_alignment(json.as_bytes());
    let tensor_entry = build_tensor_entry("test.weight", 0, &[4], 0, 16);

    let metadata_offset = HEADER_SIZE as u64;
    let tensor_index_offset = metadata_offset + metadata_bytes.len() as u64;
    let data_offset = (tensor_index_offset + tensor_entry.len() as u64).div_ceil(64) * 64;

    let header = build_apr_header(
        1,
        metadata_offset,
        json.len() as u32,
        tensor_index_offset,
        data_offset,
    );

    let mut file_data = Vec::new();
    file_data.extend_from_slice(&header);
    file_data.extend_from_slice(&metadata_bytes);
    file_data.extend_from_slice(&tensor_entry);
    file_data.resize(data_offset as usize, 0);
    file_data.extend_from_slice(&[0u8; 16]);

    let mut tmpfile = NamedTempFile::new().expect("tempfile creation failed");
    tmpfile.write_all(&file_data).expect("write failed");
    tmpfile.flush().expect("flush failed");

    let start = Instant::now();
    let result = std::panic::catch_unwind(|| realizar::apr::AprV2Model::load(tmpfile.path()));
    let elapsed = start.elapsed();

    println!("[INFO] 10MB metadata processed in {:?}", elapsed);

    match result {
        Ok(Ok(_)) => {
            println!("[PASS] Huge metadata handled gracefully");
        },
        Ok(Err(e)) => {
            println!("[PASS] Huge metadata returned Err: {e}");
        },
        Err(_) => {
            panic!("[FALSIFIED] Huge metadata caused PANIC!");
        },
    }
}

/// Attack: Invalid UTF-8 in tensor name
#[test]
fn test_falsify_metadata_invalid_utf8_tensor_name() {
    let metadata = r#"{"architecture":"test"}"#;
    let metadata_bytes = pad_to_alignment(metadata.as_bytes());

    // Build tensor entry with invalid UTF-8 name bytes
    let mut tensor_entry = Vec::new();
    let invalid_name = &[0xFF, 0xFE, 0x80, 0x81]; // Invalid UTF-8 sequence
    tensor_entry.push(invalid_name.len() as u8);
    tensor_entry.extend_from_slice(invalid_name);
    tensor_entry.push(0); // dtype = F32
    tensor_entry.push(1); // ndim = 1
    tensor_entry.extend_from_slice(&4u64.to_le_bytes()); // shape[0] = 4
    tensor_entry.extend_from_slice(&0u64.to_le_bytes()); // offset
    tensor_entry.extend_from_slice(&16u64.to_le_bytes()); // size

    let metadata_offset = HEADER_SIZE as u64;
    let tensor_index_offset = metadata_offset + metadata_bytes.len() as u64;
    let data_offset = (tensor_index_offset + tensor_entry.len() as u64).div_ceil(64) * 64;

    let header = build_apr_header(
        1,
        metadata_offset,
        metadata.len() as u32,
        tensor_index_offset,
        data_offset,
    );

    let mut file_data = Vec::new();
    file_data.extend_from_slice(&header);
    file_data.extend_from_slice(&metadata_bytes);
    file_data.extend_from_slice(&tensor_entry);
    file_data.resize(data_offset as usize, 0);
    file_data.extend_from_slice(&[0u8; 16]);

    let mut tmpfile = NamedTempFile::new().expect("tempfile creation failed");
    tmpfile.write_all(&file_data).expect("write failed");
    tmpfile.flush().expect("flush failed");

    let result = std::panic::catch_unwind(|| realizar::apr::AprV2Model::load(tmpfile.path()));

    match result {
        Ok(Ok(_)) => {
            println!("[WARN] Invalid UTF-8 tensor name was accepted");
        },
        Ok(Err(e)) => {
            println!("[PASS] Invalid UTF-8 tensor name returned Err: {e}");
        },
        Err(_) => {
            panic!("[FALSIFIED] Invalid UTF-8 tensor name caused PANIC!");
        },
    }
}

// ============================================================================
// ATTACK 5: Boundary Conditions
// ============================================================================

/// Attack: Zero-length file
#[test]
fn test_falsify_boundary_empty_file() {
    let mut tmpfile = NamedTempFile::new().expect("tempfile creation failed");
    tmpfile.flush().expect("flush failed");

    let result = std::panic::catch_unwind(|| realizar::apr::AprV2Model::load(tmpfile.path()));

    match result {
        Ok(Ok(_)) => {
            panic!("[FALSIFIED] Empty file loaded successfully!");
        },
        Ok(Err(e)) => {
            println!("[PASS] Empty file returned Err: {e}");
        },
        Err(_) => {
            panic!("[FALSIFIED] Empty file caused PANIC!");
        },
    }
}

/// Attack: File with only magic bytes
#[test]
fn test_falsify_boundary_magic_only() {
    let mut tmpfile = NamedTempFile::new().expect("tempfile creation failed");
    tmpfile.write_all(&MAGIC).expect("write failed");
    tmpfile.flush().expect("flush failed");

    let result = std::panic::catch_unwind(|| realizar::apr::AprV2Model::load(tmpfile.path()));

    match result {
        Ok(Ok(_)) => {
            panic!("[FALSIFIED] Magic-only file loaded successfully!");
        },
        Ok(Err(e)) => {
            println!("[PASS] Magic-only file returned Err: {e}");
        },
        Err(_) => {
            panic!("[FALSIFIED] Magic-only file caused PANIC!");
        },
    }
}

/// Attack: Wrong magic bytes
#[test]
fn test_falsify_boundary_wrong_magic() {
    let mut file_data = vec![0u8; HEADER_SIZE + 100];
    file_data[0..4].copy_from_slice(b"GGUF"); // Wrong magic

    let mut tmpfile = NamedTempFile::new().expect("tempfile creation failed");
    tmpfile.write_all(&file_data).expect("write failed");
    tmpfile.flush().expect("flush failed");

    let result = std::panic::catch_unwind(|| realizar::apr::AprV2Model::load(tmpfile.path()));

    match result {
        Ok(Ok(_)) => {
            panic!("[FALSIFIED] Wrong magic file loaded successfully!");
        },
        Ok(Err(e)) => {
            println!("[PASS] Wrong magic returned Err: {e}");
        },
        Err(_) => {
            panic!("[FALSIFIED] Wrong magic caused PANIC!");
        },
    }
}

/// Attack: Tensor offset beyond file size
#[test]
fn test_falsify_boundary_offset_beyond_eof() {
    let metadata = r#"{"architecture":"test"}"#;
    let metadata_bytes = pad_to_alignment(metadata.as_bytes());

    // Tensor claims to be at offset 1GB
    let tensor_entry = build_tensor_entry("beyond.weight", 0, &[4], 1_000_000_000, 16);

    let metadata_offset = HEADER_SIZE as u64;
    let tensor_index_offset = metadata_offset + metadata_bytes.len() as u64;
    let data_offset = (tensor_index_offset + tensor_entry.len() as u64).div_ceil(64) * 64;

    let header = build_apr_header(
        1,
        metadata_offset,
        metadata.len() as u32,
        tensor_index_offset,
        data_offset,
    );

    let mut file_data = Vec::new();
    file_data.extend_from_slice(&header);
    file_data.extend_from_slice(&metadata_bytes);
    file_data.extend_from_slice(&tensor_entry);
    file_data.resize(data_offset as usize, 0);
    file_data.extend_from_slice(&[0u8; 16]); // Actual data is much smaller

    let mut tmpfile = NamedTempFile::new().expect("tempfile creation failed");
    tmpfile.write_all(&file_data).expect("write failed");
    tmpfile.flush().expect("flush failed");

    let result = std::panic::catch_unwind(|| {
        let model = realizar::apr::AprV2Model::load(tmpfile.path())?;
        model.get_tensor_f32("beyond.weight")
    });

    match result {
        Ok(Ok(_)) => {
            panic!("[FALSIFIED] Offset beyond EOF returned data!");
        },
        Ok(Err(e)) => {
            println!("[PASS] Offset beyond EOF returned Err: {e}");
        },
        Err(_) => {
            panic!("[FALSIFIED] Offset beyond EOF caused PANIC!");
        },
    }
}

// ============================================================================
// Summary Test
// ============================================================================

#[test]
fn test_falsification_summary() {
    println!("\n");
    println!("╔═══════════════════════════════════════════════════════════════╗");
    println!("║           APR FORMAT FALSIFICATION SUMMARY                    ║");
    println!("╠═══════════════════════════════════════════════════════════════╣");
    println!("║                                                               ║");
    println!("║  Attack Categories:                                           ║");
    println!("║  ┌────────────────────────────────────────────────────────┐   ║");
    println!("║  │ 1. Type Safety     - Invalid dtypes, shape mismatches │   ║");
    println!("║  │ 2. Efficiency      - Many tensors, unaligned offsets  │   ║");
    println!("║  │ 3. Metadata        - Deep nesting, special floats     │   ║");
    println!("║  │ 4. Boundaries      - Empty files, wrong magic, EOF    │   ║");
    println!("║  └────────────────────────────────────────────────────────┘   ║");
    println!("║                                                               ║");
    println!("║  SUCCESS: Format is SAFE if all attacks return Result::Err    ║");
    println!("║  FAILURE: Format is UNSAFE if any attack causes panic/segfault║");
    println!("║                                                               ║");
    println!("╚═══════════════════════════════════════════════════════════════╝");
    println!();
}
