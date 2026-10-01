//! Inspect command implementation (PMAT-225)
//!
//! Toyota Way: Genchi Genbutsu - Go to the source to understand.
//! Inspect APR v2 model metadata, architecture, tensors, and structure.

use crate::error::CliError;
use crate::output;
use aprender::format::v2::{AprV2Flags, AprV2Header, AprV2Metadata, HEADER_SIZE_V2, MAGIC_V2};
use serde::Serialize;
use std::fs::File;
use std::io::{BufReader, Read, Seek, SeekFrom};
use std::path::Path;

// ============================================================================
// Data Structures
// ============================================================================

/// Model inspection result for JSON output
#[derive(Serialize)]
struct InspectResult {
    file: String,
    valid: bool,
    format: String,
    version: String,
    tensor_count: u32,
    size_bytes: u64,
    checksum_valid: bool,
    /// GH-249: Top-level architecture field for parity checker compatibility
    #[serde(skip_serializing_if = "Option::is_none")]
    architecture: Option<String>,
    /// GH-249: Top-level num_layers for parity checker compatibility
    #[serde(skip_serializing_if = "Option::is_none")]
    num_layers: Option<usize>,
    /// GH-249: Top-level num_heads for parity checker compatibility
    #[serde(skip_serializing_if = "Option::is_none")]
    num_heads: Option<usize>,
    /// GH-249: Top-level hidden_size for parity checker compatibility
    #[serde(skip_serializing_if = "Option::is_none")]
    hidden_size: Option<usize>,
    /// GH-249: Top-level vocab_size for parity checker compatibility
    #[serde(skip_serializing_if = "Option::is_none")]
    vocab_size: Option<usize>,
    flags: FlagsInfo,
    metadata: MetadataInfo,
    /// APR-05: everything a `setfit-apr-v1` artifact records about itself.
    ///
    /// `skip_serializing_if` so a plain APR's output is BYTE-UNCHANGED from before
    /// this branch existed — asserted by a golden comparison, because a new key that
    /// appeared on every model would break every downstream `jq` in the fleet.
    #[serde(skip_serializing_if = "Option::is_none")]
    setfit: Option<serde_json::Value>,
}

#[derive(Serialize)]
struct FlagsInfo {
    lz4_compressed: bool,
    zstd_compressed: bool,
    encrypted: bool,
    signed: bool,
    sharded: bool,
    quantized: bool,
    has_vocab: bool,
}

/// What `apr inspect` derived from a file's metadata block.
///
/// `pub(crate)`, along with `HeaderData`, `read_and_parse_header` and the three fields
/// marked below, for ONE reason: the four-consumer agreement table in
/// `setfit_tag_tests.rs` drives this reader directly. The plan for this change budgeted
/// a single `pub(crate)` keyword on `read_metadata`; the compiler disagreed, because
/// that function's signature names two module-private types and the facts the table
/// must assert live in private fields. Everything not needed by the table stays private.
#[derive(Serialize, Default)]
pub(crate) struct MetadataInfo {
    /// `pub(crate)`: the table asserts nothing is derived from an unread block.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) model_type: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    name: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    description: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    author: Option<String>,
    // C-APR-PROVENANCE / INV-APR-PROV-002 / FM-APR-PROV-SILENT-SKIP:
    // provenance keys MUST always serialize (null when absent) so auditors
    // never see them silently skipped. Do NOT add skip_serializing_if here.
    license: Option<String>,
    data_source: Option<String>,
    data_license: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    source: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    original_format: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    created_at: Option<String>,
    architecture: Option<String>,
    // PMAT-690 P0-K: surface the HF identity fields stamped at import
    // time so operators can verify upstream propagation (apr convert
    // → apr pretrain --init → apr export). Distinct from `architecture`
    // (lowercase family) — `hf_architecture` is the canonical class
    // name like "Qwen2ForCausalLM"; `hf_model_type` mirrors
    // `config.json::model_type`. Per C-APR-INSPECT-METADATA-PROPAGATION
    // and C-APR-CONVERT-HF-ARCH the two fields render as null when
    // absent (NOT silently skipped via skip_serializing_if) so the
    // auditor can grep for them in any apr inspect --json output.
    hf_architecture: Option<String>,
    hf_model_type: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    param_count: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    vocab_size: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    hidden_size: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    num_layers: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    num_heads: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    num_kv_heads: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    intermediate_size: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    max_position_embeddings: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    rope_theta: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    chat_template: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    chat_format: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    special_tokens: Option<serde_json::Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    source_metadata: Option<serde_json::Value>,
    /// The declared metadata length the ABSOLUTE cap refused, when it refused one.
    ///
    /// Present ONLY on a container whose header declares more metadata than
    /// [`crate::setfit_tag::MAX_TAG_METADATA_BYTES`]. `skip_serializing_if` so every
    /// legitimate artifact's `--json` output is byte-identical to what it was before
    /// the cap existed, and no `output_json` / `output_json_with_quality` call site
    /// changes.
    ///
    /// It exists because the alternative is silence: without it an over-cap container
    /// renders as a model with no `model_type` and no APR-05 section, which is
    /// indistinguishable from a plain APR that genuinely has neither. That is the
    /// `inspect` half of WR-08 — the fact that the block was REFUSED is the whole
    /// diagnosis, and defaulting throws it away.
    /// `pub(crate)`: the table asserts the refusal is DISCLOSED, not defaulted.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) metadata_over_cap_bytes: Option<u64>,
    /// The raw value at custom key `setfit`, present only on a TAGGED artifact.
    ///
    /// `#[serde(skip)]`: this is an INPUT to the APR-05 section built below, not a
    /// field of the metadata block. Serializing it here would publish the artifact
    /// document twice under two different keys — two copies of one fact, which is
    /// two values that can disagree.
    /// `pub(crate)`: the table asserts NO APR-05 section is rendered for these bytes.
    #[serde(skip)]
    pub(crate) setfit_doc: Option<serde_json::Value>,
}

/// Parsed v2 header data
///
/// `pub(crate)` only because `read_metadata` and `read_and_parse_header` name it and
/// both are now `pub(crate)`; the fields stay private.
pub(crate) struct HeaderData {
    version: (u8, u8),
    flags: AprV2Flags,
    tensor_count: u32,
    metadata_offset: u64,
    metadata_size: u32,
    #[allow(dead_code)]
    tensor_index_offset: u64,
    data_offset: u64,
    checksum_valid: bool,
}

// ============================================================================
// Command Entry Point
// ============================================================================

/// Run the inspect command
#[provable_contracts_macros::contract(
    "apr-cli-operations-v1",
    equation = "side_effect_classification"
)]
pub(crate) fn run(
    path: &Path,
    show_vocab: bool,
    show_filters: bool,
    show_weights: bool,
    json_output: bool,
    show_quality: bool,
) -> Result<(), CliError> {
    validate_path(path)?;

    // Detect format via magic bytes (Rosetta Stone dispatch)
    let format = aprender::format::rosetta::FormatType::from_magic(path)
        .or_else(|_| aprender::format::rosetta::FormatType::from_extension(path));

    match format {
        Ok(
            aprender::format::rosetta::FormatType::Gguf
            | aprender::format::rosetta::FormatType::SafeTensors,
            // GH-685: forward all flags to rosetta path (were dropped)
        ) => {
            let result = run_rosetta_inspect(path, show_vocab, show_weights, json_output);
            // GH-685: --filters on GGUF/SafeTensors — acknowledge the flag
            if show_filters && !json_output {
                println!();
                println!("  (--filters: GGUF/SafeTensors format has no security filter metadata)");
            }
            // PMAT-690 P3-A: --quality on non-APR is informational; the
            // score depends on AprV2Metadata fields not present in raw
            // GGUF/SafeTensors. Operators run `apr convert` first.
            if show_quality && !json_output {
                println!();
                println!("  (--quality: run `apr convert` to APR first for the full score)");
            }
            result
        }
        _ => {
            // Default: APR v2 inspect (existing path)
            let file = File::open(path)?;
            let file_size = file.metadata()?.len();
            let mut reader = BufReader::new(file);

            let header = read_and_parse_header(&mut reader)?;
            // aprender#2564: the header is 64 bytes and its offsets are simply
            // BELIEVED. Without this, `apr inspect` reads a 1 KiB fragment of a
            // 991 MB model and reports `valid: true, tensor_count: 291` with a
            // data offset of 3.66 MiB -- past EOF -- while validate/tensors/lint/qa
            // all reject the same file. Two commands in one binary disagreeing about
            // one file is bad; the one that says VALID being the documented
            // first-line diagnostic, and the JSON a CI gate consumes, is the defect.
            check_header_fits_file(&header, file_size)?;
            let metadata_info = read_metadata(&mut reader, &header);

            // APR-05. Built ONCE, from the document the metadata read already
            // recovered, and consumed by whichever renderer runs. Inspection is
            // READ-ONLY metadata work: no tensor is loaded and no probe is replayed,
            // because `inspect` does not classify and therefore does not need the
            // verified state APR-04 gates prediction behind. The whole file is read
            // only for the artifact's SHA-256, and only through the ONE bounded door.
            let setfit = build_setfit_inspection(path, metadata_info.setfit_doc.as_ref());

            if json_output {
                output_json_with_quality(
                    path,
                    file_size,
                    &header,
                    metadata_info,
                    show_quality,
                    setfit,
                );
            } else {
                output_text(
                    path,
                    file_size,
                    &header,
                    &metadata_info,
                    show_vocab,
                    show_filters,
                    show_weights,
                );
                output_metadata_over_cap_text(&metadata_info);
                output_setfit_text(setfit.as_ref());
                if show_quality {
                    output_quality_text(&metadata_info, &header);
                }
            }
            Ok(())
        }
    }
}

/// GGUF/SafeTensors inspect via RosettaStone
/// Print rosetta inspection report as JSON.
fn output_rosetta_json(path: &Path, report: &aprender::format::rosetta::InspectionReport) {
    if let Ok(json) = serde_json::to_string_pretty(&rosetta_json(path, report)) {
        println!("{json}");
    }
}

/// The `apr inspect --json` document for a GGUF/SafeTensors file.
///
/// `metadata` is every key in the report, unfiltered: #3733's gate reads a
/// model's declared context and memory arithmetic from it, and a second GGUF
/// reader is refused (`check_no_hand_rolled_parsers.sh`).
fn rosetta_json(
    path: &Path,
    report: &aprender::format::rosetta::InspectionReport,
) -> serde_json::Value {
    let mut json_map = serde_json::Map::new();
    json_map.insert(
        "file".to_string(),
        serde_json::Value::String(path.display().to_string()),
    );
    json_map.insert(
        "format".to_string(),
        serde_json::Value::String(report.format.to_string()),
    );
    json_map.insert(
        "file_size".to_string(),
        serde_json::Value::Number(serde_json::Number::from(report.file_size)),
    );
    json_map.insert(
        "total_params".to_string(),
        serde_json::Value::Number(serde_json::Number::from(report.total_params)),
    );
    // GH-249: Always include architecture and quantization (use "unknown" if absent)
    json_map.insert(
        "architecture".to_string(),
        serde_json::Value::String(
            report
                .architecture
                .clone()
                .unwrap_or_else(|| "unknown".to_string()),
        ),
    );
    json_map.insert(
        "quantization".to_string(),
        serde_json::Value::String(
            report
                .quantization
                .clone()
                .unwrap_or_else(|| "unknown".to_string()),
        ),
    );
    json_map.insert(
        "tensor_count".to_string(),
        serde_json::Value::Number(serde_json::Number::from(report.tensors.len())),
    );
    let metadata: serde_json::Value = report
        .metadata
        .iter()
        .map(|(k, v)| (k.clone(), serde_json::Value::String(v.clone())))
        .collect::<serde_json::Map<_, _>>()
        .into();
    json_map.insert("metadata".to_string(), metadata);
    serde_json::Value::Object(json_map)
}

/// Print rosetta inspection report as rich text.
fn output_rosetta_text(report: &aprender::format::rosetta::InspectionReport) {
    output::header("Rosetta Stone Inspection");

    let mut pairs: Vec<(&str, String)> = vec![
        ("Format", report.format.to_string()),
        ("File Size", output::format_size(report.file_size as u64)),
        ("Parameters", output::count_fmt(report.total_params)),
    ];
    if let Some(ref arch) = report.architecture {
        pairs.push(("Architecture", arch.clone()));
    }
    if let Some(ref quant) = report.quantization {
        pairs.push(("Quantization", quant.clone()));
    }
    println!("{}", output::kv_table(&pairs));

    if !report.metadata.is_empty() {
        output::subheader(&format!("Metadata ({} keys)", report.metadata.len()));
        let meta_pairs: Vec<(&str, String)> = report
            .metadata
            .iter()
            .map(|(k, v)| {
                let display_v = if v.len() > 60 {
                    format!("{}...", &v[..60])
                } else {
                    v.clone()
                };
                (k.as_str(), display_v)
            })
            .collect();
        println!("{}", output::kv_table(&meta_pairs));
    }

    output::subheader(&format!("Tensors ({} total)", report.tensors.len()));
    let mut rows: Vec<Vec<String>> = Vec::new();
    for (i, t) in report.tensors.iter().enumerate() {
        if i < 10 || i >= report.tensors.len().saturating_sub(2) {
            rows.push(vec![
                t.name.clone(),
                format!("{}", output::dtype_color(&t.dtype)),
                format!("{:?}", t.shape),
                output::format_size(t.size_bytes as u64),
            ]);
        } else if i == 10 {
            rows.push(vec![
                format!("... {} more ...", report.tensors.len().saturating_sub(12)),
                String::new(),
                String::new(),
                String::new(),
            ]);
        }
    }
    println!(
        "{}",
        output::table(&["Name", "DType", "Shape", "Size"], &rows)
    );
}

/// GH-682: Display vocabulary/tokenizer metadata from GGUF/SafeTensors models.
fn output_rosetta_vocab(report: &aprender::format::rosetta::InspectionReport) {
    let vocab_keys: Vec<(&str, &str)> = report
        .metadata
        .iter()
        .filter(|(k, _)| k.starts_with("tokenizer.ggml."))
        .map(|(k, v)| (k.as_str(), v.as_str()))
        .collect();

    if vocab_keys.is_empty() {
        println!("\n  (no tokenizer metadata present in model)");
        return;
    }

    output::subheader("Vocabulary");
    let pairs: Vec<(&str, String)> = vocab_keys
        .iter()
        .filter(|(k, _)| {
            // Show concise tokenizer info, skip large arrays
            !k.ends_with(".tokens") && !k.ends_with(".merges") && !k.ends_with(".token_type")
        })
        .map(|(k, v)| {
            let short_key = k.strip_prefix("tokenizer.ggml.").unwrap_or(k);
            (short_key, v.to_string())
        })
        .collect();
    println!("{}", output::kv_table(&pairs));

    // Show token counts for the large arrays
    for (k, v) in &vocab_keys {
        if k.ends_with(".tokens") || k.ends_with(".merges") {
            let short_key = k.strip_prefix("tokenizer.ggml.").unwrap_or(k);
            if let Some(len) = v.strip_prefix("[len=").and_then(|s| s.strip_suffix(']')) {
                println!("  {short_key}: {len} entries");
            }
        }
    }
}

// Contract: apr-inspect-flags-v1 F-INSPECT-FLAGS-001
fn run_rosetta_inspect(
    path: &Path,
    show_vocab: bool,
    show_weights: bool,
    json_output: bool,
) -> Result<(), CliError> {
    use aprender::format::rosetta::RosettaStone;

    let rosetta = RosettaStone::new();
    let report = rosetta
        .inspect(path)
        .map_err(|e| CliError::invalid_model_file(path, "Inspection failed", &e))?;

    if json_output {
        output_rosetta_json(path, &report);
    } else {
        output_rosetta_text(&report);
    }

    // GH-682: --vocab flag was dropped by rosetta dispatch path
    if show_vocab {
        output_rosetta_vocab(&report);
    }

    // GH-685: --weights shows per-tensor statistics
    if show_weights {
        output_rosetta_weights(&report);
    }

    Ok(())
}

/// GH-685: Display per-tensor weight statistics from GGUF/SafeTensors.
fn output_rosetta_weights(report: &aprender::format::rosetta::InspectionReport) {
    output::subheader("Weight Statistics");
    let mut rows: Vec<Vec<String>> = Vec::new();
    for t in &report.tensors {
        let elements: usize = t.shape.iter().product();
        rows.push(vec![
            t.name.clone(),
            format!("{}", output::dtype_color(&t.dtype)),
            format!("{elements}"),
            output::format_size(t.size_bytes as u64),
        ]);
    }
    if rows.len() > 12 {
        let total = rows.len();
        let mut display_rows: Vec<Vec<String>> = rows[..5].to_vec();
        display_rows.push(vec![
            format!("... {} more ...", total - 10),
            String::new(),
            String::new(),
            String::new(),
        ]);
        display_rows.extend_from_slice(&rows[total - 5..]);
        println!(
            "{}",
            output::table(&["Tensor", "DType", "Elements", "Size"], &display_rows)
        );
    } else {
        println!(
            "{}",
            output::table(&["Tensor", "DType", "Elements", "Size"], &rows)
        );
    }
    let total_elements: usize = report
        .tensors
        .iter()
        .map(|t| t.shape.iter().product::<usize>())
        .sum();
    let total_bytes: u64 = report.tensors.iter().map(|t| t.size_bytes as u64).sum();
    println!(
        "  Total: {} tensors, {} elements, {}",
        report.tensors.len(),
        output::count_fmt(total_elements),
        output::format_size(total_bytes)
    );
}

// ============================================================================
// Parsing
// ============================================================================

fn validate_path(path: &Path) -> Result<(), CliError> {
    if !path.exists() {
        return Err(CliError::FileNotFound(path.to_path_buf()));
    }
    if !path.is_file() {
        return Err(CliError::NotAFile(path.to_path_buf()));
    }
    Ok(())
}

/// `pub(crate)`: the agreement table needs a real header to drive `read_metadata`
/// with, and constructing one by hand would test a header this code never produced.
pub(crate) fn read_and_parse_header(reader: &mut BufReader<File>) -> Result<HeaderData, CliError> {
    let mut header_bytes = [0u8; HEADER_SIZE_V2];
    reader.read_exact(&mut header_bytes).map_err(|_| {
        CliError::InvalidFormat(
            "File too small to contain valid APR header (need 64 bytes)".to_string(),
        )
    })?;

    // Check magic - only APR\0 (v2) is supported for detailed inspection
    // BUG-INSPECT-001 FIX: Distinguish GGUF from legacy APR formats
    let magic = &header_bytes[0..4];
    if magic != MAGIC_V2 {
        if magic == output::MAGIC_GGUF {
            return Err(CliError::InvalidFormat(
                "GGUF format detected. Use 'apr inspect' with --format gguf flag \
                 or convert to APR format with 'apr import'."
                    .to_string(),
            ));
        }
        if output::is_valid_magic(magic) {
            return Err(CliError::InvalidFormat(
                "Legacy APR format detected (APRN/APR1/APR2). Only APR v2 (APR\\0) is supported. \
                 Re-import the model to create a v2 file."
                    .to_string(),
            ));
        }
        return Err(CliError::InvalidFormat(format!(
            "Invalid magic bytes: expected APR\\0, got {:02x}{:02x}{:02x}{:02x}",
            magic[0], magic[1], magic[2], magic[3]
        )));
    }

    let header = AprV2Header::from_bytes(&header_bytes)
        .map_err(|e| CliError::InvalidFormat(format!("Failed to parse v2 header: {e}")))?;

    let checksum_valid = header.verify_checksum();

    Ok(HeaderData {
        version: header.version,
        flags: header.flags,
        tensor_count: header.tensor_count,
        metadata_offset: header.metadata_offset,
        metadata_size: header.metadata_size,
        tensor_index_offset: header.tensor_index_offset,
        data_offset: header.data_offset,
        checksum_valid,
    })
}

/// Disclose, in the HUMAN output, a metadata block the absolute cap refused.
///
/// Printed only when the cap fired, so every legitimate artifact's text output is
/// unchanged — the `None` arm returns before printing anything.
///
/// It lives here rather than inside `output_metadata_text` because that function is
/// in `inspect_output_json.rs`, and this disclosure must not perturb that file: the
/// `skip_serializing_if` field it reports was designed to need no call-site change,
/// and a diff there would undercut the claim.
///
/// Without this line `apr inspect` renders a hostile container as a model with no
/// `model_type` and no APR-05 section — byte-identical to a plain APR that genuinely
/// has neither. The operator cannot tell "this file says nothing" from "I refused to
/// read what it says", and only the second is actionable.
fn output_metadata_over_cap_text(metadata: &MetadataInfo) {
    let Some(declared) = metadata.metadata_over_cap_bytes else {
        return;
    };
    println!("\n  Metadata: NOT READ");
    println!(
        "    The header declares {declared} bytes, over the {} byte (16 MiB) cap.",
        crate::setfit_tag::MAX_TAG_METADATA_BYTES
    );
    println!(
        "    The block was NOT read, so no model type, no identity fields and no \
         SetFit section could be derived from it."
    );
}

/// Read the metadata block, or report why it was not read.
///
/// `pub(crate)` for ONE reason: the four-consumer agreement table in
/// `setfit_tag_tests.rs` must drive `inspect`'s OWN reader — the surface where
/// `inspect`'s decision about a file is actually made — rather than a re-implementation
/// of it. A table that exercised only the two already-reachable consumers would be
/// precisely the "guard that does not scan the surface where the DECISION is made"
/// that CLAUDE.md rule 5 calls theater.
/// Refuse a header whose own offsets do not fit inside the file.
///
/// Every offset here is read from the 64-byte header and is attacker- or
/// truncation-controlled. A partial download is the common case: the header
/// arrives intact and describes a body that never did. The error text matches
/// the other commands' so a user who runs two of them sees one story.
fn check_header_fits_file(header: &HeaderData, file_size: u64) -> Result<(), CliError> {
    let too_big = |what: &str, need: u64| {
        CliError::InvalidFormat(format!(
            "Invalid header: file too small for {what} \
             (header claims {need} bytes, file is {file_size})"
        ))
    };

    if header.data_offset > file_size {
        return Err(too_big("tensor data", header.data_offset));
    }
    if header.tensor_index_offset > file_size {
        return Err(too_big("the tensor index", header.tensor_index_offset));
    }
    let meta_end = header
        .metadata_offset
        .saturating_add(u64::from(header.metadata_size));
    if meta_end > file_size {
        return Err(too_big("metadata", meta_end));
    }
    Ok(())
}

pub(crate) fn read_metadata(reader: &mut BufReader<File>, header: &HeaderData) -> MetadataInfo {
    if header.metadata_size == 0 {
        return MetadataInfo::default();
    }

    // Seek to metadata offset
    if reader
        .seek(SeekFrom::Start(header.metadata_offset))
        .is_err()
    {
        return MetadataInfo::default();
    }

    // TWO BOUNDS BEFORE THE ALLOCATION, and NEITHER SUBSUMES THE OTHER.
    // `metadata_size` is a u32 read out of the file under inspection, so a hostile
    // container can declare up to 4 GiB and make this allocate it before `read_exact`
    // discovers there is nothing to fill it with.
    //
    // (a) THE FILE LENGTH is tighter for a SMALL file: a block cannot be longer than
    //     the file that contains it.
    // (b) THE ABSOLUTE CAP is tighter for a LARGE one.
    //
    // This comment previously claimed the stat'd length was "a sound bound that needs
    // no invented constant", and that is the claim T-04-70 refutes. `setfit_tag.rs`'s
    // own module header names the counter-example: a 30 GB APR whose metadata block
    // declares 20 MiB passes (a) and is then read in full to answer one question. The
    // hostile version of that is cheaper still — on a sparse filesystem `truncate -s
    // 4297M` costs nothing, so (a) can be SATISFIED by a 4 GiB declaration for free
    // and `apr inspect` — the tool CLAUDE.md mandates as diagnostic step 1 — allocates
    // it and then reports success. Under a memory-limited container it is OOM-killed.
    //
    // The cap is `crate::setfit_tag::MAX_TAG_METADATA_BYTES`, REFERENCED and never
    // re-declared, so this reader and the tag detector cannot answer differently about
    // one block. The comparison is `>` in both places, so the boundary agrees too.
    //
    // Over-cap is DISCLOSED, not silently defaulted. Defaulting would render a hostile
    // container as a model with no `model_type` and no APR-05 section — the same
    // output as a plain APR that genuinely has neither — and the fact that the block
    // was REFUSED is the entire diagnosis (WR-08's `inspect` half).
    let declared = u64::from(header.metadata_size);
    let fits = reader
        .get_ref()
        .metadata()
        .is_ok_and(|m| header.metadata_offset.saturating_add(declared) <= m.len());
    if !fits {
        return MetadataInfo::default();
    }
    if declared > crate::setfit_tag::MAX_TAG_METADATA_BYTES {
        return MetadataInfo {
            metadata_over_cap_bytes: Some(declared),
            ..MetadataInfo::default()
        };
    }
    let mut metadata_bytes = vec![0u8; header.metadata_size as usize];
    if reader.read_exact(&mut metadata_bytes).is_err() {
        return MetadataInfo::default();
    }

    // Parse JSON metadata (v2 uses JSON, not msgpack)
    match AprV2Metadata::from_json(&metadata_bytes) {
        Ok(mut meta) => {
            let source_metadata = meta.custom.remove("source_metadata");
            // The TYPED TAG decides, never a tensor name and never the mere presence
            // of the custom key (D-04): an untagged APR that happens to carry a
            // `setfit` key is a plain APR here, exactly as it is to `apr predict`.
            // `remove`, not `get(..).cloned()`: `meta.custom` is owned and is not read again,
            // so the document moves out instead of being deep-copied.
            let setfit_doc = if meta.model_type == crate::setfit_tag::SETFIT_MODEL_TYPE {
                meta.custom.remove(crate::setfit_tag::SETFIT_CUSTOM_KEY)
            } else {
                None
            };

            MetadataInfo {
                setfit_doc,
                // The block was READ, so there is no over-cap fact to disclose.
                metadata_over_cap_bytes: None,
                model_type: if meta.model_type.is_empty() {
                    None
                } else {
                    Some(meta.model_type)
                },
                name: meta.name,
                description: meta.description,
                author: meta.author,
                // C-APR-PROVENANCE: these three are copied unconditionally
                // (even if None) so inspect always emits the keys.
                license: meta.license,
                data_source: meta.data_source,
                data_license: meta.data_license,
                source: meta.source,
                original_format: meta.original_format,
                created_at: meta.created_at,
                // GH-249: Always include architecture (never empty)
                architecture: meta
                    .architecture
                    .filter(|a| !a.is_empty())
                    .or_else(|| Some("unknown".to_string())),
                // PMAT-690 P0-K: copy unconditionally (even if None) so
                // operators can grep `apr inspect --json | jq .hf_architecture`
                // and distinguish "stamped" from "missing" rather than
                // having the key silently skipped.
                hf_architecture: meta.hf_architecture.filter(|a| !a.is_empty()),
                hf_model_type: meta.hf_model_type.filter(|a| !a.is_empty()),
                param_count: if meta.param_count > 0 {
                    Some(meta.param_count)
                } else {
                    None
                },
                vocab_size: meta.vocab_size,
                hidden_size: meta.hidden_size,
                num_layers: meta.num_layers,
                num_heads: meta.num_heads,
                num_kv_heads: meta.num_kv_heads,
                intermediate_size: meta.intermediate_size,
                max_position_embeddings: meta.max_position_embeddings,
                rope_theta: meta.rope_theta,
                chat_template: meta.chat_template,
                chat_format: meta.chat_format,
                special_tokens: meta
                    .special_tokens
                    .and_then(|st| serde_json::to_value(st).ok()),
                source_metadata,
            }
        }
        Err(_) => MetadataInfo::default(),
    }
}

include!("inspect_output_json.rs");
include!("inspect_setfit.rs");
include!("inspect_03.rs");
include!("inspect_tests.rs");
include!("inspect_setfit_tests.rs");
