//! Basic CLI commands for data conversion and inspection.

use std::path::{Path, PathBuf};

use arrow::util::pretty::print_batches;

use crate::{ArrowDataset, Dataset};

/// Weighted dataset inputs: each entry is (dataset, weight, display_name), plus
/// total weight.
#[cfg(feature = "shuffle")]
type MixInputs = (Vec<(ArrowDataset, f64, String)>, f64);

/// Load a dataset from a file path based on extension.
pub(crate) fn load_dataset(path: &Path) -> crate::Result<ArrowDataset> {
    let ext = path.extension().and_then(|e| e.to_str()).unwrap_or("");

    match ext {
        "parquet" => ArrowDataset::from_parquet(path),
        "csv" => ArrowDataset::from_csv(path),
        "json" | "jsonl" => ArrowDataset::from_json(path),
        ext => Err(crate::Error::unsupported_format(ext)),
    }
}

/// Save a dataset to a file path based on extension.
pub(crate) fn save_dataset(dataset: &ArrowDataset, path: &Path) -> crate::Result<()> {
    let ext = path.extension().and_then(|e| e.to_str()).unwrap_or("");

    match ext {
        "parquet" => dataset.to_parquet(path),
        "csv" => dataset.to_csv(path),
        "json" | "jsonl" => dataset.to_json(path),
        ext => Err(crate::Error::unsupported_format(ext)),
    }
}

/// Get format name from file extension.
pub(crate) fn get_format(path: &Path) -> &'static str {
    match path.extension().and_then(|e| e.to_str()) {
        Some("parquet") => "Parquet",
        Some("arrow" | "ipc") => "Arrow IPC",
        Some("csv") => "CSV",
        Some("json" | "jsonl") => "JSON",
        _ => "Unknown",
    }
}

/// Convert between data formats.
pub(crate) fn cmd_convert(input: &Path, output: &Path) -> crate::Result<()> {
    // Load input (supports parquet, csv)
    let dataset = load_dataset(input)?;

    // Save output (supports parquet, csv)
    save_dataset(&dataset, output)?;

    println!(
        "Converted {} -> {} ({} rows)",
        input.display(),
        output.display(),
        dataset.len()
    );

    Ok(())
}

/// Display dataset information.
/// ALB-099: For Parquet, reads only file metadata (footer) — zero row group
/// decoding. dhat profiling showed the old path loaded the entire dataset (69.8
/// MB for 9 MB file).
pub(crate) fn cmd_info(path: &Path) -> crate::Result<()> {
    let file_size = std::fs::metadata(path).map(|m| m.len()).unwrap_or(0);
    let ext = path.extension().and_then(|e| e.to_str()).unwrap_or("");

    println!("File: {}", path.display());
    println!("Format: {}", get_format(path));

    if ext == "parquet" {
        // Read only Parquet footer metadata — no row group decoding
        let file = std::fs::File::open(path).map_err(|e| crate::Error::io(e, path))?;
        let builder = parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder::try_new(file)
            .map_err(crate::Error::Parquet)?;
        let metadata = builder.metadata();
        let num_rows: i64 = metadata.row_groups().iter().map(|rg| rg.num_rows()).sum();
        let num_batches = metadata.num_row_groups();
        let num_columns = metadata
            .row_groups()
            .first()
            .map_or(0, |rg| rg.num_columns());
        println!("Rows: {num_rows}");
        println!("Batches: {num_batches}");
        println!("Columns: {num_columns}");
    } else {
        let dataset = load_dataset(path)?;
        println!("Rows: {}", dataset.len());
        println!("Batches: {}", dataset.num_batches());
        println!("Columns: {}", dataset.schema().fields().len());
    }

    println!("Size: {file_size} bytes");

    Ok(())
}

/// Display first N rows of a dataset.
pub(crate) fn cmd_head(path: &Path, rows: usize) -> crate::Result<()> {
    // ALB-099: For Parquet, use with_limit() to avoid loading the entire dataset.
    // dhat profiling showed `head -n 10` allocated 69.8 MB for a 9 MB/1M-row file.
    let ext = path.extension().and_then(|e| e.to_str()).unwrap_or("");
    let dataset = if ext == "parquet" {
        let file = std::fs::File::open(path).map_err(|e| crate::Error::io(e, path))?;
        let builder = parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder::try_new(file)
            .map_err(crate::Error::Parquet)?;
        let reader = builder
            .with_limit(rows)
            .build()
            .map_err(crate::Error::Parquet)?;
        let batches: Vec<arrow::record_batch::RecordBatch> = reader
            .collect::<std::result::Result<Vec<_>, _>>()
            .map_err(crate::Error::Arrow)?;
        if batches.is_empty() {
            println!("Dataset is empty");
            return Ok(());
        }
        ArrowDataset::new(batches)?
    } else {
        load_dataset(path)?
    };

    if dataset.is_empty() {
        println!("Dataset is empty");
        return Ok(());
    }

    // Collect rows into batches
    let mut collected = Vec::new();
    let mut count = 0;

    for batch in dataset.iter() {
        let take = (rows - count).min(batch.num_rows());
        if take > 0 {
            collected.push(batch.slice(0, take));
            count += take;
        }
        if count >= rows {
            break;
        }
    }

    if collected.is_empty() {
        println!("No data to display");
        return Ok(());
    }

    // Print using Arrow's pretty printer
    print_batches(&collected).map_err(crate::Error::Arrow)?;

    if count < dataset.len() {
        println!("... showing {} of {} rows", count, dataset.len());
    }

    Ok(())
}

/// Display dataset schema.
pub(crate) fn cmd_schema(path: &Path) -> crate::Result<()> {
    let dataset = load_dataset(path)?;
    let schema = dataset.schema();

    println!("Schema for {}:", path.display());
    println!();

    for (i, field) in schema.fields().iter().enumerate() {
        let nullable = if field.is_nullable() {
            "nullable"
        } else {
            "not null"
        };
        println!(
            "  {}: {} ({}) [{}]",
            i,
            field.name(),
            field.data_type(),
            nullable
        );
    }

    println!();
    println!("Total columns: {}", schema.fields().len());

    Ok(())
}

/// Parse an input spec of the form "path" or "path:weight".
#[cfg(feature = "shuffle")]
fn parse_input_spec(spec: &str) -> (PathBuf, f64) {
    if let Some((path, weight_str)) = spec.rsplit_once(':') {
        // Check if the part after : is a valid float (not a Windows drive letter)
        if let Ok(weight) = weight_str.parse::<f64>() {
            return (PathBuf::from(path), weight);
        }
    }
    (PathBuf::from(spec), 1.0)
}

/// Load input datasets from specs.
#[cfg(feature = "shuffle")]
fn load_mix_inputs(inputs: &[String]) -> crate::Result<MixInputs> {
    let mut datasets = Vec::new();
    let mut total_weight = 0.0;

    for spec in inputs {
        let (path, weight) = parse_input_spec(spec);
        if !path.exists() {
            return Err(crate::Error::io(
                std::io::Error::new(std::io::ErrorKind::NotFound, "Input file not found"),
                &path,
            ));
        }
        let dataset = load_dataset(&path)?;
        println!(
            "  Loaded {} ({} rows, weight={:.2})",
            path.display(),
            dataset.len(),
            weight
        );
        total_weight += weight;
        datasets.push((dataset, weight, path.display().to_string()));
    }
    Ok((datasets, total_weight))
}

/// Sample rows from a dataset with optional upsampling.
#[cfg(feature = "shuffle")]
#[allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]
fn sample_dataset(
    dataset: &ArrowDataset,
    rows_needed: usize,
    rng: &mut rand::rngs::StdRng,
) -> crate::Result<arrow::array::RecordBatch> {
    use rand::seq::SliceRandom;

    let available = dataset.len();
    let mut indices: Vec<usize> = (0..available).collect();
    indices.shuffle(rng);

    if rows_needed > available {
        let extra: Vec<usize> = (0..available)
            .cycle()
            .take(rows_needed - available)
            .collect();
        indices.extend(extra);
    }
    indices.truncate(rows_needed);

    let schema = dataset.schema();
    let flat_batches: Vec<_> = dataset.iter().collect();
    let concatenated = arrow::compute::concat_batches(&schema, &flat_batches)
        .map_err(|e| crate::Error::invalid_config(format!("Arrow concat error: {e}")))?;

    let take_indices: Vec<u32> = indices.iter().map(|&i| i as u32).collect();
    let index_array = arrow::array::UInt32Array::from(take_indices);

    let columns: Vec<arrow::array::ArrayRef> = (0..concatenated.num_columns())
        .map(|col_idx| {
            arrow::compute::take(concatenated.column(col_idx), &index_array, None)
                .map_err(|e| crate::Error::invalid_config(format!("Arrow take error: {e}")))
        })
        .collect::<crate::Result<Vec<_>>>()?;

    arrow::array::RecordBatch::try_new(schema, columns)
        .map_err(|e| crate::Error::invalid_config(format!("RecordBatch error: {e}")))
}

/// Mix multiple datasets with weighted sampling.
#[cfg(feature = "shuffle")]
#[allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]
pub(crate) fn cmd_mix(
    inputs: &[String],
    output: &Path,
    seed: u64,
    max_rows: usize,
) -> crate::Result<()> {
    use rand::{rngs::StdRng, SeedableRng};

    if inputs.is_empty() {
        return Err(crate::Error::invalid_config("No input files provided"));
    }

    let (datasets, total_weight) = load_mix_inputs(inputs)?;
    if total_weight == 0.0 {
        return Err(crate::Error::invalid_config("All weights are zero"));
    }

    let total_available: usize = datasets.iter().map(|(d, _, _)| d.len()).sum();
    let target_rows = if max_rows > 0 {
        max_rows
    } else {
        total_available
    };

    println!(
        "\nMixing {} datasets → {} target rows",
        datasets.len(),
        target_rows
    );

    let mut rng = StdRng::seed_from_u64(seed);
    let mut all_batches = Vec::new();
    let mut mixed_rows = 0;

    for (dataset, weight, name) in &datasets {
        let fraction = weight / total_weight;
        let rows_for_dataset = (target_rows as f64 * fraction) as usize;

        let batch = sample_dataset(dataset, rows_for_dataset, &mut rng)?;
        let count = batch.num_rows();
        all_batches.push(batch);
        mixed_rows += count;

        println!("  {} → {} rows ({:.1}%)", name, count, fraction * 100.0);
    }

    if all_batches.is_empty() {
        return Err(crate::Error::invalid_config("No data to mix"));
    }

    let mixed = ArrowDataset::new(all_batches)?;
    save_dataset(&mixed, output)?;

    println!("\nMixed {} rows → {}", mixed_rows, output.display());
    Ok(())
}

#[cfg(feature = "shuffle")]
pub(crate) fn cmd_fim(
    input: &Path,
    output: &Path,
    column: &str,
    rate: f64,
    format: &str,
    seed: u64,
) -> crate::Result<()> {
    use crate::transform::{Fim, FimFormat, Transform};

    let dataset = load_dataset(input)?;
    let fim_format = match format {
        "spm" => FimFormat::SPM,
        _ => FimFormat::PSM,
    };

    let fim = Fim::new(column)
        .with_rate(rate)
        .with_format(fim_format)
        .with_seed(seed);

    let mut all_batches = Vec::new();
    for batch in dataset.iter() {
        all_batches.push(fim.apply(batch)?);
    }

    let transformed = ArrowDataset::new(all_batches)?;
    save_dataset(&transformed, output)?;

    println!(
        "FIM transform ({} format, {:.0}% rate) applied to '{}' column",
        format.to_uppercase(),
        rate * 100.0,
        column
    );
    println!("{} rows → {}", dataset.len(), output.display());
    Ok(())
}

/// R-019: Deduplicate dataset rows by text content hash.
///
/// Uses SHA-256 content hashing for exact deduplication on the specified
/// text column. Falls back to full-row deduplication if no column specified.
pub(crate) fn cmd_dedup(input: &Path, output: &Path, column: Option<&str>) -> crate::Result<()> {
    use crate::transform::{Transform, Unique};

    let dataset = load_dataset(input)?;
    let original_rows = dataset.len();

    // Use content-hash dedup on text column, or full-row dedup
    let dedup = match column {
        Some(col) => Unique::by(vec![col]),
        None => detect_text_column_dedup(&dataset),
    };

    let mut all_batches = Vec::new();
    for batch in dataset.iter() {
        all_batches.push(dedup.apply(batch)?);
    }

    let deduped = ArrowDataset::new(all_batches)?;
    let deduped_rows = deduped.len();
    save_dataset(&deduped, output)?;

    let removed = original_rows - deduped_rows;
    println!(
        "Dedup: {} → {} rows ({} duplicates removed, {:.1}% reduction)",
        original_rows,
        deduped_rows,
        removed,
        removed as f64 / original_rows.max(1) as f64 * 100.0
    );
    Ok(())
}

/// Auto-detect text column and create Unique transform for it.
fn detect_text_column_dedup(dataset: &ArrowDataset) -> crate::transform::Unique {
    use arrow::datatypes::DataType;

    use crate::transform::Unique;

    let schema = dataset.schema();
    for name in &["text", "content", "code", "source"] {
        if let Some((_, field)) = schema.column_with_name(name) {
            if matches!(field.data_type(), DataType::Utf8 | DataType::LargeUtf8) {
                return Unique::by(vec![*name]);
            }
        }
    }
    // Fallback: dedup on all columns
    Unique::all()
}

/// R-022: Filter dataset rows by text quality signals.
///
/// Computes quality scores for each row and removes low-quality entries.
/// Signals: line length, alphanumeric ratio, duplicate line ratio, entropy.
pub(crate) fn cmd_filter_text(
    input: &Path,
    output: &Path,
    column: Option<&str>,
    min_score: f64,
    min_length: usize,
    max_length: usize,
) -> crate::Result<()> {
    use crate::transform::Transform;

    let dataset = load_dataset(input)?;
    let original_rows = dataset.len();

    let col_name = column
        .map(String::from)
        .unwrap_or_else(|| find_text_column(&dataset));

    let filter = TextQualityFilter::new(&col_name, min_score, min_length, max_length);

    let mut all_batches = Vec::new();
    for batch in dataset.iter() {
        all_batches.push(filter.apply(batch)?);
    }

    let filtered = ArrowDataset::new(all_batches)?;
    let kept = filtered.len();
    save_dataset(&filtered, output)?;

    let removed = original_rows - kept;
    println!(
        "Filter: {} → {} rows ({} removed, {:.1}% kept)",
        original_rows,
        kept,
        removed,
        kept as f64 / original_rows.max(1) as f64 * 100.0
    );
    println!(
        "  min_score={:.2} min_len={} max_len={} column='{}'",
        min_score, min_length, max_length, col_name
    );
    Ok(())
}

/// Find the first text column in a dataset.
fn find_text_column(dataset: &ArrowDataset) -> String {
    use arrow::datatypes::DataType;
    let schema = dataset.schema();
    for name in &["text", "content", "code", "source"] {
        if let Some((_, field)) = schema.column_with_name(name) {
            if matches!(field.data_type(), DataType::Utf8 | DataType::LargeUtf8) {
                return (*name).to_string();
            }
        }
    }
    // Fallback: first Utf8 column
    for field in schema.fields() {
        if matches!(field.data_type(), DataType::Utf8 | DataType::LargeUtf8) {
            return field.name().clone();
        }
    }
    "text".to_string()
}

/// Text quality filter transform.
struct TextQualityFilter {
    column: String,
    min_score: f64,
    min_length: usize,
    max_length: usize,
}

impl TextQualityFilter {
    fn new(column: &str, min_score: f64, min_length: usize, max_length: usize) -> Self {
        Self {
            column: column.to_string(),
            min_score,
            min_length,
            max_length,
        }
    }
}

impl crate::transform::Transform for TextQualityFilter {
    fn apply(&self, batch: arrow::array::RecordBatch) -> crate::Result<arrow::array::RecordBatch> {
        use arrow::{
            array::{Array, BooleanArray, StringArray},
            compute::filter_record_batch,
        };

        let schema = batch.schema();
        let col_idx = schema
            .column_with_name(&self.column)
            .map(|(i, _)| i)
            .ok_or_else(|| crate::Error::column_not_found(&self.column))?;

        let text_arr = batch
            .column(col_idx)
            .as_any()
            .downcast_ref::<StringArray>()
            .ok_or_else(|| crate::Error::column_not_found(&self.column))?;

        let mask: BooleanArray = (0..text_arr.len())
            .map(|i| {
                if text_arr.is_null(i) {
                    Some(false)
                } else {
                    let text = text_arr.value(i);
                    Some(passes_quality(
                        text,
                        self.min_score,
                        self.min_length,
                        self.max_length,
                    ))
                }
            })
            .collect();

        filter_record_batch(&batch, &mask).map_err(crate::Error::Arrow)
    }
}

/// Check if a text document passes quality thresholds.
fn passes_quality(text: &str, min_score: f64, min_len: usize, max_len: usize) -> bool {
    let len = text.len();
    if len < min_len || len > max_len {
        return false;
    }
    composite_score(text) >= min_score
}

/// Compute composite quality score (0.0-1.0) for a text document.
fn composite_score(text: &str) -> f64 {
    let s1 = score_alnum_ratio(text);
    let s2 = score_line_length(text);
    let s3 = score_dup_lines(text);
    let s4 = score_entropy(text);
    (s1 + s2 + s3 + s4) / 4.0
}

/// Alphanumeric character ratio. Below 0.3 = likely binary/garbage.
fn score_alnum_ratio(text: &str) -> f64 {
    if text.is_empty() {
        return 0.0;
    }
    let alnum = text.chars().filter(|c| c.is_alphanumeric()).count();
    let ratio = alnum as f64 / text.len() as f64;
    if ratio < 0.2 {
        0.0
    } else if ratio < 0.3 {
        ratio
    } else {
        1.0
    }
}

/// Average line length score. Ideal 30-80 chars.
fn score_line_length(text: &str) -> f64 {
    let lines: Vec<&str> = text.lines().collect();
    if lines.is_empty() {
        return 0.0;
    }
    let avg = text.len() as f64 / lines.len() as f64;
    if avg < 10.0 {
        0.2
    } else if avg > 200.0 {
        0.5
    } else {
        1.0
    }
}

/// Duplicate line ratio. High = boilerplate.
fn score_dup_lines(text: &str) -> f64 {
    use std::collections::HashSet;
    let lines: Vec<&str> = text.lines().collect();
    if lines.len() <= 1 {
        return 1.0;
    }
    let unique: HashSet<&str> = lines.iter().copied().collect();
    let dup_ratio = 1.0 - (unique.len() as f64 / lines.len() as f64);
    if dup_ratio > 0.5 {
        0.2
    } else {
        1.0 - dup_ratio
    }
}

/// Character-level Shannon entropy. Low = repetitive, high = random/binary.
fn score_entropy(text: &str) -> f64 {
    if text.is_empty() {
        return 0.0;
    }
    let mut counts = [0u32; 256];
    for &b in text.as_bytes() {
        counts[b as usize] += 1;
    }
    let len = text.len() as f64;
    let entropy: f64 = counts
        .iter()
        .filter(|&&c| c > 0)
        .map(|&c| {
            let p = f64::from(c) / len;
            -p * p.ln()
        })
        .sum();
    let e = entropy / std::f64::consts::LN_2; // bits
    if e < 2.0 {
        0.2
    } else if e > 6.5 {
        0.3
    } else {
        1.0
    }
}

#[cfg(test)]
#[allow(
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::cast_precision_loss,
    clippy::uninlined_format_args,
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::redundant_clone,
    clippy::cast_lossless,
    clippy::redundant_closure_for_method_calls,
    clippy::too_many_lines,
    clippy::float_cmp,
    clippy::similar_names,
    clippy::needless_late_init,
    clippy::redundant_pattern_matching
)]
mod tests {
    use std::sync::Arc;

    use arrow::{
        array::{Int32Array, StringArray},
        datatypes::{DataType, Field, Schema},
    };

    use super::*;

    fn create_test_parquet(path: &Path, rows: usize) {
        let schema = Arc::new(Schema::new(vec![
            Field::new("id", DataType::Int32, false),
            Field::new("name", DataType::Utf8, false),
        ]));

        let ids: Vec<i32> = (0..rows as i32).collect();
        let names: Vec<String> = ids.iter().map(|i| format!("item_{}", i)).collect();

        let batch = arrow::array::RecordBatch::try_new(
            schema,
            vec![
                Arc::new(Int32Array::from(ids)),
                Arc::new(StringArray::from(names)),
            ],
        )
        .ok()
        .unwrap_or_else(|| panic!("Should create batch"));

        let dataset = ArrowDataset::from_batch(batch)
            .ok()
            .unwrap_or_else(|| panic!("Should create dataset"));

        dataset
            .to_parquet(path)
            .ok()
            .unwrap_or_else(|| panic!("Should write parquet"));
    }

    #[test]
    fn test_cmd_info() {
        let temp_dir = tempfile::tempdir()
            .ok()
            .unwrap_or_else(|| panic!("Should create temp dir"));
        let path = temp_dir.path().join("test.parquet");
        create_test_parquet(&path, 100);

        let result = cmd_info(&path);
        assert!(result.is_ok());
    }

    #[test]
    fn test_cmd_head() {
        let temp_dir = tempfile::tempdir()
            .ok()
            .unwrap_or_else(|| panic!("Should create temp dir"));
        let path = temp_dir.path().join("test.parquet");
        create_test_parquet(&path, 100);

        let result = cmd_head(&path, 5);
        assert!(result.is_ok());
    }

    #[test]
    fn test_cmd_schema() {
        let temp_dir = tempfile::tempdir()
            .ok()
            .unwrap_or_else(|| panic!("Should create temp dir"));
        let path = temp_dir.path().join("test.parquet");
        create_test_parquet(&path, 10);

        let result = cmd_schema(&path);
        assert!(result.is_ok());
    }

    #[test]
    fn test_cmd_convert() {
        let temp_dir = tempfile::tempdir()
            .ok()
            .unwrap_or_else(|| panic!("Should create temp dir"));
        let input = temp_dir.path().join("input.parquet");
        let output = temp_dir.path().join("output.parquet");
        create_test_parquet(&input, 50);

        let result = cmd_convert(&input, &output);
        assert!(result.is_ok());

        // Verify output was created and has same data
        let original = ArrowDataset::from_parquet(&input)
            .ok()
            .unwrap_or_else(|| panic!("Should load original"));
        let converted = ArrowDataset::from_parquet(&output)
            .ok()
            .unwrap_or_else(|| panic!("Should load converted"));

        assert_eq!(original.len(), converted.len());
    }

    #[test]
    fn test_load_dataset_unsupported() {
        let path = PathBuf::from("test.xyz");
        let result = load_dataset(&path);
        assert!(result.is_err());
    }

    #[test]
    fn test_get_format() {
        assert_eq!(get_format(Path::new("test.parquet")), "Parquet");
        assert_eq!(get_format(Path::new("test.arrow")), "Arrow IPC");
        assert_eq!(get_format(Path::new("test.csv")), "CSV");
        assert_eq!(get_format(Path::new("test.json")), "JSON");
        assert_eq!(get_format(Path::new("test.unknown")), "Unknown");
    }

    #[test]
    fn test_cmd_head_with_more_rows_than_dataset() {
        let temp_dir = tempfile::tempdir()
            .ok()
            .unwrap_or_else(|| panic!("Should create temp dir"));
        let path = temp_dir.path().join("test.parquet");
        create_test_parquet(&path, 5);

        // Request more rows than exist
        let result = cmd_head(&path, 100);
        assert!(result.is_ok());
    }

    #[test]
    fn test_cmd_convert_parquet_to_csv() {
        let temp_dir = tempfile::tempdir()
            .ok()
            .unwrap_or_else(|| panic!("Should create temp dir"));
        let input = temp_dir.path().join("input.parquet");
        let output = temp_dir.path().join("output.csv");
        create_test_parquet(&input, 25);

        let result = cmd_convert(&input, &output);
        assert!(result.is_ok());
        assert!(output.exists());
    }

    #[test]
    fn test_cmd_convert_parquet_to_json() {
        let temp_dir = tempfile::tempdir()
            .ok()
            .unwrap_or_else(|| panic!("Should create temp dir"));
        let input = temp_dir.path().join("input.parquet");
        let output = temp_dir.path().join("output.json");
        create_test_parquet(&input, 15);

        let result = cmd_convert(&input, &output);
        assert!(result.is_ok());
        assert!(output.exists());
    }

    #[test]
    fn test_save_dataset_unsupported_format() {
        let temp_dir = tempfile::tempdir()
            .ok()
            .unwrap_or_else(|| panic!("Should create temp dir"));
        let input = temp_dir.path().join("data.parquet");
        let output = temp_dir.path().join("output.xyz");
        create_test_parquet(&input, 5);

        let dataset = ArrowDataset::from_parquet(&input)
            .ok()
            .unwrap_or_else(|| panic!("Should load"));

        let result = save_dataset(&dataset, &output);
        assert!(result.is_err());
    }

    #[test]
    fn test_get_format_ipc() {
        assert_eq!(get_format(Path::new("test.ipc")), "Arrow IPC");
    }

    #[test]
    fn test_get_format_jsonl() {
        assert_eq!(get_format(Path::new("test.jsonl")), "JSON");
    }

    #[test]
    fn test_get_format_no_extension() {
        assert_eq!(get_format(Path::new("testfile")), "Unknown");
    }

    #[test]
    fn test_cmd_convert_unsupported_output() {
        let temp_dir = tempfile::tempdir()
            .ok()
            .unwrap_or_else(|| panic!("Should create temp dir"));
        let input = temp_dir.path().join("input.parquet");
        let output = temp_dir.path().join("output.xyz");
        create_test_parquet(&input, 10);

        let result = cmd_convert(&input, &output);
        assert!(result.is_err());
    }

    #[test]
    fn test_load_dataset_xyz_format() {
        let temp_dir = tempfile::tempdir()
            .ok()
            .unwrap_or_else(|| panic!("Should create temp dir"));
        let path = temp_dir.path().join("data.xyz");

        std::fs::write(&path, "some data")
            .ok()
            .unwrap_or_else(|| panic!("Should write file"));

        let result = load_dataset(&path);
        assert!(result.is_err());
    }

    #[test]
    fn test_get_format_arrow() {
        assert_eq!(get_format(Path::new("test.arrow")), "Arrow IPC");
    }

    #[test]
    fn test_get_format_unknown() {
        assert_eq!(get_format(Path::new("test.feather")), "Unknown");
        assert_eq!(get_format(Path::new("test.txt")), "Unknown");
    }

    #[test]
    fn test_load_dataset_csv() {
        let temp_dir = tempfile::tempdir()
            .ok()
            .unwrap_or_else(|| panic!("Should create temp dir"));
        let parquet_path = temp_dir.path().join("data.parquet");
        let csv_path = temp_dir.path().join("data.csv");

        create_test_parquet(&parquet_path, 10);

        // Convert to CSV first
        let dataset = ArrowDataset::from_parquet(&parquet_path)
            .ok()
            .unwrap_or_else(|| panic!("Should load"));
        dataset
            .to_csv(&csv_path)
            .ok()
            .unwrap_or_else(|| panic!("Should write csv"));

        // Load from CSV
        let loaded = load_dataset(&csv_path);
        assert!(loaded.is_ok());
    }

    #[test]
    fn test_load_dataset_json() {
        let temp_dir = tempfile::tempdir()
            .ok()
            .unwrap_or_else(|| panic!("Should create temp dir"));
        let parquet_path = temp_dir.path().join("data.parquet");
        let json_path = temp_dir.path().join("data.json");

        create_test_parquet(&parquet_path, 10);

        // Convert to JSON first
        let dataset = ArrowDataset::from_parquet(&parquet_path)
            .ok()
            .unwrap_or_else(|| panic!("Should load"));
        dataset
            .to_json(&json_path)
            .ok()
            .unwrap_or_else(|| panic!("Should write json"));

        // Load from JSON
        let loaded = load_dataset(&json_path);
        assert!(loaded.is_ok());
    }

    #[test]
    fn test_load_dataset_jsonl() {
        let temp_dir = tempfile::tempdir()
            .ok()
            .unwrap_or_else(|| panic!("Should create temp dir"));
        let parquet_path = temp_dir.path().join("data.parquet");
        let jsonl_path = temp_dir.path().join("data.jsonl");

        create_test_parquet(&parquet_path, 10);

        // Convert to JSON first (jsonl is same format)
        let dataset = ArrowDataset::from_parquet(&parquet_path)
            .ok()
            .unwrap_or_else(|| panic!("Should load"));
        dataset
            .to_json(&jsonl_path)
            .ok()
            .unwrap_or_else(|| panic!("Should write jsonl"));

        // Load from JSONL
        let loaded = load_dataset(&jsonl_path);
        assert!(loaded.is_ok());
    }

    #[test]
    fn test_save_dataset_parquet() {
        let temp_dir = tempfile::tempdir()
            .ok()
            .unwrap_or_else(|| panic!("Should create temp dir"));
        let path = temp_dir.path().join("output.parquet");

        let schema = Arc::new(Schema::new(vec![Field::new("id", DataType::Int32, false)]));
        let batch = arrow::array::RecordBatch::try_new(
            schema,
            vec![Arc::new(Int32Array::from(vec![1, 2, 3]))],
        )
        .unwrap();
        let dataset = ArrowDataset::from_batch(batch).unwrap();

        let result = save_dataset(&dataset, &path);
        assert!(result.is_ok());
    }

    #[test]
    fn test_save_dataset_csv() {
        let temp_dir = tempfile::tempdir()
            .ok()
            .unwrap_or_else(|| panic!("Should create temp dir"));
        let path = temp_dir.path().join("output.csv");

        let schema = Arc::new(Schema::new(vec![Field::new("id", DataType::Int32, false)]));
        let batch = arrow::array::RecordBatch::try_new(
            schema,
            vec![Arc::new(Int32Array::from(vec![1, 2, 3]))],
        )
        .unwrap();
        let dataset = ArrowDataset::from_batch(batch).unwrap();

        let result = save_dataset(&dataset, &path);
        assert!(result.is_ok());
    }

    #[test]
    fn test_save_dataset_json() {
        let temp_dir = tempfile::tempdir()
            .ok()
            .unwrap_or_else(|| panic!("Should create temp dir"));
        let path = temp_dir.path().join("output.json");

        let schema = Arc::new(Schema::new(vec![Field::new("id", DataType::Int32, false)]));
        let batch = arrow::array::RecordBatch::try_new(
            schema,
            vec![Arc::new(Int32Array::from(vec![1, 2, 3]))],
        )
        .unwrap();
        let dataset = ArrowDataset::from_batch(batch).unwrap();

        let result = save_dataset(&dataset, &path);
        assert!(result.is_ok());
    }

    #[test]
    fn test_save_dataset_unknown_extension() {
        let temp_dir = tempfile::tempdir()
            .ok()
            .unwrap_or_else(|| panic!("Should create temp dir"));
        let path = temp_dir.path().join("output.xyz");

        let schema = Arc::new(Schema::new(vec![Field::new("id", DataType::Int32, false)]));
        let batch = arrow::array::RecordBatch::try_new(
            schema,
            vec![Arc::new(Int32Array::from(vec![1, 2, 3]))],
        )
        .unwrap();
        let dataset = ArrowDataset::from_batch(batch).unwrap();

        let result = save_dataset(&dataset, &path);
        assert!(result.is_err());
    }

    #[test]
    fn test_cmd_convert_to_csv_format() {
        let temp_dir = tempfile::tempdir()
            .ok()
            .unwrap_or_else(|| panic!("Should create temp dir"));
        let input = temp_dir.path().join("input.parquet");
        let output = temp_dir.path().join("output.csv");
        create_test_parquet(&input, 20);

        let result = cmd_convert(&input, &output);
        assert!(result.is_ok());
        assert!(output.exists());
    }

    #[test]
    fn test_cmd_convert_to_json_format() {
        let temp_dir = tempfile::tempdir()
            .ok()
            .unwrap_or_else(|| panic!("Should create temp dir"));
        let input = temp_dir.path().join("input.parquet");
        let output = temp_dir.path().join("output.json");
        create_test_parquet(&input, 20);

        let result = cmd_convert(&input, &output);
        assert!(result.is_ok());
        assert!(output.exists());
    }

    #[test]
    fn test_cmd_head_more_than_available() {
        let temp_dir = tempfile::tempdir()
            .ok()
            .unwrap_or_else(|| panic!("Should create temp dir"));
        let path = temp_dir.path().join("small.parquet");
        create_test_parquet(&path, 5);

        // Request more rows than available
        let result = cmd_head(&path, 100);
        assert!(result.is_ok());
    }

    #[test]
    fn test_load_dataset_csv_file() {
        let temp_dir = tempfile::tempdir()
            .ok()
            .unwrap_or_else(|| panic!("Should create temp dir"));
        let csv_path = temp_dir.path().join("test.csv");

        // Create a simple CSV
        std::fs::write(&csv_path, "id,name\n1,foo\n2,bar\n").unwrap();

        let result = load_dataset(&csv_path);
        assert!(result.is_ok());
    }

    #[test]
    fn test_load_dataset_json_file() {
        let temp_dir = tempfile::tempdir()
            .ok()
            .unwrap_or_else(|| panic!("Should create temp dir"));
        let json_path = temp_dir.path().join("test.json");

        // Create a simple JSON Lines file
        std::fs::write(
            &json_path,
            r#"{"id":1,"name":"foo"}
{"id":2,"name":"bar"}"#,
        )
        .unwrap();

        let result = load_dataset(&json_path);
        assert!(result.is_ok());
    }

    // === Additional CLI basic tests ===

    #[test]
    fn test_cmd_head_zero_rows() {
        let temp_dir = tempfile::tempdir()
            .ok()
            .unwrap_or_else(|| panic!("Should create temp dir"));
        let path = temp_dir.path().join("test.parquet");
        create_test_parquet(&path, 50);

        // Request 0 rows - should still work
        let result = cmd_head(&path, 0);
        assert!(result.is_ok());
    }

    #[test]
    fn test_cmd_info_small_file() {
        let temp_dir = tempfile::tempdir()
            .ok()
            .unwrap_or_else(|| panic!("Should create temp dir"));
        let path = temp_dir.path().join("small.parquet");
        create_test_parquet(&path, 5);

        let result = cmd_info(&path);
        assert!(result.is_ok());
    }

    #[test]
    fn test_cmd_info_large_file() {
        let temp_dir = tempfile::tempdir()
            .ok()
            .unwrap_or_else(|| panic!("Should create temp dir"));
        let path = temp_dir.path().join("large.parquet");
        create_test_parquet(&path, 1000);

        let result = cmd_info(&path);
        assert!(result.is_ok());
    }

    #[test]
    fn test_cmd_schema_complex() {
        let temp_dir = tempfile::tempdir()
            .ok()
            .unwrap_or_else(|| panic!("Should create temp dir"));
        let path = temp_dir.path().join("complex.parquet");

        // Create with more columns
        let schema = Arc::new(Schema::new(vec![
            Field::new("id", DataType::Int32, false),
            Field::new("name", DataType::Utf8, true),
            Field::new("value", DataType::Float64, true),
        ]));

        let batch = arrow::array::RecordBatch::try_new(
            schema,
            vec![
                Arc::new(Int32Array::from(vec![1, 2, 3])),
                Arc::new(StringArray::from(vec!["a", "b", "c"])),
                Arc::new(arrow::array::Float64Array::from(vec![1.0, 2.0, 3.0])),
            ],
        )
        .unwrap();

        let dataset = ArrowDataset::from_batch(batch).unwrap();
        dataset.to_parquet(&path).unwrap();

        let result = cmd_schema(&path);
        assert!(result.is_ok());
    }

    #[test]
    fn test_cmd_convert_csv_to_parquet() {
        let temp_dir = tempfile::tempdir()
            .ok()
            .unwrap_or_else(|| panic!("Should create temp dir"));
        let csv_path = temp_dir.path().join("input.csv");
        let parquet_path = temp_dir.path().join("output.parquet");

        std::fs::write(&csv_path, "id,name\n1,foo\n2,bar\n").unwrap();

        let result = cmd_convert(&csv_path, &parquet_path);
        assert!(result.is_ok());
        assert!(parquet_path.exists());
    }

    #[test]
    fn test_cmd_convert_json_to_parquet() {
        let temp_dir = tempfile::tempdir()
            .ok()
            .unwrap_or_else(|| panic!("Should create temp dir"));
        let json_path = temp_dir.path().join("input.json");
        let parquet_path = temp_dir.path().join("output.parquet");

        std::fs::write(
            &json_path,
            r#"{"id":1,"name":"foo"}
{"id":2,"name":"bar"}"#,
        )
        .unwrap();

        let result = cmd_convert(&json_path, &parquet_path);
        assert!(result.is_ok());
        assert!(parquet_path.exists());
    }

    #[test]
    fn test_save_dataset_jsonl() {
        let temp_dir = tempfile::tempdir()
            .ok()
            .unwrap_or_else(|| panic!("Should create temp dir"));
        let path = temp_dir.path().join("output.jsonl");

        let schema = Arc::new(Schema::new(vec![Field::new("id", DataType::Int32, false)]));
        let batch = arrow::array::RecordBatch::try_new(
            schema,
            vec![Arc::new(Int32Array::from(vec![1, 2, 3]))],
        )
        .unwrap();
        let dataset = ArrowDataset::from_batch(batch).unwrap();

        let result = save_dataset(&dataset, &path);
        assert!(result.is_ok());
    }

    #[test]
    fn test_load_dataset_no_extension() {
        let path = PathBuf::from("file_without_extension");
        let result = load_dataset(&path);
        assert!(result.is_err());
    }

    #[test]
    fn test_cmd_head_exact_rows() {
        let temp_dir = tempfile::tempdir()
            .ok()
            .unwrap_or_else(|| panic!("Should create temp dir"));
        let path = temp_dir.path().join("exact.parquet");
        create_test_parquet(&path, 10);

        // Request exact number of rows
        let result = cmd_head(&path, 10);
        assert!(result.is_ok());
    }

    #[test]
    fn test_cmd_convert_parquet_to_parquet() {
        let temp_dir = tempfile::tempdir()
            .ok()
            .unwrap_or_else(|| panic!("Should create temp dir"));
        let input = temp_dir.path().join("input.parquet");
        let output = temp_dir.path().join("output.parquet");
        create_test_parquet(&input, 20);

        let result = cmd_convert(&input, &output);
        assert!(result.is_ok());

        // Both should have same data
        let original = ArrowDataset::from_parquet(&input).unwrap();
        let converted = ArrowDataset::from_parquet(&output).unwrap();
        assert_eq!(original.len(), converted.len());
    }

    #[test]
    fn test_get_format_all_types() {
        assert_eq!(get_format(Path::new("data.parquet")), "Parquet");
        assert_eq!(get_format(Path::new("data.arrow")), "Arrow IPC");
        assert_eq!(get_format(Path::new("data.ipc")), "Arrow IPC");
        assert_eq!(get_format(Path::new("data.csv")), "CSV");
        assert_eq!(get_format(Path::new("data.json")), "JSON");
        assert_eq!(get_format(Path::new("data.jsonl")), "JSON");
        assert_eq!(get_format(Path::new("data.txt")), "Unknown");
        assert_eq!(get_format(Path::new("data.yaml")), "Unknown");
        assert_eq!(get_format(Path::new("data")), "Unknown");
    }
}
