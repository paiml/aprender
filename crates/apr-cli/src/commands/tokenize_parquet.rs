//! Parquet input adapter for `apr tokenize encode-corpus`.
//!
//! Issue #1410 follow-up. The Stack v1.2 (`bigcode/the-stack-dedup`) and
//! codeparrot Python corpora ship as parquet, not JSONL. The original
//! `encode-corpus` only accepted JSONL — this module fills the gap by
//! reading the configured `--content-field` column from each parquet shard
//! and yielding `String` values that drop into the existing tokenize loop
//! exactly the way JSONL line text does.
//!
//! Per `feedback_stack_tool_extension_not_cli_shim.md`: `apr` is canonical;
//! shelling out to `uv run --with pyarrow` to convert parquet → JSONL would
//! be muda. This is the in-tree apr extension.
//!
//! ## Memory model
//!
//! Parquet shards are streamed one row group at a time. The Stack v1.2
//! Python shards are ~200 MB each; loading all rows into RAM up-front
//! would peak at ~30 GB across 144 shards. The streaming `RowGroupIter`
//! amortizes that to ~one row group's worth of memory at any moment
//! (configurable, default 128k rows per group).
//!
//! Contract: [`contracts/apr-cli-tokenize-encode-corpus-parquet-v1.yaml`]
//! (PROPOSED). Falsifiers FALSIFY-PARQ-INPUT-001..004 are bound at
//! PARTIAL_ALGORITHM_LEVEL via the unit tests below.

use std::fs::File;
use std::path::{Path, PathBuf};

use crate::error::{CliError, Result};

/// Recognise parquet files by their `.parquet` extension. Pure function
/// so callers can use it for `collect_corpus_files` dispatch without
/// touching the filesystem.
#[must_use]
pub fn is_parquet(path: &Path) -> bool {
    path.extension()
        .and_then(|e| e.to_str())
        .map(|s| s.eq_ignore_ascii_case("parquet"))
        .unwrap_or(false)
}

/// Collect parquet files under `path`. Mirrors the JSONL `collect_jsonl_files`
/// behavior:
///
/// - if `path` is a regular `.parquet` file, returns `[path]`
/// - if `path` is a directory, returns sorted list of `*.parquet` immediate
///   children (does NOT recurse — matches the JSONL collector's flat
///   directory model)
/// - returns an error if `path` is a regular file but not `.parquet`
///
/// # Errors
///
/// - [`CliError::ValidationFailed`] if `path` cannot be stat'd, or is a
///   non-parquet regular file, or directory iteration fails.
pub fn collect_parquet_files(path: &Path) -> Result<Vec<PathBuf>> {
    let meta = std::fs::metadata(path)
        .map_err(|e| CliError::ValidationFailed(format!("Cannot stat {}: {e}", path.display())))?;
    if meta.is_file() {
        if is_parquet(path) {
            return Ok(vec![path.to_path_buf()]);
        }
        return Err(CliError::ValidationFailed(format!(
            "Corpus file {} is not a .parquet file",
            path.display()
        )));
    }
    let mut out = Vec::new();
    let entries = std::fs::read_dir(path).map_err(|e| {
        CliError::ValidationFailed(format!("Cannot read directory {}: {e}", path.display()))
    })?;
    for entry in entries {
        let entry =
            entry.map_err(|e| CliError::ValidationFailed(format!("Directory entry error: {e}")))?;
        let p = entry.path();
        if p.is_file() && is_parquet(&p) {
            out.push(p);
        }
    }
    out.sort();
    Ok(out)
}

/// Stream the `content_field` column from a single parquet shard, yielding
/// each row's value as an owned `String`. Rows whose value is null or whose
/// type is not Utf8/LargeUtf8 are skipped (matching JSONL's "skip lines
/// without the content field" behavior).
///
/// Reads one row group at a time so peak memory stays bounded by the row
/// group size (Stack v1.2 row groups are ~5MB each ≈ a few thousand
/// content strings).
///
/// # Errors
///
/// - [`CliError::ValidationFailed`] if the file cannot be opened, parsed
///   as parquet, or does not contain a column named `content_field`.
pub fn iter_parquet_content(
    path: &Path,
    content_field: &str,
) -> Result<impl Iterator<Item = Result<String>>> {
    use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;

    let file = File::open(path).map_err(|e| {
        CliError::ValidationFailed(format!("Cannot open parquet {}: {e}", path.display()))
    })?;
    let builder = ParquetRecordBatchReaderBuilder::try_new(file).map_err(|e| {
        CliError::ValidationFailed(format!(
            "Cannot read parquet {} as Arrow: {e}",
            path.display()
        ))
    })?;
    let schema = builder.schema().clone();
    let col_idx = schema.index_of(content_field).map_err(|_| {
        CliError::ValidationFailed(format!(
            "parquet {} has no column named {content_field:?} (available: {:?})",
            path.display(),
            schema.fields().iter().map(|f| f.name()).collect::<Vec<_>>()
        ))
    })?;
    let path_for_err = path.to_path_buf();
    let reader = builder.build().map_err(|e| {
        CliError::ValidationFailed(format!(
            "Cannot build parquet reader for {}: {e}",
            path.display()
        ))
    })?;
    Ok(parquet_string_iter(reader, col_idx, path_for_err))
}

/// Inner iterator: for each Arrow `RecordBatch` produced by the reader,
/// yield each row's content cell as a `String`. Pulled out into its own
/// function so callers don't need to name the impl trait chain.
fn parquet_string_iter(
    reader: parquet::arrow::arrow_reader::ParquetRecordBatchReader,
    col_idx: usize,
    path: PathBuf,
) -> impl Iterator<Item = Result<String>> {
    use arrow_array::{Array, LargeStringArray, StringArray};

    reader.flat_map(
        move |batch_result| -> Box<dyn Iterator<Item = Result<String>>> {
            let path = path.clone();
            match batch_result {
                Err(e) => Box::new(std::iter::once(Err(CliError::ValidationFailed(format!(
                    "parquet read error in {}: {e}",
                    path.display()
                ))))),
                Ok(batch) => {
                    let col = batch.column(col_idx);
                    // Try Utf8 first (most common), then LargeUtf8.
                    if let Some(arr) = col.as_any().downcast_ref::<StringArray>() {
                        let strings: Vec<Result<String>> = (0..arr.len())
                            .filter_map(|i| {
                                if arr.is_null(i) {
                                    None
                                } else {
                                    Some(Ok(arr.value(i).to_string()))
                                }
                            })
                            .collect();
                        Box::new(strings.into_iter())
                    } else if let Some(arr) = col.as_any().downcast_ref::<LargeStringArray>() {
                        let strings: Vec<Result<String>> = (0..arr.len())
                            .filter_map(|i| {
                                if arr.is_null(i) {
                                    None
                                } else {
                                    Some(Ok(arr.value(i).to_string()))
                                }
                            })
                            .collect();
                        Box::new(strings.into_iter())
                    } else {
                        Box::new(std::iter::once(Err(CliError::ValidationFailed(format!(
                            "parquet {} content column is not Utf8/LargeUtf8 (got: {:?})",
                            path.display(),
                            col.data_type()
                        )))))
                    }
                }
            }
        },
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Provenance pin: PR-A rev 1; if anyone moves the public function
    /// signatures, this test should fail to force a contract bump.
    #[test]
    fn provenance_pin_pr_a_rev1() {
        assert!(is_parquet(Path::new("/tmp/x.parquet")));
        assert!(!is_parquet(Path::new("/tmp/x.jsonl")));
    }

    #[test]
    fn is_parquet_recognises_extension_case_insensitive() {
        assert!(is_parquet(Path::new("data.parquet")));
        assert!(is_parquet(Path::new("DATA.PARQUET")));
        assert!(is_parquet(Path::new("data.Parquet")));
    }

    #[test]
    fn is_parquet_rejects_other_extensions() {
        assert!(!is_parquet(Path::new("data.json")));
        assert!(!is_parquet(Path::new("data.jsonl")));
        assert!(!is_parquet(Path::new("data")));
        assert!(!is_parquet(Path::new("data.parquet.bak")));
    }

    #[test]
    fn collect_parquet_files_single_file() {
        let tmp = tempfile::tempdir().unwrap();
        let p = tmp.path().join("a.parquet");
        std::fs::write(&p, b"PAR1").unwrap(); // dummy bytes; collect doesn't parse
        let files = collect_parquet_files(&p).unwrap();
        assert_eq!(files, vec![p]);
    }

    #[test]
    fn collect_parquet_files_directory_filters_and_sorts() {
        let tmp = tempfile::tempdir().unwrap();
        let a = tmp.path().join("01.parquet");
        let b = tmp.path().join("00.parquet");
        let other = tmp.path().join("ignore.json");
        std::fs::write(&a, b"a").unwrap();
        std::fs::write(&b, b"b").unwrap();
        std::fs::write(&other, b"c").unwrap();
        let files = collect_parquet_files(tmp.path()).unwrap();
        assert_eq!(files, vec![b, a]); // sorted ascending; .json filtered
    }

    #[test]
    fn collect_parquet_files_rejects_non_parquet_file() {
        let tmp = tempfile::tempdir().unwrap();
        let p = tmp.path().join("a.jsonl");
        std::fs::write(&p, b"x").unwrap();
        let err = collect_parquet_files(&p).unwrap_err();
        assert!(format!("{err}").contains(".parquet"));
    }

    #[test]
    fn collect_parquet_files_rejects_missing_path() {
        let tmp = tempfile::tempdir().unwrap();
        let p = tmp.path().join("does-not-exist.parquet");
        let err = collect_parquet_files(&p).unwrap_err();
        assert!(format!("{err}").contains("Cannot stat"));
    }

    /// Build a tiny in-memory parquet file with a `content` column and
    /// verify `iter_parquet_content` yields each row's string. Round-trip
    /// pin for the parquet → string adapter.
    #[test]
    fn iter_parquet_content_yields_each_row() {
        use arrow_array::{RecordBatch, StringArray};
        use parquet::arrow::ArrowWriter;
        use std::sync::Arc;

        let tmp = tempfile::tempdir().unwrap();
        let path = tmp.path().join("tiny.parquet");

        let texts = vec!["fn main() {}", "import os", "x = 42"];
        let arr = StringArray::from(texts.clone());
        let schema = arrow_array::RecordBatch::try_from_iter(vec![(
            "content",
            Arc::new(arr) as Arc<dyn arrow_array::Array>,
        )])
        .unwrap();
        {
            let f = File::create(&path).unwrap();
            let mut w = ArrowWriter::try_new(f, schema.schema(), None).unwrap();
            w.write(&schema).unwrap();
            w.close().unwrap();
        }

        let collected: Vec<String> = iter_parquet_content(&path, "content")
            .unwrap()
            .collect::<Result<Vec<_>>>()
            .unwrap();
        assert_eq!(collected, texts);
        // suppress unused-var warning on `RecordBatch` import
        let _ = std::mem::size_of::<RecordBatch>();
    }

    #[test]
    fn iter_parquet_content_unknown_column_errors_with_helpful_message() {
        use arrow_array::StringArray;
        use parquet::arrow::ArrowWriter;
        use std::sync::Arc;

        let tmp = tempfile::tempdir().unwrap();
        let path = tmp.path().join("tiny.parquet");
        let arr = StringArray::from(vec!["a", "b"]);
        let batch = arrow_array::RecordBatch::try_from_iter(vec![(
            "code",
            Arc::new(arr) as Arc<dyn arrow_array::Array>,
        )])
        .unwrap();
        {
            let f = File::create(&path).unwrap();
            let mut w = ArrowWriter::try_new(f, batch.schema(), None).unwrap();
            w.write(&batch).unwrap();
            w.close().unwrap();
        }

        let err = match iter_parquet_content(&path, "content") {
            Ok(_) => panic!("expected error for missing column"),
            Err(e) => e,
        };
        let msg = format!("{err}");
        assert!(msg.contains("no column named"));
        assert!(msg.contains("content"));
        assert!(msg.contains("code")); // available columns listed
    }

    #[test]
    fn iter_parquet_content_skips_null_rows() {
        use arrow_array::StringArray;
        use parquet::arrow::ArrowWriter;
        use std::sync::Arc;

        let tmp = tempfile::tempdir().unwrap();
        let path = tmp.path().join("tiny.parquet");
        let arr = StringArray::from(vec![Some("real_code"), None, Some("more")]);
        let batch = arrow_array::RecordBatch::try_from_iter(vec![(
            "content",
            Arc::new(arr) as Arc<dyn arrow_array::Array>,
        )])
        .unwrap();
        {
            let f = File::create(&path).unwrap();
            let mut w = ArrowWriter::try_new(f, batch.schema(), None).unwrap();
            w.write(&batch).unwrap();
            w.close().unwrap();
        }

        let collected: Vec<String> = iter_parquet_content(&path, "content")
            .unwrap()
            .collect::<Result<Vec<_>>>()
            .unwrap();
        assert_eq!(collected, vec!["real_code".to_string(), "more".to_string()]);
    }
}
