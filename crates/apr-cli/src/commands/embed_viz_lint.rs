//! `apr embed-viz-lint` — CRUX-F-18 UMAP token-embedding CSV gate.
//!
//! Reads a captured `apr debug embed-viz -o emb.csv` body and dispatches
//! the pure classifiers in `embed_viz_classifier`. Exits non-zero on
//! any failure.
//!
//! Spec: `contracts/crux-F-18-v1.yaml`. CRUX-SHIP-001 g2/g3 surface.

use std::path::{Path, PathBuf};

use super::embed_viz_classifier::{
    classify_determinism, classify_row_count, classify_schema, EmbedDeterminismOutcome,
    EmbedRowCountOutcome, EmbedSchemaOutcome,
};
use crate::error::{CliError, Result};

pub(crate) fn run(
    csv_file: &Path,
    expected_vocab_size: Option<usize>,
    csv_file_b: Option<&Path>,
    json: bool,
) -> Result<()> {
    if !csv_file.exists() {
        return Err(CliError::FileNotFound(PathBuf::from(csv_file)));
    }
    let body = std::fs::read_to_string(csv_file)?;

    let schema = classify_schema(&body);
    let row_count = expected_vocab_size.map(|n| classify_row_count(&body, n));
    let determinism = match csv_file_b {
        Some(p) => {
            if !p.exists() {
                return Err(CliError::FileNotFound(PathBuf::from(p)));
            }
            let b = std::fs::read(p)?;
            Some(classify_determinism(body.as_bytes(), &b))
        }
        None => None,
    };

    print_report(
        csv_file,
        csv_file_b,
        &schema,
        row_count.as_ref(),
        determinism.as_ref(),
        json,
    );

    if !matches!(schema, EmbedSchemaOutcome::Ok { .. }) {
        return Err(CliError::ValidationFailed(format!(
            "embed-viz-lint schema gate rejected body: {schema:?}"
        )));
    }
    if let Some(o) = &row_count {
        if !matches!(o, EmbedRowCountOutcome::Ok) {
            return Err(CliError::ValidationFailed(format!(
                "embed-viz-lint row-count gate rejected body: {o:?}"
            )));
        }
    }
    if let Some(o) = &determinism {
        if !matches!(o, EmbedDeterminismOutcome::Ok { .. }) {
            return Err(CliError::ValidationFailed(format!(
                "embed-viz-lint determinism gate rejected bodies: {o:?}"
            )));
        }
    }
    Ok(())
}

fn print_report(
    csv_file: &Path,
    csv_file_b: Option<&Path>,
    schema: &EmbedSchemaOutcome,
    row_count: Option<&EmbedRowCountOutcome>,
    determinism: Option<&EmbedDeterminismOutcome>,
    json: bool,
) {
    if json {
        let obj = serde_json::json!({
            "csv_file": csv_file.display().to_string(),
            "csv_file_b": csv_file_b.map(|p| p.display().to_string()),
            "schema": schema,
            "row_count": row_count.map(|o| format!("{o:?}")),
            "determinism": determinism.map(|o| format!("{o:?}")),
        });
        println!("{}", serde_json::to_string_pretty(&obj).unwrap_or_default());
        return;
    }
    println!("embed-viz-lint report for {}", csv_file.display());
    println!("  schema     : {schema:?}");
    if let Some(o) = row_count {
        println!("  row_count  : {o:?}");
    }
    if let Some(o) = determinism {
        println!("  determinism: {o:?}");
    }
}

#[cfg(test)]
mod cov_tests {
    use super::*;
    use std::io::Write;
    use tempfile::NamedTempFile;
    fn w(s: &str) -> NamedTempFile {
        let mut f = NamedTempFile::new().unwrap();
        f.write_all(s.as_bytes()).unwrap();
        f.flush().unwrap();
        f
    }
    #[test]
    fn missing_file_is_file_not_found() {
        let err = run(Path::new("/no/such/embed.csv"), None, None, false).unwrap_err();
        assert!(matches!(err, CliError::FileNotFound(_)));
    }
    #[test]
    fn missing_second_file_is_file_not_found() {
        let a = w("token,x,y\n");
        let err = run(a.path(), None, Some(Path::new("/no/such/b.csv")), false);
        assert!(err.is_err());
    }
    #[test]
    fn empty_csv_runs() {
        let f = w("");
        let _ = run(f.path(), None, None, true);
    }
}
