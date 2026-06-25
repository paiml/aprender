//! `apr attn-viz-lint` — CRUX-F-17 attention-visualization gate.
//!
//! Reads a captured attention dump (JSON form of the 4-D
//! `[layers][heads][rows][cols]` array) and/or an HTML heatmap output
//! and dispatches the pure classifiers in `attn_viz_classifier`. Exits
//! non-zero on any failure.
//!
//! Spec: `contracts/crux-F-17-v1.yaml`. CRUX-SHIP-001 g2/g3 surface.

use std::path::{Path, PathBuf};

use serde_json::Value;

use super::attn_viz_classifier::{
    classify_causal_mask, classify_html_heatmap_count, classify_row_softmax_normalization,
    AttnCausalMaskOutcome, AttnHtmlOutcome, AttnRowsOutcome,
};
use crate::error::{CliError, Result};

pub(crate) fn run(
    attn_file: Option<&Path>,
    html_file: Option<&Path>,
    expected_heatmaps: usize,
    tolerance: f64,
    epsilon: f64,
    json: bool,
) -> Result<()> {
    if attn_file.is_none() && html_file.is_none() {
        return Err(CliError::ValidationFailed(
            "apr attn-viz-lint: at least one of --attn-file or --html-file is required".to_string(),
        ));
    }

    let (rows_outcome, mask_outcome) = match attn_file {
        Some(p) => {
            if !p.exists() {
                return Err(CliError::FileNotFound(PathBuf::from(p)));
            }
            let body_text = std::fs::read_to_string(p)?;
            let body: Value = serde_json::from_str(&body_text).map_err(|e| {
                CliError::InvalidFormat(format!(
                    "apr attn-viz-lint: failed to parse JSON from {}: {e}",
                    p.display()
                ))
            })?;
            (
                Some(classify_row_softmax_normalization(&body, tolerance)),
                Some(classify_causal_mask(&body, epsilon)),
            )
        }
        None => (None, None),
    };

    let html_outcome = match html_file {
        Some(p) => {
            if !p.exists() {
                return Err(CliError::FileNotFound(PathBuf::from(p)));
            }
            let html = std::fs::read_to_string(p)?;
            Some(classify_html_heatmap_count(&html, expected_heatmaps))
        }
        None => None,
    };

    print_report(
        attn_file,
        html_file,
        rows_outcome.as_ref(),
        mask_outcome.as_ref(),
        html_outcome.as_ref(),
        json,
    );

    if let Some(o) = &rows_outcome {
        if !matches!(o, AttnRowsOutcome::Ok { .. }) {
            return Err(CliError::ValidationFailed(format!(
                "attn-viz-lint row-softmax gate rejected body: {o:?}"
            )));
        }
    }
    if let Some(o) = &mask_outcome {
        if !matches!(o, AttnCausalMaskOutcome::Ok) {
            return Err(CliError::ValidationFailed(format!(
                "attn-viz-lint causal-mask gate rejected body: {o:?}"
            )));
        }
    }
    if let Some(o) = &html_outcome {
        if !matches!(o, AttnHtmlOutcome::Ok { .. }) {
            return Err(CliError::ValidationFailed(format!(
                "attn-viz-lint html-heatmap-count gate rejected body: {o:?}"
            )));
        }
    }
    Ok(())
}

fn print_report(
    attn_file: Option<&Path>,
    html_file: Option<&Path>,
    rows: Option<&AttnRowsOutcome>,
    mask: Option<&AttnCausalMaskOutcome>,
    html: Option<&AttnHtmlOutcome>,
    json: bool,
) {
    if json {
        let obj = serde_json::json!({
            "attn_file": attn_file.map(|p| p.display().to_string()),
            "html_file": html_file.map(|p| p.display().to_string()),
            "row_softmax": rows.map(|o| format!("{o:?}")),
            "causal_mask": mask.map(|o| format!("{o:?}")),
            "html_heatmaps": html.map(|o| format!("{o:?}")),
        });
        println!("{}", serde_json::to_string_pretty(&obj).unwrap_or_default());
        return;
    }
    println!("attn-viz-lint report");
    if let Some(p) = attn_file {
        println!("  attn_file     : {}", p.display());
    }
    if let Some(p) = html_file {
        println!("  html_file     : {}", p.display());
    }
    if let Some(o) = rows {
        println!("  row_softmax   : {o:?}");
    }
    if let Some(o) = mask {
        println!("  causal_mask   : {o:?}");
    }
    if let Some(o) = html {
        println!("  html_heatmaps : {o:?}");
    }
}

#[cfg(test)]
mod cov_tests {
    use super::*;
    #[test]
    fn no_inputs_is_validation_failed() {
        let err = run(None, None, 0, 1e-3, 1e-9, false).unwrap_err();
        assert!(matches!(err, CliError::ValidationFailed(_)));
    }
    #[test]
    fn missing_attn_file_is_file_not_found() {
        let err = run(
            Some(Path::new("/no/such/attn.json")),
            None,
            0,
            1e-3,
            1e-9,
            false,
        )
        .unwrap_err();
        assert!(matches!(err, CliError::FileNotFound(_)));
    }
    #[test]
    fn missing_html_file_is_file_not_found() {
        let err = run(
            None,
            Some(Path::new("/no/such/viz.html")),
            0,
            1e-3,
            1e-9,
            true,
        )
        .unwrap_err();
        assert!(matches!(err, CliError::FileNotFound(_)));
    }
}
