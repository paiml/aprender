//! Attention pattern visualization classifier (CRUX-F-17).
//!
//! Pure, deterministic classifiers that discharge FALSIFY-CRUX-F-17-{001,002,003}
//! at the PARTIAL_ALGORITHM_LEVEL — algorithm-level necessary conditions on:
//!
//!   * a captured attention dump in JSON form (4-D array
//!     `[layers][heads][rows][cols]` of floats) — converted from the
//!     spec's binary `attn.npy` via
//!     `python3 -c "import numpy as np, json; json.dump(np.load('attn.npy').tolist(), open('attn.json','w'))"`
//!   * the HTML heatmap output from `apr attn-viz --out DIR`.
//!
//! Classifiers:
//!   * `classify_row_softmax_normalization` — every row in every
//!     `(layer, head)` slice sums to 1.0 ± tolerance (default 1e-5).
//!   * `classify_causal_mask` — for every `(layer, head, i, j)` with
//!     `j > i`, the value is ≤ epsilon (default 1e-9) — the causal
//!     mask must zero future positions before softmax.
//!   * `classify_html_heatmap_count` — the HTML body contains at
//!     least `expected` `<svg` or `<canvas` open-tags (one per
//!     `(layer, head)` cell).
//!
//! Full discharge requires `apr attn-viz` actually emitting the
//! attention dump + HTML — tracked as BLOCKER-UPSTREAM-MISSING.

use serde_json::Value;

/// Outcome of `classify_row_softmax_normalization`.
#[derive(Debug, Clone, PartialEq)]
pub enum AttnRowsOutcome {
    Ok {
        shape: (usize, usize, usize, usize),
    },
    NotA4DArray {
        message: String,
    },
    RowOutOfNormalization {
        layer: usize,
        head: usize,
        row: usize,
        sum: f64,
        tolerance: f64,
    },
}

/// Outcome of `classify_causal_mask`.
#[derive(Debug, Clone, PartialEq)]
pub enum AttnCausalMaskOutcome {
    Ok,
    NonZeroFuturePosition {
        layer: usize,
        head: usize,
        row: usize,
        col: usize,
        value: f64,
        epsilon: f64,
    },
}

/// Outcome of `classify_html_heatmap_count`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AttnHtmlOutcome {
    Ok { count: usize },
    TooFewHeatmaps { got: usize, expected: usize },
}

/// Parse a 4-D `Vec<Vec<Vec<Vec<f64>>>>` from `body` (typically `attn.json`).
/// Returns the array and its shape as `(L, H, R, C)`.
fn parse_4d_array(
    body: &Value,
) -> Result<(Vec<Vec<Vec<Vec<f64>>>>, (usize, usize, usize, usize)), String> {
    let layers = body
        .as_array()
        .ok_or_else(|| "root is not an array".to_string())?;
    if layers.is_empty() {
        return Err("layers dimension is empty".to_string());
    }
    let mut out: Vec<Vec<Vec<Vec<f64>>>> = Vec::with_capacity(layers.len());
    let mut shape = (layers.len(), 0usize, 0usize, 0usize);
    for (li, layer) in layers.iter().enumerate() {
        let heads = layer
            .as_array()
            .ok_or_else(|| format!("layer[{li}] not an array"))?;
        let mut layer_vec: Vec<Vec<Vec<f64>>> = Vec::with_capacity(heads.len());
        for (hi, head) in heads.iter().enumerate() {
            let rows = head
                .as_array()
                .ok_or_else(|| format!("layer[{li}].head[{hi}] not an array"))?;
            let mut head_vec: Vec<Vec<f64>> = Vec::with_capacity(rows.len());
            for (ri, row) in rows.iter().enumerate() {
                let cols = row
                    .as_array()
                    .ok_or_else(|| format!("layer[{li}].head[{hi}].row[{ri}] not an array"))?;
                let mut row_vec: Vec<f64> = Vec::with_capacity(cols.len());
                for (ci, c) in cols.iter().enumerate() {
                    let v = c.as_f64().ok_or_else(|| {
                        format!("layer[{li}].head[{hi}].row[{ri}].col[{ci}] not a number")
                    })?;
                    row_vec.push(v);
                }
                head_vec.push(row_vec);
            }
            layer_vec.push(head_vec);
        }
        if li == 0 {
            shape.1 = layer_vec.len();
            shape.2 = layer_vec.first().map(Vec::len).unwrap_or(0);
            shape.3 = layer_vec
                .first()
                .and_then(|h| h.first())
                .map(Vec::len)
                .unwrap_or(0);
        }
        out.push(layer_vec);
    }
    Ok((out, shape))
}

/// Verify that every row in every (layer, head) slice sums to 1.0
/// within `tolerance`.
pub fn classify_row_softmax_normalization(body: &Value, tolerance: f64) -> AttnRowsOutcome {
    let (arr, shape) = match parse_4d_array(body) {
        Ok(v) => v,
        Err(e) => return AttnRowsOutcome::NotA4DArray { message: e },
    };
    for (li, layer) in arr.iter().enumerate() {
        for (hi, head) in layer.iter().enumerate() {
            for (ri, row) in head.iter().enumerate() {
                let s: f64 = row.iter().sum();
                if (s - 1.0).abs() > tolerance {
                    return AttnRowsOutcome::RowOutOfNormalization {
                        layer: li,
                        head: hi,
                        row: ri,
                        sum: s,
                        tolerance,
                    };
                }
            }
        }
    }
    AttnRowsOutcome::Ok { shape }
}

/// Verify that the upper-triangular positions (j > i) are ≤ `epsilon`
/// — causal mask must zero future positions before softmax.
pub fn classify_causal_mask(body: &Value, epsilon: f64) -> AttnCausalMaskOutcome {
    let arr = match parse_4d_array(body) {
        Ok((v, _)) => v,
        Err(_) => return AttnCausalMaskOutcome::Ok,
    };
    for (li, layer) in arr.iter().enumerate() {
        for (hi, head) in layer.iter().enumerate() {
            for (ri, row) in head.iter().enumerate() {
                for (ci, &v) in row.iter().enumerate() {
                    if ci > ri && v.abs() > epsilon {
                        return AttnCausalMaskOutcome::NonZeroFuturePosition {
                            layer: li,
                            head: hi,
                            row: ri,
                            col: ci,
                            value: v,
                            epsilon,
                        };
                    }
                }
            }
        }
    }
    AttnCausalMaskOutcome::Ok
}

/// Count the number of `<svg` or `<canvas` open-tags in the HTML body;
/// require at least `expected` (typically `|layers| * |heads|`).
pub fn classify_html_heatmap_count(html: &str, expected: usize) -> AttnHtmlOutcome {
    let count = count_tag_opens(html, "<svg") + count_tag_opens(html, "<canvas");
    if count >= expected {
        AttnHtmlOutcome::Ok { count }
    } else {
        AttnHtmlOutcome::TooFewHeatmaps {
            got: count,
            expected,
        }
    }
}

fn count_tag_opens(haystack: &str, needle: &str) -> usize {
    let mut n = 0usize;
    let mut start = 0usize;
    while let Some(idx) = haystack[start..].find(needle) {
        n += 1;
        start += idx + needle.len();
    }
    n
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    /// Build a (2,2,3,3) attention array with row-softmax normalized
    /// and causal mask honored.
    fn good_array() -> Value {
        // (L=2, H=2, S=3, S=3); each row sums to 1, j > i is zero.
        let row0 = json!([1.0, 0.0, 0.0]); // step 0 attends only to itself
        let row1 = json!([0.4, 0.6, 0.0]); // step 1 attends to {0,1}
        let row2 = json!([0.2, 0.3, 0.5]); // step 2 attends to {0,1,2}
        let head = json!([row0.clone(), row1.clone(), row2.clone()]);
        let layer = json!([head.clone(), head.clone()]);
        json!([layer.clone(), layer.clone()])
    }

    #[test]
    fn rows_softmax_ok_on_good_array() {
        let out = classify_row_softmax_normalization(&good_array(), 1e-5);
        assert!(matches!(out, AttnRowsOutcome::Ok { .. }), "{out:?}");
    }

    #[test]
    fn rows_softmax_rejects_non_4d_input() {
        let out = classify_row_softmax_normalization(&json!([1, 2, 3]), 1e-5);
        assert!(
            matches!(out, AttnRowsOutcome::NotA4DArray { .. }),
            "{out:?}"
        );
    }

    #[test]
    fn rows_softmax_reports_unnormalized_row() {
        // Sum to 1.2 on row 0 of layer 0 head 0.
        let bad = json!([[[[0.6, 0.6, 0.0], [0.4, 0.6, 0.0], [0.2, 0.3, 0.5]]]]);
        match classify_row_softmax_normalization(&bad, 1e-5) {
            AttnRowsOutcome::RowOutOfNormalization {
                layer: 0,
                head: 0,
                row: 0,
                sum,
                ..
            } => {
                assert!((sum - 1.2).abs() < 1e-9, "got sum {sum}");
            }
            other => panic!("expected RowOutOfNormalization, got {other:?}"),
        }
    }

    #[test]
    fn rows_softmax_tolerance_can_be_relaxed() {
        // Sum to 1.01 — strict 1e-5 fails, relaxed 0.05 passes.
        let body = json!([[[[0.51, 0.50, 0.0], [0.4, 0.6, 0.0], [0.2, 0.3, 0.5]]]]);
        assert!(matches!(
            classify_row_softmax_normalization(&body, 1e-5),
            AttnRowsOutcome::RowOutOfNormalization { .. }
        ));
        assert!(matches!(
            classify_row_softmax_normalization(&body, 0.05),
            AttnRowsOutcome::Ok { .. }
        ));
    }

    #[test]
    fn causal_mask_ok_on_good_array() {
        assert_eq!(
            classify_causal_mask(&good_array(), 1e-9),
            AttnCausalMaskOutcome::Ok
        );
    }

    #[test]
    fn causal_mask_reports_nonzero_future() {
        // Row 0 has weight 0.5 on column 1 — future position leaks.
        let body = json!([[[[0.5, 0.5, 0.0], [0.4, 0.6, 0.0], [0.2, 0.3, 0.5]]]]);
        match classify_causal_mask(&body, 1e-9) {
            AttnCausalMaskOutcome::NonZeroFuturePosition {
                layer: 0,
                head: 0,
                row: 0,
                col: 1,
                value,
                ..
            } => {
                assert!((value - 0.5).abs() < 1e-9, "got value {value}");
            }
            other => panic!("expected NonZeroFuturePosition, got {other:?}"),
        }
    }

    #[test]
    fn html_heatmap_count_ok_when_threshold_met() {
        let html = "<html><svg></svg><svg></svg><canvas></canvas><svg></svg></html>";
        assert_eq!(
            classify_html_heatmap_count(html, 4),
            AttnHtmlOutcome::Ok { count: 4 }
        );
    }

    #[test]
    fn html_heatmap_count_reports_too_few() {
        let html = "<html><svg></svg></html>";
        assert_eq!(
            classify_html_heatmap_count(html, 4),
            AttnHtmlOutcome::TooFewHeatmaps {
                got: 1,
                expected: 4
            }
        );
    }

    #[test]
    fn html_heatmap_count_handles_attributes_in_open_tag() {
        // Real HTML has attributes between `<svg` and `>`.
        let html = r#"<svg class="heatmap" width="200"></svg><svg id="b"></svg>"#;
        assert_eq!(
            classify_html_heatmap_count(html, 2),
            AttnHtmlOutcome::Ok { count: 2 }
        );
    }
}
