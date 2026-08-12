//! Token embedding UMAP 2D classifier (CRUX-F-18).
//!
//! Pure, deterministic classifiers that discharge FALSIFY-CRUX-F-18-{001,003}
//! at the PARTIAL_ALGORITHM_LEVEL — algorithm-level necessary conditions on
//! a captured `apr debug embed-viz --seed N -o emb.csv` output:
//!
//!   * `classify_schema` — header is `token_id,token_str,x,y` (in some
//!     order); every row has the same 4 columns; `token_id` parses as
//!     non-negative int; `x` and `y` parse as finite floats.
//!   * `classify_row_count` — number of data rows equals
//!     `expected_vocab_size`.
//!   * `classify_determinism` — two captured CSVs produced under the
//!     same seed are byte-identical (no trailing-newline / whitespace
//!     drift either).
//!
//! FALSIFY-CRUX-F-18-002 (token_str round-trips via the tokenizer)
//! requires a live tokenizer decoder — tracked as
//! BLOCKER-UPSTREAM-MISSING.

/// Required CSV column headers (order-independent).
pub const F18_REQUIRED_COLUMNS: &[&str] = &["token_id", "token_str", "x", "y"];

/// Outcome of `classify_schema`.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize)]
#[serde(tag = "status", rename_all = "snake_case")]
pub enum EmbedSchemaOutcome {
    Ok {
        rows: usize,
    },
    Empty,
    MissingHeader,
    MissingColumn {
        col: &'static str,
    },
    WrongColumnCount {
        line_no: usize,
        got: usize,
        expected: usize,
    },
    TokenIdNotNonNegativeInt {
        line_no: usize,
        got: String,
    },
    CoordNotFiniteFloat {
        line_no: usize,
        col: &'static str,
        got: String,
    },
}

/// Outcome of `classify_row_count`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum EmbedRowCountOutcome {
    Ok,
    Mismatch { got: usize, expected: usize },
}

/// Outcome of `classify_determinism`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum EmbedDeterminismOutcome {
    Ok {
        bytes: usize,
    },
    LengthDiffers {
        left_bytes: usize,
        right_bytes: usize,
    },
    FirstDiffAtByte {
        offset: usize,
        left: u8,
        right: u8,
    },
}

/// Validate CSV schema (header + per-row types).
pub fn classify_schema(body: &str) -> EmbedSchemaOutcome {
    let mut lines = body.lines();
    let Some(header) = lines.next() else {
        return EmbedSchemaOutcome::Empty;
    };
    let cols: Vec<&str> = header.split(',').map(str::trim).collect();
    if cols.len() < F18_REQUIRED_COLUMNS.len() {
        return EmbedSchemaOutcome::MissingHeader;
    }
    let mut idx_token_id: Option<usize> = None;
    let mut idx_token_str: Option<usize> = None;
    let mut idx_x: Option<usize> = None;
    let mut idx_y: Option<usize> = None;
    for (i, c) in cols.iter().enumerate() {
        match *c {
            "token_id" => idx_token_id = Some(i),
            "token_str" => idx_token_str = Some(i),
            "x" => idx_x = Some(i),
            "y" => idx_y = Some(i),
            _ => {}
        }
    }
    for (val, col) in [
        (idx_token_id, "token_id"),
        (idx_token_str, "token_str"),
        (idx_x, "x"),
        (idx_y, "y"),
    ] {
        if val.is_none() {
            return EmbedSchemaOutcome::MissingColumn { col };
        }
    }
    let idx_token_id = idx_token_id.unwrap_or(0);
    let idx_x = idx_x.unwrap_or(0);
    let idx_y = idx_y.unwrap_or(0);
    let expected = cols.len();

    let mut rows = 0usize;
    for (i, line) in lines.enumerate() {
        let line_no = i + 2; // header is line 1
        if line.trim().is_empty() {
            continue;
        }
        // Split on commas; we tolerate quoted token_str strings if they don't
        // contain commas. For commas inside quoted strings, the contract spec
        // says the emitter is responsible for proper CSV quoting — the
        // classifier verifies the column count matches the header.
        let parts: Vec<&str> = line.split(',').collect();
        if parts.len() != expected {
            return EmbedSchemaOutcome::WrongColumnCount {
                line_no,
                got: parts.len(),
                expected,
            };
        }
        match parts[idx_token_id].trim().parse::<i64>() {
            Ok(v) if v >= 0 => {}
            _ => {
                return EmbedSchemaOutcome::TokenIdNotNonNegativeInt {
                    line_no,
                    got: parts[idx_token_id].to_string(),
                }
            }
        }
        for (idx, name) in [(idx_x, "x"), (idx_y, "y")] {
            let raw = parts[idx].trim();
            match raw.parse::<f64>() {
                Ok(v) if v.is_finite() => {}
                _ => {
                    return EmbedSchemaOutcome::CoordNotFiniteFloat {
                        line_no,
                        col: name,
                        got: raw.to_string(),
                    }
                }
            }
        }
        rows += 1;
    }
    EmbedSchemaOutcome::Ok { rows }
}

/// Verify the row count equals `expected_vocab_size`.
pub fn classify_row_count(body: &str, expected: usize) -> EmbedRowCountOutcome {
    let count = body
        .lines()
        .skip(1) // header
        .filter(|l| !l.trim().is_empty())
        .count();
    if count == expected {
        EmbedRowCountOutcome::Ok
    } else {
        EmbedRowCountOutcome::Mismatch {
            got: count,
            expected,
        }
    }
}

/// Verify two CSVs are byte-identical.
pub fn classify_determinism(left: &[u8], right: &[u8]) -> EmbedDeterminismOutcome {
    if left.len() != right.len() {
        return EmbedDeterminismOutcome::LengthDiffers {
            left_bytes: left.len(),
            right_bytes: right.len(),
        };
    }
    for (i, (l, r)) in left.iter().zip(right.iter()).enumerate() {
        if l != r {
            return EmbedDeterminismOutcome::FirstDiffAtByte {
                offset: i,
                left: *l,
                right: *r,
            };
        }
    }
    EmbedDeterminismOutcome::Ok { bytes: left.len() }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn good_body() -> String {
        "token_id,token_str,x,y\n0,<pad>,0.1,0.2\n1,<unk>,-0.5,1.5\n2,hello,3.14,-2.71\n"
            .to_string()
    }

    #[test]
    fn schema_ok_on_good_body() {
        let out = classify_schema(&good_body());
        assert_eq!(out, EmbedSchemaOutcome::Ok { rows: 3 });
    }

    #[test]
    fn schema_rejects_empty_body() {
        assert_eq!(classify_schema(""), EmbedSchemaOutcome::Empty);
    }

    #[test]
    fn schema_rejects_missing_column() {
        // Header missing `y`.
        let body = "token_id,token_str,x\n0,<pad>,0.1\n";
        assert!(matches!(
            classify_schema(body),
            EmbedSchemaOutcome::MissingHeader | EmbedSchemaOutcome::MissingColumn { .. }
        ));
    }

    #[test]
    fn schema_rejects_negative_token_id() {
        let body = "token_id,token_str,x,y\n-1,<pad>,0.1,0.2\n";
        assert!(matches!(
            classify_schema(body),
            EmbedSchemaOutcome::TokenIdNotNonNegativeInt { line_no: 2, .. }
        ));
    }

    #[test]
    fn schema_rejects_nonfinite_x() {
        let body = "token_id,token_str,x,y\n0,<pad>,nan,0.2\n";
        match classify_schema(body) {
            EmbedSchemaOutcome::CoordNotFiniteFloat {
                line_no: 2,
                col: "x",
                ..
            } => {}
            other => panic!("expected CoordNotFiniteFloat(x), got {other:?}"),
        }
    }

    #[test]
    fn schema_rejects_inf_y() {
        let body = "token_id,token_str,x,y\n0,<pad>,0.0,inf\n";
        match classify_schema(body) {
            EmbedSchemaOutcome::CoordNotFiniteFloat {
                line_no: 2,
                col: "y",
                ..
            } => {}
            other => panic!("expected CoordNotFiniteFloat(y), got {other:?}"),
        }
    }

    #[test]
    fn schema_accepts_reordered_columns() {
        let body = "x,y,token_id,token_str\n0.1,0.2,0,<pad>\n";
        assert_eq!(classify_schema(body), EmbedSchemaOutcome::Ok { rows: 1 });
    }

    #[test]
    fn row_count_ok_on_match() {
        assert_eq!(
            classify_row_count(&good_body(), 3),
            EmbedRowCountOutcome::Ok
        );
    }

    #[test]
    fn row_count_reports_mismatch() {
        assert_eq!(
            classify_row_count(&good_body(), 100),
            EmbedRowCountOutcome::Mismatch {
                got: 3,
                expected: 100
            }
        );
    }

    #[test]
    fn determinism_ok_on_byte_identical() {
        let b1 = b"abc\n123\n";
        let b2 = b"abc\n123\n";
        assert_eq!(
            classify_determinism(b1, b2),
            EmbedDeterminismOutcome::Ok { bytes: 8 }
        );
    }

    #[test]
    fn determinism_reports_length_diff() {
        let b1 = b"abc\n";
        let b2 = b"abc\n\n";
        assert_eq!(
            classify_determinism(b1, b2),
            EmbedDeterminismOutcome::LengthDiffers {
                left_bytes: 4,
                right_bytes: 5,
            }
        );
    }

    #[test]
    fn determinism_reports_first_diff_byte() {
        let b1 = b"abc\n123\n";
        let b2 = b"abc\n124\n";
        match classify_determinism(b1, b2) {
            EmbedDeterminismOutcome::FirstDiffAtByte {
                offset: 6,
                left,
                right,
            } => {
                assert_eq!(left, b'3');
                assert_eq!(right, b'4');
            }
            other => panic!("expected FirstDiffAtByte, got {other:?}"),
        }
    }
}
