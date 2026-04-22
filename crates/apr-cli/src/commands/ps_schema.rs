//! `apr ps` row schema + column header classifier (CRUX-A-12).
//!
//! Contract: `contracts/crux-A-12-v1.yaml`.
//!
//! Pure classifier — validates the JSON row schema and text header
//! layout without touching any live daemon. Given an arbitrary
//! `apr ps --json` payload and an `apr ps` header line, the classifier
//! answers:
//!   * does every row carry `{name, id, size_bytes, processor, until}`
//!     with correct types?  (FALSIFY-003)
//!   * is the text header a superset of `{NAME, SIZE, PROCESSOR}`?
//!     (FALSIFY-004)
//!   * is an empty-state response the literal `[]`?  (FALSIFY-001)
//!
//! FALSIFY-CRUX-A-12-002 (load visibility in `apr ps` after a real
//! generation) is NOT discharged by this module — it requires live
//! `apr serve` + `apr run`.

use serde_json::Value;

/// Required column headers in `apr ps` (ollama-parity superset).
pub const REQUIRED_COLUMNS: &[&str] = &["NAME", "SIZE", "PROCESSOR"];

/// Required JSON fields per row, with expected type category.
pub const REQUIRED_FIELDS: &[&str] = &["name", "id", "size_bytes", "processor", "until"];

/// Reason a row (or header) is rejected.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PsSchemaError {
    MissingField(String),
    WrongType {
        field: String,
        expected: &'static str,
    },
    IdTooShort(String),
    ProcessorInvalid(String),
    UntilNotRfc3339(String),
    MissingColumn(String),
    NotAnArray,
    NotAnObject,
}

impl std::fmt::Display for PsSchemaError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            PsSchemaError::MissingField(x) => write!(f, "missing required field: {x:?}"),
            PsSchemaError::WrongType { field, expected } => {
                write!(f, "field {field:?} has wrong type; expected {expected}")
            }
            PsSchemaError::IdTooShort(id) => {
                write!(f, "id must be ≥12 hex chars: {id:?}")
            }
            PsSchemaError::ProcessorInvalid(p) => {
                write!(f, "processor does not match spec: {p:?}")
            }
            PsSchemaError::UntilNotRfc3339(u) => {
                write!(f, "until is not RFC3339: {u:?}")
            }
            PsSchemaError::MissingColumn(c) => {
                write!(f, "missing required column: {c:?}")
            }
            PsSchemaError::NotAnArray => write!(f, "top-level JSON is not an array"),
            PsSchemaError::NotAnObject => write!(f, "row is not a JSON object"),
        }
    }
}

impl std::error::Error for PsSchemaError {}

/// Return true iff `s` is a lowercase-or-mixed-case hex string of at
/// least 12 chars. Ollama truncates IDs to a 12-char prefix.
fn is_hex_prefix_12(s: &str) -> bool {
    s.len() >= 12 && s.chars().all(|c| c.is_ascii_hexdigit())
}

/// Processor spec from the contract:
///   `{"100% GPU", "NN%/MM% CPU/GPU", "100% CPU"}`
/// We allow case-insensitive match and any non-negative integer pair
/// that sums to exactly 100.
fn is_valid_processor(s: &str) -> bool {
    let t = s.trim();
    if t.eq_ignore_ascii_case("100% GPU") {
        return true;
    }
    if t.eq_ignore_ascii_case("100% CPU") {
        return true;
    }
    // NN%/MM% CPU/GPU — accept any `N%/M% CPU/GPU` with N+M=100.
    if let Some((left, rest)) = t.split_once(' ') {
        if !rest.eq_ignore_ascii_case("CPU/GPU") {
            return false;
        }
        let parts: Vec<&str> = left.split('/').collect();
        if parts.len() != 2 {
            return false;
        }
        let a = parts[0]
            .strip_suffix('%')
            .and_then(|x| x.parse::<u32>().ok());
        let b = parts[1]
            .strip_suffix('%')
            .and_then(|x| x.parse::<u32>().ok());
        return matches!((a, b), (Some(a), Some(b)) if a + b == 100);
    }
    false
}

/// Minimal RFC3339 check — matches the canonical
/// `YYYY-MM-DDTHH:MM:SS` shape with optional fractional seconds and a
/// `Z` / `±HH:MM` suffix. Good enough for algorithm-level proof;
/// the integration-level check is chrono round-trip.
fn is_rfc3339(s: &str) -> bool {
    // Minimum shape: 2026-04-20T12:34:56Z (20 chars)
    if s.len() < 20 {
        return false;
    }
    let b = s.as_bytes();
    let digits_at = |i: usize| b.get(i).map(|c| c.is_ascii_digit()).unwrap_or(false);
    let char_at = |i: usize, c: u8| b.get(i) == Some(&c);
    digits_at(0)
        && digits_at(1)
        && digits_at(2)
        && digits_at(3)
        && char_at(4, b'-')
        && digits_at(5)
        && digits_at(6)
        && char_at(7, b'-')
        && digits_at(8)
        && digits_at(9)
        && (char_at(10, b'T') || char_at(10, b't'))
        && digits_at(11)
        && digits_at(12)
        && char_at(13, b':')
        && digits_at(14)
        && digits_at(15)
        && char_at(16, b':')
        && digits_at(17)
        && digits_at(18)
        && {
            // Tail: either Z / z, or ±HH:MM, or .fractional + one of above.
            let tail = &s[19..];
            let after_secs = tail.trim_start_matches(|c: char| c == '.' || c.is_ascii_digit());
            matches!(after_secs, "Z" | "z")
                || (after_secs.len() == 6
                    && matches!(&after_secs[..1], "+" | "-")
                    && after_secs[1..3].chars().all(|c| c.is_ascii_digit())
                    && &after_secs[3..4] == ":"
                    && after_secs[4..6].chars().all(|c| c.is_ascii_digit()))
        }
}

/// Validate a single `apr ps` row.
pub fn validate_ps_row(v: &Value) -> Result<(), PsSchemaError> {
    let obj = v.as_object().ok_or(PsSchemaError::NotAnObject)?;

    for field in REQUIRED_FIELDS {
        if !obj.contains_key(*field) {
            return Err(PsSchemaError::MissingField((*field).to_string()));
        }
    }

    let name = obj["name"].as_str().ok_or(PsSchemaError::WrongType {
        field: "name".into(),
        expected: "string",
    })?;
    if name.is_empty() {
        return Err(PsSchemaError::WrongType {
            field: "name".into(),
            expected: "non-empty string",
        });
    }

    let id = obj["id"].as_str().ok_or(PsSchemaError::WrongType {
        field: "id".into(),
        expected: "string",
    })?;
    if !is_hex_prefix_12(id) {
        return Err(PsSchemaError::IdTooShort(id.to_string()));
    }

    let sb = obj["size_bytes"].as_u64().ok_or(PsSchemaError::WrongType {
        field: "size_bytes".into(),
        expected: "u64",
    })?;
    if sb == 0 {
        return Err(PsSchemaError::WrongType {
            field: "size_bytes".into(),
            expected: "u64 > 0",
        });
    }

    let proc_ = obj["processor"].as_str().ok_or(PsSchemaError::WrongType {
        field: "processor".into(),
        expected: "string",
    })?;
    if !is_valid_processor(proc_) {
        return Err(PsSchemaError::ProcessorInvalid(proc_.to_string()));
    }

    let until = obj["until"].as_str().ok_or(PsSchemaError::WrongType {
        field: "until".into(),
        expected: "string",
    })?;
    if !is_rfc3339(until) {
        return Err(PsSchemaError::UntilNotRfc3339(until.to_string()));
    }

    Ok(())
}

/// Validate a full `apr ps --json` array payload.
///
/// CRUX-A-12 ALGO-003 sub-claim of FALSIFY-003: every row satisfies
/// the `{name,id,size_bytes,processor,until}` schema.
pub fn validate_ps_array(v: &Value) -> Result<(), PsSchemaError> {
    let arr = v.as_array().ok_or(PsSchemaError::NotAnArray)?;
    for row in arr {
        validate_ps_row(row)?;
    }
    Ok(())
}

/// CRUX-A-12 ALGO-001 sub-claim of FALSIFY-001: an empty `apr ps --json`
/// is the literal array `[]`, never `null` / `"{}"` / omitted field.
pub fn is_empty_ps(v: &Value) -> bool {
    v.as_array().map(|a| a.is_empty()).unwrap_or(false)
}

/// CRUX-A-12 ALGO-004 sub-claim of FALSIFY-004: the text `apr ps`
/// header is a superset of `{NAME, SIZE, PROCESSOR}`. Column names are
/// matched as whole whitespace-separated tokens so e.g. "NAMED" does
/// not accidentally match "NAME".
pub fn header_has_required_columns(header: &str) -> Result<(), PsSchemaError> {
    let tokens: Vec<&str> = header.split_whitespace().collect();
    for col in REQUIRED_COLUMNS {
        if !tokens.iter().any(|t| t == col) {
            return Err(PsSchemaError::MissingColumn((*col).to_string()));
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn good_row() -> Value {
        json!({
            "name": "qwen2.5-coder:7b-q4_k_m",
            "id": "abcdef123456",
            "size_bytes": 4_589_000_000u64,
            "processor": "100% GPU",
            "until": "2026-04-20T12:34:56Z",
        })
    }

    #[test]
    fn falsify_001_sub_claim_empty_is_empty_array() {
        // CRUX-A-12 ALGO-001 sub-claim of FALSIFY-001: `apr ps --json`
        // returns exactly `[]` when no model is loaded — not null,
        // not an object, not an empty string.
        assert!(is_empty_ps(&json!([])));
        assert!(!is_empty_ps(&Value::Null));
        assert!(!is_empty_ps(&json!({})));
        assert!(!is_empty_ps(&json!("")));
        assert!(!is_empty_ps(&json!([good_row()])));
    }

    #[test]
    fn falsify_003_sub_claim_good_row_validates() {
        // CRUX-A-12 ALGO-003 sub-claim of FALSIFY-003: a well-formed
        // row satisfies the schema.
        assert!(validate_ps_row(&good_row()).is_ok());
        assert!(validate_ps_array(&json!([good_row(), good_row()])).is_ok());
    }

    #[test]
    fn falsify_004_sub_claim_header_has_required_columns() {
        // CRUX-A-12 ALGO-004 sub-claim of FALSIFY-004: the ollama-
        // parity header superset check passes on a canonical header.
        assert!(header_has_required_columns("NAME ID SIZE PROCESSOR UNTIL").is_ok());
        assert!(header_has_required_columns("NAME\tID\tSIZE\tPROCESSOR\tUNTIL").is_ok());
    }

    #[test]
    fn missing_name_is_error() {
        let mut r = good_row();
        r.as_object_mut().unwrap().remove("name");
        let err = validate_ps_row(&r).unwrap_err();
        assert_eq!(err, PsSchemaError::MissingField("name".into()));
    }

    #[test]
    fn missing_id_is_error() {
        let mut r = good_row();
        r.as_object_mut().unwrap().remove("id");
        let err = validate_ps_row(&r).unwrap_err();
        assert_eq!(err, PsSchemaError::MissingField("id".into()));
    }

    #[test]
    fn missing_size_bytes_is_error() {
        let mut r = good_row();
        r.as_object_mut().unwrap().remove("size_bytes");
        let err = validate_ps_row(&r).unwrap_err();
        assert_eq!(err, PsSchemaError::MissingField("size_bytes".into()));
    }

    #[test]
    fn missing_processor_is_error() {
        let mut r = good_row();
        r.as_object_mut().unwrap().remove("processor");
        let err = validate_ps_row(&r).unwrap_err();
        assert_eq!(err, PsSchemaError::MissingField("processor".into()));
    }

    #[test]
    fn missing_until_is_error() {
        let mut r = good_row();
        r.as_object_mut().unwrap().remove("until");
        let err = validate_ps_row(&r).unwrap_err();
        assert_eq!(err, PsSchemaError::MissingField("until".into()));
    }

    #[test]
    fn size_bytes_zero_is_error() {
        let mut r = good_row();
        r["size_bytes"] = json!(0u64);
        let err = validate_ps_row(&r).unwrap_err();
        assert!(matches!(err, PsSchemaError::WrongType { .. }));
    }

    #[test]
    fn size_bytes_string_is_error() {
        let mut r = good_row();
        r["size_bytes"] = json!("not-a-number");
        let err = validate_ps_row(&r).unwrap_err();
        assert!(matches!(err, PsSchemaError::WrongType { .. }));
    }

    #[test]
    fn short_id_is_error() {
        let mut r = good_row();
        r["id"] = json!("abc");
        let err = validate_ps_row(&r).unwrap_err();
        assert!(matches!(err, PsSchemaError::IdTooShort(_)));
    }

    #[test]
    fn non_hex_id_is_error() {
        let mut r = good_row();
        r["id"] = json!("zzzzzzzzzzzz"); // right length, not hex
        let err = validate_ps_row(&r).unwrap_err();
        assert!(matches!(err, PsSchemaError::IdTooShort(_)));
    }

    #[test]
    fn processor_100_gpu_accepted() {
        assert!(is_valid_processor("100% GPU"));
        assert!(is_valid_processor("100% gpu"));
    }

    #[test]
    fn processor_100_cpu_accepted() {
        assert!(is_valid_processor("100% CPU"));
    }

    #[test]
    fn processor_split_accepted() {
        assert!(is_valid_processor("70%/30% CPU/GPU"));
        assert!(is_valid_processor("50%/50% cpu/gpu"));
        assert!(is_valid_processor("0%/100% CPU/GPU"));
    }

    #[test]
    fn processor_bad_sum_rejected() {
        // 70+20 != 100
        assert!(!is_valid_processor("70%/20% CPU/GPU"));
    }

    #[test]
    fn processor_garbage_rejected() {
        assert!(!is_valid_processor("somewhere on the GPU"));
        assert!(!is_valid_processor(""));
        assert!(!is_valid_processor("50% GPU"));
    }

    #[test]
    fn until_rfc3339_z_accepted() {
        assert!(is_rfc3339("2026-04-20T12:34:56Z"));
        assert!(is_rfc3339("2026-04-20T12:34:56.123Z"));
        assert!(is_rfc3339("2026-04-20T12:34:56.123456Z"));
    }

    #[test]
    fn until_rfc3339_offset_accepted() {
        assert!(is_rfc3339("2026-04-20T12:34:56+00:00"));
        assert!(is_rfc3339("2026-04-20T12:34:56-08:00"));
        assert!(is_rfc3339("2026-04-20T12:34:56.5+09:00"));
    }

    #[test]
    fn until_not_rfc3339_rejected() {
        assert!(!is_rfc3339("2026-04-20"));
        assert!(!is_rfc3339("not a timestamp"));
        assert!(!is_rfc3339(""));
        assert!(!is_rfc3339("2026-04-20T12:34:56")); // no Z / offset
    }

    #[test]
    fn bad_until_propagates_row_error() {
        let mut r = good_row();
        r["until"] = json!("next tuesday");
        let err = validate_ps_row(&r).unwrap_err();
        assert!(matches!(err, PsSchemaError::UntilNotRfc3339(_)));
    }

    #[test]
    fn validate_rejects_non_array_top_level() {
        let err = validate_ps_array(&json!({})).unwrap_err();
        assert_eq!(err, PsSchemaError::NotAnArray);
    }

    #[test]
    fn validate_rejects_row_that_is_not_object() {
        let err = validate_ps_array(&json!([42])).unwrap_err();
        assert_eq!(err, PsSchemaError::NotAnObject);
    }

    #[test]
    fn header_missing_name_rejected() {
        let err = header_has_required_columns("ID SIZE PROCESSOR UNTIL").unwrap_err();
        assert_eq!(err, PsSchemaError::MissingColumn("NAME".into()));
    }

    #[test]
    fn header_missing_size_rejected() {
        let err = header_has_required_columns("NAME ID PROCESSOR").unwrap_err();
        assert_eq!(err, PsSchemaError::MissingColumn("SIZE".into()));
    }

    #[test]
    fn header_missing_processor_rejected() {
        let err = header_has_required_columns("NAME ID SIZE UNTIL").unwrap_err();
        assert_eq!(err, PsSchemaError::MissingColumn("PROCESSOR".into()));
    }

    #[test]
    fn header_named_not_a_name_match() {
        // "NAMED" is not "NAME" — whole-token match only.
        let err = header_has_required_columns("NAMED SIZE PROCESSOR").unwrap_err();
        assert_eq!(err, PsSchemaError::MissingColumn("NAME".into()));
    }

    #[test]
    fn header_case_matters() {
        // Columns are ALL-CAPS by convention — lowercase rejected
        // (matches ollama exactly).
        let err = header_has_required_columns("name size processor").unwrap_err();
        assert!(matches!(err, PsSchemaError::MissingColumn(_)));
    }

    #[test]
    fn empty_array_passes_validation() {
        // An empty `apr ps` response is trivially valid.
        assert!(validate_ps_array(&json!([])).is_ok());
    }

    #[test]
    fn two_row_array_validates_each_row() {
        let arr = json!([good_row(), good_row()]);
        assert!(validate_ps_array(&arr).is_ok());
    }

    #[test]
    fn second_row_malformed_is_detected() {
        let mut bad = good_row();
        bad["size_bytes"] = json!(0u64);
        let arr = json!([good_row(), bad]);
        assert!(validate_ps_array(&arr).is_err());
    }

    #[test]
    fn is_empty_ps_rejects_non_array_types() {
        // If the daemon ever emits a non-array payload, this is a
        // contract violation — `is_empty_ps` must NOT silently return
        // true for `null` or `{}`.
        assert!(!is_empty_ps(&Value::Null));
        assert!(!is_empty_ps(&json!({})));
        assert!(!is_empty_ps(&json!(0)));
    }
}
