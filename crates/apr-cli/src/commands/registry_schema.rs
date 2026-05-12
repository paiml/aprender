//! Pure JSON-schema validators + name resolver for the local model
//! registry (`apr ls` / `apr show` / `apr rm`) under CRUX-A-09.
//!
//! Contract: `contracts/crux-A-09-v1.yaml`.
//!
//! Pure classifier — takes already-parsed `serde_json::Value`s and name
//! lookups and returns `Result<(), String>` or `Result<&str, ...>`.
//! Zero I/O, no filesystem, no global state. The integration-level
//! claims ("`apr ls --json` actually emits one of these", "`apr rm`
//! atomically removes files") are discharged by separate
//! registry-backed harnesses (follow-up).

use serde_json::Value;

/// The required top-level fields of an `apr ls --json` array element.
/// Matches CRUX-A-09 `list_json_schema` formula.
pub const LS_REQUIRED_FIELDS: &[&str] = &["name", "size_bytes", "sha256", "quant"];

/// The required top-level fields of an `apr show NAME --json` object.
/// Matches CRUX-A-09 `show_json_schema` formula.
pub const SHOW_REQUIRED_FIELDS: &[&str] = &["arch", "params", "tensor_histogram", "size_bytes"];

/// Return Ok(()) iff `v` is a single `apr ls` entry with well-typed
/// required fields. Error string names the offending field for
/// operator-visible diagnostics.
///
/// CRUX-A-09 ALGO-001 sub-claim of FALSIFY-001: the schema validator
/// rejects malformed entries (missing name, non-hex sha256, wrong
/// types) so the integration-level `apr ls --json` output can be
/// mechanically validated against the same predicate.
pub fn validate_ls_entry(v: &Value) -> Result<(), String> {
    let obj = v
        .as_object()
        .ok_or_else(|| "ls entry must be a JSON object".to_string())?;

    for field in LS_REQUIRED_FIELDS {
        if !obj.contains_key(*field) {
            return Err(format!("ls entry missing required field '{field}'"));
        }
    }

    match obj.get("name") {
        Some(Value::String(s)) if !s.is_empty() => {}
        _ => return Err("ls entry 'name' must be a non-empty string".to_string()),
    }

    match obj.get("size_bytes") {
        Some(Value::Number(n)) if n.as_u64().is_some() => {}
        _ => return Err("ls entry 'size_bytes' must be a u64 number".to_string()),
    }

    match obj.get("sha256") {
        Some(Value::String(s)) if is_hex64(s) => {}
        _ => return Err("ls entry 'sha256' must be a 64-char lowercase hex string".to_string()),
    }

    match obj.get("quant") {
        Some(Value::String(s)) if !s.is_empty() => {}
        _ => return Err("ls entry 'quant' must be a non-empty string".to_string()),
    }

    Ok(())
}

/// Return Ok(()) iff `v` is a JSON array of valid `apr ls` entries.
/// CRUX-A-09 ALGO-001 sub-claim of FALSIFY-001.
pub fn validate_ls_array(v: &Value) -> Result<(), String> {
    let arr = v
        .as_array()
        .ok_or_else(|| "apr ls --json output must be a JSON array".to_string())?;
    for (i, entry) in arr.iter().enumerate() {
        validate_ls_entry(entry).map_err(|e| format!("ls[{i}]: {e}"))?;
    }
    Ok(())
}

/// Return Ok(()) iff `v` is a valid `apr show NAME --json` object.
/// CRUX-A-09 ALGO-002 sub-claim of FALSIFY-002.
pub fn validate_show_object(v: &Value) -> Result<(), String> {
    let obj = v
        .as_object()
        .ok_or_else(|| "apr show --json output must be a JSON object".to_string())?;

    for field in SHOW_REQUIRED_FIELDS {
        if !obj.contains_key(*field) {
            return Err(format!("show output missing required field '{field}'"));
        }
    }

    match obj.get("arch") {
        Some(Value::String(s)) if !s.is_empty() => {}
        _ => return Err("show 'arch' must be a non-empty string".to_string()),
    }

    match obj.get("params") {
        Some(Value::Number(n)) if n.as_u64().map(|x| x > 0).unwrap_or(false) => {}
        _ => return Err("show 'params' must be a u64 > 0".to_string()),
    }

    match obj.get("tensor_histogram") {
        Some(Value::Object(_)) => {}
        _ => return Err("show 'tensor_histogram' must be a JSON object".to_string()),
    }

    match obj.get("size_bytes") {
        Some(Value::Number(n)) if n.as_u64().map(|x| x > 0).unwrap_or(false) => {}
        _ => return Err("show 'size_bytes' must be a u64 > 0".to_string()),
    }

    Ok(())
}

/// Error variants returned by `resolve_name` when a registry lookup
/// fails. Kept small and operator-readable.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ResolveError {
    /// Name did not match any registry entry.
    NotFound(String),
}

impl std::fmt::Display for ResolveError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ResolveError::NotFound(name) => {
                write!(f, "model '{name}' not found in local registry")
            }
        }
    }
}

/// Look up `name` in the provided registry entry list. Returns the
/// matching entry on success. Pure — caller supplies the slice so the
/// function is filesystem-free.
///
/// CRUX-A-09 ALGO-005 sub-claim of FALSIFY-005: unknown names MUST
/// produce a non-Ok result whose message contains "not found", which is
/// the algorithm-level precondition for the integration-level
/// `apr show does-not-exist` exits non-zero with an explanatory error.
pub fn resolve_name<'a, S: AsRef<str>>(
    name: &str,
    entries: &'a [S],
) -> Result<&'a str, ResolveError> {
    for e in entries {
        if e.as_ref() == name {
            return Ok(e.as_ref());
        }
    }
    Err(ResolveError::NotFound(name.to_string()))
}

/// Render the `apr rm NAME --dry-run` plan line for a single registry
/// entry. CRUX-A-09 ALGO-004 sub-claim of FALSIFY-004: the dry-run
/// output MUST contain a "would remove" / "would delete" / "dry run"
/// keyword so automation can assert dry-run-only semantics.
pub fn rm_dry_run_plan_line(name: &str) -> String {
    format!("would remove model '{name}' (dry-run: no changes applied)")
}

fn is_hex64(s: &str) -> bool {
    s.len() == 64
        && s.chars()
            .all(|c| c.is_ascii_hexdigit() && !c.is_uppercase())
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn good_entry() -> Value {
        json!({
            "name": "qwen2.5-coder-7b-apache-q4k",
            "size_bytes": 4_536_000_000u64,
            "sha256": "a".repeat(64),
            "quant": "q4_k_m",
        })
    }

    #[test]
    fn ls_entry_accepts_good_entry() {
        assert!(validate_ls_entry(&good_entry()).is_ok());
    }

    #[test]
    fn ls_entry_rejects_missing_name() {
        let mut v = good_entry();
        v.as_object_mut().unwrap().remove("name");
        let err = validate_ls_entry(&v).unwrap_err();
        assert!(err.contains("name"));
    }

    #[test]
    fn ls_entry_rejects_missing_sha256() {
        let mut v = good_entry();
        v.as_object_mut().unwrap().remove("sha256");
        let err = validate_ls_entry(&v).unwrap_err();
        assert!(err.contains("sha256"));
    }

    #[test]
    fn ls_entry_rejects_non_hex_sha256() {
        let mut v = good_entry();
        v["sha256"] = json!("not-a-hex-string");
        assert!(validate_ls_entry(&v).is_err());
    }

    #[test]
    fn ls_entry_rejects_uppercase_sha256() {
        // Lowercase-only matches `sha256sum` canonical output.
        let mut v = good_entry();
        v["sha256"] = json!("A".repeat(64));
        assert!(validate_ls_entry(&v).is_err());
    }

    #[test]
    fn ls_entry_rejects_empty_name() {
        let mut v = good_entry();
        v["name"] = json!("");
        assert!(validate_ls_entry(&v).is_err());
    }

    #[test]
    fn ls_entry_rejects_non_object() {
        assert!(validate_ls_entry(&json!("a string")).is_err());
        assert!(validate_ls_entry(&json!([])).is_err());
    }

    #[test]
    fn ls_array_accepts_empty_array() {
        // Empty registry is a legitimate state.
        assert!(validate_ls_array(&json!([])).is_ok());
    }

    #[test]
    fn ls_array_accepts_array_of_good_entries() {
        let arr = json!([good_entry(), good_entry()]);
        assert!(validate_ls_array(&arr).is_ok());
    }

    #[test]
    fn ls_array_rejects_non_array() {
        assert!(validate_ls_array(&json!({"not": "an array"})).is_err());
    }

    #[test]
    fn ls_array_error_names_offending_index() {
        let bad = json!([
            good_entry(),
            {"name": "", "size_bytes": 1, "sha256": "a".repeat(64), "quant": "f16"},
        ]);
        let err = validate_ls_array(&bad).unwrap_err();
        assert!(err.starts_with("ls[1]"), "unexpected: {err}");
    }

    fn good_show() -> Value {
        json!({
            "arch": "qwen2",
            "params": 7_000_000_000u64,
            "tensor_histogram": {"q4_k": 290, "q6_k": 1},
            "size_bytes": 4_536_000_000u64,
        })
    }

    #[test]
    fn show_object_accepts_good_object() {
        assert!(validate_show_object(&good_show()).is_ok());
    }

    #[test]
    fn show_object_rejects_missing_arch() {
        let mut v = good_show();
        v.as_object_mut().unwrap().remove("arch");
        assert!(validate_show_object(&v).is_err());
    }

    #[test]
    fn show_object_rejects_zero_params() {
        let mut v = good_show();
        v["params"] = json!(0);
        assert!(validate_show_object(&v).is_err());
    }

    #[test]
    fn show_object_rejects_non_object_histogram() {
        let mut v = good_show();
        v["tensor_histogram"] = json!([1, 2, 3]);
        assert!(validate_show_object(&v).is_err());
    }

    #[test]
    fn show_object_accepts_empty_histogram() {
        // Empty histogram is legal: pre-import model with zero tensors.
        let mut v = good_show();
        v["tensor_histogram"] = json!({});
        assert!(validate_show_object(&v).is_ok());
    }

    #[test]
    fn resolve_name_ok_on_hit() {
        let entries = vec!["a".to_string(), "b".to_string()];
        assert_eq!(resolve_name("b", &entries).unwrap(), "b");
    }

    #[test]
    fn resolve_name_err_on_miss() {
        let entries = vec!["a".to_string()];
        let err = resolve_name("zzz", &entries).unwrap_err();
        match err {
            ResolveError::NotFound(n) => assert_eq!(n, "zzz"),
        }
    }

    #[test]
    fn resolve_name_err_message_contains_not_found() {
        // CRUX-A-09 ALGO-005 sub-claim of FALSIFY-005: integration
        // test greps for "not found" in stderr; the Display impl is the
        // algorithm-level guarantor of that substring.
        let entries: Vec<String> = vec![];
        let err = resolve_name("nope", &entries).unwrap_err();
        let msg = err.to_string();
        assert!(
            msg.to_lowercase().contains("not found"),
            "unexpected message: {msg}",
        );
    }

    #[test]
    fn resolve_name_err_message_names_missing_model() {
        let entries: Vec<String> = vec![];
        let msg = resolve_name("xyz-model", &entries).unwrap_err().to_string();
        assert!(
            msg.contains("xyz-model"),
            "message should name missing model: {msg}"
        );
    }

    #[test]
    fn resolve_name_on_empty_registry_errs() {
        let entries: Vec<String> = vec![];
        assert!(resolve_name("anything", &entries).is_err());
    }

    #[test]
    fn resolve_name_is_deterministic() {
        let entries = vec!["a".to_string(), "b".to_string()];
        let a = resolve_name("a", &entries).unwrap();
        let b = resolve_name("a", &entries).unwrap();
        assert_eq!(a, b);
    }

    #[test]
    fn rm_dry_run_plan_line_contains_would_remove() {
        // CRUX-A-09 ALGO-004 sub-claim of FALSIFY-004: integration
        // test greps `would (remove|delete)|dry.?run` — the classifier
        // guarantees the "would remove" + "dry-run" substrings are
        // present in the line we emit for each candidate entry.
        let line = rm_dry_run_plan_line("some-model");
        assert!(line.to_lowercase().contains("would remove"));
        assert!(line.to_lowercase().contains("dry-run"));
        assert!(line.contains("some-model"));
    }

    #[test]
    fn rm_dry_run_plan_line_is_deterministic() {
        let a = rm_dry_run_plan_line("m");
        let b = rm_dry_run_plan_line("m");
        assert_eq!(a, b);
    }

    #[test]
    fn ls_required_fields_stable() {
        // Downstream jq expressions depend on these exact names.
        assert_eq!(
            LS_REQUIRED_FIELDS,
            &["name", "size_bytes", "sha256", "quant"]
        );
    }

    #[test]
    fn show_required_fields_stable() {
        assert_eq!(
            SHOW_REQUIRED_FIELDS,
            &["arch", "params", "tensor_histogram", "size_bytes"]
        );
    }

    #[test]
    fn is_hex64_accepts_canonical() {
        assert!(is_hex64(&"0".repeat(64)));
        assert!(is_hex64(&"a".repeat(64)));
        assert!(is_hex64(&"f".repeat(64)));
    }

    #[test]
    fn is_hex64_rejects_wrong_length() {
        assert!(!is_hex64(&"a".repeat(63)));
        assert!(!is_hex64(&"a".repeat(65)));
        assert!(!is_hex64(""));
    }

    #[test]
    fn is_hex64_rejects_non_hex_chars() {
        // 64 chars total, but 'z' is not hex.
        let mut s = "a".repeat(63);
        s.push('z');
        assert!(!is_hex64(&s));
    }

    #[test]
    fn is_hex64_rejects_uppercase() {
        assert!(!is_hex64(&"A".repeat(64)));
    }
}
