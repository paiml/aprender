//! Simple JSON field extraction for benchmark results.

/// Find the raw value substring after a JSON key.
pub(crate) fn json_value_after_key<'a>(json: &'a str, key: &str) -> Option<&'a str> {
    let pattern = format!(r#""{}":"#, key);
    let start = json.find(&pattern)? + pattern.len();
    Some(&json[start..])
}

/// Extract string value from simple JSON.
pub(crate) fn extract_json_string(json: &str, key: &str) -> Option<String> {
    let rest = json_value_after_key(json, key)?;
    let unquoted = rest.strip_prefix('"')?;
    let end = unquoted.find('"')?;
    Some(unquoted[..end].to_string())
}

/// Extract number value from simple JSON.
pub(crate) fn extract_json_number(json: &str, key: &str) -> Option<f64> {
    let rest = json_value_after_key(json, key)?;
    let end = rest.find([',', '}']).unwrap_or(rest.len());
    rest[..end].trim().parse().ok()
}
