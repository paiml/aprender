//! Modelfile DSL parser (CRUX-K-11).

use super::{MessageEntry, ModelfileConfig};
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

/// Parser error with file:line:col location for FALSIFY-CRUX-K-11-003.
#[derive(Debug)]
pub struct ModelfileError {
    /// Source file path (or `<inline>` when parsing from a string buffer).
    pub source: PathBuf,
    /// 1-indexed line number where the offending token starts.
    pub line: usize,
    /// 1-indexed column number where the offending token starts.
    pub col: usize,
    /// Human-readable explanation.
    pub message: String,
}

impl std::fmt::Display for ModelfileError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{}:{}:{}: {}",
            self.source.display(),
            self.line,
            self.col,
            self.message
        )
    }
}

impl std::error::Error for ModelfileError {}

const KNOWN_DIRECTIVES: &[&str] = &[
    "FROM",
    "PARAMETER",
    "TEMPLATE",
    "SYSTEM",
    "LICENSE",
    "MESSAGE",
    "ADAPTER",
];

/// Parse a Modelfile from disk.
///
/// # Errors
///
/// See [`parse_modelfile_str`].
pub fn parse_modelfile(path: &Path) -> Result<ModelfileConfig, ModelfileError> {
    let text = std::fs::read_to_string(path).map_err(|e| ModelfileError {
        source: path.to_path_buf(),
        line: 0,
        col: 0,
        message: format!("read failed: {e}"),
    })?;
    parse_modelfile_str(&text, path)
}

/// Parse Modelfile text. `source_path` is used only for error reporting.
///
/// # Errors
///
/// Returns `ModelfileError` when:
/// - FROM directive is missing
/// - An unknown directive appears
/// - A triple-quoted block is not terminated
/// - A directive has no value (e.g. bare `FROM`)
pub fn parse_modelfile_str(
    text: &str,
    source_path: &Path,
) -> Result<ModelfileConfig, ModelfileError> {
    let mut config = ModelfileConfig {
        from: String::new(),
        parameters: BTreeMap::new(),
        template: None,
        system: None,
        license: None,
        messages: Vec::new(),
        adapter: None,
    };

    let lines: Vec<&str> = text.lines().collect();
    let mut i: usize = 0;
    while i < lines.len() {
        let raw_line = lines[i];
        let trimmed = raw_line.trim_start();
        let leading_ws = raw_line.len() - trimmed.len();

        // Skip blank lines and full-line comments.
        if trimmed.is_empty() || trimmed.starts_with('#') {
            i += 1;
            continue;
        }

        // Split directive | value at first whitespace.
        let (directive_raw, rest_after_directive) = match trimmed.split_once(char::is_whitespace) {
            Some((d, r)) => (d, r.trim_start()),
            None => (trimmed, ""),
        };

        let directive_upper = directive_raw.to_uppercase();
        let directive_col = leading_ws + 1;

        if !KNOWN_DIRECTIVES.contains(&directive_upper.as_str()) {
            return Err(ModelfileError {
                source: source_path.to_path_buf(),
                line: i + 1,
                col: directive_col,
                message: format!(
                    "unknown directive `{directive_raw}`; expected one of {}",
                    KNOWN_DIRECTIVES.join(", ")
                ),
            });
        }

        // Capture the value — triple-quoted block or single-line.
        let (value, consumed) = if rest_after_directive.starts_with("\"\"\"") {
            read_triple_quoted_block(&lines, i, rest_after_directive, source_path)?
        } else {
            (rest_after_directive.to_string(), 0)
        };
        let trimmed_value = value.trim().to_string();

        match directive_upper.as_str() {
            "FROM" => {
                if trimmed_value.is_empty() {
                    return Err(ModelfileError {
                        source: source_path.to_path_buf(),
                        line: i + 1,
                        col: directive_col,
                        message: "FROM directive requires a value".into(),
                    });
                }
                config.from = trimmed_value;
            }
            "PARAMETER" => {
                let (k, v) = parse_parameter_kv(&trimmed_value).ok_or_else(|| ModelfileError {
                    source: source_path.to_path_buf(),
                    line: i + 1,
                    col: directive_col,
                    message: "PARAMETER requires `<name> <value>`".into(),
                })?;
                config.parameters.insert(k, v);
            }
            "TEMPLATE" => {
                config.template = Some(trimmed_value);
            }
            "SYSTEM" => {
                config.system = Some(trimmed_value);
            }
            "LICENSE" => {
                config.license = Some(trimmed_value);
            }
            "MESSAGE" => {
                let (role, content) =
                    parse_message_value(&trimmed_value).ok_or_else(|| ModelfileError {
                        source: source_path.to_path_buf(),
                        line: i + 1,
                        col: directive_col,
                        message: "MESSAGE requires `<role> <content>`".into(),
                    })?;
                config.messages.push(MessageEntry { role, content });
            }
            "ADAPTER" => {
                config.adapter = Some(trimmed_value);
            }
            _ => unreachable!("KNOWN_DIRECTIVES guard"),
        }

        i += 1 + consumed;
    }

    if config.from.is_empty() {
        return Err(ModelfileError {
            source: source_path.to_path_buf(),
            line: 1,
            col: 1,
            message: "FROM directive is required but was not found".into(),
        });
    }

    Ok(config)
}

/// Read a triple-quoted `"""..."""` block. Returns the inner content (without
/// the wrapping quotes) plus the number of EXTRA lines consumed beyond the
/// directive line.
fn read_triple_quoted_block(
    lines: &[&str],
    start_line: usize,
    rest_after_directive: &str,
    source_path: &Path,
) -> Result<(String, usize), ModelfileError> {
    let after_open = rest_after_directive.strip_prefix("\"\"\"").unwrap_or("");

    // Inline triple-quoted on the same line: `SYSTEM """text"""`.
    if let Some(close_idx) = after_open.find("\"\"\"") {
        return Ok((after_open[..close_idx].to_string(), 0));
    }

    // Multi-line: accumulate from next line until we find the closing `"""`.
    let mut buf = String::new();
    if !after_open.is_empty() {
        buf.push_str(after_open);
        buf.push('\n');
    }
    let mut extra: usize = 0;
    for (offset, line) in lines.iter().enumerate().skip(start_line + 1) {
        extra = offset - start_line;
        if let Some(close_idx) = line.find("\"\"\"") {
            buf.push_str(&line[..close_idx]);
            return Ok((buf, extra));
        }
        buf.push_str(line);
        buf.push('\n');
    }

    Err(ModelfileError {
        source: source_path.to_path_buf(),
        line: start_line + 1,
        col: 1,
        message: "unterminated triple-quoted block (missing closing `\"\"\"`)".into(),
    })
}

fn parse_parameter_kv(value: &str) -> Option<(String, serde_json::Value)> {
    let (name, raw) = value.split_once(char::is_whitespace)?;
    let name = name.trim().to_string();
    let raw = raw.trim();
    if name.is_empty() {
        return None;
    }
    // Type the value: numeric → Number; "true"/"false" → Bool; else String.
    let v: serde_json::Value = if let Ok(i) = raw.parse::<i64>() {
        serde_json::Value::from(i)
    } else if let Ok(f) = raw.parse::<f64>() {
        serde_json::Value::from(f)
    } else if raw.eq_ignore_ascii_case("true") {
        serde_json::Value::Bool(true)
    } else if raw.eq_ignore_ascii_case("false") {
        serde_json::Value::Bool(false)
    } else {
        // Strip optional surrounding quotes on string values.
        let stripped = raw.trim_matches('"').to_string();
        serde_json::Value::String(stripped)
    };
    Some((name, v))
}

fn parse_message_value(value: &str) -> Option<(String, String)> {
    let (role, content) = value.split_once(char::is_whitespace)?;
    let role = role.trim().to_string();
    let content = content.trim().to_string();
    if role.is_empty() || content.is_empty() {
        return None;
    }
    Some((role, content))
}
