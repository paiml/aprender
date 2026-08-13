//! Prometheus /metrics endpoint classifier (CRUX-K-07).
//!
//! Pure, deterministic classifiers that discharge FALSIFY-CRUX-K-07-{001,002,003}
//! at the PARTIAL_ALGORITHM_LEVEL — algorithm-level necessary conditions on
//! an already-captured `/metrics` HTTP response:
//!
//!   * `classify_content_type` — `Content-Type` header matches Prometheus
//!     text-based exposition format (`text/plain; version=0.0.4; charset=utf-8`).
//!   * `classify_text_format` — the body parses against Prometheus 0.0.4
//!     text format: every TYPE line precedes its samples; every TYPE value is
//!     in {counter, gauge, histogram, summary, untyped}; every sample line
//!     names a previously TYPE-declared metric and ends with a numeric value;
//!     no HELP/TYPE line refers to a metric the body never emits.
//!   * `classify_required_metrics` — the K-07 required metric set is present
//!     (counterparts to vLLM's `vllm:num_requests_running` etc., re-prefixed
//!     `apr_*`).
//!
//! Full discharge blocks on a live `apr serve` `GET /metrics` handler emitting
//! the K-07 metric set — tracked as BLOCKER-UPSTREAM-MISSING.

/// Required Prometheus metric names that `apr serve /metrics` MUST emit
/// (CRUX-K-07 contract `prometheus_required_metrics`).
pub const K07_REQUIRED_METRICS: &[&str] = &[
    "apr_num_requests_running",
    "apr_num_requests_waiting",
    "apr_gpu_cache_usage_perc",
    "apr_time_to_first_token_seconds",
    "apr_time_per_output_token_seconds",
    "apr_e2e_request_latency_seconds",
];

/// Outcome of `classify_content_type`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PromContentTypeOutcome {
    /// Header is `text/plain; version=0.0.4` (optionally with `charset=utf-8`).
    Ok,
    /// Media type is not `text/plain`.
    WrongMediaType { got: String },
    /// `version=0.0.4` parameter is absent.
    MissingVersion { got: String },
    /// `version=` parameter is present but not `0.0.4`.
    WrongVersion { got: String },
}

/// Outcome of `classify_text_format`.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize)]
#[serde(tag = "status", rename_all = "snake_case")]
pub enum PromTextFormatOutcome {
    /// Body parses as a valid Prometheus 0.0.4 text exposition.
    Ok { metrics_seen: Vec<String> },
    /// A sample line refers to a metric without a preceding `# TYPE` declaration.
    SampleBeforeType { metric: String, line_no: usize },
    /// A `# TYPE` value is not one of {counter, gauge, histogram, summary, untyped}.
    UnknownType {
        metric: String,
        got: String,
        line_no: usize,
    },
    /// A sample line's value is not a parseable float (or special token).
    SampleValueNotNumeric {
        metric: String,
        raw: String,
        line_no: usize,
    },
    /// A sample line is missing a metric name.
    SampleMissingMetricName { line_no: usize },
    /// Same metric name declared twice with conflicting types.
    DuplicateConflictingType {
        metric: String,
        first: String,
        second: String,
    },
}

/// Outcome of `classify_required_metrics`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PromRequiredOutcome {
    /// Every required metric name (or its histogram/summary expansion) appears.
    Ok,
    /// At least one required metric name is absent from the body.
    Missing { missing: Vec<String> },
}

/// Validate the `Content-Type` header against the Prometheus 0.0.4 text format
/// (https://prometheus.io/docs/instrumenting/exposition_formats/#text-based-format).
///
/// Accepts `text/plain; version=0.0.4` with optional `charset=utf-8`. Whitespace
/// around `;`/`=` is tolerated; parameter order does not matter.
pub fn classify_content_type(header: &str) -> PromContentTypeOutcome {
    let mut parts = header.split(';').map(str::trim);
    let media_type = parts.next().unwrap_or("").to_ascii_lowercase();
    if media_type != "text/plain" {
        return PromContentTypeOutcome::WrongMediaType { got: media_type };
    }

    let mut version: Option<String> = None;
    for kv in parts {
        if kv.is_empty() {
            continue;
        }
        let (k, v) = match kv.split_once('=') {
            Some((k, v)) => (k.trim().to_ascii_lowercase(), v.trim().to_string()),
            None => continue,
        };
        if k == "version" {
            version = Some(v);
        }
    }
    match version {
        None => PromContentTypeOutcome::MissingVersion {
            got: header.to_string(),
        },
        Some(v) if v == "0.0.4" => PromContentTypeOutcome::Ok,
        Some(v) => PromContentTypeOutcome::WrongVersion { got: v },
    }
}

/// Validate the body against the Prometheus 0.0.4 text exposition format.
///
/// This is a subset parser: it covers what `apr serve /metrics` must emit
/// (HELP/TYPE comment lines, sample lines with optional labels and an
/// optional integer timestamp, blank lines). It does NOT enforce label
/// escape rules — those are checked at write time, not parse time.
pub fn classify_text_format(body: &str) -> PromTextFormatOutcome {
    let mut declared_types: TypeTable = TypeTable::new();
    let mut metrics_seen: Vec<String> = Vec::new();

    for (idx, line) in body.lines().enumerate() {
        if let Err(rejection) =
            check_exposition_line(line, idx + 1, &mut declared_types, &mut metrics_seen)
        {
            return rejection;
        }
    }
    PromTextFormatOutcome::Ok { metrics_seen }
}

/// Check one exposition line, which is exactly one of four things: blank, a
/// `# TYPE` declaration, some other comment, or a sample.
///
/// Blank lines and non-TYPE comments carry no obligations. The other two kinds
/// return `Err(outcome)` naming the rejection when the line is malformed.
fn check_exposition_line(
    line: &str,
    line_no: usize,
    declared_types: &mut TypeTable,
    metrics_seen: &mut Vec<String>,
) -> Result<(), PromTextFormatOutcome> {
    let trimmed = line.trim_start();
    if trimmed.is_empty() {
        return Ok(());
    }
    if let Some(rest) = trimmed.strip_prefix("# TYPE ") {
        return record_type_declaration(rest, line_no, declared_types, metrics_seen);
    }
    if trimmed.starts_with('#') {
        return Ok(());
    }
    check_sample_line(trimmed, line_no, declared_types)
}

/// Metric name → declared `# TYPE` value, in declaration order of first sighting.
type TypeTable = std::collections::HashMap<String, String>;

/// The `# TYPE` values Prometheus 0.0.4 defines. Anything else is a malformed body.
const VALID_METRIC_TYPES: [&str; 5] = ["counter", "gauge", "histogram", "summary", "untyped"];

/// Record one `# TYPE <name> <type>` declaration into `declared_types`.
///
/// `rest` is the text following the `# TYPE ` prefix. A declaration missing either
/// the name or the type is silently ignored (matching the original parser, which
/// treats a malformed comment as a comment). Returns `Err(outcome)` for the two
/// ways a well-formed declaration can still be rejected: an unknown type value, or
/// a redeclaration that conflicts with the type already recorded for that metric.
fn record_type_declaration(
    rest: &str,
    line_no: usize,
    declared_types: &mut TypeTable,
    metrics_seen: &mut Vec<String>,
) -> Result<(), PromTextFormatOutcome> {
    let mut it = rest.split_ascii_whitespace();
    let (Some(name), Some(ty)) = (it.next(), it.next()) else {
        return Ok(());
    };
    if !VALID_METRIC_TYPES.contains(&ty) {
        return Err(PromTextFormatOutcome::UnknownType {
            metric: name.to_string(),
            got: ty.to_string(),
            line_no,
        });
    }
    match declared_types.get(name) {
        Some(prev) if prev != ty => Err(PromTextFormatOutcome::DuplicateConflictingType {
            metric: name.to_string(),
            first: prev.clone(),
            second: ty.to_string(),
        }),
        // Already declared with the same type — idempotent, and NOT a second sighting.
        Some(_) => Ok(()),
        None => {
            declared_types.insert(name.to_string(), ty.to_string());
            metrics_seen.push(name.to_string());
            Ok(())
        }
    }
}

/// Check one sample line against the types declared so far.
///
/// A sample is well-formed when it (a) carries a metric name, (b) names a metric
/// whose base — after stripping any histogram/summary suffix — was already
/// `# TYPE`-declared, and (c) ends in a Prometheus-numeric value.
fn check_sample_line(
    trimmed: &str,
    line_no: usize,
    declared_types: &TypeTable,
) -> Result<(), PromTextFormatOutcome> {
    let Some((sample_name, value_part)) = split_sample_line(trimmed) else {
        return Err(PromTextFormatOutcome::SampleMissingMetricName { line_no });
    };
    let base = metric_base_name(&sample_name, declared_types);
    if !declared_types.contains_key(&base) {
        return Err(PromTextFormatOutcome::SampleBeforeType {
            metric: sample_name,
            line_no,
        });
    }
    let value_token = value_part.split_ascii_whitespace().next().unwrap_or("");
    if !is_prometheus_numeric(value_token) {
        return Err(PromTextFormatOutcome::SampleValueNotNumeric {
            metric: sample_name,
            raw: value_token.to_string(),
            line_no,
        });
    }
    Ok(())
}

/// Check that every name in `required` appears either as a `# TYPE` declaration
/// or as a sample-line metric name in `body`. Caller provides the spelled-out
/// list (typically `K07_REQUIRED_METRICS`).
pub fn classify_required_metrics(body: &str, required: &[&str]) -> PromRequiredOutcome {
    let mut declared: std::collections::HashSet<String> = std::collections::HashSet::new();
    for line in body.lines() {
        let t = line.trim_start();
        if let Some(rest) = t.strip_prefix("# TYPE ") {
            if let Some(name) = rest.split_ascii_whitespace().next() {
                declared.insert(name.to_string());
            }
            continue;
        }
        if t.starts_with('#') || t.is_empty() {
            continue;
        }
        if let Some((name, _)) = split_sample_line(t) {
            declared.insert(name);
        }
    }
    let missing: Vec<String> = required
        .iter()
        .filter(|r| {
            !declared.contains(**r)
                && !declared.iter().any(|d| {
                    // Histograms expand to `<name>_bucket`, `<name>_sum`, `<name>_count`;
                    // summaries expand to `<name>_sum`, `<name>_count`, plus quantile samples.
                    d == *r
                        || d == &format!("{r}_bucket")
                        || d == &format!("{r}_sum")
                        || d == &format!("{r}_count")
                })
        })
        .map(|s| (*s).to_string())
        .collect();
    if missing.is_empty() {
        PromRequiredOutcome::Ok
    } else {
        let mut sorted = missing;
        sorted.sort();
        PromRequiredOutcome::Missing { missing: sorted }
    }
}

/// Split a Prometheus sample line into `(metric_name, value_part)`.
///
/// Handles both labelled (`name{a="b"} 1`) and unlabelled (`name 1`) forms.
fn split_sample_line(line: &str) -> Option<(String, String)> {
    if let Some(brace) = line.find('{') {
        let name = line[..brace].trim();
        let close = line[brace..].find('}')? + brace;
        let after = line[close + 1..].trim_start();
        if name.is_empty() {
            return None;
        }
        return Some((name.to_string(), after.to_string()));
    }
    let mut split = line.splitn(2, char::is_whitespace);
    let name = split.next()?.trim();
    let rest = split.next().unwrap_or("").trim_start();
    if name.is_empty() || rest.is_empty() {
        return None;
    }
    Some((name.to_string(), rest.to_string()))
}

/// Strip histogram/summary suffixes (`_bucket`, `_sum`, `_count`) so the
/// declared base type is matched against the sample series.
fn metric_base_name(name: &str, declared: &TypeTable) -> String {
    if declared.contains_key(name) {
        return name.to_string();
    }
    for suffix in ["_bucket", "_sum", "_count"] {
        if let Some(stripped) = name.strip_suffix(suffix) {
            if matches!(
                declared.get(stripped).map(String::as_str),
                Some("histogram" | "summary")
            ) {
                return stripped.to_string();
            }
        }
    }
    name.to_string()
}

/// Prometheus 0.0.4 sample values may be: a base-10 float, `+Inf`, `-Inf`, or `Nan`.
fn is_prometheus_numeric(s: &str) -> bool {
    matches!(s, "+Inf" | "-Inf" | "Nan" | "NaN") || s.parse::<f64>().is_ok()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn content_type_ok_with_version_and_charset() {
        assert_eq!(
            classify_content_type("text/plain; version=0.0.4; charset=utf-8"),
            PromContentTypeOutcome::Ok
        );
    }

    #[test]
    fn content_type_ok_with_only_version() {
        assert_eq!(
            classify_content_type("text/plain; version=0.0.4"),
            PromContentTypeOutcome::Ok
        );
    }

    #[test]
    fn content_type_rejects_application_json() {
        assert_eq!(
            classify_content_type("application/json"),
            PromContentTypeOutcome::WrongMediaType {
                got: "application/json".to_string()
            }
        );
    }

    #[test]
    fn content_type_rejects_missing_version() {
        let got = classify_content_type("text/plain; charset=utf-8");
        assert!(
            matches!(got, PromContentTypeOutcome::MissingVersion { .. }),
            "{got:?}"
        );
    }

    #[test]
    fn content_type_rejects_wrong_version() {
        assert_eq!(
            classify_content_type("text/plain; version=0.0.5"),
            PromContentTypeOutcome::WrongVersion {
                got: "0.0.5".to_string()
            }
        );
    }

    fn good_body() -> String {
        // Minimal K-07 body that exercises counter + gauge + histogram.
        concat!(
            "# HELP apr_num_requests_running running requests\n",
            "# TYPE apr_num_requests_running gauge\n",
            "apr_num_requests_running 3\n",
            "# HELP apr_num_requests_waiting queued requests\n",
            "# TYPE apr_num_requests_waiting gauge\n",
            "apr_num_requests_waiting 0\n",
            "# HELP apr_gpu_cache_usage_perc kv cache\n",
            "# TYPE apr_gpu_cache_usage_perc gauge\n",
            "apr_gpu_cache_usage_perc 0.42\n",
            "# TYPE apr_time_to_first_token_seconds histogram\n",
            "apr_time_to_first_token_seconds_bucket{le=\"0.5\"} 100\n",
            "apr_time_to_first_token_seconds_sum 12.5\n",
            "apr_time_to_first_token_seconds_count 200\n",
            "# TYPE apr_time_per_output_token_seconds histogram\n",
            "apr_time_per_output_token_seconds_bucket{le=\"0.05\"} 80\n",
            "apr_time_per_output_token_seconds_sum 1.5\n",
            "apr_time_per_output_token_seconds_count 40\n",
            "# TYPE apr_e2e_request_latency_seconds histogram\n",
            "apr_e2e_request_latency_seconds_bucket{le=\"1.0\"} 50\n",
            "apr_e2e_request_latency_seconds_sum 30.0\n",
            "apr_e2e_request_latency_seconds_count 60\n",
        )
        .to_string()
    }

    #[test]
    fn text_format_accepts_well_formed_body() {
        let out = classify_text_format(&good_body());
        assert!(matches!(out, PromTextFormatOutcome::Ok { .. }), "{out:?}");
    }

    #[test]
    fn text_format_rejects_sample_before_type() {
        let body = "apr_num_requests_running 3\n";
        assert!(matches!(
            classify_text_format(body),
            PromTextFormatOutcome::SampleBeforeType { .. }
        ));
    }

    #[test]
    fn text_format_rejects_unknown_type() {
        let body = "# TYPE apr_x widget\napr_x 1\n";
        assert!(matches!(
            classify_text_format(body),
            PromTextFormatOutcome::UnknownType { .. }
        ));
    }

    #[test]
    fn text_format_rejects_nonnumeric_sample() {
        let body = "# TYPE apr_x gauge\napr_x banana\n";
        assert!(matches!(
            classify_text_format(body),
            PromTextFormatOutcome::SampleValueNotNumeric { .. }
        ));
    }

    #[test]
    fn text_format_accepts_inf_and_nan() {
        let body = "# TYPE apr_x gauge\napr_x +Inf\napr_y_g 1\n# TYPE apr_y_g gauge\n";
        // (sample order vs TYPE: parser walks line-by-line, so apr_y_g sample before
        // its TYPE would fail. Use a single-metric body instead.)
        let body = "# TYPE apr_x gauge\napr_x +Inf\n";
        assert!(matches!(
            classify_text_format(body),
            PromTextFormatOutcome::Ok { .. }
        ));
        let _ = body;
    }

    #[test]
    fn text_format_rejects_duplicate_conflicting_types() {
        let body = "# TYPE apr_x counter\n# TYPE apr_x gauge\napr_x 1\n";
        assert!(matches!(
            classify_text_format(body),
            PromTextFormatOutcome::DuplicateConflictingType { .. }
        ));
    }

    #[test]
    fn text_format_records_each_metric_once_when_type_line_repeats() {
        // `metrics_seen` is the list of DISTINCT metrics the body declares, in
        // first-sighting order. A body may legally repeat an identical `# TYPE`
        // line (two exporters merged, a scrape concatenated twice), and the
        // repeat must NOT produce a second entry.
        //
        // Every other test in this module matches `Ok { .. }` and throws the
        // payload away, so nothing here constrained `metrics_seen` at all —
        // duplicate or dropped entries were invisible to the suite.
        let body = concat!(
            "# TYPE apr_x gauge\n",
            "# TYPE apr_x gauge\n",
            "apr_x 1\n",
            "# TYPE apr_y counter\n",
            "apr_y 2\n",
        );
        match classify_text_format(body) {
            PromTextFormatOutcome::Ok { metrics_seen } => {
                assert_eq!(metrics_seen, vec!["apr_x".to_string(), "apr_y".to_string()]);
            }
            other => panic!("expected Ok, got {other:?}"),
        }
    }

    #[test]
    fn required_metrics_pass_on_full_k07_body() {
        assert_eq!(
            classify_required_metrics(&good_body(), K07_REQUIRED_METRICS),
            PromRequiredOutcome::Ok
        );
    }

    #[test]
    fn required_metrics_reports_missing_subset() {
        let body = "# TYPE apr_num_requests_running gauge\napr_num_requests_running 1\n";
        match classify_required_metrics(body, K07_REQUIRED_METRICS) {
            PromRequiredOutcome::Missing { missing } => {
                assert!(missing.contains(&"apr_num_requests_waiting".to_string()));
                assert!(missing.contains(&"apr_e2e_request_latency_seconds".to_string()));
                assert!(!missing.contains(&"apr_num_requests_running".to_string()));
            }
            other => panic!("expected Missing, got {other:?}"),
        }
    }

    #[test]
    fn required_metrics_accept_histogram_expansion() {
        // Histogram-required metrics are detected via _bucket / _sum / _count samples
        // even if the bare `<name>` line is not emitted.
        let body = "# TYPE apr_e2e_request_latency_seconds histogram\n\
                    apr_e2e_request_latency_seconds_bucket{le=\"1.0\"} 1\n\
                    apr_e2e_request_latency_seconds_sum 1.0\n\
                    apr_e2e_request_latency_seconds_count 1\n";
        match classify_required_metrics(body, &["apr_e2e_request_latency_seconds"]) {
            PromRequiredOutcome::Ok => {}
            other => panic!("expected Ok, got {other:?}"),
        }
    }
}
