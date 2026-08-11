// GH-2395: option resolution for `apr profile`, kept pure so it can be falsified
// without loading a model.
//
// Every defect these functions guard shipped in 0.63.0:
//   * `--json` is a global flag; `apr bench --json` honours it and `apr profile
//     --json` printed the human table.
//   * an unparseable `--format` silently degraded to Human, so `--format jsonn`
//     produced a human table that a `| jq` pipeline choked on.
//   * an unrecognised `--focus` silently degraded to the full unfiltered report
//     with exit 0, so a typo answered a different question than the one asked.

/// Resolve the effective output format from `--format` plus the global `--json`.
///
/// `--json` wins, because it is the flag the rest of the CLI uses for
/// machine-readable output.
pub(crate) fn resolve_output_format(format: &str, json: bool) -> Result<OutputFormat, CliError> {
    if json {
        return Ok(OutputFormat::Json);
    }
    format.parse().map_err(|_| {
        CliError::ValidationFailed(format!(
            "Unknown --format value '{format}'. Valid values: human (text), json, flamegraph (svg)"
        ))
    })
}

/// Build the `Hardware:` label for the roofline block.
///
/// The model string usually already carries the vendor ("AMD Ryzen Threadripper
/// 7960X"), so repeating it would print "AMD AMD Ryzen …".
pub(crate) fn cpu_hardware_label(
    vendor: &str,
    model: &str,
    cores: usize,
    simd_bits: usize,
) -> String {
    let name = if model.is_empty() || model == "Unknown" {
        vendor.to_string()
    } else if model.to_lowercase().starts_with(&vendor.to_lowercase()) {
        model.to_string()
    } else {
        format!("{vendor} {model}")
    };
    format!("{name} ({cores} cores, {simd_bits}-bit SIMD)")
}

/// Resolve `--focus`, rejecting values that name no focus area.
pub(crate) fn resolve_focus(focus: Option<&str>) -> Result<ProfileFocus, CliError> {
    match focus {
        None => Ok(ProfileFocus::All),
        Some(f) => f.parse().map_err(|_| {
            CliError::ValidationFailed(format!(
                "Unknown --focus value '{f}'. Valid values: all, attention, mlp (ffn), matmul (gemm), embedding"
            ))
        }),
    }
}
