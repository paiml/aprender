//! `GET /v1/capability` — what this build can and cannot run on the GPU, and why
//! (aprender#3856 row 3).
//!
//! The body is the embedded `apr-model-capability-v1` contract, section for
//! section: the same `ops`, `quant_types` and `op_implementation` that
//! `apr capability --json` and the `apr.capability` MCP tool report. The
//! contract is the origin; this handler only serializes it.
//!
//! The YAML is this crate's own byte-identical mirror of
//! `contracts/apr-model-capability-v1.yaml`: `include_str!` cannot reach
//! outside a crate once it is packaged, and aprender-serve does not depend on
//! apr-cli. `crates/apr-cli/tests/capability_mirror.rs` fails the build if the
//! mirror drifts from the source.
//!
//! A contract that does not parse, or lacks `ops` / `quant_types`, is a 500
//! naming the defect, never a 200 with empty lists: an empty `ops` would read
//! as "nothing is unsupported".

use axum::{
    http::StatusCode,
    response::{IntoResponse, Response},
    Json,
};

use super::ErrorResponse;

/// The embedded contract, this crate's mirror of the repo-root source.
pub const CONTRACT_YAML: &str = include_str!("../../contracts/apr-model-capability-v1.yaml");

/// Repo-relative path of the mirror [`CONTRACT_YAML`] was embedded from.
pub const EMBEDDED_FROM: &str = "crates/aprender-serve/contracts/apr-model-capability-v1.yaml";

/// The capability report as JSON, built from [`CONTRACT_YAML`].
///
/// # Errors
///
/// The contract does not parse, or `ops` / `quant_types` is missing or not a
/// sequence.
pub fn capability_report() -> Result<serde_json::Value, String> {
    report_from_yaml(CONTRACT_YAML)
}

fn report_from_yaml(yaml: &str) -> Result<serde_json::Value, String> {
    let doc: serde_json::Value = serde_yaml_ng::from_str(yaml)
        .map_err(|e| format!("apr-model-capability-v1 does not parse: {e}"))?;
    let section = |key: &str| {
        doc.get(key)
            .filter(|v| v.is_array())
            .cloned()
            .ok_or_else(|| format!("apr-model-capability-v1 has no `{key}` sequence"))
    };
    let ops = section("ops")?;
    let quant_types = section("quant_types")?;
    Ok(serde_json::json!({
        "source": "contracts/apr-model-capability-v1.yaml",
        "embedded_from": EMBEDDED_FROM,
        "ops": ops,
        "quant_types": quant_types,
        "op_implementation": doc.get("op_implementation").cloned().unwrap_or(serde_json::Value::Null),
    }))
}

/// Handler for `GET /v1/capability`.
pub(crate) async fn capability_handler() -> Response {
    match capability_report() {
        Ok(body) => Json(body).into_response(),
        Err(error) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(ErrorResponse { error }),
        )
            .into_response(),
    }
}

#[cfg(test)]
mod tests {
    use super::report_from_yaml;

    #[test]
    fn a_contract_without_ops_is_an_error_not_an_empty_list() {
        let err = report_from_yaml("quant_types: []\n").expect_err("no ops must fail");
        assert!(err.contains("`ops`"), "{err}");
    }

    #[test]
    fn a_contract_whose_ops_is_not_a_sequence_is_an_error() {
        let err = report_from_yaml("ops: {}\nquant_types: []\n").expect_err("map ops must fail");
        assert!(err.contains("`ops`"), "{err}");
    }

    #[test]
    fn an_unparseable_contract_is_an_error() {
        let err = report_from_yaml("ops: [\n").expect_err("bad yaml must fail");
        assert!(err.contains("does not parse"), "{err}");
    }
}
