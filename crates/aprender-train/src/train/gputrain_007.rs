//! FALSIFY-GPUTRAIN-007 / INV-GPUTRAIN-007 / GATE-GPUTRAIN-006 — algorithm-level
//! PARTIAL discharge.
//!
//! Spec: `docs/specifications/aprender-train/ship-two-models-spec.md` §14
//! (task #132 CUDA training backend gap).
//!
//! Contract: `contracts/entrenar/gpu-training-backend-v1.yaml` v1.0.0 → v1.1.0
//! binds INV-GPUTRAIN-007 at PARTIAL_ALGORITHM_LEVEL via two pure functions:
//!
//!   1. `verdict_from_version_json_keys(present_keys) -> Gputrain007Verdict`
//!      — schema gate: Pass iff every key in
//!      `AC_GPUTRAIN_007_REQUIRED_VERSION_JSON_KEYS` appears in the
//!      presented-keys slice (extra unknown keys are tolerated; missing
//!      required keys are not).
//!
//!   2. `verdict_from_version_json_fields(&VersionJsonCudaFields) -> Verdict`
//!      — field-shape gate: Pass iff `visible_devices.len() <= 16` (matches
//!      INV-GPUTRAIN-001's `cuda:0..cuda:15` grammar, 16 max) AND NOT
//!      `(cuda_feature && !cuda_runtime_available)` (claiming CUDA was
//!      compiled in while the runtime is absent is exactly the
//!      FM-GPUTRAIN-STALE-BUILD failure mode — the binary must either not
//!      advertise the feature or confirm the runtime).
//!
//! INV-GPUTRAIN-007 states: "`apr --version --json` reports
//! `{cuda_feature, cuda_runtime_available, visible_devices[]}`." Operators
//! must be able to tell "compiled without cuda" apart from "compiled with
//! cuda but no GPU visible" without reading a stack trace.
//!
//! The compute-heavy portion (spawning `apr --version --json` as a
//! subprocess and parsing its stdout) is intentionally out of scope here;
//! the schema+shape rules are what the live gate calls after the JSON has
//! been deserialized, so this binding is JSON-library agnostic (we take a
//! `&[&str]` of keys, not a raw string).

/// Required top-level keys in `apr --version --json` output. Order is
/// irrelevant; extra keys (e.g. future `tensorrt_feature`) are tolerated,
/// but these three are load-bearing for operator triage.
pub const AC_GPUTRAIN_007_REQUIRED_VERSION_JSON_KEYS: &[&str] =
    &["cuda_feature", "cuda_runtime_available", "visible_devices"];

/// Deserialized shape of the CUDA-related block of `apr --version --json`.
/// Library-agnostic: caller is responsible for parsing JSON into this
/// shape (serde_json, hand-rolled, whatever) and then running the field-
/// shape gate.
#[derive(Debug, Clone)]
pub struct VersionJsonCudaFields {
    /// Whether the binary was built with `--features cuda`.
    pub cuda_feature: bool,
    /// Whether, at startup, `cudaGetDeviceCount` returned a non-zero
    /// device count AND the runtime didn't error.
    pub cuda_runtime_available: bool,
    /// Human-readable name/index pairs for each visible device, e.g.
    /// `["0:RTX 4090", "1:RTX 4090"]`. Upper bound 16 matches
    /// INV-GPUTRAIN-001 grammar's `:0..:15` range.
    pub visible_devices: Vec<String>,
}

/// Binary verdict for FALSIFY-GPUTRAIN-007 / GATE-GPUTRAIN-006.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Gputrain007Verdict {
    /// JSON contains all required keys (schema gate) OR field values
    /// pass every sanity rail (shape gate).
    Pass,
    /// A required key is missing, device-count exceeds the grammar
    /// limit, or the (cuda_feature, cuda_runtime_available) pair is in
    /// the advertised-but-missing inconsistent state.
    Fail,
}

/// Schema gate: given the list of top-level keys actually present in
/// `apr --version --json`, Pass iff every one of
/// `AC_GPUTRAIN_007_REQUIRED_VERSION_JSON_KEYS` is represented.
/// Unknown / extra keys are silently tolerated (forward-compatible).
#[must_use]
pub fn verdict_from_version_json_keys(present_keys: &[&str]) -> Gputrain007Verdict {
    for required in AC_GPUTRAIN_007_REQUIRED_VERSION_JSON_KEYS {
        if !present_keys.contains(required) {
            return Gputrain007Verdict::Fail;
        }
    }
    Gputrain007Verdict::Pass
}

/// Field-shape gate: given the parsed shape, Pass iff
///   1. `visible_devices.len() <= 16` (matches INV-GPUTRAIN-001 grammar
///      `cuda:0..cuda:15`, 16 max), and
///   2. NOT `(cuda_feature && !cuda_runtime_available)` — the advertised-
///      but-missing inconsistent state from FM-GPUTRAIN-STALE-BUILD.
///
/// The other three (cuda_feature, cuda_runtime_available) combinations
/// are all operationally valid:
///   - `(false, false)`: CPU-only build. Fine.
///   - `(false, true)`: CUDA runtime present but build didn't enable it.
///      Also fine; operator just needs a cuda-feature build to use it.
///   - `(true, true)`: CUDA fully wired. Baseline.
#[must_use]
pub fn verdict_from_version_json_fields(fields: &VersionJsonCudaFields) -> Gputrain007Verdict {
    // INV-GPUTRAIN-001 grammar allows cuda:0..cuda:15 — 16 indices max.
    if fields.visible_devices.len() > 16 {
        return Gputrain007Verdict::Fail;
    }
    // FM-GPUTRAIN-STALE-BUILD: advertised feature but missing runtime is
    // the footgun that cost 14 minutes on lambda-labs. Fail-closed.
    if fields.cuda_feature && !fields.cuda_runtime_available {
        return Gputrain007Verdict::Fail;
    }
    Gputrain007Verdict::Pass
}

// ─────────────────────────────────────────────────────────────
// Unit tests — FALSIFY-GPUTRAIN-007 algorithm-level proof
// ─────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    /// FALSIFY-GPUTRAIN-007 algorithm-level PARTIAL discharge: prove the
    /// version-JSON schema + field-shape invariants. Any mutation that
    /// allows a missing required key, silently accepts too many visible
    /// devices, or lets the stale-build state Pass must break this test
    /// before `apr --version --json` is ever shelled out.
    #[test]
    fn falsify_gputrain_007_version_json_schema_and_shape() {
        // Section 1: all required keys present — baseline Pass for the
        // schema gate. Mirrors the minimum JSON output on any shipping
        // build (CUDA or not).
        assert_eq!(
            verdict_from_version_json_keys(&[
                "cuda_feature",
                "cuda_runtime_available",
                "visible_devices",
            ]),
            Gputrain007Verdict::Pass,
            "all three required keys present must Pass",
        );
        // Extra unknown keys (future-proof): tolerated per spec
        // comment about forward compatibility.
        assert_eq!(
            verdict_from_version_json_keys(&[
                "cuda_feature",
                "cuda_runtime_available",
                "visible_devices",
                "version",
                "sha",
                "tensorrt_feature",
            ]),
            Gputrain007Verdict::Pass,
            "extra unknown keys must not Fail (forward-compat)",
        );

        // Section 2: each required key missing in turn. Three separate
        // sub-mutations of the output schema; each must Fail. Catches
        // any mutation to the required-keys list (e.g. removing one by
        // accident during refactor).
        assert_eq!(
            verdict_from_version_json_keys(&["cuda_runtime_available", "visible_devices",]),
            Gputrain007Verdict::Fail,
            "missing `cuda_feature` must Fail",
        );
        assert_eq!(
            verdict_from_version_json_keys(&["cuda_feature", "visible_devices"]),
            Gputrain007Verdict::Fail,
            "missing `cuda_runtime_available` must Fail",
        );
        assert_eq!(
            verdict_from_version_json_keys(&["cuda_feature", "cuda_runtime_available",]),
            Gputrain007Verdict::Fail,
            "missing `visible_devices` must Fail",
        );
        // All three missing (minimal empty JSON).
        assert_eq!(
            verdict_from_version_json_keys(&[]),
            Gputrain007Verdict::Fail,
            "empty present-keys slice must Fail",
        );
        // Only completely unrelated keys present.
        assert_eq!(
            verdict_from_version_json_keys(&["version", "sha"]),
            Gputrain007Verdict::Fail,
            "only unrelated keys present must Fail",
        );

        // Section 3: field-shape happy paths. Three of four
        // (cuda_feature, cuda_runtime_available) combinations are all
        // operationally valid.
        //   (false, false) — pure CPU build.
        //   (false, true)  — CUDA runtime present but build didn't
        //                    enable; not a bug, just a rebuild needed.
        //   (true, true)   — fully wired.
        for (feat, runtime) in [(false, false), (false, true), (true, true)] {
            let fields = VersionJsonCudaFields {
                cuda_feature: feat,
                cuda_runtime_available: runtime,
                visible_devices: vec!["0:RTX 4090".to_string()],
            };
            assert_eq!(
                verdict_from_version_json_fields(&fields),
                Gputrain007Verdict::Pass,
                "consistent (cuda_feature={feat}, runtime_available={runtime}) must Pass",
            );
        }
        // Zero visible devices is Pass on a CPU-only build; grammar
        // allows up to 16. Lower bound is 0.
        let fields = VersionJsonCudaFields {
            cuda_feature: false,
            cuda_runtime_available: false,
            visible_devices: vec![],
        };
        assert_eq!(
            verdict_from_version_json_fields(&fields),
            Gputrain007Verdict::Pass,
            "empty visible_devices on CPU-only build must Pass",
        );

        // Section 4: claims-feature-without-runtime — the stale-build
        // footgun. Must Fail. THIS IS THE NOVEL SANITY RAIL — if a
        // binary advertises `cuda_feature: true` it must also prove
        // `cuda_runtime_available: true`, or operators risk another
        // task #126.
        let stale = VersionJsonCudaFields {
            cuda_feature: true,
            cuda_runtime_available: false,
            visible_devices: vec![],
        };
        assert_eq!(
            verdict_from_version_json_fields(&stale),
            Gputrain007Verdict::Fail,
            "cuda_feature=true + runtime_available=false must Fail \
             (FM-GPUTRAIN-STALE-BUILD: advertised but missing)",
        );
        // Same inconsistency even if visible_devices accidentally got
        // populated (e.g. cached from a previous build).
        let stale_with_stale_devices = VersionJsonCudaFields {
            cuda_feature: true,
            cuda_runtime_available: false,
            visible_devices: vec!["0:RTX 4090".to_string()],
        };
        assert_eq!(
            verdict_from_version_json_fields(&stale_with_stale_devices),
            Gputrain007Verdict::Fail,
            "advertised feature + missing runtime must Fail regardless of visible_devices",
        );

        // Section 5: too-many visible devices. INV-GPUTRAIN-001 grammar
        // allows indices 0..=15 (16 max). Boundary: 16 Pass, 17 Fail.
        let sixteen = VersionJsonCudaFields {
            cuda_feature: true,
            cuda_runtime_available: true,
            visible_devices: (0..16).map(|i| format!("{i}:device")).collect(),
        };
        assert_eq!(
            verdict_from_version_json_fields(&sixteen),
            Gputrain007Verdict::Pass,
            "exactly 16 visible devices must Pass (grammar max)",
        );
        let seventeen = VersionJsonCudaFields {
            cuda_feature: true,
            cuda_runtime_available: true,
            visible_devices: (0..17).map(|i| format!("{i}:device")).collect(),
        };
        assert_eq!(
            verdict_from_version_json_fields(&seventeen),
            Gputrain007Verdict::Fail,
            "17 visible devices must Fail (exceeds cuda:0..cuda:15 grammar)",
        );
        let many = VersionJsonCudaFields {
            cuda_feature: true,
            cuda_runtime_available: true,
            visible_devices: (0..100).map(|i| format!("{i}:device")).collect(),
        };
        assert_eq!(
            verdict_from_version_json_fields(&many),
            Gputrain007Verdict::Fail,
            "100 visible devices must Fail (well past grammar)",
        );

        // Section 6: combined happy-path shape — the three required
        // keys present AND the field values consistent. Catches a
        // refactor that split the two gates into separate codepaths
        // and forgot to call one of them.
        let happy_fields = VersionJsonCudaFields {
            cuda_feature: true,
            cuda_runtime_available: true,
            visible_devices: vec!["0:RTX 4090".to_string(), "1:RTX 4090".to_string()],
        };
        assert_eq!(
            verdict_from_version_json_fields(&happy_fields),
            Gputrain007Verdict::Pass,
            "consistent 2-device CUDA build must Pass field-shape gate",
        );
        assert_eq!(
            verdict_from_version_json_keys(&[
                "cuda_feature",
                "cuda_runtime_available",
                "visible_devices",
            ]),
            Gputrain007Verdict::Pass,
            "matching 3-key schema must Pass schema gate",
        );

        // Section 7: provenance pin — the required-keys slice is
        // load-bearing. If a future spec amendment adds a 4th key
        // (e.g. `rocm_feature`), this slice and the YAML rule must
        // move together. The byte-literal shape prevents accidental
        // reordering from changing behaviour.
        assert_eq!(
            AC_GPUTRAIN_007_REQUIRED_VERSION_JSON_KEYS.len(),
            3,
            "required key count is 3 \
             (spec §14.4 / gpu-training-backend-v1 INV-GPUTRAIN-007)",
        );
        assert!(
            AC_GPUTRAIN_007_REQUIRED_VERSION_JSON_KEYS.contains(&"cuda_feature"),
            "`cuda_feature` is a required key",
        );
        assert!(
            AC_GPUTRAIN_007_REQUIRED_VERSION_JSON_KEYS.contains(&"cuda_runtime_available"),
            "`cuda_runtime_available` is a required key",
        );
        assert!(
            AC_GPUTRAIN_007_REQUIRED_VERSION_JSON_KEYS.contains(&"visible_devices"),
            "`visible_devices` is a required key",
        );
    }
}
