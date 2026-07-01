//! Beat-sklearn NMI parity: anti-drift pin between the contract YAML and the
//! Rust-pinned scikit-learn 1.9.0 oracle constants.
//!
//! The clustering-metric parity itself is exercised by the unit tests in
//! `crates/aprender-core/src/metrics/mod.rs` (module `tests_nmi`). This
//! integration test guards the *contract*: it `include_str!`s
//! `contracts/beat-sklearn-nmi-v1.yaml` at compile time so the pinned oracle
//! values cannot silently drift away from the implementation.
//!
//! Oracle (scikit-learn 1.9.0, numpy float64), generated offline:
//!   uv run --with scikit-learn --with numpy python3 -c "
//!   from sklearn.metrics import normalized_mutual_info_score as nmi, mutual_info_score
//!   print(nmi([0,0,1,1,2,2],[0,0,1,2,2,2],average_method='arithmetic'))  # 0.7396673768007592
//!   print(mutual_info_score([0,0,1,1,2,2],[0,0,1,2,2,2]))                # 0.7803552045207032
//!   "

use aprender::metrics::{mutual_info_score, normalized_mutual_info_score};

const CONTRACT_YAML: &str = include_str!("../../../contracts/beat-sklearn-nmi-v1.yaml");

/// The full-precision sklearn 1.9.0 oracle values, as they appear verbatim in
/// the contract YAML.
const ORACLE_NMI_LITERAL: &str = "0.7396673768007592";
const ORACLE_MI_LITERAL: &str = "0.7803552045207032";

/// FALSIFY-METRIC-NMI-004: the contract-pinned oracle equals the
/// implementation's oracle, AND the live implementation reproduces it.
#[test]
fn contract_oracle_constants_pinned() {
    // 1. The contract YAML literally contains the pinned sklearn oracle values.
    assert!(
        CONTRACT_YAML.contains(ORACLE_NMI_LITERAL),
        "contract YAML missing pinned NMI oracle {ORACLE_NMI_LITERAL}"
    );
    assert!(
        CONTRACT_YAML.contains(ORACLE_MI_LITERAL),
        "contract YAML missing pinned MI oracle {ORACLE_MI_LITERAL}"
    );

    // 2. The live implementation reproduces those exact oracle values (1e-4 f32).
    let t = [0usize, 0, 1, 1, 2, 2];
    let p = [0usize, 0, 1, 2, 2, 2];
    let nmi_oracle: f32 = ORACLE_NMI_LITERAL
        .parse()
        .expect("NMI oracle literal parses");
    let mi_oracle: f32 = ORACLE_MI_LITERAL.parse().expect("MI oracle literal parses");
    assert!(
        (normalized_mutual_info_score(&t, &p) - nmi_oracle).abs() < 1e-4,
        "NMI drift: impl {} vs oracle {nmi_oracle}",
        normalized_mutual_info_score(&t, &p)
    );
    assert!(
        (mutual_info_score(&t, &p) - mi_oracle).abs() < 1e-4,
        "MI drift: impl {} vs oracle {mi_oracle}",
        mutual_info_score(&t, &p)
    );
}
