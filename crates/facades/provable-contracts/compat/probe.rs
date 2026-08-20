//! Non-vacuity probe for the `provable-contracts` facade.
//!
//! The 27 vendored 0.3.1 example programs are the compatibility corpus. This
//! file is the CONTROL that proves the corpus is being compiled at all, and
//! that a broken re-export is a compile error rather than a silent pass.
//!
//! Two directions, both automated:
//!
//! * GREEN — default features. Real work is done through `provable_contracts::`
//!   paths: a contract is parsed, validated and scored. A facade re-exporting
//!   an empty module compiles but fails these assertions.
//! * RED — `--features __facade_probe_mutant` names an item that does not
//!   exist behind the facade. `scripts/check_facade_compat.sh --self-test`
//!   requires that build to FAIL. If it ever passes, the compile half of the
//!   gate is not consulting the compiler and every green run above is theater.

// The mutant arm. Compiled only under the private probe feature; the gate's
// self-test asserts this fails to compile, which is the whole point of it.
#[cfg(feature = "__facade_probe_mutant")]
use provable_contracts::this_module_was_never_exported::nothing;

use provable_contracts::schema::{parse_contract_str, validate_contract};
use provable_contracts::scoring::score_contract;

/// A minimal contract, inline so the probe depends on no repository path.
const MINIMAL: &str = r"
metadata:
  version: 1.0.0
  description: facade probe fixture
  references:
    - 'facade probe — inline fixture, cites itself'
  kind: registry
";

#[test]
fn parse_validate_and_score_all_resolve_through_the_facade() {
    let contract = parse_contract_str(MINIMAL).expect("0.3.1 parse_contract_str signature drifted");
    assert_eq!(
        contract.metadata.version, "1.0.0",
        "Metadata.version is no longer reachable as a public field"
    );

    let errors = validate_contract(&contract);
    assert!(
        errors.is_empty(),
        "a registry contract with references should validate clean; got {errors:?}"
    );

    // 0.3.1 signature: score_contract(&Contract, Option<&BindingRegistry>, &str)
    let score = score_contract(&contract, None, "facade-probe");
    assert_eq!(score.stem, "facade-probe");
    assert!(
        (0.0..=1.0).contains(&score.composite),
        "ContractScore.composite left [0,1]: {}",
        score.composite
    );
}

#[cfg(feature = "__facade_probe_mutant")]
#[test]
fn unreachable_mutant_arm() {
    let _ = nothing;
}
