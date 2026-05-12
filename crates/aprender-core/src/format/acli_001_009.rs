// SHIP-TWO-001 — `apr-cli-v1` algorithm-level PARTIAL discharge for
// FALSIFY-ACLI-001..009 (closes 9/9 sweep).
//
// Contract: `contracts/apr-cli-v1.yaml`.
//
// The runtime falsification harness exists per the contract evidence;
// this file pins the algorithm-level rule against drift.

// ===========================================================================
// ACLI-001 — Parse determinism: parse(argv) yields identical struct N times
// ===========================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Acli001Verdict { Pass, Fail }

/// Pass iff every Debug-rendered Cli string in `parsed_repeats` is
/// byte-identical AND `parsed_repeats.len() >= min_iterations`.
#[must_use]
pub fn verdict_from_parse_determinism(
    parsed_repeats: &[String],
    min_iterations: usize,
) -> Acli001Verdict {
    if parsed_repeats.len() < min_iterations || min_iterations == 0 {
        return Acli001Verdict::Fail;
    }
    for w in parsed_repeats.windows(2) {
        if w[0] != w[1] { return Acli001Verdict::Fail; }
    }
    Acli001Verdict::Pass
}

// ===========================================================================
// ACLI-002 — Diagnostic commands exempt from contract gate (no exit 5)
// ===========================================================================

pub const AC_ACLI_002_CONTRACT_GATE_EXIT_CODE: i32 = 5;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Acli002Verdict { Pass, Fail }

/// Pass iff `exit_code != 5` for a diagnostic command on a corrupt file.
/// (Other non-zero exits like 4 = InvalidFormat are acceptable; the
/// contract only forbids exit 5 = ContractValidationFailed.)
#[must_use]
pub const fn verdict_from_diagnostic_bypass_contract_gate(exit_code: i32) -> Acli002Verdict {
    if exit_code == AC_ACLI_002_CONTRACT_GATE_EXIT_CODE { Acli002Verdict::Fail } else { Acli002Verdict::Pass }
}

// ===========================================================================
// ACLI-003 — RAII tempfile cleanup
// ===========================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Acli003Verdict { Pass, Fail }

/// Pass iff `tempfile_existed_in_scope == true` AND
/// `tempfile_existed_after_drop == false`.
#[must_use]
pub const fn verdict_from_raii_cleanup(
    existed_in_scope: bool,
    existed_after_drop: bool,
) -> Acli003Verdict {
    if existed_in_scope && !existed_after_drop { Acli003Verdict::Pass } else { Acli003Verdict::Fail }
}

// ===========================================================================
// ACLI-004 — index.json takes priority over shard files
// ===========================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ResolvedTo { IndexJson, ShardFile, Other }

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Acli004Verdict { Pass, Fail }

#[must_use]
pub const fn verdict_from_shard_index_priority(
    has_index: bool,
    has_shard: bool,
    resolved: ResolvedTo,
) -> Acli004Verdict {
    if has_index && has_shard {
        match resolved {
            ResolvedTo::IndexJson => Acli004Verdict::Pass,
            _ => Acli004Verdict::Fail,
        }
    } else {
        Acli004Verdict::Fail
    }
}

// ===========================================================================
// ACLI-005 — Global --json flag propagates to all subcommands
// ===========================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Acli005Verdict { Pass, Fail }

/// Pass iff:
///   - When global flag is set: cli.json == true regardless of subcommand-local flag
///   - When local subcommand flag is set: subcommand_json == true
#[must_use]
#[allow(clippy::fn_params_excessive_bools)]
pub const fn verdict_from_global_json_propagation(
    global_flag_set: bool,
    cli_json: bool,
    local_flag_set: bool,
    subcommand_json: bool,
) -> Acli005Verdict {
    if global_flag_set && !cli_json { return Acli005Verdict::Fail; }
    if local_flag_set && !subcommand_json { return Acli005Verdict::Fail; }
    Acli005Verdict::Pass
}

// ===========================================================================
// ACLI-006 — Alias commands parse to identical Cli variants
// ===========================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Acli006Verdict { Pass, Fail }

/// Pass iff both alias debug strings are byte-identical AND non-empty.
/// (e.g. parsing `apr ls` and `apr list` should produce the same Debug
/// rendering of the parsed Cli struct.)
#[must_use]
pub fn verdict_from_alias_equivalence(primary_dbg: &str, alias_dbg: &str) -> Acli006Verdict {
    if primary_dbg.is_empty() || alias_dbg.is_empty() { return Acli006Verdict::Fail; }
    if primary_dbg == alias_dbg { Acli006Verdict::Pass } else { Acli006Verdict::Fail }
}

// ===========================================================================
// ACLI-007 — `train plan` purity: does NOT create output dir
// ===========================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Acli007Verdict { Pass, Fail }

#[must_use]
pub const fn verdict_from_train_plan_purity(output_dir_existed_after: bool) -> Acli007Verdict {
    if output_dir_existed_after { Acli007Verdict::Fail } else { Acli007Verdict::Pass }
}

// ===========================================================================
// ACLI-008 — --skip-contract bypasses the contract gate (no exit 5)
// ===========================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Acli008Verdict { Pass, Fail }

#[must_use]
pub const fn verdict_from_skip_contract_bypass(exit_code_with_skip: i32) -> Acli008Verdict {
    if exit_code_with_skip == AC_ACLI_002_CONTRACT_GATE_EXIT_CODE { Acli008Verdict::Fail } else { Acli008Verdict::Pass }
}

// ===========================================================================
// ACLI-009 — Tokenizer vocab size matches requested
// ===========================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Acli009Verdict { Pass, Fail }

#[must_use]
pub const fn verdict_from_vocab_size(observed: u64, requested: u64) -> Acli009Verdict {
    if requested == 0 { return Acli009Verdict::Fail; }
    if observed == requested { Acli009Verdict::Pass } else { Acli009Verdict::Fail }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ACLI-001
    #[test] fn acli001_pass_1000_identical() {
        let s = "Cli { json: false }".to_string();
        let r: Vec<_> = (0..1000).map(|_| s.clone()).collect();
        assert_eq!(verdict_from_parse_determinism(&r, 1000), Acli001Verdict::Pass);
    }
    #[test] fn acli001_fail_drift() {
        let mut r: Vec<_> = (0..1000).map(|_| "A".to_string()).collect();
        r[500] = "B".to_string();
        assert_eq!(verdict_from_parse_determinism(&r, 1000), Acli001Verdict::Fail);
    }
    #[test] fn acli001_fail_too_few() {
        let r = vec!["A".to_string(); 5];
        assert_eq!(verdict_from_parse_determinism(&r, 1000), Acli001Verdict::Fail);
    }
    #[test] fn acli001_fail_zero_min() {
        let r = vec!["A".to_string(); 5];
        assert_eq!(verdict_from_parse_determinism(&r, 0), Acli001Verdict::Fail);
    }

    // ACLI-002
    #[test] fn acli002_pass_invalid_format() {
        // Exit 4 = InvalidFormat (not 5), diagnostic ran successfully.
        assert_eq!(verdict_from_diagnostic_bypass_contract_gate(4), Acli002Verdict::Pass);
    }
    #[test] fn acli002_pass_success() {
        assert_eq!(verdict_from_diagnostic_bypass_contract_gate(0), Acli002Verdict::Pass);
    }
    #[test] fn acli002_fail_contract_gate_blocked() {
        assert_eq!(verdict_from_diagnostic_bypass_contract_gate(5), Acli002Verdict::Fail);
    }

    // ACLI-003
    #[test] fn acli003_pass_clean() {
        assert_eq!(verdict_from_raii_cleanup(true, false), Acli003Verdict::Pass);
    }
    #[test] fn acli003_fail_leak() {
        assert_eq!(verdict_from_raii_cleanup(true, true), Acli003Verdict::Fail);
    }
    #[test] fn acli003_fail_never_existed() {
        assert_eq!(verdict_from_raii_cleanup(false, false), Acli003Verdict::Fail);
    }

    // ACLI-004
    #[test] fn acli004_pass_index_priority() {
        assert_eq!(
            verdict_from_shard_index_priority(true, true, ResolvedTo::IndexJson),
            Acli004Verdict::Pass
        );
    }
    #[test] fn acli004_fail_shard_picked() {
        assert_eq!(
            verdict_from_shard_index_priority(true, true, ResolvedTo::ShardFile),
            Acli004Verdict::Fail
        );
    }
    #[test] fn acli004_fail_no_index() {
        assert_eq!(
            verdict_from_shard_index_priority(false, true, ResolvedTo::ShardFile),
            Acli004Verdict::Fail
        );
    }

    // ACLI-005
    #[test] fn acli005_pass_both_set() {
        assert_eq!(
            verdict_from_global_json_propagation(true, true, true, true),
            Acli005Verdict::Pass
        );
    }
    #[test] fn acli005_pass_neither_set() {
        assert_eq!(
            verdict_from_global_json_propagation(false, false, false, false),
            Acli005Verdict::Pass
        );
    }
    #[test] fn acli005_fail_global_set_but_cli_false() {
        assert_eq!(
            verdict_from_global_json_propagation(true, false, false, false),
            Acli005Verdict::Fail
        );
    }
    #[test] fn acli005_fail_local_set_but_subcmd_false() {
        assert_eq!(
            verdict_from_global_json_propagation(false, false, true, false),
            Acli005Verdict::Fail
        );
    }

    // ACLI-006
    #[test] fn acli006_pass_match() {
        assert_eq!(
            verdict_from_alias_equivalence("Commands::List", "Commands::List"),
            Acli006Verdict::Pass
        );
    }
    #[test] fn acli006_fail_drift() {
        assert_eq!(
            verdict_from_alias_equivalence("Commands::List", "Commands::Ls"),
            Acli006Verdict::Fail
        );
    }
    #[test] fn acli006_fail_empty() {
        assert_eq!(verdict_from_alias_equivalence("", "Commands::List"), Acli006Verdict::Fail);
    }

    // ACLI-007
    #[test] fn acli007_pass_no_dir_created() {
        assert_eq!(verdict_from_train_plan_purity(false), Acli007Verdict::Pass);
    }
    #[test] fn acli007_fail_dir_created() {
        assert_eq!(verdict_from_train_plan_purity(true), Acli007Verdict::Fail);
    }

    // ACLI-008
    #[test] fn acli008_pass_skip_works() {
        // Exit 0 — skip-contract bypassed gate.
        assert_eq!(verdict_from_skip_contract_bypass(0), Acli008Verdict::Pass);
    }
    #[test] fn acli008_pass_other_error() {
        // Exit 4 — different error class, but NOT 5.
        assert_eq!(verdict_from_skip_contract_bypass(4), Acli008Verdict::Pass);
    }
    #[test] fn acli008_fail_gate_still_triggered() {
        assert_eq!(verdict_from_skip_contract_bypass(5), Acli008Verdict::Fail);
    }

    // ACLI-009
    #[test] fn acli009_pass_match() {
        assert_eq!(verdict_from_vocab_size(512, 512), Acli009Verdict::Pass);
    }
    #[test] fn acli009_fail_off() {
        assert_eq!(verdict_from_vocab_size(511, 512), Acli009Verdict::Fail);
    }
    #[test] fn acli009_fail_zero_requested() {
        assert_eq!(verdict_from_vocab_size(0, 0), Acli009Verdict::Fail);
    }

    // Provenance pin
    #[test] fn provenance_gate_exit_code() {
        assert_eq!(AC_ACLI_002_CONTRACT_GATE_EXIT_CODE, 5);
    }
}
