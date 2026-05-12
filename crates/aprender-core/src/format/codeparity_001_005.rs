// `apr-code-parity-v1` algorithm-level PARTIAL discharge for
// FALSIFY-CODE-PARITY-001..005.
//
// Contract: `contracts/apr-code-parity-v1.yaml`.
//
// CODE-PARITY-001: row-by-row mechanical audit (every row's cross-check passes)
// CODE-PARITY-002: headline aggregate invariant (counts == row tally)
// CODE-PARITY-003: prose ↔ YAML drift check (md table matches YAML)
// CODE-PARITY-004: P0 closure gate (closed ticket → row status updated)
// CODE-PARITY-005: epic closure (shipped ≥ 9 AND missing ≤ 4)

/// CODE-PARITY-005 epic-close thresholds (per the YAML acceptance_criterion).
pub const AC_CODEPARITY_SHIPPED_FLOOR: u32 = 9;
pub const AC_CODEPARITY_MISSING_CEIL: u32 = 4;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CodeParityVerdict {
    Pass,
    Fail,
}

/// CODE-PARITY-001: every row's cross-check passes.
#[must_use]
pub fn verdict_from_mechanical_audit(
    rows_total: u32,
    rows_passing: u32,
) -> CodeParityVerdict {
    if rows_total == 0 {
        return CodeParityVerdict::Fail;
    }
    if rows_passing == rows_total {
        CodeParityVerdict::Pass
    } else {
        CodeParityVerdict::Fail
    }
}

/// CODE-PARITY-002: headline counts match row tally.
#[must_use]
pub fn verdict_from_headline_aggregate(
    headline_shipped: u32,
    headline_partial: u32,
    headline_missing: u32,
    actual_shipped: u32,
    actual_partial: u32,
    actual_missing: u32,
) -> CodeParityVerdict {
    if headline_shipped == actual_shipped
        && headline_partial == actual_partial
        && headline_missing == actual_missing
    {
        CodeParityVerdict::Pass
    } else {
        CodeParityVerdict::Fail
    }
}

/// CODE-PARITY-003: prose ↔ YAML drift — every (category, status, ticket) tuple matches.
#[must_use]
pub fn verdict_from_prose_yaml_drift(
    md_rows: &[(&str, &str, &str)],
    yaml_rows: &[(&str, &str, &str)],
) -> CodeParityVerdict {
    if md_rows.is_empty() || yaml_rows.is_empty() {
        return CodeParityVerdict::Fail;
    }
    if md_rows.len() != yaml_rows.len() {
        return CodeParityVerdict::Fail;
    }
    for (md, ym) in md_rows.iter().zip(yaml_rows.iter()) {
        if md != ym {
            return CodeParityVerdict::Fail;
        }
    }
    CodeParityVerdict::Pass
}

/// CODE-PARITY-004: P0 closure — when ticket closed in roadmap, row status
/// must be updated AND headline.shipped incremented.
#[must_use]
pub fn verdict_from_p0_closure(
    ticket_closed_in_roadmap: bool,
    row_status_updated: bool,
    headline_shipped_incremented: bool,
) -> CodeParityVerdict {
    if !ticket_closed_in_roadmap {
        return CodeParityVerdict::Pass; // gate not applicable
    }
    if row_status_updated && headline_shipped_incremented {
        CodeParityVerdict::Pass
    } else {
        CodeParityVerdict::Fail
    }
}

/// CODE-PARITY-005: epic closeable iff `shipped ≥ 9` AND `missing ≤ 4`.
#[must_use]
pub fn verdict_from_epic_closure(
    shipped: u32,
    missing: u32,
    epic_marked_closed: bool,
) -> CodeParityVerdict {
    let eligible = shipped >= AC_CODEPARITY_SHIPPED_FLOOR
        && missing <= AC_CODEPARITY_MISSING_CEIL;
    if eligible {
        CodeParityVerdict::Pass
    } else if !epic_marked_closed {
        // Not eligible but epic remains open — gate satisfied.
        CodeParityVerdict::Pass
    } else {
        // Not eligible AND epic claimed closed — regression class.
        CodeParityVerdict::Fail
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // -----------------------------------------------------------------
    // Section 1: Provenance pin.
    // -----------------------------------------------------------------
    #[test]
    fn provenance_constants() {
        assert_eq!(AC_CODEPARITY_SHIPPED_FLOOR, 9);
        assert_eq!(AC_CODEPARITY_MISSING_CEIL, 4);
    }

    // -----------------------------------------------------------------
    // Section 2: CODE-PARITY-001 mechanical audit.
    // -----------------------------------------------------------------
    #[test]
    fn fcp001_pass_all_rows_passing() {
        let v = verdict_from_mechanical_audit(20, 20);
        assert_eq!(v, CodeParityVerdict::Pass);
    }

    #[test]
    fn fcp001_fail_one_row_failing() {
        let v = verdict_from_mechanical_audit(20, 19);
        assert_eq!(v, CodeParityVerdict::Fail);
    }

    #[test]
    fn fcp001_fail_zero_rows() {
        let v = verdict_from_mechanical_audit(0, 0);
        assert_eq!(v, CodeParityVerdict::Fail);
    }

    // -----------------------------------------------------------------
    // Section 3: CODE-PARITY-002 headline.
    // -----------------------------------------------------------------
    #[test]
    fn fcp002_pass_match() {
        let v = verdict_from_headline_aggregate(14, 3, 4, 14, 3, 4);
        assert_eq!(v, CodeParityVerdict::Pass);
    }

    #[test]
    fn fcp002_fail_drift_in_shipped() {
        let v = verdict_from_headline_aggregate(15, 3, 4, 14, 3, 4);
        assert_eq!(v, CodeParityVerdict::Fail);
    }

    #[test]
    fn fcp002_fail_drift_in_missing() {
        let v = verdict_from_headline_aggregate(14, 3, 4, 14, 3, 5);
        assert_eq!(v, CodeParityVerdict::Fail);
    }

    // -----------------------------------------------------------------
    // Section 4: CODE-PARITY-003 prose drift.
    // -----------------------------------------------------------------
    #[test]
    fn fcp003_pass_identical_rows() {
        let md = vec![
            ("MCP-CLIENT", "shipped", "PMAT-CODE-MCP-CLIENT-001"),
            ("SLASH-PARITY", "shipped", "PMAT-SLASH-001"),
        ];
        let yaml = md.clone();
        let v = verdict_from_prose_yaml_drift(&md, &yaml);
        assert_eq!(v, CodeParityVerdict::Pass);
    }

    #[test]
    fn fcp003_fail_status_drift() {
        let md = vec![("MCP-CLIENT", "shipped", "T-1")];
        let yaml = vec![("MCP-CLIENT", "partial", "T-1")];
        let v = verdict_from_prose_yaml_drift(&md, &yaml);
        assert_eq!(v, CodeParityVerdict::Fail);
    }

    #[test]
    fn fcp003_fail_length_mismatch() {
        let md = vec![("MCP-CLIENT", "shipped", "T-1")];
        let yaml = vec![("MCP-CLIENT", "shipped", "T-1"), ("X", "y", "z")];
        let v = verdict_from_prose_yaml_drift(&md, &yaml);
        assert_eq!(v, CodeParityVerdict::Fail);
    }

    // -----------------------------------------------------------------
    // Section 5: CODE-PARITY-004 P0 closure.
    // -----------------------------------------------------------------
    #[test]
    fn fcp004_pass_closed_with_updates() {
        let v = verdict_from_p0_closure(true, true, true);
        assert_eq!(v, CodeParityVerdict::Pass);
    }

    #[test]
    fn fcp004_pass_ticket_open() {
        let v = verdict_from_p0_closure(false, false, false);
        assert_eq!(v, CodeParityVerdict::Pass);
    }

    #[test]
    fn fcp004_fail_closed_without_row_update() {
        let v = verdict_from_p0_closure(true, false, true);
        assert_eq!(v, CodeParityVerdict::Fail);
    }

    #[test]
    fn fcp004_fail_closed_without_headline_bump() {
        let v = verdict_from_p0_closure(true, true, false);
        assert_eq!(v, CodeParityVerdict::Fail);
    }

    // -----------------------------------------------------------------
    // Section 6: CODE-PARITY-005 epic closure.
    // -----------------------------------------------------------------
    #[test]
    fn fcp005_pass_eligible_and_closed() {
        let v = verdict_from_epic_closure(14, 4, true);
        assert_eq!(v, CodeParityVerdict::Pass);
    }

    #[test]
    fn fcp005_pass_eligible_at_threshold() {
        let v = verdict_from_epic_closure(9, 4, true);
        assert_eq!(v, CodeParityVerdict::Pass);
    }

    #[test]
    fn fcp005_pass_not_eligible_open() {
        let v = verdict_from_epic_closure(8, 5, false);
        assert_eq!(v, CodeParityVerdict::Pass);
    }

    #[test]
    fn fcp005_fail_not_eligible_but_closed() {
        // shipped < 9
        let v = verdict_from_epic_closure(8, 4, true);
        assert_eq!(v, CodeParityVerdict::Fail);
    }

    #[test]
    fn fcp005_fail_too_many_missing() {
        let v = verdict_from_epic_closure(14, 5, true);
        assert_eq!(v, CodeParityVerdict::Fail);
    }

    // -----------------------------------------------------------------
    // Section 7: Mutation surveys + realistic.
    // -----------------------------------------------------------------
    #[test]
    fn mutation_survey_005_eligibility_band() {
        for shipped in [0_u32, 8, 9, 10, 14] {
            for missing in [0_u32, 4, 5, 10] {
                let eligible = shipped >= 9 && missing <= 4;
                let v = verdict_from_epic_closure(shipped, missing, true);
                let want = if eligible {
                    CodeParityVerdict::Pass
                } else {
                    CodeParityVerdict::Fail
                };
                assert_eq!(v, want, "(s={shipped}, m={missing})");
            }
        }
    }

    #[test]
    fn realistic_healthy_passes_all_5() {
        let v1 = verdict_from_mechanical_audit(21, 21);
        let v2 = verdict_from_headline_aggregate(14, 3, 4, 14, 3, 4);
        let md = vec![("MCP-CLIENT", "shipped", "T-1")];
        let v3 = verdict_from_prose_yaml_drift(&md, &md);
        let v4 = verdict_from_p0_closure(true, true, true);
        let v5 = verdict_from_epic_closure(14, 4, true);
        for v in [v1, v2, v3, v4, v5] {
            assert_eq!(v, CodeParityVerdict::Pass);
        }
    }

    #[test]
    fn realistic_pre_fix_all_5_failures() {
        // 5 simultaneous regressions:
        //  1: a row's cross-check fails (matrix lied or stale)
        //  2: headline drifted by +1 shipped without re-tally
        //  3: prose status drifted from yaml
        //  4: P0 ticket closed but row not updated
        //  5: epic marked closed despite shipped=8 (< 9)
        let v1 = verdict_from_mechanical_audit(21, 19);
        let v2 = verdict_from_headline_aggregate(15, 3, 4, 14, 3, 4);
        let md = vec![("MCP-CLIENT", "shipped", "T-1")];
        let yaml = vec![("MCP-CLIENT", "partial", "T-1")];
        let v3 = verdict_from_prose_yaml_drift(&md, &yaml);
        let v4 = verdict_from_p0_closure(true, false, true);
        let v5 = verdict_from_epic_closure(8, 4, true);
        for v in [v1, v2, v3, v4, v5] {
            assert_eq!(v, CodeParityVerdict::Fail);
        }
    }
}
