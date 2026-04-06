use regex::Regex;

use super::super::super::types::{PtxBug, PtxBugClass};
use super::super::PtxBugAnalyzer;

impl PtxBugAnalyzer {
    /// Detect missing kernel entry point
    pub(in crate::ptx::bugs::analyzer) fn detect_missing_entry_point(
        &self,
        ptx: &str,
        _lines: &[&str],
    ) -> Vec<PtxBug> {
        let mut bugs = Vec::new();

        let entry_pattern =
            Regex::new(r"\.entry\s+\w+").expect("invariant: regex pattern is valid");
        let has_entry = entry_pattern.is_match(ptx);

        // Only flag if PTX has some content but no entry point
        if !ptx.trim().is_empty() && !has_entry {
            bugs.push(PtxBug {
                class: PtxBugClass::MissingEntryPoint,
                line: 0,
                instruction: String::new(),
                message: "No kernel entry point (.entry) found".to_string(),
                fix: Some("Add .entry <kernel_name>(...) declaration".to_string()),
            });
        }

        bugs
    }

    /// Detect redundant register moves (P2)
    /// Pattern: mov %rx, %ry followed by mov %rz, %rx (could use %ry directly)
    pub(in crate::ptx::bugs::analyzer) fn detect_redundant_moves(
        &self,
        _ptx: &str,
        lines: &[&str],
    ) -> Vec<PtxBug> {
        let mut bugs = Vec::new();

        // Look for mov chains: mov %a, %b; mov %c, %a; -> should be mov %c, %b
        let mov_pattern = Regex::new(r"^\s*mov\.\w+\s+(%\w+),\s*(%\w+)")
            .expect("invariant: regex pattern is valid");

        let mut last_mov: Option<(usize, String, String)> = None; // (line, dest, src)

        for (line_num, line) in lines.iter().enumerate() {
            let trimmed = line.trim();

            if let Some(caps) = mov_pattern.captures(trimmed) {
                let dest =
                    caps.get(1).expect("invariant: capture group exists").as_str().to_string();
                let src =
                    caps.get(2).expect("invariant: capture group exists").as_str().to_string();

                // Check if src matches previous dest (redundant chain)
                if let Some((prev_line, prev_dest, _prev_src)) = &last_mov {
                    if &src == prev_dest {
                        bugs.push(PtxBug {
                            class: PtxBugClass::RedundantMoves,
                            line: line_num + 1,
                            instruction: format!(
                                "mov chain at lines {} and {}",
                                prev_line + 1,
                                line_num + 1
                            ),
                            message: format!(
                                "Redundant move: {} copied to {} then to another register",
                                prev_dest, dest
                            ),
                            fix: Some("Combine mov chain into single mov".to_string()),
                        });
                    }
                }

                last_mov = Some((line_num, dest, src));
            } else {
                // Reset on non-mov instruction
                last_mov = None;
            }
        }

        bugs
    }

    /// Detect high register pressure (P1)
    /// >64 registers per thread reduces occupancy and may cause spills
    pub(in crate::ptx::bugs::analyzer) fn detect_high_register_pressure(
        &self,
        ptx: &str,
        _lines: &[&str],
    ) -> Vec<PtxBug> {
        let mut bugs = Vec::new();

        // Count register declarations: .reg .type %name<count>
        let reg_pattern =
            Regex::new(r"\.reg\s+\.\w+\s+%\w+<(\d+)>").expect("invariant: regex pattern is valid");
        let total_regs: usize = reg_pattern
            .captures_iter(ptx)
            .filter_map(|c| c.get(1).and_then(|m| m.as_str().parse::<usize>().ok()))
            .sum();

        // Threshold: >64 registers is problematic for occupancy
        // SM_89 has 65536 regs/SM, 64 regs/thread allows 32 warps (100% occupancy)
        if total_regs > 64 {
            let occupancy = 65536 / (total_regs * 32);
            let occupancy_pct = (occupancy as f32 / 32.0 * 100.0).min(100.0);
            bugs.push(PtxBug {
                class: PtxBugClass::HighRegisterPressure,
                line: 0,
                instruction: format!("{} register banks declared", total_regs),
                message: format!(
                    "High register pressure: {} registers limits occupancy to {:.0}%",
                    total_regs, occupancy_pct
                ),
                fix: Some("Reduce live variables or split into multiple kernels".to_string()),
            });
        }

        bugs
    }

    /// Detect predicate register overflow (P1)
    /// PTX has 8 predicate registers (p0-p7), exceeding this causes spills
    pub(in crate::ptx::bugs::analyzer) fn detect_predicate_overflow(
        &self,
        ptx: &str,
        _lines: &[&str],
    ) -> Vec<PtxBug> {
        let mut bugs = Vec::new();

        // Pattern: .reg .pred %p<count>
        let pred_pattern =
            Regex::new(r"\.reg\s+\.pred\s+%p<(\d+)>").expect("invariant: regex pattern is valid");
        if let Some(caps) = pred_pattern.captures(ptx) {
            if let Ok(pred_count) =
                caps.get(1).expect("invariant: capture group exists").as_str().parse::<usize>()
            {
                if pred_count > 8 {
                    bugs.push(PtxBug {
                        class: PtxBugClass::PredicateOverflow,
                        line: 0,
                        instruction: format!(".reg .pred %p<{}>", pred_count),
                        message: format!(
                            "Predicate overflow: {} predicates declared (max 8 hardware registers)",
                            pred_count
                        ),
                        fix: Some(
                            "Reduce predicate usage by combining conditions or using branches"
                                .to_string(),
                        ),
                    });
                }
            }
        }

        bugs
    }

    /// Detect placeholder/incomplete code (P1)
    /// Comments like "omitted", "simplified", "placeholder" indicate incomplete kernels
    pub(in crate::ptx::bugs::analyzer) fn detect_placeholder_code(
        &self,
        _ptx: &str,
        lines: &[&str],
    ) -> Vec<PtxBug> {
        let mut bugs = Vec::new();

        // Patterns indicating incomplete code
        let placeholder_patterns = [
            "omitted",
            "simplified",
            "placeholder",
            "todo",
            "fixme",
            "not implemented",
            "for now",
            "for brevity",
        ];

        for (line_num, line) in lines.iter().enumerate() {
            let lower = line.to_lowercase();
            // Only check comments
            if lower.contains("//") {
                for pattern in &placeholder_patterns {
                    if lower.contains(pattern) {
                        bugs.push(PtxBug {
                            class: PtxBugClass::PlaceholderCode,
                            line: line_num + 1,
                            instruction: line.trim().to_string(),
                            message: format!("Placeholder code detected: contains '{}'", pattern),
                            fix: Some(
                                "Implement complete kernel or use trueno-gpu generation"
                                    .to_string(),
                            ),
                        });
                        break; // Only report once per line
                    }
                }
            }
        }

        bugs
    }
}
