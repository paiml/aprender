use regex::Regex;
use std::collections::HashSet;

use super::super::super::types::{PtxBug, PtxBugClass};
use super::super::PtxBugAnalyzer;

impl PtxBugAnalyzer {
    /// Detect shared memory accessed with 64-bit register
    pub(in crate::ptx::bugs::analyzer) fn detect_shared_mem_u64(
        &self,
        _ptx: &str,
        lines: &[&str],
    ) -> Vec<PtxBug> {
        let mut bugs = Vec::new();
        // Pattern: st.shared.* [%rd*] or ld.shared.* [%rd*]
        let pattern = Regex::new(r"(?:st|ld)\.shared\.[^\[]+\[%rd\d+")
            .expect("invariant: regex pattern is valid");

        for (line_num, line) in lines.iter().enumerate() {
            let trimmed = line.trim();
            if pattern.is_match(trimmed) {
                bugs.push(PtxBug {
                    class: PtxBugClass::SharedMemU64Addressing,
                    line: line_num + 1,
                    instruction: trimmed.to_string(),
                    message: "Shared memory accessed with 64-bit register. Use 32-bit addressing."
                        .to_string(),
                    fix: Some("Replace %rd* with %r* for shared memory addressing".to_string()),
                });
            }
        }

        bugs
    }

    /// Detect missing barrier sync between shared memory operations (PARITY-114)
    pub(in crate::ptx::bugs::analyzer) fn detect_missing_barrier_sync(
        &self,
        ptx: &str,
        lines: &[&str],
    ) -> Vec<PtxBug> {
        let mut bugs = Vec::new();

        if !self.strict {
            return bugs;
        }

        // Check if shared memory is ACTUALLY used (st.shared or ld.shared operations)
        // Note: We don't flag just `.shared` declarations - only actual load/store operations
        // This prevents false positives for kernels that declare shared memory but use warp shuffles
        let has_st_shared = ptx.contains("st.shared");
        let has_ld_shared = ptx.contains("ld.shared");
        let uses_shared_ops = has_st_shared || has_ld_shared;
        let has_barrier = ptx.contains("bar.sync");

        if uses_shared_ops && !has_barrier {
            bugs.push(PtxBug {
                class: PtxBugClass::MissingBarrierSync,
                line: 0,
                instruction: String::new(),
                message: "Shared memory used but no bar.sync found. Race condition possible."
                    .to_string(),
                fix: Some("Add bar.sync 0; between st.shared and ld.shared operations".to_string()),
            });
        }

        // More precise detection: find st.shared followed by ld.shared without bar.sync
        let st_shared = Regex::new(r"st\.shared").expect("invariant: regex pattern is valid");
        let ld_shared = Regex::new(r"ld\.shared").expect("invariant: regex pattern is valid");
        let bar_sync = Regex::new(r"bar\.sync").expect("invariant: regex pattern is valid");

        let mut last_st_shared_line: Option<usize> = None;

        for (line_num, line) in lines.iter().enumerate() {
            let trimmed = line.trim();

            if st_shared.is_match(trimmed) {
                last_st_shared_line = Some(line_num);
            } else if bar_sync.is_match(trimmed) {
                last_st_shared_line = None; // Reset after barrier
            } else if ld_shared.is_match(trimmed) {
                if let Some(st_line) = last_st_shared_line {
                    // ld.shared after st.shared without bar.sync
                    bugs.push(PtxBug {
                        class: PtxBugClass::MissingBarrierSync,
                        line: line_num + 1,
                        instruction: format!(
                            "st.shared at line {}, ld.shared at line {}",
                            st_line + 1,
                            line_num + 1
                        ),
                        message: "ld.shared follows st.shared without barrier synchronization"
                            .to_string(),
                        fix: Some(format!(
                            "Add bar.sync 0; between lines {} and {}",
                            st_line + 1,
                            line_num + 1
                        )),
                    });
                }
            }
        }

        bugs
    }

    /// Detect register spills to local memory
    pub(in crate::ptx::bugs::analyzer) fn detect_register_spills(
        &self,
        ptx: &str,
        lines: &[&str],
    ) -> Vec<PtxBug> {
        let mut bugs = Vec::new();

        // Spills manifest as .local memory usage
        let local_pattern = Regex::new(r"\.local").expect("invariant: regex pattern is valid");
        let spill_count = local_pattern.find_iter(ptx).count();

        if spill_count > 0 {
            // Find the first .local declaration
            let mut first_local_line = 0;
            for (line_num, line) in lines.iter().enumerate() {
                if local_pattern.is_match(line) {
                    first_local_line = line_num + 1;
                    break;
                }
            }

            bugs.push(PtxBug {
                class: PtxBugClass::RegisterSpills,
                line: first_local_line,
                instruction: format!("{} .local declarations", spill_count),
                message: format!(
                    "{} potential register spills detected. High latency local memory access.",
                    spill_count
                ),
                fix: Some("Reduce live variables or increase register allocation".to_string()),
            });
        }

        bugs
    }

    /// Detect unoptimized memory access patterns (P2)
    /// Patterns: strided access, non-vectorized loads, etc.
    pub(in crate::ptx::bugs::analyzer) fn detect_unoptimized_memory(
        &self,
        ptx: &str,
        lines: &[&str],
    ) -> Vec<PtxBug> {
        let mut bugs = Vec::new();

        // Pattern 1: Multiple single-element loads that could be vectorized
        // ld.global.f32 x4 in sequence could be ld.global.v4.f32
        let single_load =
            Regex::new(r"ld\.global\.f32").expect("invariant: regex pattern is valid");
        let vector_load =
            Regex::new(r"ld\.global\.v[24]\.f32").expect("invariant: regex pattern is valid");

        let single_loads = single_load.find_iter(ptx).count();
        let vector_loads = vector_load.find_iter(ptx).count();

        // If there are many single loads but no vector loads, suggest vectorization
        if single_loads >= 4 && vector_loads == 0 {
            bugs.push(PtxBug {
                class: PtxBugClass::UnoptimizedMemoryPattern,
                line: 0,
                instruction: format!("{} single f32 loads, 0 vector loads", single_loads),
                message: "Multiple single-element loads could potentially be vectorized"
                    .to_string(),
                fix: Some(
                    "Consider using ld.global.v2.f32 or ld.global.v4.f32 for consecutive addresses"
                        .to_string(),
                ),
            });
        }

        // Pattern 2: Look for non-coalesced access hints
        // Strided access: base + i * stride where stride != sizeof(element)
        let strided_pattern = Regex::new(r"mul\.wide\.[us]32\s+%\w+,\s*%\w+,\s*(\d+)")
            .expect("invariant: regex pattern is valid");
        let mut suspicious_strides = Vec::new();

        // Known quantization block strides (not bugs - legitimate data layouts)
        // Q4_K: 144 bytes, Q5_K: 176 bytes, Q6_K: 210 bytes, Q8_K: 256 bytes
        let quantization_strides: HashSet<u32> = [144, 176, 210, 256, 512].into_iter().collect();

        for (line_num, line) in lines.iter().enumerate() {
            if let Some(caps) = strided_pattern.captures(line) {
                if let Ok(stride) = caps
                    .get(1)
                    .expect("invariant: capture group exists")
                    .as_str()
                    .parse::<u32>()
                {
                    // Suspicious if stride is not standard and not a known quantization block size
                    // Standard: 4 (f32), 8 (f64), 2 (f16), 1 (byte), or multiple of 4
                    if stride > 8 && stride % 4 != 0 && !quantization_strides.contains(&stride) {
                        suspicious_strides.push((line_num + 1, stride));
                    }
                }
            }
        }

        if !suspicious_strides.is_empty() && self.strict {
            bugs.push(PtxBug {
                class: PtxBugClass::UnoptimizedMemoryPattern,
                line: suspicious_strides[0].0,
                instruction: format!("Stride {} detected", suspicious_strides[0].1),
                message: "Non-standard stride may indicate strided (non-coalesced) access"
                    .to_string(),
                fix: Some("Consider restructuring data layout for coalesced access".to_string()),
            });
        }

        bugs
    }

    /// Detect missing thread bounds check (P1)
    /// Kernels should check tid < size before accessing memory
    pub(in crate::ptx::bugs::analyzer) fn detect_missing_bounds_check(
        &self,
        ptx: &str,
        _lines: &[&str],
    ) -> Vec<PtxBug> {
        let mut bugs = Vec::new();

        // Only check if there are memory operations
        let has_global_mem = ptx.contains("ld.global") || ptx.contains("st.global");
        if !has_global_mem {
            return bugs;
        }

        // Check for common bounds check patterns
        let has_tid = ptx.contains("%tid.") || ptx.contains("%ntid.");
        let has_setp_lt = ptx.contains("setp.lt") || ptx.contains("setp.ge");
        let has_predicated_branch = Regex::new(r"@%p\d+\s+bra")
            .expect("invariant: regex pattern is valid")
            .is_match(ptx);

        // If kernel uses tid and global memory but has no bounds check
        if has_tid && !has_setp_lt && !has_predicated_branch {
            bugs.push(PtxBug {
                class: PtxBugClass::MissingBoundsCheck,
                line: 0,
                instruction: "No setp.lt/ge with predicated branch found".to_string(),
                message: "Kernel accesses global memory but may lack thread bounds checking"
                    .to_string(),
                fix: Some("Add: setp.lt.u32 %p0, %tid, %size; @%p0 bra do_work;".to_string()),
            });
        }

        bugs
    }
}
