impl MqsCalculator {
    /// Check gateway conditions (G0-G4)
    fn check_gateways(&self, evidence: &[Evidence]) -> Vec<GatewayResult> {
        vec![
            check_g0_model_integrity(evidence),
            check_g1_model_loads(evidence),
            check_g2_basic_inference(evidence),
            check_g3_no_crashes(evidence),
            check_g4_no_garbage(evidence),
        ]
    }

    /// Calculate category scores from evidence
    fn calculate_categories(&self, evidence: &[Evidence]) -> CategoryScores {
        // Tally pass/total per category using a map
        let mut tallies: HashMap<String, (usize, usize)> = HashMap::new();
        for e in evidence {
            let cat = Self::extract_category(&e.gate_id);
            let key = match cat.as_str() {
                "QUAL" | "PERF" | "STAB" | "COMP" | "EDGE" | "REGR" => cat,
                other => {
                    eprintln!("[WARN] calculate_categories: extract_category returned unknown '{other}' for gate_id '{}', defaulting to QUAL", e.gate_id);
                    "QUAL".to_string()
                }
            };
            // Skipped/Timeout tests don't count toward category scoring
            // (Popper: only Corroborated/Falsified outcomes are evidence)
            if matches!(e.outcome, Outcome::Skipped | Outcome::Timeout) {
                continue;
            }
            let entry = tallies.entry(key).or_insert((0, 0));
            entry.1 += 1;
            if e.outcome == Outcome::Corroborated {
                entry.0 += 1;
            }
        }

        let score_for = |cat: &str, max: u32| -> u32 {
            let &(pass, total) = tallies.get(cat).unwrap_or(&(0, 0));
            Self::proportional_score(pass, total, max)
        };

        // Categories with 0 tests score zero (Popper: untested ≠ qualified)
        CategoryScores {
            qual: score_for("QUAL", CategoryScores::MAX_QUAL),
            perf: score_for("PERF", CategoryScores::MAX_PERF),
            stab: score_for("STAB", CategoryScores::MAX_STAB),
            comp: score_for("COMP", CategoryScores::MAX_COMP),
            edge: score_for("EDGE", CategoryScores::MAX_EDGE),
            regr: score_for("REGR", CategoryScores::MAX_REGR),
        }
    }

    /// Extract MQS category from gate ID.
    ///
    /// Maps gate IDs to the 6 MQS categories via:
    /// 1. Prefix lookup table (longest prefix first)
    /// 2. Serve battery suffix mapping (`F-A{1-6}-SUFFIX-001`)
    /// 3. Fallback: `F-{CATEGORY}-xxx` pattern
    ///
    /// Public so markdown report can reuse (DRY — single source of truth).
    #[must_use]
    pub fn extract_category(gate_id: &str) -> String {
        // Prefix -> category mapping (order matters: longer prefixes first)
        const PREFIX_MAP: &[(&str, &str)] = &[
            ("F-CONV-RT", "REGR"),
            ("F-CONV-IDEM", "REGR"),
            ("F-CONV-COM", "REGR"),
            ("F-CONV", "COMP"),
            ("F-CONTRACT", "COMP"),
            ("F-GOLDEN-RULE", "REGR"), // Convert→infer→diff → regression
            ("F-INT", "STAB"),         // Process integrity → stability
            ("F-SEC", "STAB"),         // Security/DoS detection → stability
            ("F-NUM", "STAB"),         // Numerical stability → stability
            ("F-PROFILE", "PERF"),     // Performance profiling → performance
            ("F-OLLAMA-003", "PERF"),  // TTFT comparison → performance
            ("F-OLLAMA-004", "COMP"),  // API endpoint parity → compatibility
            ("F-OLLAMA-005", "COMP"),  // GGUF loadability → compatibility
            ("F-OLLAMA-PULL", "COMP"), // Ollama pull → compatibility
            ("F-OLLAMA", "QUAL"),      // Output match → quality (after specific prefixes)
            ("F-HF-PARITY", "QUAL"),   // HF parity checks → quality
            ("F-LAYOUT", "STAB"),      // Layout contract → stability
            ("F-PERF-", "PERF"),       // Performance CI gates (F-PERF-001..F-PERF-006)
            ("G0-", "STAB"),
            ("G3-", "STAB"),           // G3 crash/panic evidence → stability
            ("T1-QUANT", "COMP"),   // Quantize → compatibility
            ("T2-IMPORT", "COMP"),  // Import → compatibility
            ("T3-PRUNE", "QUAL"),   // Prune → quality
            ("T4-DISTILL", "QUAL"), // Distill → quality
        ];

        for &(prefix, cat) in PREFIX_MAP {
            if gate_id.starts_with(prefix) {
                return cat.to_string();
            }
        }

        // Serve battery: F-A{1-6}-SUFFIX-001 → map suffix to category.
        // COMP/CHAT/STREAM/CSTREAM/INFO/MODELS/TMPL → API compatibility (COMP)
        // ERR → stability (STAB)
        // METRICS/PERF → performance (PERF)
        // CHARS → edge cases (EDGE)
        // rest (001, STOP, EOS, DETERM, MULTI, TOK, MAXTOK, SCHEMA) → quality (QUAL)
        if let Some(suffix) = Self::extract_serve_suffix(gate_id) {
            return match suffix {
                "COMP" | "CHAT" | "STREAM" | "CSTREAM" | "INFO" | "MODELS" | "TMPL" => "COMP",
                "ERR" => "STAB",
                "METRICS" | "PERF" => "PERF",
                "CHARS" => "EDGE",
                _ => "QUAL", // 001, STOP, EOS, DETERM, MULTI, TOK, MAXTOK, SCHEMA
            }
            .to_string();
        }

        // Parse F-{CATEGORY}-xxx pattern
        const CATEGORIES: [&str; 6] = ["QUAL", "PERF", "STAB", "COMP", "EDGE", "REGR"];
        gate_id
            .split('-')
            .nth(1)
            .map(str::to_uppercase)
            .filter(|s| CATEGORIES.contains(&s.as_str()))
            .unwrap_or_else(|| {
                eprintln!("[WARN] extract_category: unrecognised gate_id '{gate_id}', defaulting to QUAL");
                "QUAL".to_string()
            })
    }

    /// Extract the suffix from a serve battery gate ID.
    ///
    /// Pattern: `F-A{1-6}-SUFFIX-001` → returns SUFFIX.
    /// For `F-A{1-6}-001` (no suffix), returns `"001"`.
    /// Returns None if gate_id doesn't match this pattern.
    fn extract_serve_suffix(gate_id: &str) -> Option<&str> {
        let parts: Vec<&str> = gate_id.split('-').collect();
        // F-A5-STREAM-001 → ["F", "A5", "STREAM", "001"] (4 parts, suffix = parts[2])
        // F-A5-001 → ["F", "A5", "001"] (3 parts, suffix = parts[2] = "001")
        if parts.len() < 3 {
            return None;
        }
        let modality = parts[1];
        if !matches!(modality, "A1" | "A2" | "A3" | "A4" | "A5" | "A6") {
            return None;
        }
        Some(parts[2])
    }

    /// Calculate proportional score. Untested categories score zero.
    /// Rationale (Popper): absence of evidence means the category was never
    /// subjected to falsification — it cannot earn points it never defended.
    fn proportional_score(passed: usize, total: usize, max: u32) -> u32 {
        if total == 0 {
            return 0;
        }
        // Clamp passed to total to prevent corrupted evidence from inflating scores
        let clamped = passed.min(total);
        let ratio = clamped as f64 / total as f64;
        (ratio * f64::from(max)).round() as u32
    }

    /// Normalize raw score to 0-100 using logarithmic scaling
    /// This makes achieving 100/100 extremely difficult
    #[cfg(test)]
    fn normalize_score(&self, raw: u32, pre_penalty: u32) -> f64 {
        self.normalize_score_with_max(raw, pre_penalty, CategoryScores::MAX_TOTAL)
    }

    /// Normalize with explicit max possible score.
    /// When proof bonus is active, max includes the bonus cap.
    fn normalize_score_with_max(&self, raw: u32, pre_penalty: u32, max_possible: u32) -> f64 {
        if pre_penalty == 0 || max_possible == 0 {
            return 0.0;
        }

        let ratio = f64::from(raw) / f64::from(max_possible);

        // Apply logarithmic scaling to make high scores harder
        // f(x) = 100 * (log(1 + 9x) / log(10))
        // This maps [0,1] to [0,100] with diminishing returns
        let normalized = 100.0 * (1.0 + 9.0 * ratio).ln() / 10_f64.ln();

        // Clamp to valid range
        normalized.clamp(0.0, 100.0)
    }

    /// Calculate letter grade from normalized score
    fn calculate_grade(score: f64) -> String {
        const GRADE_TABLE: &[(f64, &str)] = &[
            (97.0, "A+"),
            (93.0, "A"),
            (90.0, "A-"),
            (87.0, "B+"),
            (83.0, "B"),
            (80.0, "B-"),
            (77.0, "C+"),
            (73.0, "C"),
            (70.0, "C-"),
            (67.0, "D+"),
            (63.0, "D"),
            (60.0, "D-"),
        ];
        GRADE_TABLE
            .iter()
            .find(|(threshold, _)| score >= *threshold)
            .map_or_else(|| "F".to_string(), |(_, grade)| (*grade).to_string())
    }
}

/// G0: Model integrity — all G0-* sub-gates (INTEGRITY, DIM, FORMAT, VALIDATE,
/// TENSOR, LAYOUT, PULL). Scored independently of the executor's Jidoka early
/// returns so `score`/`report` CLI catch every G0 failure.
fn check_g0_model_integrity(evidence: &[Evidence]) -> GatewayResult {
    const TITLE: &str = "Model integrity (config/tensor/format/layout)";
    let g0_failures: Vec<&Evidence> = evidence
        .iter()
        .filter(|e| e.gate_id.starts_with("G0-") && e.outcome.is_fail())
        .collect();
    if g0_failures.is_empty() {
        return GatewayResult::passed("G0", TITLE);
    }
    let error_details: Vec<&str> = g0_failures.iter().map(|e| e.reason.as_str()).collect();
    GatewayResult::failed(
        "G0",
        TITLE,
        format!(
            "{} G0 check(s) failed: {}",
            g0_failures.len(),
            error_details.join("; ")
        ),
    )
}

/// G1: Model loads successfully. No gate_id prefix exists for G1; we infer it
/// from whether ALL non-G0 outcomes failed — that indicates the model never
/// loaded. A single non-failure proves it loaded.
fn check_g1_model_loads(evidence: &[Evidence]) -> GatewayResult {
    const TITLE: &str = "Model loads successfully";
    let non_g0_evidence: Vec<&Evidence> = evidence
        .iter()
        .filter(|e| !e.gate_id.starts_with("G0-"))
        .collect();
    let all_non_g0_failed =
        !non_g0_evidence.is_empty() && non_g0_evidence.iter().all(|e| e.outcome.is_fail());
    if all_non_g0_failed {
        GatewayResult::failed(
            "G1",
            TITLE,
            "All test attempts failed — model may not have loaded",
        )
    } else {
        GatewayResult::passed("G1", TITLE)
    }
}

/// G2: Basic inference works — any G2-prefixed evidence with a failure outcome.
fn check_g2_basic_inference(evidence: &[Evidence]) -> GatewayResult {
    const TITLE: &str = "Basic inference works";
    let has_inference_failure = evidence
        .iter()
        .any(|e| e.gate_id.starts_with("G2") && e.outcome.is_fail());
    if has_inference_failure {
        GatewayResult::failed("G2", TITLE, "Inference failed")
    } else {
        GatewayResult::passed("G2", TITLE)
    }
}

/// G3: No crashes — any evidence with `Outcome::Crashed`.
fn check_g3_no_crashes(evidence: &[Evidence]) -> GatewayResult {
    const TITLE: &str = "No crashes";
    let crash_count = evidence
        .iter()
        .filter(|e| e.outcome == Outcome::Crashed)
        .count();
    if crash_count > 0 {
        GatewayResult::failed("G3", TITLE, format!("{crash_count} crash(es) detected"))
    } else {
        GatewayResult::passed("G3", TITLE)
    }
}

/// G4: Output is not garbage. The GarbageOracle emits evidence with gate_ids
/// like `F-A1-001` (QUAL category), not `G4-*`. Detect via
/// `scenario.oracle_type == "garbage"`. Threshold: >25% garbage-oracle failures.
fn check_g4_no_garbage(evidence: &[Evidence]) -> GatewayResult {
    const TITLE: &str = "Output is not garbage";
    let garbage_tests: Vec<&Evidence> = evidence
        .iter()
        .filter(|e| e.scenario.oracle_type == "garbage")
        .collect();
    let garbage_failures = garbage_tests.iter().filter(|e| e.outcome.is_fail()).count();
    let garbage_total = garbage_tests.len();
    if garbage_total > 0 && garbage_failures * 4 > garbage_total {
        GatewayResult::failed(
            "G4",
            TITLE,
            format!(
                "{garbage_failures}/{garbage_total} garbage-oracle tests failed (>25% threshold)"
            ),
        )
    } else {
        GatewayResult::passed("G4", TITLE)
    }
}
