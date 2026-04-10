impl MqsCalculator {
    /// Check gateway conditions (G0-G4)
    fn check_gateways(&self, evidence: &[Evidence]) -> Vec<GatewayResult> {
        let mut results = Vec::new();

        // G0: Model integrity — all G0-* sub-gates (INTEGRITY, DIM, FORMAT, VALIDATE,
        // TENSOR, LAYOUT, PULL). The executor enforces Jidoka early returns, but when
        // evidence is scored independently (via `score` or `report` CLI), we must catch
        // ALL G0 failures, not just G0-INTEGRITY.
        let g0_failures: Vec<&Evidence> = evidence
            .iter()
            .filter(|e| e.gate_id.starts_with("G0-") && e.outcome.is_fail())
            .collect();
        if g0_failures.is_empty() {
            results.push(GatewayResult::passed(
                "G0",
                "Model integrity (config/tensor/format/layout)",
            ));
        } else {
            let error_details: Vec<&str> = g0_failures
                .iter()
                .map(|e| e.reason.as_str())
                .collect();
            results.push(GatewayResult::failed(
                "G0",
                "Model integrity (config/tensor/format/layout)",
                format!(
                    "{} G0 check(s) failed: {}",
                    g0_failures.len(),
                    error_details.join("; ")
                ),
            ));
        }

        // G1: Model loads successfully
        // No evidence uses "G1-*" gate_id prefix directly. Instead, a model that
        // fails to load causes ALL tests to fail. G1 fails when all non-G0 outcomes
        // are failures (Falsified/Timeout/Crashed) with zero successes — indicating
        // the model never loaded. If any test succeeds, the model loaded.
        let non_g0_evidence: Vec<&Evidence> = evidence
            .iter()
            .filter(|e| !e.gate_id.starts_with("G0-"))
            .collect();
        let all_non_g0_failed = !non_g0_evidence.is_empty()
            && non_g0_evidence
                .iter()
                .all(|e| e.outcome.is_fail());
        if all_non_g0_failed {
            results.push(GatewayResult::failed(
                "G1",
                "Model loads successfully",
                "All test attempts failed — model may not have loaded",
            ));
        } else {
            results.push(GatewayResult::passed("G1", "Model loads successfully"));
        }

        // G2: Basic inference works
        let has_inference_failure = evidence
            .iter()
            .any(|e| e.gate_id.starts_with("G2") && e.outcome.is_fail());
        if has_inference_failure {
            results.push(GatewayResult::failed(
                "G2",
                "Basic inference works",
                "Inference failed",
            ));
        } else {
            results.push(GatewayResult::passed("G2", "Basic inference works"));
        }

        // G3: No crashes
        let crash_count = evidence
            .iter()
            .filter(|e| e.outcome == Outcome::Crashed)
            .count();
        if crash_count > 0 {
            results.push(GatewayResult::failed(
                "G3",
                "No crashes",
                format!("{crash_count} crash(es) detected"),
            ));
        } else {
            results.push(GatewayResult::passed("G3", "No crashes"));
        }

        // G4: Output is not garbage
        // The GarbageOracle produces evidence with gate_ids like "F-A1-001" (QUAL
        // category), not "G4-*". Detect garbage by checking oracle_type == "garbage"
        // on falsified evidence. Threshold: >25% garbage across garbage-oracle tests.
        let garbage_tests: Vec<&Evidence> = evidence
            .iter()
            .filter(|e| e.scenario.oracle_type == "garbage")
            .collect();
        let garbage_failures = garbage_tests
            .iter()
            .filter(|e| e.outcome.is_fail())
            .count();
        let garbage_total = garbage_tests.len();
        if garbage_total > 0 && garbage_failures * 4 > garbage_total {
            // More than 25% garbage output (use multiplication to avoid integer truncation)
            results.push(GatewayResult::failed(
                "G4",
                "Output is not garbage",
                format!(
                    "{garbage_failures}/{garbage_total} garbage-oracle tests failed (>{} threshold)",
                    "25%"
                ),
            ));
        } else {
            results.push(GatewayResult::passed("G4", "Output is not garbage"));
        }

        results
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
