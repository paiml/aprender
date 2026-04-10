
impl PatternDetector {
    /// Create detector with all patterns enabled
    #[must_use]
    pub fn new() -> Self {
        Self {
            patterns: BugPattern::all().to_vec(),
        }
    }

    /// Create detector with only P0 (critical) patterns
    #[must_use]
    pub fn critical_only() -> Self {
        Self {
            patterns: BugPattern::by_severity("P0"),
        }
    }

    /// Check for SilentFallbackWrongResource pattern
    ///
    /// Detection: Compare output from primary resource vs fallback resource.
    /// If outputs differ significantly, fallback used wrong resource.
    #[must_use]
    pub fn check_fallback_consistency(&self, primary_output: &str, fallback_output: &str) -> bool {
        // If fallback produces wildly different output, it found wrong resource
        let similarity = self.jaccard_similarity(primary_output, fallback_output);
        similarity > 0.8 // Require >80% token overlap
    }

    /// Check for MissingPostTransformValidation pattern
    ///
    /// Detection: Look for NaN, Inf, or extreme values in transformed data.
    #[must_use]
    pub fn check_tensor_validity(&self, values: &[f32]) -> TensorValidityResult {
        let mut nan_count = 0;
        let mut inf_count = 0;
        let mut zero_count = 0;
        let mut sum = 0.0f64;

        for &v in values {
            if v.is_nan() {
                nan_count += 1;
            } else if v.is_infinite() {
                inf_count += 1;
            } else if v == 0.0 {
                zero_count += 1;
            }
            sum += f64::from(v);
        }

        let mean = if values.is_empty() {
            0.0
        } else {
            sum / values.len() as f64
        };

        TensorValidityResult {
            nan_count,
            inf_count,
            zero_count,
            total: values.len(),
            mean,
            is_valid: nan_count == 0 && inf_count == 0 && mean.abs() < 100.0,
        }
    }

    /// Check for MissingCompanionData pattern
    ///
    /// Detection: Verify expected companion files exist alongside primary file.
    #[must_use]
    pub fn check_companion_files(
        &self,
        primary_path: &std::path::Path,
        required_companions: &[&str],
    ) -> CompanionCheckResult {
        let parent = primary_path.parent();
        let mut missing = Vec::new();
        let mut found = Vec::new();

        for companion in required_companions {
            let companion_path = parent.map(|p| p.join(companion));
            if companion_path.is_some_and(|p| p.exists()) {
                found.push((*companion).to_string());
            } else {
                missing.push((*companion).to_string());
            }
        }

        let all_present = found.len() == required_companions.len();
        CompanionCheckResult {
            missing,
            found,
            all_present,
        }
    }

    /// Check for PathTraversal pattern
    ///
    /// Detection: Reject paths containing traversal sequences.
    #[must_use]
    pub fn check_path_safety(&self, path: &str) -> PathSafetyResult {
        let issues = vec![
            ("../", "Parent directory traversal"),
            ("..\\", "Parent directory traversal (Windows)"),
            ("/etc/", "System directory access"),
            ("C:\\Windows", "System directory access (Windows)"),
            ("\x00", "Null byte injection"),
        ];

        let mut violations = Vec::new();
        for (pattern, description) in issues {
            if path.contains(pattern) {
                violations.push(PathViolation {
                    pattern: pattern.to_string(),
                    description: description.to_string(),
                });
            }
        }

        PathSafetyResult {
            is_safe: violations.is_empty(),
            violations,
        }
    }

    /// Check for PromptInjection pattern
    ///
    /// Detection: Look for unescaped special tokens in user input.
    #[must_use]
    pub fn check_prompt_safety(&self, prompt: &str) -> PromptSafetyResult {
        let dangerous_patterns = vec![
            ("<|", "Special token start"),
            ("|>", "Special token end"),
            ("<s>", "BOS token"),
            ("</s>", "EOS token"),
            ("[INST]", "Instruction marker"),
            ("[/INST]", "Instruction end marker"),
            ("<<SYS>>", "System prompt marker"),
            ("<</SYS>>", "System prompt end"),
        ];

        let mut found_patterns = Vec::new();
        for (pattern, description) in dangerous_patterns {
            if prompt.contains(pattern) {
                found_patterns.push(PromptPattern {
                    pattern: pattern.to_string(),
                    description: description.to_string(),
                });
            }
        }

        PromptSafetyResult {
            is_safe: found_patterns.is_empty(),
            found_patterns,
        }
    }

    /// Simple Jaccard similarity for token comparison
    fn jaccard_similarity(&self, a: &str, b: &str) -> f64 {
        let tokens_a: std::collections::HashSet<&str> = a.split_whitespace().collect();
        let tokens_b: std::collections::HashSet<&str> = b.split_whitespace().collect();

        if tokens_a.is_empty() && tokens_b.is_empty() {
            return 1.0;
        }

        let intersection = tokens_a.intersection(&tokens_b).count();
        let union = tokens_a.union(&tokens_b).count();

        if union == 0 {
            0.0
        } else {
            intersection as f64 / union as f64
        }
    }

    // =========================================================================
    // Numerical Stability Checks (F-NUM-001..004)
    // =========================================================================

    /// Check attention entropy (F-NUM-001)
    ///
    /// Attention should not collapse (entropy ≈ 0) or explode (uniform).
    /// Valid range: 0.1 < entropy < 0.9 * max_entropy
    #[must_use]
    pub fn check_attention_entropy(&self, attention_weights: &[f32]) -> NumericalStabilityResult {
        if attention_weights.is_empty() {
            return NumericalStabilityResult {
                gate_id: "F-NUM-001".to_string(),
                is_valid: false,
                value: 0.0,
                expected_range: (0.1, f64::MAX),
                description: "Empty attention weights".to_string(),
            };
        }

        // Calculate entropy: -sum(p * log(p))
        let sum: f32 = attention_weights.iter().sum();
        if sum <= 0.0 || sum.is_nan() {
            return NumericalStabilityResult {
                gate_id: "F-NUM-001".to_string(),
                is_valid: false,
                value: 0.0,
                expected_range: (0.1, f64::MAX),
                description: "Invalid attention sum".to_string(),
            };
        }

        let mut entropy = 0.0f64;
        for &w in attention_weights {
            let p = f64::from(w / sum);
            if p > 0.0 {
                entropy -= p * p.ln();
            }
        }

        // Max entropy for uniform distribution
        let max_entropy = (attention_weights.len() as f64).ln();
        let normalized_entropy = if max_entropy > 0.0 {
            entropy / max_entropy
        } else {
            0.0
        };

        // Valid: not collapsed (>0.1) and not uniform (< 0.95)
        let is_valid = normalized_entropy > 0.1 && normalized_entropy < 0.95;

        NumericalStabilityResult {
            gate_id: "F-NUM-001".to_string(),
            is_valid,
            value: normalized_entropy,
            expected_range: (0.1, 0.95),
            description: if is_valid {
                "Attention entropy in valid range".to_string()
            } else if normalized_entropy <= 0.1 {
                "Attention collapsed (entropy too low)".to_string()
            } else {
                "Attention exploded (nearly uniform)".to_string()
            },
        }
    }

    /// Check LayerNorm output (F-NUM-002)
    ///
    /// LayerNorm output should have mean ≈ 0 and std ≈ 1
    #[must_use]
    pub fn check_layernorm_output(&self, values: &[f32]) -> NumericalStabilityResult {
        if values.is_empty() {
            return NumericalStabilityResult {
                gate_id: "F-NUM-002".to_string(),
                is_valid: false,
                value: 0.0,
                expected_range: (-0.001, 0.001),
                description: "Empty LayerNorm output".to_string(),
            };
        }

        let n = values.len() as f64;
        let sum: f64 = values.iter().map(|&v| f64::from(v)).sum();
        let mean = sum / n;

        let variance: f64 = values
            .iter()
            .map(|&v| {
                let diff = f64::from(v) - mean;
                diff * diff
            })
            .sum::<f64>()
            / n;
        let std_dev = variance.sqrt();

        // Check: mean should be close to 0, std should be close to 1
        let mean_ok = mean.abs() < 0.001;
        let std_ok = (std_dev - 1.0).abs() < 0.05;
        let is_valid = mean_ok && std_ok;

        NumericalStabilityResult {
            gate_id: "F-NUM-002".to_string(),
            is_valid,
            value: mean,
            expected_range: (-0.001, 0.001),
            description: if is_valid {
                format!("LayerNorm valid: mean={mean:.6}, std={std_dev:.4}")
            } else {
                format!("LayerNorm drift: mean={mean:.6} (want ≈0), std={std_dev:.4} (want ≈1)")
            },
        }
    }

    /// Check softmax output (F-NUM-003)
    ///
    /// Softmax output must sum to 1.0 ± 1e-6
    #[must_use]
    pub fn check_softmax_sum(&self, probabilities: &[f32]) -> NumericalStabilityResult {
        let sum: f64 = probabilities.iter().map(|&p| f64::from(p)).sum();
        let tolerance = 1e-6;
        let is_valid = (sum - 1.0).abs() < tolerance;

        NumericalStabilityResult {
            gate_id: "F-NUM-003".to_string(),
            is_valid,
            value: sum,
            expected_range: (1.0 - tolerance, 1.0 + tolerance),
            description: if is_valid {
                format!("Softmax sum valid: {sum:.9}")
            } else {
                format!("Softmax sum invalid: {sum:.9} (expected 1.0 ± {tolerance})")
            },
        }
    }

    /// Check token probabilities (F-NUM-004)
    ///
    /// All probabilities must be in range [0, 1]
    #[must_use]
    pub fn check_probability_range(&self, probabilities: &[f32]) -> NumericalStabilityResult {
        let mut invalid_count = 0;
        let mut min_val = f64::MAX;
        let mut max_val = f64::MIN;

        for &p in probabilities {
            let pf = f64::from(p);
            if !(0.0..=1.0).contains(&pf) || pf.is_nan() {
                invalid_count += 1;
            }
            if pf < min_val {
                min_val = pf;
            }
            if pf > max_val {
                max_val = pf;
            }
        }

        let is_valid = invalid_count == 0;

        NumericalStabilityResult {
            gate_id: "F-NUM-004".to_string(),
            is_valid,
            value: if invalid_count > 0 {
                f64::from(invalid_count)
            } else {
                0.0
            },
            expected_range: (0.0, 1.0),
            description: if is_valid {
                format!("Probabilities valid: range [{min_val:.6}, {max_val:.6}]")
            } else {
                format!("Invalid probabilities: {invalid_count} out of range [0,1]")
            },
        }
    }

    // =========================================================================
    // DoS Protection (F-SEC-003)
    // =========================================================================

    /// Check input for DoS attack patterns (F-SEC-003)
    ///
    /// Detects: zip bombs, token floods, excessive repetition, oversized inputs
    #[must_use]
    pub fn check_dos_protection(
        &self,
        input: &str,
        config: &DosProtectionConfig,
    ) -> DosCheckResult {
        let mut violations = Vec::new();

        // Check 1: Input length limit
        if input.len() > config.max_input_bytes {
            violations.push(DosViolation {
                check: "input_length".to_string(),
                description: format!(
                    "Input too large: {} bytes (max: {})",
                    input.len(),
                    config.max_input_bytes
                ),
                severity: "P0".to_string(),
            });
        }

        // Check 2: Token count estimate (rough: 4 chars per token)
        let estimated_tokens = input.len() / 4;
        if estimated_tokens > config.max_tokens {
            violations.push(DosViolation {
                check: "token_count".to_string(),
                description: format!(
                    "Too many tokens: ~{} (max: {})",
                    estimated_tokens, config.max_tokens
                ),
                severity: "P0".to_string(),
            });
        }

        // Check 3: Repetition detection (potential zip bomb pattern)
        let repetition_ratio = self.calculate_repetition_ratio(input);
        if repetition_ratio > config.max_repetition_ratio {
            violations.push(DosViolation {
                check: "repetition".to_string(),
                description: format!(
                    "Excessive repetition: {:.1}% (max: {:.1}%)",
                    repetition_ratio * 100.0,
                    config.max_repetition_ratio * 100.0
                ),
                severity: "P1".to_string(),
            });
        }

        // Check 4: Expansion ratio (compressed data that expands)
        let unique_chars: std::collections::HashSet<char> = input.chars().collect();
        let expansion_ratio = input.len() as f64 / (unique_chars.len().max(1) as f64);
        if expansion_ratio > config.max_expansion_ratio {
            violations.push(DosViolation {
                check: "expansion".to_string(),
                description: format!(
                    "High expansion ratio: {:.1}x (max: {:.1}x)",
                    expansion_ratio, config.max_expansion_ratio
                ),
                severity: "P1".to_string(),
            });
        }

        DosCheckResult {
            gate_id: "F-SEC-003".to_string(),
            is_safe: violations.is_empty(),
            violations,
            input_bytes: input.len(),
            estimated_tokens,
            repetition_ratio,
            expansion_ratio,
        }
    }

    /// Calculate ratio of repeated n-grams in input
    fn calculate_repetition_ratio(&self, input: &str) -> f64 {
        if input.len() < 10 {
            return 0.0;
        }

        // Use 4-grams for repetition detection
        let ngram_size = 4;
        let mut ngrams: std::collections::HashMap<&str, usize> = std::collections::HashMap::new();

        for i in 0..input.len().saturating_sub(ngram_size) {
            if let Some(ngram) = input.get(i..i + ngram_size) {
                *ngrams.entry(ngram).or_insert(0) += 1;
            }
        }

        let total_ngrams = ngrams.values().sum::<usize>();
        let repeated_ngrams: usize = ngrams.values().filter(|&&c| c > 1).map(|c| c - 1).sum();

        if total_ngrams == 0 {
            0.0
        } else {
            repeated_ngrams as f64 / total_ngrams as f64
        }
    }
}
