
/// Compute score based on generalization gap.
fn compute_gap_score(gap: f64) -> f32 {
    if gap < 0.05 {
        5.0
    } else if gap < 0.1 {
        3.0
    } else if gap < 0.2 {
        1.0
    } else {
        0.0
    }
}

/// Score Dimension 3: Model Complexity (15 points)
fn score_model_complexity(metadata: &ModelMetadata, findings: &mut Vec<Finding>) -> DimensionScore {
    let mut dim = DimensionScore::new(15.0);

    // 3.1 Parameter efficiency (5 points)
    score_parameter_efficiency(metadata, &mut dim, findings);

    // 3.2 Model interpretability (5 points)
    let model_type = metadata.model_type.unwrap_or(ScoredModelType::Other);
    dim.add_score("interpretability", model_type.interpretability_score(), 5.0);

    // 3.3 Feature importance available (5 points)
    score_feature_importance(metadata, &mut dim, findings);

    dim
}

/// Score parameter efficiency based on params/sample ratio.
fn score_parameter_efficiency(
    metadata: &ModelMetadata,
    dim: &mut DimensionScore,
    findings: &mut Vec<Finding>,
) {
    let Some(n_params) = metadata.n_parameters else {
        return;
    };
    let Some(training) = &metadata.training else {
        return;
    };
    let Some(n_samples) = training.n_samples else {
        return;
    };

    let params_per_sample = n_params as f64 / n_samples as f64;
    let efficiency_score = compute_efficiency_score(params_per_sample);
    dim.add_score("parameter_efficiency", efficiency_score, 5.0);

    if params_per_sample > 1.0 {
        findings.push(Finding::Info {
            message: format!(
                "High parameter count relative to data: {params_per_sample:.2} params/sample"
            ),
            recommendation: "Consider feature selection or simpler model architecture".to_string(),
        });
    }
}

/// Compute efficiency score from params/sample ratio.
fn compute_efficiency_score(params_per_sample: f64) -> f32 {
    // Rule of thumb: < 0.1 params/sample is efficient
    if params_per_sample < 0.1 {
        5.0
    } else if params_per_sample < 0.5 {
        4.0
    } else if params_per_sample < 1.0 {
        3.0
    } else if params_per_sample < 5.0 {
        2.0
    } else {
        1.0
    }
}

/// Score feature importance availability.
fn score_feature_importance(
    metadata: &ModelMetadata,
    dim: &mut DimensionScore,
    findings: &mut Vec<Finding>,
) {
    if metadata.flags.has_feature_importance {
        dim.add_score("feature_importance", 5.0, 5.0);
    } else {
        findings.push(Finding::Info {
            message: "No feature importance information available".to_string(),
            recommendation: "Include feature importance for model interpretability".to_string(),
        });
    }
}

/// Score Dimension 4: Documentation & Provenance (15 points)
fn score_documentation_provenance(
    metadata: &ModelMetadata,
    findings: &mut Vec<Finding>,
    _critical: &mut Vec<CriticalIssue>,
) -> DimensionScore {
    let mut dim = DimensionScore::new(15.0);
    let mut name_desc_score = 0.0;

    // 4.1 Model name and description (3 points)
    if metadata.model_name.is_some() {
        name_desc_score += 1.5;
    }
    if metadata.description.is_some() {
        name_desc_score += 1.5;
    }
    dim.add_score("name_description", name_desc_score, 3.0);

    // 4.2 Training provenance (4 points)
    let mut provenance_score = 0.0;
    if let Some(training) = &metadata.training {
        if training.source.is_some() {
            provenance_score += 1.0;
        }
        if training.n_samples.is_some() {
            provenance_score += 1.0;
        }
        if training.duration_ms.is_some() {
            provenance_score += 1.0;
        }
        if training.random_seed.is_some() {
            provenance_score += 1.0;
        }
    }
    dim.add_score("training_provenance", provenance_score, 4.0);

    if provenance_score < 2.0 {
        findings.push(Finding::Warning {
            message: "Incomplete training provenance".to_string(),
            recommendation: "Record data source, sample count, training duration, and random seed"
                .to_string(),
        });
    }

    // 4.3 Hyperparameters documented (4 points)
    let hp_count = metadata.hyperparameters.len();
    let hp_score = (hp_count as f32 / 5.0).min(1.0) * 4.0;
    dim.add_score("hyperparameters", hp_score, 4.0);

    // 4.4 Model card present (4 points)
    if metadata.flags.has_model_card {
        dim.add_score("model_card", 4.0, 4.0);
    } else {
        findings.push(Finding::Info {
            message: "No model card attached".to_string(),
            recommendation:
                "Add model card for comprehensive documentation (see Mitchell et al. 2019)"
                    .to_string(),
        });
    }

    dim
}

/// Score Dimension 5: Reproducibility (15 points)
fn score_reproducibility(metadata: &ModelMetadata, findings: &mut Vec<Finding>) -> DimensionScore {
    let mut dim = DimensionScore::new(15.0);

    // 5.1 Random seed recorded (5 points)
    if metadata
        .training
        .as_ref()
        .is_some_and(|t| t.random_seed.is_some())
    {
        dim.add_score("random_seed", 5.0, 5.0);
    } else {
        findings.push(Finding::Warning {
            message: "No random seed recorded".to_string(),
            recommendation: "Set and record random seed for reproducibility".to_string(),
        });
    }

    // 5.2 Framework version recorded (3 points)
    if metadata.aprender_version.is_some() {
        dim.add_score("framework_version", 3.0, 3.0);
    }

    // 5.3 Data preprocessing documented (4 points)
    if metadata.flags.has_preprocessing_steps {
        dim.add_score("preprocessing", 4.0, 4.0);
    }

    // 5.4 Checksum/hash for integrity (3 points)
    // Always present in valid .apr files, give full points
    dim.add_score("checksum", 3.0, 3.0);

    dim
}

/// Score Dimension 6: Security & Safety (10 points)
fn score_security_safety(
    metadata: &ModelMetadata,
    config: &ScoringConfig,
    findings: &mut Vec<Finding>,
    critical: &mut Vec<CriticalIssue>,
) -> DimensionScore {
    let mut dim = DimensionScore::new(10.0);

    // 6.1 Model signed (4 points)
    if metadata.flags.is_signed {
        dim.add_score("signed", 4.0, 4.0);
    } else if config.require_signed {
        critical.push(CriticalIssue {
            severity: Severity::High,
            message: "Model is not signed".to_string(),
            action: "Sign model with Ed25519 key for deployment".to_string(),
        });
    } else {
        findings.push(Finding::Info {
            message: "Model is not cryptographically signed".to_string(),
            recommendation: "Consider signing models for production deployment".to_string(),
        });
    }

    // 6.2 No sensitive data in metadata (3 points)
    // Check for potential secrets in metadata
    let has_secrets = metadata.custom.keys().any(|k| {
        let k_lower = k.to_lowercase();
        k_lower.contains("password")
            || k_lower.contains("secret")
            || k_lower.contains("api_key")
            || k_lower.contains("token")
    });

    if has_secrets {
        critical.push(CriticalIssue {
            severity: Severity::Critical,
            message: "Potential secrets detected in model metadata".to_string(),
            action: "Remove all sensitive data from model metadata before distribution".to_string(),
        });
    } else {
        dim.add_score("no_secrets", 3.0, 3.0);
    }

    // 6.3 Input validation documented (3 points)
    let has_input_bounds = metadata.custom.contains_key("input_bounds")
        || metadata.custom.contains_key("input_schema")
        || metadata.custom.contains_key("feature_ranges");

    if has_input_bounds {
        dim.add_score("input_validation", 3.0, 3.0);
    } else {
        findings.push(Finding::Info {
            message: "No input validation bounds documented".to_string(),
            recommendation: "Document expected input ranges for safe inference".to_string(),
        });
    }

    dim
}

#[cfg(test)]
mod tests;
