//! ModelRegistry trait implementation for InMemoryRegistry

use chrono::Utc;
use std::collections::HashMap;

use super::super::comparison::VersionComparison;
use super::super::error::{RegistryError, Result};
use super::super::policy::PolicyCheckResult;
use super::super::stage::ModelStage;
use super::super::traits::ModelRegistry;
use super::super::transition::StageTransition;
use super::super::version::ModelVersion;
use super::registry::InMemoryRegistry;

impl ModelRegistry for InMemoryRegistry {
    fn register_model(&mut self, name: &str, artifact_uri: &str) -> Result<ModelVersion> {
        let version = self.next_version(name);
        let model = ModelVersion::new(name, version, artifact_uri);

        self.models.entry(name.to_string()).or_default().insert(version, model.clone());

        Ok(model)
    }

    fn get_model(&self, name: &str, version: u32) -> Result<ModelVersion> {
        self.models
            .get(name)
            .and_then(|versions| versions.get(&version))
            .cloned()
            .ok_or_else(|| RegistryError::VersionNotFound(name.to_string(), version))
    }

    fn get_latest(&self, name: &str) -> Result<ModelVersion> {
        self.models
            .get(name)
            .and_then(|versions| {
                let max_version = versions.keys().max()?;
                versions.get(max_version)
            })
            .cloned()
            .ok_or_else(|| RegistryError::ModelNotFound(name.to_string()))
    }

    fn get_latest_by_stage(&self, name: &str, stage: ModelStage) -> Option<ModelVersion> {
        self.models.get(name).and_then(|versions| {
            versions.values().filter(|m| m.stage == stage).max_by_key(|m| m.version).cloned()
        })
    }

    fn list_versions(&self, name: &str) -> Result<Vec<ModelVersion>> {
        self.models
            .get(name)
            .map(|versions| {
                let mut v: Vec<_> = versions.values().cloned().collect();
                v.sort_by_key(|m| m.version);
                v
            })
            .ok_or_else(|| RegistryError::ModelNotFound(name.to_string()))
    }

    fn transition_stage(
        &mut self,
        name: &str,
        version: u32,
        target_stage: ModelStage,
        user: Option<&str>,
    ) -> Result<()> {
        let model = self
            .models
            .get_mut(name)
            .and_then(|versions| versions.get_mut(&version))
            .ok_or_else(|| RegistryError::VersionNotFound(name.to_string(), version))?;

        if !model.stage.can_transition_to(target_stage) {
            return Err(RegistryError::InvalidTransition(model.stage, target_stage));
        }

        let from_stage = model.stage;
        model.stage = target_stage;
        model.promoted_at = Some(Utc::now());
        model.promoted_by = user.map(ToString::to_string);

        // Record transition
        self.transitions.push(StageTransition {
            model_name: name.to_string(),
            version,
            from_stage,
            to_stage: target_stage,
            timestamp: Utc::now(),
            user: user.map(ToString::to_string),
            reason: None,
        });

        Ok(())
    }

    fn compare_versions(&self, name: &str, v1: u32, v2: u32) -> Result<VersionComparison> {
        let m1 = self.get_model(name, v1)?;
        let m2 = self.get_model(name, v2)?;

        let mut metric_diffs = HashMap::new();
        let mut v2_better_count = 0;
        let mut total_comparisons = 0;

        // Compare all metrics from both versions
        let all_metrics: std::collections::HashSet<_> =
            m1.metrics.keys().chain(m2.metrics.keys()).collect();

        for metric in all_metrics {
            let val1 = m1.metrics.get(metric).copied().unwrap_or(0.0);
            let val2 = m2.metrics.get(metric).copied().unwrap_or(0.0);
            let diff = val2 - val1;
            metric_diffs.insert(metric.clone(), diff);

            // Assume higher is better for most metrics
            if diff > 0.0 {
                v2_better_count += 1;
            }
            total_comparisons += 1;
        }

        let v2_is_better = total_comparisons > 0 && v2_better_count > total_comparisons / 2;

        let summary = if v2_is_better {
            format!(
                "Version {v2} is better than {v1} on {v2_better_count}/{total_comparisons} metrics"
            )
        } else {
            format!("Version {v2} is not definitively better than {v1}")
        };

        Ok(VersionComparison { v1, v2, metric_diffs, v2_is_better, summary })
    }

    fn log_metrics(
        &mut self,
        name: &str,
        version: u32,
        metrics: HashMap<String, f64>,
    ) -> Result<()> {
        let model = self
            .models
            .get_mut(name)
            .and_then(|versions| versions.get_mut(&version))
            .ok_or_else(|| RegistryError::VersionNotFound(name.to_string(), version))?;

        model.metrics.extend(metrics);
        Ok(())
    }

    fn get_transition_history(&self, name: &str) -> Result<Vec<StageTransition>> {
        let history: Vec<_> =
            self.transitions.iter().filter(|t| t.model_name == name).cloned().collect();

        if history.is_empty() && !self.models.contains_key(name) {
            return Err(RegistryError::ModelNotFound(name.to_string()));
        }

        Ok(history)
    }

    fn set_policy(&mut self, policy: super::super::policy::PromotionPolicy) {
        self.policies.insert(policy.target_stage, policy);
    }

    fn get_policy(&self, stage: ModelStage) -> Option<&super::super::policy::PromotionPolicy> {
        self.policies.get(&stage)
    }

    fn can_promote(
        &self,
        name: &str,
        version: u32,
        target_stage: ModelStage,
        approvals: u32,
    ) -> Result<PolicyCheckResult> {
        let model = self.get_model(name, version)?;

        // Check stage transition validity
        if !model.stage.can_transition_to(target_stage) {
            return Ok(PolicyCheckResult {
                passed: false,
                failed_requirements: vec![format!(
                    "Cannot transition from {} to {}",
                    model.stage, target_stage
                )],
            });
        }

        // Check policy if exists
        if let Some(policy) = self.policies.get(&target_stage) {
            Ok(policy.check(&model, approvals))
        } else {
            // No policy = always allowed
            Ok(PolicyCheckResult { passed: true, failed_requirements: Vec::new() })
        }
    }
}
