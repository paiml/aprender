//! Experiment to `ExperimentSpec` conversion (split out of loader.rs for the pre-commit
//! complexity limit, #6025).

use super::ExperimentYaml;
use crate::edd::experiment::{
    ExperimentHypothesis, ExperimentSpec, ExperimentSpecBuilder, FalsificationAction,
    FalsificationCriterion,
};

impl ExperimentYaml {
    /// Convert to an `ExperimentSpec`.
    ///
    /// # Errors
    /// Returns error if required fields are missing.
    pub fn to_experiment_spec(&self) -> Result<ExperimentSpec, String> {
        let mut builder = ExperimentSpec::builder()
            .name(&self.metadata.name)
            .seed(self.reproducibility.seed)
            .description(&self.metadata.description);

        // Add EMC reference
        if let Some(ref emc) = self.equation_model_card {
            if !emc.emc_ref.is_empty() {
                builder = builder.emc_reference(&emc.emc_ref);
            }
        }

        // Add hypothesis
        if let Some(ref hyp) = self.hypothesis {
            let hypothesis =
                ExperimentHypothesis::new(&hyp.null_hypothesis, &hyp.alternative_hypothesis);
            builder = builder.hypothesis(hypothesis);
        }

        builder = self.add_duration(builder);
        builder = self.add_falsification(builder);

        builder.build()
    }

    fn add_duration(&self, builder: ExperimentSpecBuilder) -> ExperimentSpecBuilder {
        let Some(ref sim) = self.simulation else {
            return builder;
        };
        let Some(ref dur) = sim.duration else {
            return builder;
        };
        builder
            .warmup(dur.warmup)
            .run_length(dur.simulation)
            .replications(dur.replications)
    }

    fn add_falsification(&self, mut builder: ExperimentSpecBuilder) -> ExperimentSpecBuilder {
        let Some(ref fals) = self.falsification else {
            return builder;
        };
        for crit in &fals.criteria {
            let action = match crit.severity.to_lowercase().as_str() {
                "critical" => FalsificationAction::RejectModel,
                "major" => FalsificationAction::Stop,
                "minor" => FalsificationAction::Warn,
                _ => FalsificationAction::FlagReview,
            };
            let criterion = FalsificationCriterion::new(&crit.name, &crit.condition, action);
            builder = builder.add_falsification_criterion(criterion);
        }
        builder
    }
}
