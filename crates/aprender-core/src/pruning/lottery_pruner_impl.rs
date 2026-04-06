
impl Pruner for LotteryTicketPruner {
    fn generate_mask(
        &self,
        scores: &ImportanceScores,
        target_sparsity: f32,
        pattern: SparsityPattern,
    ) -> Result<SparsityMask, PruningError> {
        match pattern {
            SparsityPattern::Unstructured => {
                generate_unstructured_mask(&scores.values, target_sparsity)
            }
            _ => Err(PruningError::InvalidSparsity {
                value: target_sparsity,
                constraint: format!("LTH only supports unstructured pruning, got {pattern:?}"),
            }),
        }
    }

    fn apply_mask(
        &self,
        module: &mut dyn Module,
        mask: &SparsityMask,
    ) -> Result<PruningResult, PruningError> {
        let mut params = module.parameters_mut();
        if params.is_empty() {
            return Err(PruningError::NoParameters {
                module: "module".to_string(),
            });
        }

        let weights = &mut *params[0];
        let total = weights.data().len();

        mask.apply(weights)?;

        let zeros = weights.data().iter().filter(|&&v| v == 0.0).count();
        let achieved_sparsity = zeros as f32 / total as f32;

        Ok(PruningResult::new(achieved_sparsity, zeros, total))
    }

    fn importance(&self) -> &dyn super::importance::Importance {
        &self.importance
    }

    fn name(&self) -> &'static str {
        "lottery_ticket_pruner"
    }
}

#[cfg(test)]
#[path = "lottery_tests.rs"]
mod tests;
