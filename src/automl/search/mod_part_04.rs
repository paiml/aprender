
impl<S, P> SearchStrategy<P> for ActiveLearningSearch<S>
where
    S: SearchStrategy<P>,
    P: ParamKey,
{
    fn suggest(&mut self, space: &SearchSpace<P>, n: usize) -> Vec<Trial<P>> {
        // If we should stop, return empty
        if self.should_stop() {
            return Vec::new();
        }
        self.base.suggest(space, n)
    }

    fn update(&mut self, results: &[TrialResult<P>]) {
        // Collect scores for uncertainty estimation
        for result in results {
            self.scores.push(result.score);
        }

        // Update uncertainty estimate
        self.compute_uncertainty();

        // Forward to base strategy
        self.base.update(results);
    }
}

#[cfg(test)]
mod tests;
