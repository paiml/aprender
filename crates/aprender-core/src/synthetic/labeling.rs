
impl<T: Clone + std::fmt::Debug> SyntheticGenerator for WeakSupervisionGenerator<T> {
    type Input = T;
    type Output = LabeledSample<T>;

    fn generate(&self, seeds: &[T], config: &SyntheticConfig) -> Result<Vec<LabeledSample<T>>> {
        if self.labeling_functions.is_empty() {
            return Ok(Vec::new());
        }

        let target = config.target_count(seeds.len());
        let mut results = Vec::with_capacity(target.min(seeds.len()));

        for sample in seeds {
            if results.len() >= target {
                break;
            }

            // Collect votes from all LFs
            let votes = self.collect_votes(sample);

            // Aggregate votes
            if let Some((label, confidence)) = self.aggregate_votes(&votes) {
                // Check quality threshold
                if confidence >= config.quality_threshold {
                    let num_votes = votes.iter().filter(|(_, v, _)| !v.is_abstain()).count();
                    let vote_details: Vec<_> = votes
                        .iter()
                        .map(|(name, vote, _)| (name.clone(), *vote))
                        .collect();

                    results.push(
                        LabeledSample::new(sample.clone(), label, confidence)
                            .with_votes(num_votes, vote_details),
                    );
                }
            }
        }

        Ok(results)
    }

    fn quality_score(&self, generated: &LabeledSample<T>, _seed: &T) -> f32 {
        // Quality is the confidence in the label
        generated.confidence
    }

    fn diversity_score(&self, batch: &[LabeledSample<T>]) -> f32 {
        if batch.is_empty() {
            return 0.0;
        }

        // Diversity is based on label distribution entropy
        let mut label_counts: HashMap<i32, usize> = HashMap::new();
        for sample in batch {
            *label_counts.entry(sample.label).or_insert(0) += 1;
        }

        let n = batch.len() as f32;
        let mut entropy = 0.0;

        for count in label_counts.values() {
            let p = *count as f32 / n;
            if p > 0.0 {
                entropy -= p * p.ln();
            }
        }

        // Normalize by max entropy (uniform distribution over unique labels)
        let max_entropy = (label_counts.len() as f32).ln();
        if max_entropy > f32::EPSILON {
            (entropy / max_entropy).clamp(0.0, 1.0)
        } else {
            0.0
        }
    }
}

// ============================================================================
// Built-in Labeling Functions
// ============================================================================

/// Keyword-based labeling function for text.
#[derive(Debug)]
pub struct KeywordLF {
    name: String,
    keywords: Vec<String>,
    label: LabelVote,
    weight: f32,
}

impl KeywordLF {
    /// Create a new keyword labeling function.
    #[must_use]
    pub fn new(name: impl Into<String>, keywords: &[&str], label: LabelVote) -> Self {
        Self {
            name: name.into(),
            keywords: keywords.iter().map(|s| (*s).to_string()).collect(),
            label,
            weight: 1.0,
        }
    }

    /// Set the weight.
    #[must_use]
    pub fn with_weight(mut self, weight: f32) -> Self {
        self.weight = weight.max(0.0);
        self
    }
}

impl LabelingFunction<String> for KeywordLF {
    fn name(&self) -> &str {
        &self.name
    }

    fn apply(&self, sample: &String) -> LabelVote {
        let lower = sample.to_lowercase();
        if self.keywords.iter().any(|kw| lower.contains(kw)) {
            self.label
        } else {
            LabelVote::Abstain
        }
    }

    fn weight(&self) -> f32 {
        self.weight
    }
}

/// Length-based labeling function for text.
#[derive(Debug)]
pub struct LengthLF {
    name: String,
    min_len: usize,
    max_len: usize,
    label: LabelVote,
}

impl LengthLF {
    /// Create a new length labeling function.
    #[must_use]
    pub fn new(name: impl Into<String>, min_len: usize, max_len: usize, label: LabelVote) -> Self {
        Self {
            name: name.into(),
            min_len,
            max_len,
            label,
        }
    }
}

impl LabelingFunction<String> for LengthLF {
    fn name(&self) -> &str {
        &self.name
    }

    fn apply(&self, sample: &String) -> LabelVote {
        let len = sample.len();
        if len >= self.min_len && len <= self.max_len {
            self.label
        } else {
            LabelVote::Abstain
        }
    }
}

/// Regex-based labeling function for text.
#[derive(Debug)]
pub struct PatternLF {
    name: String,
    pattern: String,
    label: LabelVote,
}

impl PatternLF {
    /// Create a new pattern labeling function (simple substring match).
    #[must_use]
    pub fn new(name: impl Into<String>, pattern: impl Into<String>, label: LabelVote) -> Self {
        Self {
            name: name.into(),
            pattern: pattern.into(),
            label,
        }
    }
}

impl LabelingFunction<String> for PatternLF {
    fn name(&self) -> &str {
        &self.name
    }

    fn apply(&self, sample: &String) -> LabelVote {
        if sample.contains(&self.pattern) {
            self.label
        } else {
            LabelVote::Abstain
        }
    }
}
