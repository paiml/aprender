
impl CorpusProvenance {
    /// Create new provenance tracker
    #[must_use]
    pub fn new() -> Self {
        Self {
            sources: HashMap::new(),
            final_size: 0,
            duplicates_removed: 0,
        }
    }

    /// Add source contribution
    pub fn add_source(&mut self, name: &str, original: usize, effective: usize) {
        self.sources.insert(name.to_string(), (original, effective));
    }

    /// Set final merged size
    pub fn set_final_size(&mut self, size: usize) {
        self.final_size = size;
    }
}

impl Default for CorpusProvenance {
    fn default() -> Self {
        Self::new()
    }
}

/// Merge multiple data sources with configurable weighting
///
/// Used by ruchy Oracle to combine:
/// - Synthetic data
/// - Hand-crafted corpus
/// - Examples corpus
/// - Production corpus
#[derive(Debug)]
pub struct CorpusMerger {
    /// Sources to merge
    sources: Vec<CorpusSource>,
    /// Enable deduplication
    deduplicate: bool,
    /// Random seed for shuffling
    shuffle_seed: Option<u64>,
}

impl CorpusMerger {
    /// Create a new corpus merger
    #[must_use]
    pub fn new() -> Self {
        Self {
            sources: Vec::new(),
            deduplicate: true,
            shuffle_seed: None,
        }
    }

    /// Add a source
    pub fn add_source(&mut self, source: CorpusSource) -> &mut Self {
        self.sources.push(source);
        self
    }

    /// Set deduplication flag
    #[must_use]
    pub fn deduplicate(mut self, enable: bool) -> Self {
        self.deduplicate = enable;
        self
    }

    /// Set shuffle seed
    #[must_use]
    pub fn shuffle_seed(mut self, seed: u64) -> Self {
        self.shuffle_seed = Some(seed);
        self
    }

    /// Merge all sources into unified dataset
    pub fn merge(&self) -> Result<(CorpusBuffer, CorpusProvenance)> {
        let mut provenance = CorpusProvenance::new();
        let mut all_samples: Vec<(Sample, u8)> = Vec::new(); // (sample, priority)

        // Collect all samples with weights applied
        for source in &self.sources {
            let original_count = source.samples.len();
            let effective_count = (original_count as f64 * source.weight).round() as usize;

            // Sample with replacement if weight > 1, otherwise take all
            if source.weight >= 1.0 {
                // Take all and potentially repeat
                let repeats = source.weight.floor() as usize;
                let remainder = source.weight.fract();

                for sample in &source.samples {
                    for _ in 0..repeats {
                        let mut s = sample.clone();
                        s.weight *= source.weight;
                        all_samples.push((s, source.priority));
                    }
                }

                // Partial sampling for remainder
                let extra = (source.samples.len() as f64 * remainder).round() as usize;
                for sample in source.samples.iter().take(extra) {
                    let mut s = sample.clone();
                    s.weight *= source.weight;
                    all_samples.push((s, source.priority));
                }
            } else {
                // Subsample
                let take = (source.samples.len() as f64 * source.weight).round() as usize;
                for sample in source.samples.iter().take(take) {
                    all_samples.push((sample.clone(), source.priority));
                }
            }

            provenance.add_source(&source.name, original_count, effective_count);
        }

        // Sort by priority (higher first) for deduplication
        all_samples.sort_by(|a, b| b.1.cmp(&a.1));

        // Create buffer and deduplicate
        let config = CorpusBufferConfig {
            max_size: all_samples.len(),
            deduplicate: self.deduplicate,
            policy: EvictionPolicy::FIFO,
            seed: self.shuffle_seed,
        };

        let mut buffer = CorpusBuffer::with_config(config);
        let mut duplicates = 0;

        for (sample, _) in all_samples {
            if !buffer.add(sample) {
                duplicates += 1;
            }
        }

        provenance.duplicates_removed = duplicates;
        provenance.set_final_size(buffer.len());

        // Shuffle if seed provided
        if let Some(seed) = self.shuffle_seed {
            buffer.rng_state = seed;
            // Simple Fisher-Yates shuffle
            let n = buffer.samples.len();
            for i in (1..n).rev() {
                let j = (buffer.next_random() as usize) % (i + 1);
                buffer.samples.swap(i, j);
            }
        }

        Ok((buffer, provenance))
    }
}

impl Default for CorpusMerger {
    fn default() -> Self {
        Self::new()
    }
}
