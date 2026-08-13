
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
        let mut all_samples: Vec<(Sample, u8)> = Vec::new();

        for source in &self.sources {
            collect_source_samples(source, &mut all_samples, &mut provenance);
        }

        // Sort by priority (higher first) so dedup keeps the highest-priority copy.
        all_samples.sort_by_key(|b| std::cmp::Reverse(b.1));

        let (mut buffer, duplicates) = self.build_buffer(&all_samples);
        provenance.duplicates_removed = duplicates;
        provenance.set_final_size(buffer.len());

        if let Some(seed) = self.shuffle_seed {
            shuffle_buffer(&mut buffer, seed);
        }

        Ok((buffer, provenance))
    }

    fn build_buffer(
        &self,
        all_samples: &[(Sample, u8)],
    ) -> (CorpusBuffer, usize) {
        let config = CorpusBufferConfig {
            max_size: all_samples.len(),
            deduplicate: self.deduplicate,
            policy: EvictionPolicy::FIFO,
            seed: self.shuffle_seed,
        };
        let mut buffer = CorpusBuffer::with_config(config);
        let mut duplicates = 0;
        for (sample, _) in all_samples {
            if !buffer.add(sample.clone()) {
                duplicates += 1;
            }
        }
        (buffer, duplicates)
    }
}

/// Expand one source's samples into the merge buffer per its `weight` and
/// record the provenance row.
fn collect_source_samples(
    source: &CorpusSource,
    all_samples: &mut Vec<(Sample, u8)>,
    provenance: &mut CorpusProvenance,
) {
    let original_count = source.samples.len();
    let effective_count = (original_count as f64 * source.weight).round() as usize;

    if source.weight >= 1.0 {
        expand_oversampled_source(source, all_samples);
    } else {
        subsample_source(source, all_samples);
    }

    provenance.add_source(&source.name, original_count, effective_count);
}

/// Oversample: replicate each sample `floor(weight)` times then add a
/// partial pass for the fractional remainder.
fn expand_oversampled_source(source: &CorpusSource, all_samples: &mut Vec<(Sample, u8)>) {
    let repeats = source.weight.floor() as usize;
    let remainder = source.weight.fract();

    for sample in &source.samples {
        for _ in 0..repeats {
            let mut s = sample.clone();
            s.weight *= source.weight;
            all_samples.push((s, source.priority));
        }
    }

    let extra = (source.samples.len() as f64 * remainder).round() as usize;
    for sample in source.samples.iter().take(extra) {
        let mut s = sample.clone();
        s.weight *= source.weight;
        all_samples.push((s, source.priority));
    }
}

/// Subsample: take the leading `len * weight` samples without re-weighting.
fn subsample_source(source: &CorpusSource, all_samples: &mut Vec<(Sample, u8)>) {
    let take = (source.samples.len() as f64 * source.weight).round() as usize;
    for sample in source.samples.iter().take(take) {
        all_samples.push((sample.clone(), source.priority));
    }
}

/// In-place Fisher-Yates shuffle seeded by `seed`.
fn shuffle_buffer(buffer: &mut CorpusBuffer, seed: u64) {
    buffer.rng_state = seed;
    let n = buffer.samples.len();
    for i in (1..n).rev() {
        let j = (buffer.next_random() as usize) % (i + 1);
        buffer.samples.swap(i, j);
    }
}

impl Default for CorpusMerger {
    fn default() -> Self {
        Self::new()
    }
}
