
impl VoiceConverter for AutoVcConverter {
    fn config(&self) -> &VoiceConversionConfig {
        &self.config
    }

    fn convert(
        &self,
        source_audio: &[f32],
        _source_embedding: Option<&SpeakerEmbedding>,
        target_embedding: &SpeakerEmbedding,
    ) -> VoiceResult<ConversionResult> {
        if source_audio.is_empty() {
            return Err(VoiceError::InvalidAudio("Empty source audio".to_string()));
        }

        // Validate target embedding dimension
        if target_embedding.dim() != self.config.speaker_dim {
            return Err(VoiceError::DimensionMismatch {
                expected: self.config.speaker_dim,
                got: target_embedding.dim(),
            });
        }

        // Extract content features (speaker-independent)
        let content = self.extract_content(source_audio)?;

        // Synthesize with target speaker
        let converted_audio = self.synthesize(&content, target_embedding)?;

        let mut result = ConversionResult::new(converted_audio, self.config.sample_rate);

        // Estimate quality metrics (placeholder - would use neural network)
        result = result.with_metrics(0.85, 0.15, 0.75);

        Ok(result)
    }

    fn extract_content(&self, audio: &[f32]) -> VoiceResult<Vec<Vec<f32>>> {
        if audio.is_empty() {
            return Err(VoiceError::InvalidAudio("Empty audio".to_string()));
        }

        // Simulated content extraction (would use neural network)
        let frame_samples = self.config.frame_samples();
        let num_frames = (audio.len() / frame_samples).max(1);
        let content_dim = self.config.content_dim;

        // Create bottleneck features (downsampled content)
        let downsampled_frames = num_frames / self.downsample_factor.max(1);
        let downsampled_frames = downsampled_frames.max(1);

        let content: Vec<Vec<f32>> = (0..downsampled_frames)
            .map(|i| {
                // Simplified: average energy over frame range as placeholder
                let start_sample = i * self.downsample_factor * frame_samples;
                let end_sample =
                    ((i + 1) * self.downsample_factor * frame_samples).min(audio.len());

                let energy = if end_sample > start_sample {
                    audio[start_sample..end_sample]
                        .iter()
                        .map(|x| x * x)
                        .sum::<f32>()
                        .sqrt()
                        / (end_sample - start_sample) as f32
                } else {
                    0.0
                };

                // Create content vector (placeholder)
                vec![energy; content_dim]
            })
            .collect();

        Ok(content)
    }

    fn synthesize(
        &self,
        content: &[Vec<f32>],
        _speaker: &SpeakerEmbedding,
    ) -> VoiceResult<Vec<f32>> {
        if content.is_empty() {
            return Err(VoiceError::InvalidAudio(
                "Empty content features".to_string(),
            ));
        }

        // Simulated synthesis (would use neural vocoder)
        let frame_samples = self.config.frame_samples();
        let num_output_frames = content.len() * self.downsample_factor;
        let output_len = num_output_frames * frame_samples;

        // Generate placeholder audio (would use decoder + vocoder)
        let audio: Vec<f32> = (0..output_len)
            .map(|i| {
                let frame_idx = (i / frame_samples) / self.downsample_factor.max(1);
                let frame_idx = frame_idx.min(content.len().saturating_sub(1));

                // Use content energy to modulate output
                let energy = content
                    .get(frame_idx)
                    .and_then(|f| f.first())
                    .copied()
                    .unwrap_or(0.0);

                // Generate simple sine wave modulated by energy
                let t = i as f32 / self.config.sample_rate as f32;
                let freq = 200.0; // Base frequency
                (2.0 * std::f32::consts::PI * freq * t).sin() * energy * 0.5
            })
            .collect();

        Ok(audio)
    }
}

impl Default for AutoVcConverter {
    fn default() -> Self {
        Self::default_autovc()
    }
}

// ============================================================================
// PPG-based Converter
// ============================================================================

/// PPG (Phonetic Posteriorgram) based voice converter.
///
/// Uses ASR-derived phonetic features as the content bottleneck:
/// - More explicit phonetic representation
/// - Better for cross-lingual conversion
/// - Requires pre-trained ASR model for PPG extraction
#[derive(Debug, Clone)]
pub struct PpgConverter {
    /// Configuration
    config: VoiceConversionConfig,
    /// Number of phoneme classes (typical: 144 for multilingual)
    num_phonemes: usize,
}

impl PpgConverter {
    /// Create a new PPG-based converter
    #[must_use]
    pub fn new(config: VoiceConversionConfig, num_phonemes: usize) -> Self {
        Self {
            config,
            num_phonemes,
        }
    }

    /// Create with default PPG config
    #[must_use]
    pub fn default_ppg() -> Self {
        Self::new(VoiceConversionConfig::ppg_based(), 144)
    }

    /// Get number of phoneme classes
    #[must_use]
    pub fn num_phonemes(&self) -> usize {
        self.num_phonemes
    }
}

impl VoiceConverter for PpgConverter {
    fn config(&self) -> &VoiceConversionConfig {
        &self.config
    }

    fn convert(
        &self,
        source_audio: &[f32],
        _source_embedding: Option<&SpeakerEmbedding>,
        target_embedding: &SpeakerEmbedding,
    ) -> VoiceResult<ConversionResult> {
        if source_audio.is_empty() {
            return Err(VoiceError::InvalidAudio("Empty source audio".to_string()));
        }

        // Extract PPG features
        let ppg = self.extract_content(source_audio)?;

        // Synthesize with target speaker
        let converted_audio = self.synthesize(&ppg, target_embedding)?;

        let mut result = ConversionResult::new(converted_audio, self.config.sample_rate);
        result = result.with_metrics(0.80, 0.10, 0.80);

        Ok(result)
    }

    fn extract_content(&self, audio: &[f32]) -> VoiceResult<Vec<Vec<f32>>> {
        if audio.is_empty() {
            return Err(VoiceError::InvalidAudio("Empty audio".to_string()));
        }

        // Simulated PPG extraction (would use ASR model)
        let frame_samples = self.config.frame_samples();
        let num_frames = (audio.len() / frame_samples).max(1);

        // Generate PPG (phonetic posteriors)
        let ppg: Vec<Vec<f32>> = (0..num_frames)
            .map(|i| {
                let start = i * frame_samples;
                let end = ((i + 1) * frame_samples).min(audio.len());

                // Simple energy-based placeholder
                let energy = if end > start {
                    audio[start..end].iter().map(|x| x.abs()).sum::<f32>() / (end - start) as f32
                } else {
                    0.0
                };

                // Create sparse phonetic posterior (placeholder)
                let mut posteriors = vec![0.0f32; self.num_phonemes];
                let dominant_phoneme = ((energy * 100.0) as usize) % self.num_phonemes;
                if let Some(p) = posteriors.get_mut(dominant_phoneme) {
                    *p = 0.8;
                }
                // Spread remaining probability
                for p in &mut posteriors {
                    *p += 0.2 / self.num_phonemes as f32;
                }
                posteriors
            })
            .collect();

        Ok(ppg)
    }

    fn synthesize(
        &self,
        content: &[Vec<f32>],
        _speaker: &SpeakerEmbedding,
    ) -> VoiceResult<Vec<f32>> {
        if content.is_empty() {
            return Err(VoiceError::InvalidAudio("Empty PPG features".to_string()));
        }

        // Simulated synthesis (would use acoustic model + vocoder)
        let frame_samples = self.config.frame_samples();
        let output_len = content.len() * frame_samples;

        let audio: Vec<f32> = (0..output_len)
            .map(|i| {
                let frame_idx = (i / frame_samples).min(content.len().saturating_sub(1));
                let posteriors = &content[frame_idx];

                // Find dominant phoneme
                let (max_idx, max_val) = posteriors
                    .iter()
                    .enumerate()
                    .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                    .unwrap_or((0, &0.0));

                // Generate frequency based on phoneme class
                let base_freq = 100.0 + (max_idx as f32 * 10.0);
                let t = i as f32 / self.config.sample_rate as f32;
                (2.0 * std::f32::consts::PI * base_freq * t).sin() * max_val * 0.3
            })
            .collect();

        Ok(audio)
    }
}

impl Default for PpgConverter {
    fn default() -> Self {
        Self::default_ppg()
    }
}

// ============================================================================
// Pitch Conversion Utilities
// ============================================================================

/// Convert F0 (fundamental frequency) contour from source to target speaker.
///
/// # Arguments
/// * `f0_contour` - Source F0 values in Hz (0.0 for unvoiced)
/// * `source_mean` - Mean F0 of source speaker
/// * `source_std` - Standard deviation of source F0
/// * `target_mean` - Mean F0 of target speaker
/// * `target_std` - Standard deviation of target F0
///
/// # Returns
/// Converted F0 contour
#[must_use]
pub fn convert_f0(
    f0_contour: &[f32],
    source_mean: f32,
    source_std: f32,
    target_mean: f32,
    target_std: f32,
) -> Vec<f32> {
    if f0_contour.is_empty() || source_std <= 0.0 {
        return f0_contour.to_vec();
    }

    let target_std = if target_std <= 0.0 {
        source_std
    } else {
        target_std
    };

    f0_contour
        .iter()
        .map(|&f0| {
            if f0 <= 0.0 {
                0.0 // Preserve unvoiced
            } else {
                // Linear transformation: (f0 - src_mean) / src_std * tgt_std + tgt_mean
                let normalized = (f0 - source_mean) / source_std;
                (normalized * target_std + target_mean).max(50.0) // Clamp to reasonable range
            }
        })
        .collect()
}

/// Calculate F0 statistics from contour (mean, std of voiced frames).
///
/// # Returns
/// (mean, std) of voiced F0 values
#[must_use]
pub fn f0_statistics(f0_contour: &[f32]) -> (f32, f32) {
    let voiced: Vec<f32> = f0_contour.iter().copied().filter(|&f| f > 0.0).collect();

    if voiced.is_empty() {
        return (0.0, 0.0);
    }

    let mean = voiced.iter().sum::<f32>() / voiced.len() as f32;
    let variance = voiced.iter().map(|&f| (f - mean).powi(2)).sum::<f32>() / voiced.len() as f32;
    let std = variance.sqrt();

    (mean, std)
}

/// Convert pitch ratio to semitone shift.
#[must_use]
pub fn ratio_to_semitones(ratio: f32) -> f32 {
    if ratio <= 0.0 {
        0.0
    } else {
        12.0 * ratio.log2()
    }
}

/// Convert semitone shift to pitch ratio.
#[must_use]
pub fn semitones_to_ratio(semitones: f32) -> f32 {
    2.0_f32.powf(semitones / 12.0)
}

// ============================================================================
// Quality Metrics
// ============================================================================

/// Calculate voice conversion quality metrics.
///
/// # Arguments
/// * `source_embedding` - Original speaker embedding
/// * `target_embedding` - Target speaker embedding
/// * `converted_embedding` - Embedding of converted audio
///
/// # Returns
/// (`source_similarity`, `target_similarity`, `conversion_score`)
#[must_use]
pub fn conversion_quality(
    source_embedding: &SpeakerEmbedding,
    target_embedding: &SpeakerEmbedding,
    converted_embedding: &SpeakerEmbedding,
) -> (f32, f32, f32) {
    // Calculate cosine similarities
    let source_sim = cosine_similarity(converted_embedding.as_slice(), source_embedding.as_slice());
    let target_sim = cosine_similarity(converted_embedding.as_slice(), target_embedding.as_slice());

    // Conversion score: high target similarity, low source similarity
    let conversion_score = target_sim * (1.0 - source_sim);

    (source_sim, target_sim, conversion_score)
}

/// ONE PATH: Delegates to `nn::functional::cosine_similarity_slice` (UCBD §4).
fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    crate::nn::functional::cosine_similarity_slice(a, b)
}
