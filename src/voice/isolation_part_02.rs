
impl SpectralSubtractionIsolator {
    /// Create a new spectral subtraction isolator
    #[must_use]
    pub fn new(config: IsolationConfig) -> Self {
        Self {
            config,
            over_subtraction: 1.5,
        }
    }

    /// Set over-subtraction factor
    #[must_use]
    pub fn with_over_subtraction(mut self, factor: f32) -> Self {
        self.over_subtraction = factor.clamp(0.5, 5.0);
        self
    }

    /// Get over-subtraction factor
    #[must_use]
    pub fn over_subtraction(&self) -> f32 {
        self.over_subtraction
    }

    /// Compute STFT magnitude spectrum (simplified)
    fn compute_stft(&self, audio: &[f32]) -> Vec<Vec<f32>> {
        let fft_size = self.config.fft_size;
        let hop_length = self.config.hop_length;
        let freq_bins = self.config.freq_bins();

        let num_frames = (audio.len().saturating_sub(fft_size)) / hop_length + 1;
        let num_frames = num_frames.max(1);

        (0..num_frames)
            .map(|i| {
                let start = i * hop_length;
                let end = (start + fft_size).min(audio.len());

                // Compute energy per frequency bin (simplified - real impl uses FFT)
                let mut magnitudes = vec![0.0f32; freq_bins];

                if end > start {
                    let frame = &audio[start..end];

                    // Simple energy-based approximation
                    let energy: f32 = frame.iter().map(|x| x * x).sum::<f32>().sqrt();

                    // Distribute energy across bins (placeholder)
                    for (bin, mag) in magnitudes.iter_mut().enumerate() {
                        // Low-pass shape approximation
                        let freq_factor = 1.0 / (1.0 + (bin as f32 / 20.0));
                        *mag = energy * freq_factor;
                    }
                }

                magnitudes
            })
            .collect()
    }

    /// Apply spectral subtraction to magnitudes
    fn subtract_noise(
        &self,
        magnitudes: &[Vec<f32>],
        noise_profile: &NoiseProfile,
    ) -> Vec<Vec<f32>> {
        let strength = self.config.reduction_strength;
        let floor = self.config.spectral_floor;
        let alpha = self.over_subtraction;

        magnitudes
            .iter()
            .map(|frame| {
                frame
                    .iter()
                    .enumerate()
                    .map(|(bin, &mag)| {
                        let noise = noise_profile.noise_magnitude(bin) * alpha * strength;
                        (mag - noise).max(mag * floor)
                    })
                    .collect()
            })
            .collect()
    }

    /// Reconstruct audio from magnitudes (simplified overlap-add)
    fn reconstruct(&self, magnitudes: &[Vec<f32>]) -> Vec<f32> {
        if magnitudes.is_empty() {
            return vec![];
        }

        let hop_length = self.config.hop_length;
        let fft_size = self.config.fft_size;
        let output_len = magnitudes.len() * hop_length + fft_size;

        let mut output = vec![0.0f32; output_len];

        for (i, frame_mags) in magnitudes.iter().enumerate() {
            let start = i * hop_length;

            // Simplified reconstruction: use magnitude as envelope
            let avg_mag = if frame_mags.is_empty() {
                0.0
            } else {
                frame_mags.iter().sum::<f32>() / frame_mags.len() as f32
            };

            for j in 0..fft_size.min(output_len - start) {
                let t = j as f32 / self.config.sample_rate as f32;
                let freq = 200.0; // Simplified: single frequency
                let sample = (2.0 * std::f32::consts::PI * freq * t).sin() * avg_mag * 0.5;

                // Windowing
                let window =
                    0.5 * (1.0 - (2.0 * std::f32::consts::PI * j as f32 / fft_size as f32).cos());

                output[start + j] += sample * window;
            }
        }

        output
    }
}

impl VoiceIsolator for SpectralSubtractionIsolator {
    fn config(&self) -> &IsolationConfig {
        &self.config
    }

    fn isolate(&self, audio: &[f32]) -> VoiceResult<IsolationResult> {
        if audio.is_empty() {
            return Err(VoiceError::InvalidAudio("Empty audio".to_string()));
        }

        // Estimate noise from initial frames
        let noise_samples = self.config.noise_frames * self.config.fft_size;
        let noise_audio = if audio.len() >= noise_samples {
            &audio[..noise_samples]
        } else {
            audio
        };

        let noise_profile = self.estimate_noise(noise_audio)?;
        self.isolate_with_profile(audio, &noise_profile)
    }

    fn isolate_with_profile(
        &self,
        audio: &[f32],
        noise_profile: &NoiseProfile,
    ) -> VoiceResult<IsolationResult> {
        if audio.is_empty() {
            return Err(VoiceError::InvalidAudio("Empty audio".to_string()));
        }

        // Compute STFT
        let magnitudes = self.compute_stft(audio);

        // Apply spectral subtraction
        let clean_magnitudes = self.subtract_noise(&magnitudes, noise_profile);

        // Reconstruct
        let clean_audio = self.reconstruct(&clean_magnitudes);

        // Estimate SNR improvement
        let input_snr = estimate_snr(audio);
        let output_snr = estimate_snr(&clean_audio);

        Ok(IsolationResult::new(clean_audio, self.config.sample_rate)
            .with_snr(input_snr, output_snr)
            .with_noise_floor(noise_profile.mean_spectrum.clone()))
    }

    fn estimate_noise(&self, noise_audio: &[f32]) -> VoiceResult<NoiseProfile> {
        if noise_audio.is_empty() {
            return Err(VoiceError::InvalidAudio(
                "Empty noise audio for estimation".to_string(),
            ));
        }

        let frames = self.compute_stft(noise_audio);
        Ok(NoiseProfile::from_frames(&frames, self.config.sample_rate))
    }
}

impl Default for SpectralSubtractionIsolator {
    fn default() -> Self {
        Self::new(IsolationConfig::default())
    }
}

// ============================================================================
// Wiener Filter Isolator
// ============================================================================

/// Wiener filter based voice isolator.
///
/// Optimal linear filter that minimizes mean squared error.
/// Gain = S(f) / (S(f) + N(f)) where S = signal, N = noise
#[derive(Debug, Clone)]
pub struct WienerFilterIsolator {
    /// Configuration
    config: IsolationConfig,
    /// A priori SNR smoothing factor
    smoothing: f32,
}

impl WienerFilterIsolator {
    /// Create a new Wiener filter isolator
    #[must_use]
    pub fn new(config: IsolationConfig) -> Self {
        Self {
            config,
            smoothing: 0.98,
        }
    }

    /// Set smoothing factor for a priori SNR estimation
    #[must_use]
    pub fn with_smoothing(mut self, factor: f32) -> Self {
        self.smoothing = factor.clamp(0.0, 1.0);
        self
    }

    /// Compute Wiener gain
    fn compute_gain(signal_power: f32, noise_power: f32) -> f32 {
        if noise_power <= 0.0 {
            1.0
        } else {
            let snr = signal_power / noise_power;
            snr / (snr + 1.0)
        }
    }
}

impl VoiceIsolator for WienerFilterIsolator {
    fn config(&self) -> &IsolationConfig {
        &self.config
    }

    fn isolate(&self, audio: &[f32]) -> VoiceResult<IsolationResult> {
        if audio.is_empty() {
            return Err(VoiceError::InvalidAudio("Empty audio".to_string()));
        }

        // For Wiener filter, we need noise estimate
        let noise_samples = self.config.noise_frames * self.config.fft_size;
        let noise_audio = if audio.len() >= noise_samples {
            &audio[..noise_samples]
        } else {
            audio
        };

        let noise_profile = self.estimate_noise(noise_audio)?;
        self.isolate_with_profile(audio, &noise_profile)
    }

    fn isolate_with_profile(
        &self,
        audio: &[f32],
        noise_profile: &NoiseProfile,
    ) -> VoiceResult<IsolationResult> {
        if audio.is_empty() {
            return Err(VoiceError::InvalidAudio("Empty audio".to_string()));
        }

        let strength = self.config.reduction_strength;
        let floor = self.config.spectral_floor;

        // Apply Wiener filter frame by frame
        let hop_length = self.config.hop_length;
        let fft_size = self.config.fft_size;
        let num_frames = (audio.len().saturating_sub(fft_size)) / hop_length + 1;
        let num_frames = num_frames.max(1);

        let mut output = vec![0.0f32; audio.len()];

        for i in 0..num_frames {
            let start = i * hop_length;
            let end = (start + fft_size).min(audio.len());

            for (j, sample) in audio[start..end].iter().enumerate() {
                let bin = j % noise_profile.mean_spectrum.len().max(1);
                let noise_power = noise_profile.noise_magnitude(bin).powi(2);
                let signal_power = sample.powi(2);

                let gain = Self::compute_gain(signal_power, noise_power * strength);
                let gain = gain.max(floor);

                if start + j < output.len() {
                    output[start + j] = sample * gain;
                }
            }
        }

        let input_snr = estimate_snr(audio);
        let output_snr = estimate_snr(&output);

        Ok(IsolationResult::new(output, self.config.sample_rate).with_snr(input_snr, output_snr))
    }

    fn estimate_noise(&self, noise_audio: &[f32]) -> VoiceResult<NoiseProfile> {
        if noise_audio.is_empty() {
            return Err(VoiceError::InvalidAudio(
                "Empty noise audio for estimation".to_string(),
            ));
        }

        // Compute magnitude frames
        let hop_length = self.config.hop_length;
        let fft_size = self.config.fft_size;
        let freq_bins = self.config.freq_bins();

        let num_frames = (noise_audio.len().saturating_sub(fft_size)) / hop_length + 1;
        let num_frames = num_frames.max(1);

        let frames: Vec<Vec<f32>> = (0..num_frames)
            .map(|i| {
                let start = i * hop_length;
                let end = (start + fft_size).min(noise_audio.len());

                // Energy per bin (simplified)
                let frame = &noise_audio[start..end];
                let energy: f32 = frame.iter().map(|x| x * x).sum::<f32>().sqrt();

                (0..freq_bins)
                    .map(|bin| {
                        let freq_factor = 1.0 / (1.0 + (bin as f32 / 20.0));
                        energy * freq_factor
                    })
                    .collect()
            })
            .collect();

        Ok(NoiseProfile::from_frames(&frames, self.config.sample_rate))
    }
}

impl Default for WienerFilterIsolator {
    fn default() -> Self {
        Self::new(IsolationConfig::default())
    }
}

// ============================================================================
// Utility Functions
// ============================================================================

/// Estimate signal-to-noise ratio in dB.
///
/// Uses simple energy-based estimation.
#[must_use]
pub fn estimate_snr(audio: &[f32]) -> f32 {
    if audio.is_empty() {
        return 0.0;
    }

    // Compute signal energy
    let signal_energy: f32 = audio.iter().map(|x| x * x).sum();
    let signal_rms = (signal_energy / audio.len() as f32).sqrt();

    // Estimate noise as minimum energy segments
    let frame_size = 256.min(audio.len());
    let num_frames = audio.len() / frame_size;

    if num_frames == 0 {
        return 20.0; // Default assumption
    }

    let frame_energies: Vec<f32> = (0..num_frames)
        .map(|i| {
            let start = i * frame_size;
            let end = (start + frame_size).min(audio.len());
            let energy: f32 = audio[start..end].iter().map(|x| x * x).sum();
            (energy / (end - start) as f32).sqrt()
        })
        .collect();

    // Noise floor as 10th percentile of frame energies
    let mut sorted_energies = frame_energies.clone();
    sorted_energies.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    let noise_idx = (num_frames / 10).max(0);
    let noise_rms = sorted_energies.get(noise_idx).copied().unwrap_or(0.0001);

    // SNR in dB
    if noise_rms > 0.0 && signal_rms > 0.0 {
        20.0 * (signal_rms / noise_rms).log10()
    } else {
        0.0
    }
}

/// Compute spectral entropy (measure of noise-likeness).
///
/// Higher entropy = more noise-like.
#[must_use]
pub fn spectral_entropy(magnitudes: &[f32]) -> f32 {
    if magnitudes.is_empty() {
        return 0.0;
    }

    let sum: f32 = magnitudes.iter().sum();
    if sum <= 0.0 {
        return 0.0;
    }

    // Normalize to probability distribution
    let probabilities: Vec<f32> = magnitudes.iter().map(|&m| m / sum).collect();

    // Compute entropy
    let entropy: f32 = probabilities
        .iter()
        .filter(|&&p| p > 0.0)
        .map(|&p| -p * p.ln())
        .sum();

    // Normalize by maximum possible entropy
    let max_entropy = (magnitudes.len() as f32).ln();
    if max_entropy > 0.0 {
        entropy / max_entropy
    } else {
        0.0
    }
}

/// Detect voice activity in audio.
///
/// Returns vector of booleans (true = speech, false = silence/noise).
#[must_use]
pub fn detect_voice_activity(audio: &[f32], frame_size: usize, threshold: f32) -> Vec<bool> {
    if audio.is_empty() || frame_size == 0 {
        return vec![];
    }

    let num_frames = (audio.len() + frame_size - 1) / frame_size;

    (0..num_frames)
        .map(|i| {
            let start = i * frame_size;
            let end = (start + frame_size).min(audio.len());

            // Compute frame energy
            let energy: f32 = audio[start..end].iter().map(|x| x * x).sum();
            let rms = (energy / (end - start).max(1) as f32).sqrt();

            rms > threshold
        })
        .collect()
}
