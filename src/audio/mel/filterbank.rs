
impl MelFilterbank {
    /// Create a new mel filterbank from configuration
    ///
    /// # Arguments
    /// * `config` - Mel spectrogram configuration
    ///
    /// # Panics
    /// Panics if configuration parameters are zero
    #[must_use]
    pub fn new(config: &MelConfig) -> Self {
        assert!(config.n_mels > 0, "n_mels must be positive");
        assert!(config.n_fft > 0, "n_fft must be positive");
        assert!(config.sample_rate > 0, "sample_rate must be positive");

        let n_freqs = config.n_fft / 2 + 1;

        // Compute mel filterbank matrix
        let filters = Self::compute_filterbank(
            config.n_mels,
            config.n_fft,
            config.sample_rate,
            config.fmin,
            config.fmax,
        );

        // Precompute Hann window
        let window = Self::hann_window(config.n_fft);

        Self {
            config: config.clone(),
            filters,
            n_freqs,
            window,
        }
    }

    /// Create a mel filterbank from pre-computed filter weights
    ///
    /// This is useful when using filterbank weights from a model file
    /// that need to match exactly (e.g., for ASR parity testing).
    ///
    /// # Arguments
    /// * `filters` - Pre-computed filterbank matrix (n_mels x n_freqs) in row-major order
    /// * `config` - Configuration (n_mels, n_fft, etc.)
    ///
    /// # Panics
    /// Panics if filter dimensions don't match configuration
    #[must_use]
    pub fn from_filters(filters: Vec<f32>, config: &MelConfig) -> Self {
        let n_freqs = config.n_fft / 2 + 1;
        assert_eq!(
            filters.len(),
            config.n_mels * n_freqs,
            "filterbank size mismatch: expected {} x {} = {}, got {}",
            config.n_mels,
            n_freqs,
            config.n_mels * n_freqs,
            filters.len()
        );

        let window = Self::hann_window(config.n_fft);

        Self {
            config: config.clone(),
            filters,
            n_freqs,
            window,
        }
    }

    /// Compute the mel filterbank matrix with Slaney area normalization
    ///
    /// Creates triangular filters spaced on the mel scale, normalized so that
    /// each filter has unit area (Slaney normalization). This matches librosa's
    /// `norm='slaney'` and OpenAI Whisper's filterbank.
    ///
    /// # References
    /// - Slaney, M. (1998). Auditory Toolbox. Technical Report #1998-010.
    fn compute_filterbank(
        n_mels: usize,
        n_fft: usize,
        sample_rate: u32,
        fmin: f32,
        fmax: f32,
    ) -> Vec<f32> {
        let n_freqs = n_fft / 2 + 1;
        let mut filters = vec![0.0_f32; n_mels * n_freqs];

        // Convert to mel scale
        let mel_min = Self::hz_to_mel(fmin);
        let mel_max = Self::hz_to_mel(fmax);

        // Create n_mels + 2 points evenly spaced on mel scale
        let mel_points: Vec<f32> = (0..=n_mels + 1)
            .map(|i| mel_min + (mel_max - mel_min) * (i as f32) / ((n_mels + 1) as f32))
            .collect();

        // Convert mel points back to Hz
        let hz_points: Vec<f32> = mel_points.iter().map(|&m| Self::mel_to_hz(m)).collect();

        // Convert Hz to FFT bin indices
        let bin_points: Vec<usize> = hz_points
            .iter()
            .map(|&f| ((n_fft as f32 + 1.0) * f / sample_rate as f32).floor() as usize)
            .collect();

        // Create triangular filters with Slaney area normalization
        for m in 0..n_mels {
            let f_m_minus = bin_points[m];
            let f_m = bin_points[m + 1];
            let f_m_plus = bin_points[m + 2];

            // Slaney normalization factor: 2 / (hz_high - hz_low)
            // This ensures each triangular filter has unit area
            let hz_low = hz_points[m];
            let hz_high = hz_points[m + 2];
            let bandwidth = hz_high - hz_low;
            let slaney_norm = if bandwidth > 0.0 {
                2.0 / bandwidth
            } else {
                1.0
            };

            // Rising slope
            for k in f_m_minus..f_m {
                if k < n_freqs && f_m > f_m_minus {
                    let slope = (k - f_m_minus) as f32 / (f_m - f_m_minus) as f32;
                    filters[m * n_freqs + k] = slope * slaney_norm;
                }
            }

            // Falling slope
            for k in f_m..f_m_plus {
                if k < n_freqs && f_m_plus > f_m {
                    let slope = (f_m_plus - k) as f32 / (f_m_plus - f_m) as f32;
                    filters[m * n_freqs + k] = slope * slaney_norm;
                }
            }
        }

        filters
    }

    /// Convert frequency in Hz to mel scale
    ///
    /// Uses the formula: mel = 2595 * log10(1 + f/700)
    #[inline]
    #[must_use]
    pub fn hz_to_mel(hz: f32) -> f32 {
        2595.0 * (1.0 + hz / 700.0).log10()
    }

    /// Convert mel scale to frequency in Hz
    ///
    /// Uses the formula: f = 700 * (10^(mel/2595) - 1)
    #[inline]
    #[must_use]
    pub fn mel_to_hz(mel: f32) -> f32 {
        700.0 * (10.0_f32.powf(mel / 2595.0) - 1.0)
    }

    /// Compute Hann window
    fn hann_window(size: usize) -> Vec<f32> {
        (0..size)
            .map(|n| 0.5 * (1.0 - (2.0 * PI * n as f32 / size as f32).cos()))
            .collect()
    }

    /// Compute mel spectrogram from audio samples
    ///
    /// # Arguments
    /// * `audio` - Audio samples (mono, f32, at target sample rate)
    ///
    /// # Returns
    /// Mel spectrogram as a flattened 2D matrix (n_frames x n_mels) in row-major order
    ///
    /// # Errors
    /// Returns error if audio processing fails
    pub fn compute(&self, audio: &[f32]) -> AudioResult<Vec<f32>> {
        if audio.is_empty() {
            return Ok(Vec::new());
        }

        let hop_length = self.config.hop_length;
        if hop_length == 0 {
            return Err(AudioError::InvalidParameters(
                "hop_length must be positive".into(),
            ));
        }

        if self.config.center_pad {
            self.compute_center_padded(audio, hop_length)
        } else {
            self.compute_unpadded(audio, hop_length)
        }
    }

    /// Compute mel spectrogram with center padding (librosa/Whisper mode)
    ///
    /// Pads n_fft/2 zeros on each side so that n_frames = audio_len / hop_length.
    fn compute_center_padded(
        &self,
        audio: &[f32],
        hop_length: usize,
    ) -> AudioResult<Vec<f32>> {
        let pad_len = self.config.n_fft / 2;
        let padded_len = audio.len() + 2 * pad_len;
        let mut padded_audio = vec![0.0_f32; padded_len];
        padded_audio[pad_len..pad_len + audio.len()].copy_from_slice(audio);

        let n_frames = audio.len() / hop_length;
        if n_frames == 0 {
            return Ok(Vec::new());
        }

        let mut mel_spec = vec![0.0_f32; n_frames * self.config.n_mels];

        for frame_idx in 0..n_frames {
            let start = frame_idx * hop_length;
            let power_spec = self.compute_power_spectrum(&padded_audio, start);

            for mel_idx in 0..self.config.n_mels {
                let mut mel_energy = 0.0_f32;
                for (freq_idx, &power) in power_spec.iter().enumerate() {
                    mel_energy += self.filters[mel_idx * self.n_freqs + freq_idx] * power;
                }
                let log_mel = (mel_energy.max(1e-10)).log10();
                mel_spec[frame_idx * self.config.n_mels + mel_idx] = log_mel;
            }
        }

        self.normalize_whisper(&mut mel_spec);
        Ok(mel_spec)
    }

    /// Compute mel spectrogram without center padding (original mode)
    ///
    /// n_frames = (audio_len - n_fft) / hop_length + 1
    fn compute_unpadded(
        &self,
        audio: &[f32],
        hop_length: usize,
    ) -> AudioResult<Vec<f32>> {
        let n_frames = if audio.len() >= self.config.n_fft {
            (audio.len() - self.config.n_fft) / hop_length + 1
        } else {
            0
        };

        if n_frames == 0 {
            return Ok(Vec::new());
        }

        // Output buffer (n_frames x n_mels)
        let mut mel_spec = vec![0.0_f32; n_frames * self.config.n_mels];

        // Process each frame
        for frame_idx in 0..n_frames {
            let start = frame_idx * hop_length;

            // Apply window and compute power spectrum
            let power_spec = self.compute_power_spectrum(audio, start);

            // Apply mel filterbank
            for mel_idx in 0..self.config.n_mels {
                let mut mel_energy = 0.0_f32;
                for (freq_idx, &power) in power_spec.iter().enumerate() {
                    mel_energy += self.filters[mel_idx * self.n_freqs + freq_idx] * power;
                }

                // Apply log compression with floor to avoid log(0)
                let log_mel = (mel_energy.max(1e-10)).log10();
                mel_spec[frame_idx * self.config.n_mels + mel_idx] = log_mel;
            }
        }

        // Apply normalization (Whisper-style)
        self.normalize_whisper(&mut mel_spec);

        Ok(mel_spec)
    }

    /// Compute power spectrum for a single frame
    fn compute_power_spectrum(&self, audio: &[f32], start: usize) -> Vec<f32> {
        use rustfft::{num_complex::Complex, FftPlanner};

        let mut planner = FftPlanner::new();
        let fft = planner.plan_fft_forward(self.config.n_fft);

        // Apply window and prepare FFT input
        let mut fft_input: Vec<Complex<f32>> = (0..self.config.n_fft)
            .map(|i| {
                let sample = if start + i < audio.len() {
                    audio[start + i]
                } else {
                    0.0
                };
                Complex::new(sample * self.window[i], 0.0)
            })
            .collect();

        // Compute FFT
        fft.process(&mut fft_input);

        // Compute power spectrum (magnitude squared)
        fft_input
            .iter()
            .take(self.n_freqs)
            .map(Complex::norm_sqr)
            .collect()
    }

    /// Apply Whisper-style normalization
    #[allow(clippy::unused_self)]
    fn normalize_whisper(&self, mel_spec: &mut [f32]) {
        if mel_spec.is_empty() {
            return;
        }

        let max_val = mel_spec.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
        for x in mel_spec {
            *x = (*x).max(max_val - 8.0);
            *x = (*x + 4.0) / 4.0;
        }
    }

    /// Normalize mel spectrogram with global mean/std
    ///
    /// Applies: (x - mean) / std
    pub fn normalize_global(&self, mel_spec: &mut [f32]) {
        if mel_spec.is_empty() {
            return;
        }

        let mean = mel_spec.iter().sum::<f32>() / mel_spec.len() as f32;
        let variance =
            mel_spec.iter().map(|&x| (x - mean).powi(2)).sum::<f32>() / mel_spec.len() as f32;
        let std = variance.sqrt().max(1e-10);

        for x in mel_spec {
            *x = (*x - mean) / std;
        }
    }

    /// Get configuration
    #[must_use]
    pub fn config(&self) -> &MelConfig {
        &self.config
    }

    /// Get the number of mel channels
    #[must_use]
    pub fn n_mels(&self) -> usize {
        self.config.n_mels
    }

    /// Get the FFT size
    #[must_use]
    pub fn n_fft(&self) -> usize {
        self.config.n_fft
    }

    /// Get the hop length
    #[must_use]
    pub fn hop_length(&self) -> usize {
        self.config.hop_length
    }

    /// Get the sample rate
    #[must_use]
    pub fn sample_rate(&self) -> u32 {
        self.config.sample_rate
    }

    /// Get the number of frequency bins
    #[must_use]
    pub fn n_freqs(&self) -> usize {
        self.n_freqs
    }

    /// Get the filterbank matrix (n_mels x n_freqs) in row-major order
    #[must_use]
    pub fn filters(&self) -> &[f32] {
        &self.filters
    }

    /// Apply filterbank to power spectrum (scalar implementation)
    #[must_use]
    pub fn apply_filterbank(&self, power_spec: &[f32]) -> Vec<f32> {
        let mut mel_energies = vec![0.0_f32; self.config.n_mels];
        let spec_len = power_spec.len().min(self.n_freqs);

        for (mel_idx, mel_energy) in mel_energies.iter_mut().enumerate() {
            let mut energy = 0.0_f32;
            for (freq_idx, &spec_val) in power_spec.iter().take(spec_len).enumerate() {
                energy += self.filters[mel_idx * self.n_freqs + freq_idx] * spec_val;
            }
            *mel_energy = energy;
        }

        mel_energies
    }

    /// Calculate number of frames for given audio length
    #[must_use]
    pub fn num_frames(&self, audio_len: usize) -> usize {
        if self.config.center_pad {
            audio_len / self.config.hop_length
        } else if audio_len >= self.config.n_fft {
            (audio_len - self.config.n_fft) / self.config.hop_length + 1
        } else {
            0
        }
    }
}

#[cfg(test)]
mod tests;
