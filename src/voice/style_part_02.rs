
impl StyleTransfer for AutoVcTransfer {
    fn transfer(&self, source_audio: &[f32], _target_style: &StyleVector) -> VoiceResult<Vec<f32>> {
        if source_audio.is_empty() {
            return Err(VoiceError::InvalidAudio("empty source audio".to_string()));
        }
        Err(VoiceError::NotImplemented(
            "AutoVC requires model weights".to_string(),
        ))
    }

    fn transfer_from_reference(
        &self,
        source_audio: &[f32],
        reference_audio: &[f32],
    ) -> VoiceResult<Vec<f32>> {
        if source_audio.is_empty() {
            return Err(VoiceError::InvalidAudio("empty source audio".to_string()));
        }
        if reference_audio.is_empty() {
            return Err(VoiceError::InvalidAudio(
                "empty reference audio".to_string(),
            ));
        }
        Err(VoiceError::NotImplemented(
            "AutoVC requires model weights".to_string(),
        ))
    }

    fn config(&self) -> &StyleConfig {
        &self.config
    }
}

// ============================================================================
// Utility Functions
// ============================================================================

/// Compute prosody distance between two styles.
///
/// Focuses on pitch, energy, and rhythm differences.
#[must_use]
pub fn prosody_distance(a: &StyleVector, b: &StyleVector) -> f32 {
    if a.prosody.len() != b.prosody.len() {
        return f32::MAX;
    }
    if a.rhythm.len() != b.rhythm.len() {
        return f32::MAX;
    }

    let prosody_dist: f32 = a
        .prosody
        .iter()
        .zip(b.prosody.iter())
        .map(|(x, y)| (x - y).powi(2))
        .sum();

    let rhythm_dist: f32 = a
        .rhythm
        .iter()
        .zip(b.rhythm.iter())
        .map(|(x, y)| (x - y).powi(2))
        .sum();

    (prosody_dist + rhythm_dist).sqrt()
}

/// Compute timbre distance between two styles.
///
/// Focuses on spectral envelope differences.
#[must_use]
pub fn timbre_distance(a: &StyleVector, b: &StyleVector) -> f32 {
    if a.timbre.len() != b.timbre.len() {
        return f32::MAX;
    }

    let dist: f32 = a
        .timbre
        .iter()
        .zip(b.timbre.iter())
        .map(|(x, y)| (x - y).powi(2))
        .sum();

    dist.sqrt()
}

/// Compute total style distance (Euclidean).
#[must_use]
pub fn style_distance(a: &StyleVector, b: &StyleVector) -> f32 {
    if a.dim() != b.dim() {
        return f32::MAX;
    }

    let flat_a = a.to_flat();
    let flat_b = b.to_flat();

    let dist: f32 = flat_a
        .iter()
        .zip(flat_b.iter())
        .map(|(x, y)| (x - y).powi(2))
        .sum();

    dist.sqrt()
}

/// Create style from speaker embedding (approximate).
///
/// Maps speaker embedding to style vector space.
/// Useful when only speaker embeddings are available.
#[must_use]
pub fn style_from_embedding(embedding: &SpeakerEmbedding, config: &StyleConfig) -> StyleVector {
    let emb_slice = embedding.as_slice();
    let emb_len = emb_slice.len();

    // Simple projection: split embedding into style components
    let prosody_len = config.prosody_dim.min(emb_len);
    let timbre_len = config.timbre_dim.min(emb_len.saturating_sub(prosody_len));
    let rhythm_len = config
        .rhythm_dim
        .min(emb_len.saturating_sub(prosody_len + timbre_len));

    let mut prosody = vec![0.0_f32; config.prosody_dim];
    let mut timbre = vec![0.0_f32; config.timbre_dim];
    let mut rhythm = vec![0.0_f32; config.rhythm_dim];

    // Copy available values
    prosody[..prosody_len].copy_from_slice(&emb_slice[..prosody_len]);

    if timbre_len > 0 {
        timbre[..timbre_len].copy_from_slice(&emb_slice[prosody_len..prosody_len + timbre_len]);
    }

    if rhythm_len > 0 {
        rhythm[..rhythm_len].copy_from_slice(
            &emb_slice[prosody_len + timbre_len..prosody_len + timbre_len + rhythm_len],
        );
    }

    StyleVector::new(prosody, timbre, rhythm)
}

/// Average multiple style vectors.
///
/// # Errors
/// Returns error if styles have different dimensions or list is empty.
pub fn average_styles(styles: &[StyleVector]) -> VoiceResult<StyleVector> {
    if styles.is_empty() {
        return Err(VoiceError::InvalidConfig(
            "cannot average empty style list".to_string(),
        ));
    }

    let first = &styles[0];
    let prosody_len = first.prosody.len();
    let timbre_len = first.timbre.len();
    let rhythm_len = first.rhythm.len();

    // Verify all dimensions match
    for style in styles.iter().skip(1) {
        if style.prosody.len() != prosody_len {
            return Err(VoiceError::DimensionMismatch {
                expected: prosody_len,
                got: style.prosody.len(),
            });
        }
        if style.timbre.len() != timbre_len {
            return Err(VoiceError::DimensionMismatch {
                expected: timbre_len,
                got: style.timbre.len(),
            });
        }
        if style.rhythm.len() != rhythm_len {
            return Err(VoiceError::DimensionMismatch {
                expected: rhythm_len,
                got: style.rhythm.len(),
            });
        }
    }

    let count = styles.len() as f32;

    let mut prosody = vec![0.0_f32; prosody_len];
    let mut timbre = vec![0.0_f32; timbre_len];
    let mut rhythm = vec![0.0_f32; rhythm_len];

    for style in styles {
        for (i, &v) in style.prosody.iter().enumerate() {
            prosody[i] += v / count;
        }
        for (i, &v) in style.timbre.iter().enumerate() {
            timbre[i] += v / count;
        }
        for (i, &v) in style.rhythm.iter().enumerate() {
            rhythm[i] += v / count;
        }
    }

    Ok(StyleVector::new(prosody, timbre, rhythm))
}
