//! WASM bindings for noise generator
//!
//! This module provides wasm-bindgen exports for browser integration.
//! Compile with `--features audio-noise-wasm` and target `wasm32-unknown-unknown`.

#[cfg(feature = "audio-noise-wasm")]
use wasm_bindgen::prelude::*;

#[cfg(feature = "audio-noise-wasm")]
use super::binaural::BinauralGenerator;
#[cfg(feature = "audio-noise-wasm")]
use super::config::{BinauralPreset, NoiseConfig, NoiseType};

/// WASM-exported noise generator with binaural support
#[cfg(feature = "audio-noise-wasm")]
#[wasm_bindgen]
pub struct NoiseGeneratorWasm {
    inner: BinauralGenerator,
    buffer_l: Vec<f32>,
    buffer_r: Vec<f32>,
}

#[cfg(feature = "audio-noise-wasm")]
impl std::fmt::Debug for NoiseGeneratorWasm {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("NoiseGeneratorWasm")
            .field("inner", &self.inner)
            .field("buffer_l_len", &self.buffer_l.len())
            .field("buffer_r_len", &self.buffer_r.len())
            .finish()
    }
}

#[cfg(feature = "audio-noise-wasm")]
#[wasm_bindgen]
impl NoiseGeneratorWasm {
    /// Create a new generator with the specified noise type
    /// Valid types: "white", "pink", "brown", "blue", "violet"
    #[wasm_bindgen(constructor)]
    pub fn new(noise_type: &str) -> Result<NoiseGeneratorWasm, JsValue> {
        let nt = NoiseType::from_name(noise_type).ok_or_else(|| {
            JsValue::from_str(&format!(
                "Unknown noise type: {}. Valid: white, pink, brown, blue, violet",
                noise_type
            ))
        })?;

        let config = NoiseConfig::new(nt);
        let buffer_size = config.buffer_size;

        let inner =
            BinauralGenerator::new(config, 0.0).map_err(|e| JsValue::from_str(&e.to_string()))?;

        Ok(Self {
            inner,
            buffer_l: vec![0.0; buffer_size],
            buffer_r: vec![0.0; buffer_size],
        })
    }

    /// Generate mono audio samples
    pub fn generate(&mut self, length: usize) -> Vec<f32> {
        let buffer_size = self.inner.config().buffer_size;
        let mut output = Vec::with_capacity(length);

        while output.len() < length {
            if self
                .inner
                .generate_stereo(&mut self.buffer_l, &mut self.buffer_r)
                .is_ok()
            {
                // Average left and right for mono
                for i in 0..buffer_size {
                    if output.len() < length {
                        output.push((self.buffer_l[i] + self.buffer_r[i]) * 0.5);
                    }
                }
            } else {
                break;
            }
        }

        output
    }

    /// Generate interleaved stereo samples [L, R, L, R, ...]
    pub fn generate_stereo(&mut self, length: usize) -> Vec<f32> {
        let buffer_size = self.inner.config().buffer_size;
        let mut output = Vec::with_capacity(length * 2);

        while output.len() < length * 2 {
            if self
                .inner
                .generate_stereo(&mut self.buffer_l, &mut self.buffer_r)
                .is_ok()
            {
                for i in 0..buffer_size {
                    if output.len() < length * 2 {
                        output.push(self.buffer_l[i]);
                        output.push(self.buffer_r[i]);
                    }
                }
            } else {
                break;
            }
        }

        output
    }

    /// Set spectral slope in dB/octave (-12 to +12)
    pub fn set_spectral_slope(&mut self, slope: f32) {
        if let Ok(new_config) = self.inner.config().clone().with_spectral_slope(slope) {
            let _ = self.inner.update_config(new_config);
        }
    }

    /// Set texture (0.0 = smooth, 1.0 = grainy)
    pub fn set_texture(&mut self, texture: f32) {
        if let Ok(new_config) = self.inner.config().clone().with_texture(texture) {
            let _ = self.inner.update_config(new_config);
        }
    }

    /// Set modulation parameters
    pub fn set_modulation(&mut self, depth: f32, rate: f32) {
        if let Ok(new_config) = self.inner.config().clone().with_modulation(depth, rate) {
            let _ = self.inner.update_config(new_config);
        }
    }

    /// Set binaural beat offset in Hz (0 for mono, 1-40 for binaural)
    pub fn set_binaural_offset(&mut self, hz: f32) {
        self.inner.set_frequency_offset(hz);
    }

    /// Set binaural preset
    pub fn set_binaural_preset(&mut self, preset: &str) {
        let offset = match preset.to_lowercase().as_str() {
            "delta" => BinauralPreset::Delta.frequency(),
            "theta" => BinauralPreset::Theta.frequency(),
            "alpha" => BinauralPreset::Alpha.frequency(),
            "beta" => BinauralPreset::Beta.frequency(),
            "gamma" => BinauralPreset::Gamma.frequency(),
            _ => return,
        };
        self.inner.set_frequency_offset(offset);
    }

    /// Get current configuration as JSON
    pub fn to_json(&self) -> String {
        serde_json::to_string(self.inner.config()).unwrap_or_else(|_| "{}".to_string())
    }

    /// Reset generator state
    pub fn reset(&mut self) {
        self.inner.reset();
    }

    /// Get current playback time in seconds
    pub fn time(&self) -> f64 {
        self.inner.time()
    }

    /// Get buffer size
    pub fn buffer_size(&self) -> usize {
        self.inner.config().buffer_size
    }

    /// Get sample rate
    pub fn sample_rate(&self) -> u32 {
        self.inner.config().sample_rate
    }
}

/// Get noise generator version
#[cfg(feature = "audio-noise-wasm")]
#[wasm_bindgen]
pub fn noise_version() -> String {
    env!("CARGO_PKG_VERSION").to_string()
}

// Stub implementations for non-WASM builds
#[cfg(not(feature = "audio-noise-wasm"))]
pub struct NoiseGeneratorWasm;

#[cfg(not(feature = "audio-noise-wasm"))]
pub fn noise_version() -> String {
    env!("CARGO_PKG_VERSION").to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    // Note: WASM tests require wasm32 target
    // These tests verify the non-WASM stubs compile

    #[test]
    fn test_noise_version() {
        let version = noise_version();
        assert!(!version.is_empty());
    }

    // The following tests only run when audio-noise-wasm feature is enabled
    // and we're not on wasm32 target (for unit testing)
    #[cfg(all(feature = "audio-noise-wasm", not(target_arch = "wasm32")))]
    mod wasm_tests {
        use super::*;

        #[test]
        fn test_wasm_generator_new() {
            let gen = NoiseGeneratorWasm::new("brown");
            assert!(gen.is_ok());
        }

        #[test]
        fn test_wasm_generator_invalid_type() {
            // On native (non-WASM) targets, JsValue::from_str() aborts because
            // wasm-bindgen's JS runtime isn't available. We can't test the Err path
            // on native — instead verify the validation logic directly.
            use crate::audio::noise::config::NoiseType;
            assert!(NoiseType::from_name("invalid").is_none());
            assert!(NoiseType::from_name("white").is_some());
            assert!(NoiseType::from_name("pink").is_some());
            assert!(NoiseType::from_name("brown").is_some());
        }

        #[test]
        fn test_wasm_generator_generate_mono() {
            let mut gen = NoiseGeneratorWasm::new("brown").unwrap();
            let samples = gen.generate(1024);
            assert_eq!(samples.len(), 1024);
        }

        #[test]
        fn test_wasm_generator_generate_stereo() {
            let mut gen = NoiseGeneratorWasm::new("brown").unwrap();
            let samples = gen.generate_stereo(1024);
            assert_eq!(samples.len(), 2048); // Interleaved
        }

        #[test]
        fn test_wasm_generator_set_methods() {
            let mut gen = NoiseGeneratorWasm::new("white").unwrap();

            gen.set_spectral_slope(-3.0);
            gen.set_texture(0.8);
            gen.set_modulation(0.5, 2.0);
            gen.set_binaural_offset(4.0);
            gen.set_binaural_preset("alpha");

            // Should not panic
        }

        #[test]
        fn test_wasm_generator_to_json() {
            let gen = NoiseGeneratorWasm::new("brown").unwrap();
            let json = gen.to_json();
            assert!(json.contains("Brown"));
        }

        #[test]
        fn test_wasm_generator_reset() {
            let mut gen = NoiseGeneratorWasm::new("brown").unwrap();
            gen.generate(1024);
            assert!(gen.time() > 0.0);

            gen.reset();
            assert!((gen.time() - 0.0).abs() < f64::EPSILON);
        }

        #[test]
        fn test_wasm_generator_accessors() {
            let gen = NoiseGeneratorWasm::new("brown").unwrap();
            assert_eq!(gen.buffer_size(), 1024);
            assert_eq!(gen.sample_rate(), 44100);
        }
    }
}
