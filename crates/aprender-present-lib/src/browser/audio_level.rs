//! Microphone samples as the waveform widget wants them (aprender#6891).
//!
//! `AnalyserNode::getByteTimeDomainData` yields unsigned bytes with 128 as silence. The
//! widget wants floats in [-1, 1] with 0 as silence. `bytes_to_samples` is that mapping and
//! touches no browser API, so it is tested natively; `browser::audio_source` (wasm32) only
//! feeds it.

/// Byte value `getByteTimeDomainData` reports for silence.
pub const SILENCE_BYTE: u8 = 128;

/// Replace `out` with `bytes` mapped to floats: `(b - 128) / 128`, so 128 is exactly 0.0,
/// 0 is -1.0 and 255 is just under 1.0. Every result is finite and in [-1, 1].
pub fn bytes_to_samples(bytes: &[u8], out: &mut Vec<f32>) {
    out.clear();
    out.extend(
        bytes
            .iter()
            .map(|&b| (f32::from(b) - f32::from(SILENCE_BYTE)) / f32::from(SILENCE_BYTE)),
    );
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;

    #[test]
    fn silence_maps_to_exact_zero() {
        let mut out = vec![9.0];
        bytes_to_samples(&[SILENCE_BYTE; 4], &mut out);
        assert_eq!(out, vec![0.0; 4]);
    }

    #[test]
    fn extremes_map_to_edges() {
        let mut out = Vec::new();
        bytes_to_samples(&[0, 255], &mut out);
        assert_eq!(out[0], -1.0);
        assert!((out[1] - 0.992_187_5).abs() < 1e-7);
    }

    #[test]
    fn empty_input_clears_output() {
        let mut out = vec![1.0, 2.0];
        bytes_to_samples(&[], &mut out);
        assert!(out.is_empty());
    }

    proptest! {
        #[test]
        fn every_sample_is_in_unit_range(bytes in proptest::collection::vec(any::<u8>(), 0..512)) {
            let mut out = Vec::new();
            bytes_to_samples(&bytes, &mut out);
            prop_assert_eq!(out.len(), bytes.len());
            prop_assert!(out.iter().all(|s| s.is_finite() && (-1.0..=1.0).contains(s)));
        }

        #[test]
        fn mapping_is_monotone(a in any::<u8>(), b in any::<u8>()) {
            let (mut x, mut y) = (Vec::new(), Vec::new());
            bytes_to_samples(&[a], &mut x);
            bytes_to_samples(&[b], &mut y);
            prop_assert_eq!(a <= b, x[0] <= y[0]);
        }
    }
}
