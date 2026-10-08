//! Laya's calibrated temperature buckets and the tempered softmax.
//!
//! Port of `laya/common.py` `temp_bucket` + `clamp_temperature` and Laya's lookup
//! (`agent.py`: `temperature_by_options.get(temp_bucket(qt, k), temperature[qt])`).
//! The clamp bounds mirror `laya-finetune-gate-v1` `calibration_temp_min` /
//! `calibration_temp_max` (a test reads them from the YAML): a fitted temperature
//! below 0.5 sharpens logits instead of softening them and is never applied.

use super::{AgentConfig, QType};

/// Lower clamp bound (Laya `TEMP_MIN`).
pub const TEMP_MIN: f64 = 0.5;
/// Upper clamp bound (Laya `TEMP_MAX`).
pub const TEMP_MAX: f64 = 5.0;

/// Laya's option-count bucket: `"{qtype}:2"`, `":3-5"`, `":6-10"` or `":11+"`.
#[must_use]
pub fn bucket_key(qtype: QType, k: usize) -> String {
    let size = match k {
        0..=2 => "2",
        3..=5 => "3-5",
        6..=10 => "6-10",
        _ => "11+",
    };
    format!("{}:{size}", qtype.name())
}

/// A usable temperature: `t` confined to `[TEMP_MIN, TEMP_MAX]`, or 1.0 when it is
/// not finite (Laya `clamp_temperature`).
#[must_use]
pub fn clamp_temperature(t: f64) -> f64 {
    if t.is_finite() {
        t.clamp(TEMP_MIN, TEMP_MAX)
    } else {
        1.0
    }
}

/// The temperature Laya applies to a `qtype` question with `k` options, as the f64 the
/// agent config declares: `temperature_by_options[bucket]`, else `temperature[qtype]`,
/// else 1.0 — clamped. The ONE definition: [`temperature_for`] casts it for the forward,
/// and load rung 4 compares `manifest.calibration.t_applied` to it bit for bit.
#[must_use]
pub(crate) fn applied_temperature_f64(agent: &AgentConfig, qtype: QType, k: usize) -> f64 {
    let t = agent
        .temperature_by_options
        .get(&bucket_key(qtype, k))
        .or_else(|| agent.temperature.get(qtype.index()))
        .copied()
        .unwrap_or(1.0);
    clamp_temperature(t)
}

/// The temperature Laya applies to a `qtype` question with `k` options:
/// `temperature_by_options[bucket]`, else `temperature[qtype]`, else 1.0 — clamped.
#[must_use]
pub fn temperature_for(agent: &AgentConfig, qtype: QType, k: usize) -> f32 {
    // Laya divides an f32 logit tensor by this Python float; the division runs in f32.
    applied_temperature_f64(agent, qtype, k) as f32
}

/// `softmax(z / t)` with the spike's f64 accumulation (max-shifted in f32).
#[must_use]
pub fn softmax_t(z: &[f32], t: f32) -> Vec<f32> {
    let m = z.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b / t));
    let e: Vec<f64> = z.iter().map(|&v| f64::from(v / t - m).exp()).collect();
    let s: f64 = e.iter().sum();
    e.iter().map(|v| (v / s) as f32).collect()
}

#[cfg(test)]
mod tests {
    use super::{bucket_key, clamp_temperature, softmax_t, temperature_for, TEMP_MAX, TEMP_MIN};
    use crate::laya::{AgentConfig, QType};
    use crate::test_support::constant_f64;

    fn agent(json: &str) -> AgentConfig {
        AgentConfig::from_json_bytes(json.as_bytes()).expect("agent config parses")
    }

    #[test]
    fn bucket_keys() {
        assert_eq!(bucket_key(QType::Choice, 2), "choice:2");
        assert_eq!(bucket_key(QType::Choice, 3), "choice:3-5");
        assert_eq!(bucket_key(QType::Choice, 5), "choice:3-5");
        assert_eq!(bucket_key(QType::Choice, 6), "choice:6-10");
        assert_eq!(bucket_key(QType::Choice, 10), "choice:6-10");
        assert_eq!(bucket_key(QType::Choice, 11), "choice:11+");
        assert_eq!(bucket_key(QType::Score, 4), "score:3-5");
        assert_eq!(bucket_key(QType::Noul, 2), "noul:2");
    }

    #[test]
    fn clamp_bounds_and_non_finite() {
        assert_eq!(clamp_temperature(0.1), 0.5);
        assert_eq!(clamp_temperature(9.0), 5.0);
        assert_eq!(clamp_temperature(1.75), 1.75);
        assert_eq!(clamp_temperature(f64::NAN), 1.0);
        assert_eq!(clamp_temperature(f64::INFINITY), 1.0);
        assert_eq!(clamp_temperature(f64::NEG_INFINITY), 1.0);
    }

    /// Bucket first, then `temperature[qtype]`, then 1.0 — each clamped.
    #[test]
    fn lookup_falls_back_in_laya_order() {
        let a = agent(
            r#"{"head_layers":2,"temperature":[1.1,1.2,9.0],"temperature_by_options":{"choice:3-5":0.1}}"#,
        );
        assert_eq!(
            temperature_for(&a, QType::Choice, 3),
            0.5,
            "bucket hit, clamped up"
        );
        assert_eq!(
            temperature_for(&a, QType::Choice, 2),
            1.1_f32,
            "no bucket: temperature[0]"
        );
        assert_eq!(
            temperature_for(&a, QType::Noul, 2),
            5.0,
            "temperature[2] clamped down"
        );
        let short = agent(r#"{"head_layers":2,"temperature":[1.3]}"#);
        assert_eq!(
            temperature_for(&short, QType::Score, 3),
            1.0,
            "absent entry: 1.0"
        );
        let bare = agent(r#"{"head_layers":2}"#);
        assert_eq!(
            temperature_for(&bare, QType::Choice, 3),
            1.0,
            "Laya default 1.0"
        );
    }

    /// The clamp mirrors the gate contract; a drift in either is caught here.
    #[test]
    fn bounds_mirror_the_gate_contract() {
        assert_eq!(
            TEMP_MIN,
            constant_f64("laya-finetune-gate-v1.yaml", "calibration_temp_min")
        );
        assert_eq!(
            TEMP_MAX,
            constant_f64("laya-finetune-gate-v1.yaml", "calibration_temp_max")
        );
    }

    #[test]
    fn softmax_t_is_a_distribution_that_softens() {
        let z = [2.0f32, 0.0, -1.0];
        let p1 = softmax_t(&z, 1.0);
        let p5 = softmax_t(&z, 5.0);
        let sum: f32 = p1.iter().sum();
        assert!((sum - 1.0).abs() < 1e-6, "sums to 1: {sum}");
        assert!(
            p5[0] < p1[0],
            "a higher temperature softens the top probability"
        );
        assert!(p1.windows(2).all(|w| w[0] > w[1]), "order preserved");
    }
}
