//! Input classes (KTEST-001 §3.3): typical inputs plus the adversarial ones that expose what
//! typical inputs hide. Every input carries its seed, so a failure reproduces from its receipt.
//!
//! Quantized-block extremes (absmax at the scale limit, all-zero blocks, minimum scale) belong to
//! the dequant fixtures, not here: this module generates the f32 values a kernel consumes.

use crate::rng::SplitMix64;

/// One §3.3 input class.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum InputClass {
    /// Uniform in [−1, 1).
    Uniform,
    /// Standard normal.
    Normal,
    /// ±10^u, u uniform in [−4, 4]: large dynamic range.
    WideRange,
    /// (x, −x) pairs at wide range, shuffled: the exact sum is 0 (plus one odd term).
    Cancellation,
    /// Every element equal: softmax of an all-equal row, variance 0 in a norm.
    AllEqual,
    /// Logits in [80, 100): exp overflows f32 above ~88.72 unless the max is subtracted.
    NearOverflow,
    /// Subnormal magnitudes with random signs: flush-to-zero.
    Subnormal,
    /// Standard normal with one NaN and one ±Inf planted: propagation.
    NanInf,
}

impl InputClass {
    /// Every class, in a fixed order.
    pub const ALL: [Self; 8] = [
        Self::Uniform,
        Self::Normal,
        Self::WideRange,
        Self::Cancellation,
        Self::AllEqual,
        Self::NearOverflow,
        Self::Subnormal,
        Self::NanInf,
    ];

    /// The name a receipt records.
    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Self::Uniform => "uniform",
            Self::Normal => "normal",
            Self::WideRange => "wide-range",
            Self::Cancellation => "cancellation",
            Self::AllEqual => "all-equal",
            Self::NearOverflow => "near-overflow",
            Self::Subnormal => "subnormal",
            Self::NanInf => "nan-inf",
        }
    }
}

/// Generated values, with what generated them.
#[derive(Debug, Clone, PartialEq)]
pub struct Input {
    pub class: InputClass,
    pub seed: u64,
    pub data: Vec<f32>,
}

#[allow(clippy::cast_possible_truncation)] // the point: an f32 input
fn wide(rng: &mut SplitMix64) -> f32 {
    let mag = 10f64.powf(rng.unit() * 8.0 - 4.0);
    let sign = if rng.next_u64() & 1 == 0 { 1.0 } else { -1.0 };
    (sign * mag) as f32
}

/// `n` values of `class`, reproducible from `seed`.
#[must_use]
pub fn generate(class: InputClass, n: usize, seed: u64) -> Input {
    let mut rng = SplitMix64::new(seed);
    let data = match class {
        InputClass::Uniform => (0..n).map(|_| rng.uniform(-1.0, 1.0)).collect(),
        InputClass::Normal => (0..n).map(|_| rng.normal()).collect(),
        InputClass::WideRange => (0..n).map(|_| wide(&mut rng)).collect(),
        InputClass::Cancellation => {
            let mut v = Vec::with_capacity(n);
            while v.len() + 1 < n {
                let x = wide(&mut rng);
                v.extend([x, -x]);
            }
            if v.len() < n {
                v.push(wide(&mut rng));
            }
            for i in (1..v.len()).rev() {
                v.swap(i, rng.below(i + 1));
            }
            v
        }
        InputClass::AllEqual => vec![rng.uniform(-10.0, 10.0); n],
        InputClass::NearOverflow => (0..n).map(|_| rng.uniform(80.0, 100.0)).collect(),
        InputClass::Subnormal => (0..n)
            .map(|_| {
                let x = rng.uniform(0.0, 1.0) * f32::MIN_POSITIVE;
                if rng.next_u64() & 1 == 0 {
                    x
                } else {
                    -x
                }
            })
            .collect(),
        InputClass::NanInf => {
            let mut v: Vec<f32> = (0..n).map(|_| rng.normal()).collect();
            if n > 0 {
                let i = rng.below(n);
                v[i] = f32::NAN;
                if n > 1 {
                    let j = (i + 1 + rng.below(n - 1)) % n;
                    v[j] = if rng.next_u64() & 1 == 0 {
                        f32::INFINITY
                    } else {
                        f32::NEG_INFINITY
                    };
                }
            }
            v
        }
    };
    Input { class, seed, data }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reproducible_from_the_seed() {
        for class in InputClass::ALL {
            for n in [0, 1, 2, 7, 64] {
                let (a, b) = (generate(class, n, 42), generate(class, n, 42));
                assert_eq!(a.data.len(), n, "{}", class.name());
                let bits = |i: &Input| i.data.iter().map(|v| v.to_bits()).collect::<Vec<_>>();
                assert_eq!(bits(&a), bits(&b), "{} n = {n}", class.name());
            }
        }
        assert_ne!(
            generate(InputClass::Uniform, 8, 1).data,
            generate(InputClass::Uniform, 8, 2).data
        );
    }

    #[test]
    fn each_class_has_its_property() {
        let n = 1001;
        let g = |c| generate(c, n, 7).data;
        assert!(g(InputClass::Uniform)
            .iter()
            .all(|v| (-1.0..1.0).contains(v)));
        let w = g(InputClass::WideRange);
        assert!(w.iter().all(|v| (1e-4..=1e4).contains(&v.abs())));
        assert!(w.iter().any(|v| v.abs() < 1e-3) && w.iter().any(|v| v.abs() > 1e3));
        // every element but the odd one has its exact negation somewhere else in the row
        let c = g(InputClass::Cancellation);
        let unpaired = c.iter().filter(|&&v| !c.contains(&-v)).count();
        assert_eq!(unpaired, 1, "n = {n}: 500 (x, −x) pairs + 1");
        assert_ne!(c[..2], [c[0], -c[0]], "pairs are shuffled apart");
        let e = g(InputClass::AllEqual);
        assert!(e.iter().all(|v| v.to_bits() == e[0].to_bits()));
        let o = g(InputClass::NearOverflow);
        assert!(o.iter().all(|v| (80.0..100.0).contains(v)));
        assert!(
            o.iter().any(|v| v.exp().is_infinite()),
            "some exp must overflow f32"
        );
        let s = g(InputClass::Subnormal);
        assert!(s.iter().all(|v| v.is_subnormal() || *v == 0.0));
        assert!(s.iter().filter(|v| v.is_subnormal()).count() > n / 2);
        let ni = g(InputClass::NanInf);
        assert_eq!(ni.iter().filter(|v| v.is_nan()).count(), 1);
        assert_eq!(ni.iter().filter(|v| v.is_infinite()).count(), 1);
        let one = generate(InputClass::NanInf, 1, 3).data;
        assert!(one[0].is_nan());
    }
}
