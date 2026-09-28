//! KTEST-001 §7 falsifiers F-1 and F-6: plant the defect, show the class set turns it RED, and
//! show the correct kernel stays GREEN on the same classes (the anti-vacuity arm).

use crate::error_model::{Dtype, ErrorModel};
use crate::inputs::{generate, InputClass};
use crate::margin::{judge_model, screen, Failure, Verdict};
use crate::oracle;
use crate::shapes::{dims, Dim};
use crate::Refusal;

const T: usize = 8;
const V: usize = 4;
/// What sits past the end of the row: the neighbouring memory an out-of-bounds read returns.
const GUARD: f32 = 1.0e3;
const SEED: u64 = 0x4B54_4553_5403;

/// A tiled f32 sum over `buf[..d]`. `tail_overread = 1` plants F-1: the tail loop reads one
/// element past d. `buf` is the row plus guard elements, so the over-read returns [`GUARD`]
/// instead of panicking, as device memory would.
fn tiled_sum(buf: &[f32], d: usize, tail_overread: usize) -> f32 {
    let full = d / T * T;
    let mut acc = 0.0_f32;
    for tile in buf[..full].chunks_exact(T) {
        acc += tile.iter().sum::<f32>();
    }
    if d % T != 0 {
        for &v in &buf[full..d + tail_overread] {
            acc += v;
        }
    }
    acc
}

fn sum_verdict(d: Dim, class: InputClass, tail_overread: usize) -> Verdict {
    let mut buf = generate(class, d.value, SEED).data;
    buf.extend([GUARD; T]);
    let row = &buf[..d.value];
    let yhat = tiled_sum(&buf, d.value, tail_overread);
    let model = ErrorModel::Sum {
        n: d.value,
        acc: Dtype::F32,
        out: Dtype::F32,
        ftz: false,
    };
    judge_model(
        &[yhat],
        &[oracle::sum(row)],
        &[oracle::sum_abs(row)],
        &model,
    )
    .expect("f32 sum over the class set has a bound")
    .verdict
}

/// F-1: off-by-one in a tile tail. RED on exactly the classes with a tail; a suite that only
/// tests multiples of T (T, 2T, "powers of two") never sees it.
#[test]
fn f1_tile_tail_overread_is_red_on_every_tail_class() {
    let set = dims(T, V);
    let red: Vec<usize> = set
        .iter()
        .filter(|&&d| sum_verdict(d, InputClass::Uniform, 1) != Verdict::Pass)
        .map(|d| d.value)
        .collect();
    let tails: Vec<usize> = set.iter().map(|d| d.value).filter(|v| v % T != 0).collect();
    assert_eq!(red, tails, "RED must be exactly the tail classes");
    assert!(red.contains(&(T + 1)), "the spec's named class T+1");
    assert!(red.contains(&(2 * T + 1)), "and the multi-tile tail");
    for d in set.iter().filter(|d| d.value % T == 0) {
        assert_eq!(
            sum_verdict(*d, InputClass::Uniform, 1),
            Verdict::Pass,
            "d = {}",
            d.value
        );
    }
}

/// F-1 anti-vacuity: the correct tail passes on every shape class × every input class.
#[test]
fn f1_correct_tail_passes_every_class() {
    for d in dims(T, V) {
        for class in InputClass::ALL {
            assert_eq!(
                sum_verdict(d, class, 0),
                Verdict::Pass,
                "d = {} ({:?}), input {} seed {SEED:#x}",
                d.value,
                d.class,
                class.name()
            );
        }
    }
}

/// Softmax in f32. `subtract_max = false` is the F-6 defect: what fast-math reassociation of
/// the max-subtraction leaves behind.
fn softmax_f32(x: &[f32], subtract_max: bool) -> Vec<f32> {
    let m = if subtract_max {
        x.iter().copied().fold(f32::NEG_INFINITY, f32::max)
    } else {
        0.0
    };
    let e: Vec<f32> = x.iter().map(|&v| (v - m).exp()).collect();
    let s: f32 = e.iter().sum();
    e.iter().map(|&v| v / s).collect()
}

fn softmax_screen(n: usize, class: InputClass, subtract_max: bool) -> Verdict {
    let x = generate(class, n, SEED).data;
    screen(&softmax_f32(&x, subtract_max), &oracle::softmax(&x)).expect("same lengths")
}

/// F-6: softmax without max-subtraction. The near-overflow class makes exp overflow to Inf, and
/// Inf/Inf is NaN: RED by the NaN rule, which needs no error model. On typical inputs the
/// defect is invisible, which is why the class exists.
///
/// Exactly: RED iff some single exp(xᵢ) overflows (xᵢ > ln f32::MAX ≈ 88.72). When only the SUM
/// overflows, every output becomes 0: finite and wrong, which only a margin can catch, and
/// EM-SMX has no bound yet (S-1). Every class with ≥ T logits is RED at this seed.
#[test]
fn f6_softmax_without_max_subtraction_is_red_on_near_overflow() {
    for d in dims(T, V).into_iter().filter(|d| d.value > 0) {
        let x = generate(InputClass::NearOverflow, d.value, SEED).data;
        let overflows = x.iter().any(|v| v.exp().is_infinite());
        let red = matches!(
            softmax_screen(d.value, InputClass::NearOverflow, false),
            Verdict::Fail(Failure::UnexpectedNaN { .. })
        );
        assert_eq!(red, overflows, "d = {}", d.value);
        assert!(red || d.value < T, "d = {} must be RED", d.value);
        assert_eq!(
            softmax_screen(d.value, InputClass::Uniform, false),
            Verdict::Pass,
            "the defect hides on uniform inputs, d = {}",
            d.value
        );
    }
}

/// F-6 anti-vacuity: the max-subtracted kernel breaks no NaN/Inf rule on any class. Its margin
/// is refused, not granted: EM-SMX has no implemented bound (S-1), so the harness does not pass it.
#[test]
fn f6_max_subtracted_softmax_screens_clean_and_its_margin_is_refused() {
    for d in dims(T, V) {
        for class in InputClass::ALL {
            assert_eq!(
                softmax_screen(d.value, class, true),
                Verdict::Pass,
                "d = {}, input {}",
                d.value,
                class.name()
            );
        }
    }
    let x = generate(InputClass::NearOverflow, 16, SEED).data;
    let y = oracle::softmax(&x);
    assert_eq!(
        judge_model(&softmax_f32(&x, true), &y, &y, &ErrorModel::Softmax),
        Err(Refusal::NotImplemented { id: "EM-SMX" })
    );
}
