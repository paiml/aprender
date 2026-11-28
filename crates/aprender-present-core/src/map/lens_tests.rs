//! Tests for the lens registry: domain checks, `color_t` on both scales, non-finite and
//! non-positive values, and the registry's grid and duplicate checks.

use super::*;

fn linear(values: Vec<Option<f64>>) -> Lens {
    Lens::new(1, "lin", Scale::Linear, (0.0, 100.0), values).expect("valid linear lens")
}

fn log(values: Vec<Option<f64>>) -> Lens {
    Lens::new(2, "log", Scale::Log, (1.0, 1000.0), values).expect("valid log lens")
}

#[test]
fn domain_with_lo_at_or_above_hi_is_refused_at_construction() {
    for domain in [(0.0, 0.0), (5.0, 1.0)] {
        let err = Lens::new(1, "x", Scale::Linear, domain, vec![]).expect_err("empty domain");
        assert_eq!(err, MapError::EmptyDomain, "domain {domain:?}");
    }
    let err = Lens::new(2, "x", Scale::Log, (10.0, 10.0), vec![]).expect_err("empty domain");
    assert_eq!(err, MapError::EmptyDomain);
}

#[test]
fn domain_must_be_finite() {
    let nan = Lens::new(1, "x", Scale::Linear, (f64::NAN, 1.0), vec![]);
    assert_eq!(nan, Err(MapError::NonFinite("domain")));
    let inf = Lens::new(1, "x", Scale::Linear, (0.0, f64::INFINITY), vec![]);
    assert_eq!(inf, Err(MapError::NonFinite("domain")));
}

#[test]
fn log_domain_must_be_strictly_positive() {
    let zero = Lens::new(2, "x", Scale::Log, (0.0, 10.0), vec![]);
    assert_eq!(zero, Err(MapError::NonPositive("domain")));
    let negative = Lens::new(2, "x", Scale::Log, (-1.0, 10.0), vec![]);
    assert_eq!(negative, Err(MapError::NonPositive("domain")));
}

#[test]
fn accessors_report_what_was_built() {
    let lens =
        Lens::new(7, "pop", Scale::Log, (1.0, 10.0), vec![Some(2.0), None]).expect("valid lens");
    assert_eq!(lens.id(), 7);
    assert_eq!(lens.name(), "pop");
    assert_eq!(lens.scale(), Scale::Log);
    assert_eq!(lens.domain(), (1.0, 10.0));
    assert_eq!(lens.values(), &[Some(2.0), None]);
}

#[test]
fn linear_color_t_maps_the_domain_onto_the_unit_interval_and_clamps() {
    let lens = linear(vec![]);
    assert_eq!(lens.color_t(0.0), Some(0.0));
    assert_eq!(lens.color_t(50.0), Some(0.5));
    assert_eq!(lens.color_t(100.0), Some(1.0));
    assert_eq!(lens.color_t(-5.0), Some(0.0));
    assert_eq!(lens.color_t(150.0), Some(1.0));
}

#[test]
fn log_color_t_gives_equal_steps_in_log10_space() {
    let lens = log(vec![]);
    assert_eq!(lens.color_t(1.0), Some(0.0));
    assert_eq!(lens.color_t(1000.0), Some(1.0));
    let t10 = lens.color_t(10.0).expect("10 has a position");
    assert!((t10 - 1.0 / 3.0).abs() < 1e-12, "t(10) = {t10}");
    let t100 = lens.color_t(100.0).expect("100 has a position");
    assert!((t100 - 2.0 / 3.0).abs() < 1e-12, "t(100) = {t100}");
}

#[test]
fn log_color_t_refuses_non_positive_and_non_finite_values() {
    let lens = log(vec![]);
    for v in [0.0, -0.0, -3.0, f64::NEG_INFINITY, f64::NAN, f64::INFINITY] {
        assert_eq!(lens.color_t(v), None, "v = {v}");
    }
}

#[test]
fn non_finite_values_have_no_position_on_either_scale() {
    for lens in [linear(vec![]), log(vec![])] {
        assert_eq!(lens.color_t(f64::NAN), None);
        assert_eq!(lens.color_t(f64::INFINITY), None);
    }
}

#[test]
fn color_t_is_never_nan_and_stays_in_the_unit_interval() {
    let probes = [
        f64::MIN_POSITIVE,
        1e-300,
        0.5,
        1.0,
        1e3,
        1e300,
        f64::MAX,
        -1e300,
        f64::MIN,
    ];
    for lens in [linear(vec![]), log(vec![])] {
        for v in probes {
            if let Some(t) = lens.color_t(v) {
                assert!(
                    t.is_finite() && (0.0..=1.0).contains(&t),
                    "{:?} v = {v} gave t = {t}",
                    lens.scale()
                );
            }
        }
    }
    assert_eq!(linear(vec![]).color_t(1e300), Some(1.0));
    assert_eq!(log(vec![]).color_t(f64::MIN_POSITIVE), Some(0.0));
}

#[test]
fn registry_refuses_a_lens_whose_values_do_not_cover_the_grid() {
    let mut reg = LensRegistry::new(3);
    let short = linear(vec![Some(1.0); 2]);
    assert_eq!(reg.add(short), Err(MapError::Mismatch("lens")));
    assert!(reg.is_empty());
    assert_eq!(reg.cells(), 3);
}

#[test]
fn registry_refuses_a_duplicate_id_and_returns_registered_lenses() {
    let mut reg = LensRegistry::new(2);
    reg.add(linear(vec![None, Some(3.0)])).expect("first lens");
    let dup = reg.add(linear(vec![None, None]));
    assert_eq!(dup, Err(MapError::Duplicate("lens")));
    assert_eq!(reg.len(), 1);
    assert_eq!(reg.get(1).map(Lens::name), Some("lin"));
    assert!(reg.get(9).is_none());
}
