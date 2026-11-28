//! Tests for the legend histogram and the brush. The oracles run over every cell of the 117 x 90
//! grid (10,530 cells) with one deterministic value pattern.

use std::collections::BTreeSet;

use crate::map::{Brush, EntityId, HexGrid, Histogram, Lens, MapError, Scale, HISTOGRAM_BINS};

const CELLS: u32 = 117 * 90;

/// Deterministic value pattern: every 11th cell has no data, every 13th cell holds a
/// non-positive value, and the rest hold a value in `[0.5, 100.5)`.
fn pattern(cell: u32) -> Option<f64> {
    if cell % 11 == 0 {
        None
    } else if cell % 13 == 0 {
        Some(-f64::from(cell))
    } else {
        Some(f64::from((cell * 37) % 1000) / 10.0 + 0.5)
    }
}

fn pattern_values() -> Vec<Option<f64>> {
    (0..CELLS).map(pattern).collect()
}

fn linear() -> Lens {
    Lens::new(1, "lin", Scale::Linear, (0.0, 100.0), pattern_values()).expect("linear lens")
}

fn log() -> Lens {
    Lens::new(2, "log", Scale::Log, (1.0, 1000.0), pattern_values()).expect("log lens")
}

/// Independent oracle for the position of a value on the lens scale. It reads only the lens
/// domain and scale, never `color_t`, so the expectation cannot inherit a bug from the code
/// under test.
fn oracle_t(lens: &Lens, v: f64) -> Option<f64> {
    let (lo, hi) = lens.domain();
    let t = match lens.scale() {
        Scale::Linear if v.is_finite() => (v - lo) / (hi - lo),
        Scale::Log if v.is_finite() && v > 0.0 => {
            (v.log10() - lo.log10()) / (hi.log10() - lo.log10())
        }
        _ => return None,
    };
    Some(t.clamp(0.0, 1.0))
}

/// Brute-force bin counts: every cell is visited, and each cell with data is placed by scanning
/// the bins from the bottom instead of computing an index.
fn oracle_counts(lens: &Lens, bins: usize) -> (Vec<u32>, u64) {
    let mut counts = vec![0u32; bins];
    let mut with_data = 0u64;
    for cell in 0..CELLS {
        let Some(v) = lens.values()[cell as usize] else {
            continue;
        };
        let Some(t) = oracle_t(lens, v) else {
            continue;
        };
        with_data += 1;
        let bin = (0..bins)
            .find(|&k| t < (k + 1) as f64 / bins as f64)
            .unwrap_or(bins - 1);
        counts[bin] += 1;
    }
    (counts, with_data)
}

#[test]
fn the_oracle_grid_is_the_117_by_90_map() {
    let grid = HexGrid::map_default(1.0).expect("grid");
    assert_eq!(grid.cell_count(), CELLS);
    assert_eq!(pattern_values().len(), 10_530);
}

#[test]
fn linear_histogram_counts_each_bin_and_its_edges_span_the_domain() {
    let values = (0..100).map(|i| Some(f64::from(i))).collect();
    let lens = Lens::new(1, "lin", Scale::Linear, (0.0, 100.0), values).expect("lens");
    let h = Histogram::of(&lens, 10).expect("histogram");
    assert_eq!(h.counts(), [10u32; 10].as_slice());
    assert_eq!(h.total(), 100);
    assert_eq!(h.edges().len(), 11);
    assert_eq!(h.edges()[0], 0.0);
    assert_eq!(h.edges()[10], 100.0);
}

#[test]
fn log_histogram_edges_are_equal_steps_in_log10_space() {
    let h = Histogram::of(&log(), 3).expect("histogram");
    let e = h.edges();
    assert_eq!(e.len(), 4);
    assert!((e[0] - 1.0).abs() < 1e-12, "lo edge {}", e[0]);
    assert!((e[3] - 1000.0).abs() < 1e-9, "hi edge {}", e[3]);
    for w in e.windows(2) {
        let step = w[1].log10() - w[0].log10();
        assert!((step - 1.0).abs() < 1e-9, "log10 step {step}");
    }
}

#[test]
fn histogram_refuses_zero_bins() {
    assert_eq!(
        Histogram::of(&linear(), 0),
        Err(MapError::NonPositive("bins"))
    );
}

#[test]
fn bin_counts_match_the_brute_force_oracle_over_every_cell() {
    for lens in [linear(), log()] {
        let h = Histogram::of(&lens, HISTOGRAM_BINS).expect("histogram");
        let (want, with_data) = oracle_counts(&lens, HISTOGRAM_BINS);
        assert_eq!(h.counts(), want.as_slice(), "{:?}", lens.scale());
        assert_eq!(
            h.total(),
            with_data,
            "{:?} sums to cells with data",
            lens.scale()
        );
        assert!(with_data > 0);
    }
}

#[test]
fn brush_bounds_are_inclusive() {
    let values = vec![Some(9.0), Some(10.0), Some(15.0), Some(20.0), Some(20.5)];
    let lens = Lens::new(1, "lin", Scale::Linear, (0.0, 100.0), values).expect("lens");
    let brush = Brush::new(10.0, 20.0).expect("brush");
    assert!(brush.contains(10.0), "lo is inside");
    assert!(brush.contains(20.0), "hi is inside");
    assert!(!brush.contains(9.999_999), "just below lo is outside");
    assert!(!brush.contains(20.000_001), "just above hi is outside");
    let want: BTreeSet<EntityId> = [1, 2, 3].into_iter().collect();
    assert_eq!(brush.select(&lens), want);
}

#[test]
fn a_degenerate_brush_selects_exactly_its_value() {
    let brush = Brush::new(7.0, 7.0).expect("brush");
    let want: BTreeSet<EntityId> = (0..CELLS)
        .filter(|&c| pattern(c) == Some(7.0))
        .map(u64::from)
        .collect();
    assert!(!want.is_empty());
    assert_eq!(brush.select(&linear()), want);
}

#[test]
fn brush_refuses_inverted_and_non_finite_bounds() {
    assert_eq!(Brush::new(5.0, 4.0), Err(MapError::InvertedBounds));
    assert_eq!(Brush::new(f64::NAN, 4.0), Err(MapError::NonFinite("brush")));
    assert_eq!(
        Brush::new(0.0, f64::INFINITY),
        Err(MapError::NonFinite("brush"))
    );
}

#[test]
fn brush_selection_matches_the_brute_force_oracle_over_every_cell() {
    for lens in [linear(), log()] {
        for (lo, hi) in [(0.0, 50.0), (-5.0, 5.0), (1.0, 1.0), (30.5, 99.5)] {
            let brush = Brush::new(lo, hi).expect("brush");
            let want: BTreeSet<EntityId> = (0..CELLS)
                .filter(|&c| match pattern(c) {
                    Some(v) => oracle_t(&lens, v).is_some() && (lo..=hi).contains(&v),
                    None => false,
                })
                .map(u64::from)
                .collect();
            assert_eq!(brush.select(&lens), want, "{:?} [{lo}, {hi}]", lens.scale());
        }
    }
}
