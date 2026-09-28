//! Shape classes (KTEST-001 §4): the dimension values a kernel is tested at, derived from the
//! registered tile `T` and vector width `V`, never hand-picked.
//!
//! Theorem K2 (v1.1, `cls4`): a loop over d elements in tiles of T has control flow that depends
//! only on (min(⌊d/T⌋, 2), d mod T = 0). {T−1, T, T+1, 2T} land in four different classes
//! (`classes4_distinct`). That model has SIX classes, though, and 0 plus those four reach five:
//! (≥ 2 tiles, with a tail) is only reached by a value like 2T+1, where a tail bug meets the
//! accumulator carry. So this generator adds 2T+1, and `dims_cover_every_reachable_class` proves
//! the set reaches all six. The same holds for the vector width, so V gets the same values.

/// Why a dimension value is in the test set (the §4 table's rows).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum ShapeClass {
    /// 0: empty launches, division by 0.
    Empty,
    /// 1, V−1: scalar-tail loops.
    SubVector,
    /// V, V+1, 2V, 2V+1: SIMD remainder masks (2V, 2V+1: `cls4` applied to the vector loop).
    VectorEdge,
    /// T−1, T, T+1: tile tails and off-by-one (the #749 bug class).
    TileEdge,
    /// 2T, 2T+1: accumulator carry and unroll bugs that need ≥ 2 full tiles, without and with a
    /// tail.
    MultiTile,
    /// A device maximum ±1 (grid y ≤ 65,535, workgroup storage, binding size).
    DeviceLimit,
    /// Element indices around 2³¹: i32 index overflow in large KV caches.
    Index2p31,
    /// A real model shape (hidden, `head_dim`, vocab, GQA ratio).
    Production,
}

/// One dimension value to test, with the class that put it there (recorded in the receipt).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Dim {
    pub class: ShapeClass,
    pub value: usize,
}

/// The `cls4` control-flow class of a d-long loop in tiles of `tile`: (full tiles capped at 2,
/// no tail?).
///
/// # Panics
/// If `tile == 0`.
#[must_use]
pub const fn cls4(d: usize, tile: usize) -> (usize, bool) {
    assert!(tile > 0, "cls4: tile must be positive");
    let full = d / tile;
    (if full < 2 { full } else { 2 }, d % tile == 0)
}

fn push(out: &mut Vec<Dim>, class: ShapeClass, value: usize) {
    if !out.iter().any(|d| d.value == value) {
        out.push(Dim { class, value });
    }
}

/// The standard per-dimension class set for a kernel with tile `tile` and vector width `vec`.
/// A value reached by two classes is listed once, under the first.
///
/// # Panics
/// If `tile` or `vec` is 0: a registry entry without a tile or vector width is a registry bug.
#[must_use]
pub fn dims(tile: usize, vec: usize) -> Vec<Dim> {
    assert!(
        tile > 0 && vec > 0,
        "shapes::dims: tile and vec must be positive"
    );
    let mut out = Vec::new();
    push(&mut out, ShapeClass::Empty, 0);
    push(&mut out, ShapeClass::SubVector, 1);
    push(&mut out, ShapeClass::SubVector, vec - 1);
    for v in [vec, vec + 1, 2 * vec, 2 * vec + 1] {
        push(&mut out, ShapeClass::VectorEdge, v);
    }
    for t in [tile - 1, tile, tile + 1] {
        push(&mut out, ShapeClass::TileEdge, t);
    }
    push(&mut out, ShapeClass::MultiTile, 2 * tile);
    push(&mut out, ShapeClass::MultiTile, 2 * tile + 1);
    out
}

/// limit − 1, limit, limit + 1 for a device maximum: silent clamping and split-dispatch bugs.
#[must_use]
pub fn device_limit(limit: usize) -> Vec<Dim> {
    let mut out = Vec::new();
    for v in [limit.saturating_sub(1), limit, limit.saturating_add(1)] {
        push(&mut out, ShapeClass::DeviceLimit, v);
    }
    out
}

/// 2³¹ − 1 and 2³¹: the last index an i32 holds and the first it does not.
#[must_use]
pub const fn index_overflow() -> [Dim; 2] {
    const I31: usize = 1 << 31;
    [
        Dim {
            class: ShapeClass::Index2p31,
            value: I31 - 1,
        },
        Dim {
            class: ShapeClass::Index2p31,
            value: I31,
        },
    ]
}

/// The real model shapes, tagged as production.
#[must_use]
pub fn production(shapes: &[usize]) -> Vec<Dim> {
    let mut out = Vec::new();
    for &v in shapes {
        push(&mut out, ShapeClass::Production, v);
    }
    out
}

/// Start offsets that break an `align`-element alignment assumption (NEON/AVX loads, 16-byte
/// WGSL): 1 and align − 1. Empty when every offset is aligned (`align` ≤ 1).
#[must_use]
pub fn misaligned_offsets(align: usize) -> Vec<usize> {
    let mut out = Vec::new();
    for o in [1, align.saturating_sub(1)] {
        if o > 0 && o < align && !out.contains(&o) {
            out.push(o);
        }
    }
    out
}

/// Every combination of one value per axis (nested tilings: §4 "the product of classes").
#[must_use]
pub fn grid(axes: &[Vec<Dim>]) -> Vec<Vec<Dim>> {
    axes.iter().fold(vec![Vec::new()], |acc, axis| {
        acc.iter()
            .flat_map(|prefix| {
                axis.iter().map(move |&d| {
                    let mut p = prefix.clone();
                    p.push(d);
                    p
                })
            })
            .collect()
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeSet;

    /// [Lean] `classes4_distinct`: {T−1, T, T+1, 2T} are four different classes for T ≥ 2.
    #[test]
    fn classes4_distinct() {
        for t in 2..=256 {
            let c: BTreeSet<_> = [t - 1, t, t + 1, 2 * t]
                .iter()
                .map(|&d| cls4(d, t))
                .collect();
            assert_eq!(c.len(), 4, "T = {t}");
        }
    }

    /// [Lean] `zero_class`: (0 full tiles, no tail) only for d = 0.
    #[test]
    fn zero_class() {
        for t in 1..=64 {
            for d in 0..=5 * t {
                assert_eq!(cls4(d, t) == (0, true), d == 0, "d = {d}, T = {t}");
            }
        }
    }

    /// [Lean] `classes_covered`, for both loops: every class any d reaches, for the tile loop and
    /// for the vector loop, is reached by some value in `dims`.
    #[test]
    fn dims_cover_every_reachable_class() {
        for t in 1..=64 {
            for v in 1..=16 {
                let set = dims(t, v);
                for (loop_width, name) in [(t, "tile"), (v, "vec")] {
                    let have: BTreeSet<_> = set.iter().map(|d| cls4(d.value, loop_width)).collect();
                    for d in 0..=5 * loop_width {
                        assert!(
                            have.contains(&cls4(d, loop_width)),
                            "{name} loop, T = {t}, V = {v}: class of d = {d} not covered"
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn dims_are_unique_and_classed() {
        let d = dims(32, 8);
        let values: Vec<_> = d.iter().map(|d| d.value).collect();
        assert_eq!(values, vec![0, 1, 7, 8, 9, 16, 17, 31, 32, 33, 64, 65]);
        assert_eq!(d[0].class, ShapeClass::Empty);
        assert_eq!(d.last().map(|d| d.class), Some(ShapeClass::MultiTile));
        // V = 1: V − 1 = 0 is already Empty, and V = 1 already SubVector
        let v1: Vec<_> = dims(4, 1).iter().map(|d| d.value).collect();
        assert_eq!(v1, vec![0, 1, 2, 3, 4, 5, 8, 9]);
    }

    /// Why 2T+1 is in the set: without it the (≥ 2 tiles, tail) class is never reached.
    #[test]
    fn spec_set_without_2t_plus_1_misses_a_class() {
        let t = 8;
        let spec: BTreeSet<_> = [0, t - 1, t, t + 1, 2 * t]
            .iter()
            .map(|&d| cls4(d, t))
            .collect();
        assert_eq!(spec.len(), 5);
        assert!(!spec.contains(&(2, false)));
        assert_eq!(cls4(2 * t + 1, t), (2, false));
    }

    #[test]
    fn limits_offsets_and_grid() {
        let l: Vec<_> = device_limit(65_535).iter().map(|d| d.value).collect();
        assert_eq!(l, vec![65_534, 65_535, 65_536]);
        assert_eq!(device_limit(0).len(), 2, "0 − 1 saturates onto 0");
        assert_eq!(
            index_overflow().map(|d| d.value),
            [0x7FFF_FFFF, 0x8000_0000]
        );
        assert_eq!(misaligned_offsets(4), vec![1, 3]);
        assert_eq!(misaligned_offsets(2), vec![1]);
        assert!(misaligned_offsets(1).is_empty());
        let p = production(&[4096, 128, 4096]);
        assert_eq!(p.len(), 2);
        let g = grid(&[dims(4, 2), dims(8, 4), production(&[128])]);
        assert_eq!(g.len(), dims(4, 2).len() * dims(8, 4).len());
        assert!(g.iter().all(|c| c.len() == 3));
    }
}
