//! L1 index maps (KTEST-001 §2 L1, §6.1): the addresses a kernel reads and writes, written as pure
//! `const fn`s so the kernel that uses them and the verifier that proves them share one definition.
//!
//! Three maps, each with what is proved about it:
//! - [`store_index`] (K3): thread `t` writes lane `j < s` of its stride-`s` slice. Injective (no two
//!   (thread, lane) pairs write one address: race-free) and in bounds. Lean: `store_index_injective`,
//!   `store_index_in_bounds`; Kani: `store_index_injective_and_in_bounds`.
//! - [`tile_index`] / [`tail_range`]: a d-long loop in tiles of T reads full tiles, then the tail.
//!   Every read is in bounds and every element is read exactly once. Kani:
//!   `tiled_loop_reads_each_element_once`; F-1 (the tail reads d+1) is caught by
//!   `f1_tail_overread_is_caught`.
//! - [`vector_mask`]: the active lanes of vector chunk `c` of a d-long loop in vectors of V (the
//!   SIMD remainder mask). Active lanes are in bounds and cover every element once. Kani:
//!   `vector_masks_cover_each_element_once`.
//!
//! The `#[cfg(test)]` tests check the same properties exhaustively over small bounds, so the
//! properties run in every `cargo test`; the Kani harnesses (`cargo kani -p aprender-kernel-oracle`)
//! prove them for every value up to their bounds.

/// K3: the address thread `t` writes for lane `j` of its stride-`s` slice.
#[must_use]
pub const fn store_index(t: usize, s: usize, j: usize) -> usize {
    t * s + j
}

/// The number of full tiles in a d-long loop.
///
/// # Panics
/// If `tile == 0`.
#[must_use]
pub const fn full_tiles(d: usize, tile: usize) -> usize {
    assert!(tile > 0, "index_map: tile must be positive");
    d / tile
}

/// The element lane `lane` of full tile `k` reads.
#[must_use]
pub const fn tile_index(k: usize, tile: usize, lane: usize) -> usize {
    k * tile + lane
}

/// The tail's elements, as `start..end`: empty when `tile` divides d.
///
/// # Panics
/// If `tile == 0`.
#[must_use]
pub const fn tail_range(d: usize, tile: usize) -> (usize, usize) {
    (full_tiles(d, tile) * tile, d)
}

/// The active-lane mask of vector chunk `chunk` in a d-long loop over vectors of `vec` lanes:
/// bit `l` is set iff element `chunk * vec + l` exists. Full chunks are all ones, the remainder
/// chunk has the low `d mod vec` bits set, and chunks past the end are empty.
///
/// # Panics
/// If `vec` is 0 or above 64 (a mask is one `u64`).
#[must_use]
pub const fn vector_mask(d: usize, vec: usize, chunk: usize) -> u64 {
    assert!(vec > 0 && vec <= 64, "index_map: vec must be in 1..=64");
    let start = chunk.saturating_mul(vec);
    let active = if start >= d {
        0
    } else if d - start >= vec {
        vec
    } else {
        d - start
    };
    if active == 64 {
        u64::MAX
    } else {
        (1u64 << active) - 1
    }
}

/// The number of vector chunks a d-long loop runs (the last one may be partial).
///
/// # Panics
/// If `vec == 0`.
#[must_use]
pub const fn vector_chunks(d: usize, vec: usize) -> usize {
    assert!(vec > 0, "index_map: vec must be positive");
    d.div_ceil(vec)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Every element index a tiled loop reads, in order. `tail_overread` plants F-1.
    fn tiled_reads(d: usize, tile: usize, tail_overread: usize) -> Vec<usize> {
        let mut reads = Vec::new();
        for k in 0..full_tiles(d, tile) {
            for lane in 0..tile {
                reads.push(tile_index(k, tile, lane));
            }
        }
        let (start, end) = tail_range(d, tile);
        if start != end {
            reads.extend(start..end + tail_overread);
        }
        reads
    }

    #[test]
    fn store_index_is_injective_and_in_bounds() {
        for n in 0..=8 {
            for s in 0..=8 {
                let mut seen = vec![false; n * s];
                for t in 0..n {
                    for j in 0..s {
                        let i = store_index(t, s, j);
                        assert!(i < n * s, "n = {n}, s = {s}: ({t}, {j}) → {i}");
                        assert!(!seen[i], "n = {n}, s = {s}: {i} written twice");
                        seen[i] = true;
                    }
                }
                assert!(seen.iter().all(|&w| w), "n = {n}, s = {s}: a gap");
            }
        }
    }

    #[test]
    fn tiled_loop_reads_each_element_once() {
        for tile in 1..=16 {
            for d in 0..=5 * tile {
                let reads = tiled_reads(d, tile, 0);
                assert_eq!(reads, (0..d).collect::<Vec<_>>(), "d = {d}, T = {tile}");
            }
        }
    }

    /// F-1 at L1: the over-reading tail reads index d exactly when a tail exists.
    #[test]
    fn f1_tail_overread_reads_out_of_bounds_iff_a_tail_exists() {
        for tile in 1..=16 {
            for d in 0..=5 * tile {
                let oob = tiled_reads(d, tile, 1).iter().any(|&i| i >= d);
                assert_eq!(oob, d % tile != 0, "d = {d}, T = {tile}");
            }
        }
    }

    #[test]
    fn vector_masks_cover_each_element_once() {
        for vec in [1, 2, 3, 4, 7, 8, 16, 63, 64] {
            for d in 0..=3 * vec + 1 {
                let mut covered = 0usize;
                for c in 0..=vector_chunks(d, vec) {
                    let m = vector_mask(d, vec, c);
                    for l in 0..vec {
                        if (m >> l) & 1 == 1 {
                            assert_eq!(c * vec + l, covered, "d = {d}, V = {vec}, chunk {c}");
                            covered += 1;
                        }
                    }
                    if vec < 64 {
                        assert_eq!(m >> vec, 0, "no lane past V");
                    }
                }
                assert_eq!(covered, d, "d = {d}, V = {vec}");
                if d % vec != 0 {
                    let last = vector_mask(d, vec, d / vec);
                    assert_eq!(last.count_ones() as usize, d % vec, "the remainder mask");
                }
            }
        }
    }
}

/// Kani proofs (`cargo kani -p aprender-kernel-oracle`). Bounds are chosen so every tail class of
/// `shapes::cls4` is reachable (d up to 3·T + 1) and CBMC finishes in seconds.
#[cfg(kani)]
mod proofs {
    use super::*;

    const MAX: usize = 8;

    #[kani::proof]
    fn store_index_injective_and_in_bounds() {
        let (n, s): (usize, usize) = (kani::any(), kani::any());
        kani::assume(n <= MAX && s <= MAX);
        let (t1, j1, t2, j2): (usize, usize, usize, usize) =
            (kani::any(), kani::any(), kani::any(), kani::any());
        kani::assume(t1 < n && t2 < n && j1 < s && j2 < s);
        let (a, b) = (store_index(t1, s, j1), store_index(t2, s, j2));
        assert!(a < n * s && b < n * s);
        assert!(a != b || (t1 == t2 && j1 == j2));
    }

    /// Every read of the tiled loop is in bounds, and each element is read exactly once.
    #[kani::proof]
    #[kani::unwind(10)]
    fn tiled_loop_reads_each_element_once() {
        let (d, tile): (usize, usize) = (kani::any(), kani::any());
        kani::assume(tile >= 1 && tile <= MAX && d <= 3 * tile + 1);
        let mut next = 0;
        for k in 0..full_tiles(d, tile) {
            for lane in 0..tile {
                let i = tile_index(k, tile, lane);
                assert!(i < d && i == next);
                next += 1;
            }
        }
        let (start, end) = tail_range(d, tile);
        for i in start..end {
            assert!(i < d && i == next);
            next += 1;
        }
        assert!(next == d);
    }

    /// F-1 caught by Kani: a tail that reads one past its end must break the in-bounds assertion.
    /// `should_panic` makes this harness fail if Kani finds NO out-of-bounds read.
    #[kani::proof]
    #[kani::should_panic]
    #[kani::unwind(10)]
    fn f1_tail_overread_is_caught() {
        let (d, tile): (usize, usize) = (kani::any(), kani::any());
        kani::assume(tile >= 1 && tile <= MAX && d <= 3 * tile + 1);
        let (start, end) = tail_range(d, tile);
        if start != end {
            for i in start..end + 1 {
                assert!(i < d, "tail read out of bounds");
            }
        }
    }

    /// The remainder masks: active lanes are in bounds, and the chunks cover 0..d exactly.
    #[kani::proof]
    #[kani::unwind(10)]
    fn vector_masks_cover_each_element_once() {
        let (d, vec, c): (usize, usize, usize) = (kani::any(), kani::any(), kani::any());
        kani::assume(vec >= 1 && vec <= MAX && d <= 3 * vec + 1 && c <= vector_chunks(d, vec));
        let m = vector_mask(d, vec, c);
        let mut active = 0;
        for l in 0..vec {
            if (m >> l) & 1 == 1 {
                assert!(c * vec + l < d);
                active += 1;
            } else {
                assert!(c * vec + l >= d);
            }
        }
        assert!(m >> vec == 0);
        let expect = if c * vec >= d {
            0
        } else {
            (d - c * vec).min(vec)
        };
        assert!(active == expect);
    }
}
