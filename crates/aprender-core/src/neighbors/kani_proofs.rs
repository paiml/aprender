//! Kani proof for the k-best selection every index reports through
//! (contracts/neighbor-index-v1.yaml, KANI-NBR-001).

use super::{key_cmp, KBest};
use std::cmp::Ordering;

/// For every order of 3 offers, `KBest::new(k)` holds exactly the `k` smallest
/// `(distance, index)` keys, ascending. This is the invariant that makes a
/// tree's answer independent of the order it visits points in.
///
/// The offer order is one of the 6 permutations and the distances are drawn
/// from a concrete set, so every tie pattern among three keys occurs. One
/// harness per `k`: a symbolic `k`, a symbolic `[f32; 3]`, or all `k` in one
/// harness each exhausted a 24 GiB cap in CBMC's propositional reduction.
fn check_kbest(k: usize) {
    const LEVELS: [f32; 3] = [0.0, 1.0, 2.5];
    const ORDERS: [[usize; 3]; 6] = [
        [0, 1, 2],
        [0, 2, 1],
        [1, 0, 2],
        [1, 2, 0],
        [2, 0, 1],
        [2, 1, 0],
    ];
    // Plain indexing, not `array::map`: its MaybeUninit guard is costly in CBMC.
    let pick: [u8; 3] = kani::any();
    kani::assume(pick[0] < 3 && pick[1] < 3 && pick[2] < 3);
    let dist = [
        LEVELS[pick[0] as usize],
        LEVELS[pick[1] as usize],
        LEVELS[pick[2] as usize],
    ];
    let o: u8 = kani::any();
    kani::assume(o < 6);
    let order = ORDERS[o as usize];

    let mut best = KBest::new(k);
    for &i in &order {
        best.offer(dist[i], i);
    }
    let got = best.into_sorted();
    // The expected answer by rank: `rank[j]` counts the keys smaller than j's.
    // Indices make every key distinct, so the ranks are a permutation of 0..3
    // and "got[t] is the key of rank t" pins both the kept set and its order.
    let key = |j: usize| (dist[j], j);
    let mut rank = [0usize; 3];
    for (j, r) in rank.iter_mut().enumerate() {
        for m in 0..3 {
            if key_cmp(key(m), key(j)) == Ordering::Less {
                *r += 1;
            }
        }
    }
    assert!(got.len() == k);
    // Index with the concrete `k`, not `got.iter()`: iterating a Vec whose
    // length CBMC does not pin exhausted a 16 GiB cap.
    for t in 0..k {
        let nb = &got[t];
        assert!(nb.index < 3);
        assert!(rank[nb.index] == t);
        assert!(nb.distance.to_bits() == dist[nb.index].to_bits());
    }
}

#[kani::proof]
#[kani::unwind(5)]
fn verify_kbest_k0() {
    check_kbest(0);
}

#[kani::proof]
#[kani::unwind(5)]
fn verify_kbest_k1() {
    check_kbest(1);
}

#[kani::proof]
#[kani::unwind(5)]
fn verify_kbest_k2() {
    check_kbest(2);
}

#[kani::proof]
#[kani::unwind(5)]
fn verify_kbest_k3() {
    check_kbest(3);
}
