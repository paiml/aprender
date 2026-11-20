//! Determinism core (`contracts/simular/sim-determinism-core-v1.yaml`, PMAT-7288 D1).
//!
//! Three pieces, each a falsifiable claim in the contract:
//!
//! * [`mix`] / [`keyed_draw`]: an order-independent keyed RNG. A draw is a pure function of
//!   `(seed, entity, event, k)`; no generator state advances, so evaluation order cannot matter.
//! * [`run`]: a fixed-step kernel in integer nanoseconds. Each entity reads only the previous
//!   snapshot and writes a fresh buffer, which is sorted by entity id before it is hashed.
//! * a BLAKE3 hash chain over the sorted snapshots, linked to the previous step.

/// Domain tag for the per-step chain hash.
const STEP_DOMAIN: &[u8] = b"simular.determinism.step.v1";
/// Domain tag for the genesis value `h_(-1)`.
const GENESIS_DOMAIN: &[u8] = b"simular.determinism.genesis.v1";

/// splitmix64 finalizer applied to `x + golden-gamma`. A bijection on `u64`.
#[must_use]
pub fn mix(x: u64) -> u64 {
    let z = x.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    let z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// `draw(seed, entity, event, k) = mix(mix(mix(mix(seed) xor entity) xor event) xor k)`.
#[must_use]
pub fn keyed_draw(seed: u64, entity: u64, event: u64, k: u64) -> u64 {
    mix(mix(mix(mix(seed) ^ entity) ^ event) ^ k)
}

/// One simulated entity: a unique id and one state word.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Entity {
    pub id: u64,
    pub state: u64,
}

/// What to run: seed, step length in integer nanoseconds, step count and initial entities.
#[derive(Debug, Clone)]
pub struct Spec {
    pub seed: u64,
    pub dt_ns: u64,
    pub steps: u64,
    pub entities: Vec<Entity>,
}

/// A refusal to run (the contract preconditions).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SpecError {
    ZeroDt,
    DuplicateEntityId,
}

/// A single-bit corruption of one state word, used only to falsify chain linkage.
#[derive(Debug, Clone, Copy)]
pub struct Tamper {
    pub step: u64,
    pub index: usize,
    pub bit: u32,
}

/// `h_(-1)`: the base case of the chain, a function of the seed alone.
#[must_use]
pub fn genesis(seed: u64) -> [u8; 32] {
    let mut h = blake3::Hasher::new();
    h.update(GENESIS_DOMAIN);
    h.update(&seed.to_le_bytes());
    *h.finalize().as_bytes()
}

/// `h_k = H(domain, h_(k-1), k, t_k, states_k)`, `t_k = k * dt_ns`, states sorted by id.
fn step_hash(prev: &[u8; 32], k: u64, t_ns: u64, sorted: &[Entity]) -> [u8; 32] {
    let mut h = blake3::Hasher::new();
    h.update(STEP_DOMAIN);
    h.update(prev);
    h.update(&k.to_le_bytes());
    h.update(&t_ns.to_le_bytes());
    h.update(&(sorted.len() as u64).to_le_bytes());
    for e in sorted {
        h.update(&e.id.to_le_bytes());
        h.update(&e.state.to_le_bytes());
    }
    *h.finalize().as_bytes()
}

/// Run the kernel. `order(k, n)` gives the evaluation order (a permutation of `0..n`) for step
/// `k`; the result must not depend on it. Returns `[h_0, ..., h_(steps-1)]`.
///
/// # Errors
/// [`SpecError::ZeroDt`] if `dt_ns == 0`; [`SpecError::DuplicateEntityId`] on a repeated id.
pub fn run_with(
    spec: &Spec,
    tamper: Option<Tamper>,
    mut order: impl FnMut(u64, usize) -> Vec<usize>,
) -> Result<Vec<[u8; 32]>, SpecError> {
    if spec.dt_ns == 0 {
        return Err(SpecError::ZeroDt);
    }
    let mut ids: Vec<u64> = spec.entities.iter().map(|e| e.id).collect();
    ids.sort_unstable();
    if ids.windows(2).any(|w| w[0] == w[1]) {
        return Err(SpecError::DuplicateEntityId);
    }
    let mut snapshot = spec.entities.clone();
    let mut prev = genesis(spec.seed);
    let mut chain = Vec::new();
    for k in 0..spec.steps {
        let total = snapshot.iter().fold(0u64, |a, e| a.wrapping_add(e.state));
        let mut fresh = Vec::with_capacity(snapshot.len());
        for i in order(k, snapshot.len()) {
            let e = snapshot[i];
            let draw = keyed_draw(spec.seed, e.id, 0, k);
            fresh.push(Entity {
                id: e.id,
                state: mix(e.state ^ draw ^ total),
            });
        }
        // Canonicalise: the hash must not see evaluation order.
        fresh.sort_by_key(|e| e.id);
        if let Some(t) = tamper {
            if t.step == k {
                fresh[t.index].state ^= 1u64 << (t.bit % 64);
            }
        }
        prev = step_hash(&prev, k, k.wrapping_mul(spec.dt_ns), &fresh);
        chain.push(prev);
        snapshot = fresh;
    }
    Ok(chain)
}

/// [`run_with`] in natural order.
///
/// # Errors
/// As [`run_with`].
pub fn run(spec: &Spec) -> Result<Vec<[u8; 32]>, SpecError> {
    run_with(spec, None, |_, n| (0..n).collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn spec(seed: u64) -> Spec {
        Spec {
            seed,
            dt_ns: 1_000_000,
            steps: 200,
            entities: (0..8)
                .map(|i| Entity {
                    id: 100 + i,
                    state: mix(i),
                })
                .collect(),
        }
    }

    fn last(c: &[[u8; 32]]) -> [u8; 32] {
        *c.last().expect("non-empty chain")
    }

    fn hex(h: &[u8; 32]) -> String {
        h.iter().map(|b| format!("{b:02x}")).collect()
    }

    /// Fisher-Yates driven by the keyed draw, so the shuffle itself is reproducible.
    fn shuffled(salt: u64, k: u64, n: usize) -> Vec<usize> {
        let mut v: Vec<usize> = (0..n).collect();
        for i in (1..n).rev() {
            let j = (keyed_draw(salt, i as u64, 7, k) % (i as u64 + 1)) as usize;
            v.swap(i, j);
        }
        v
    }

    #[test]
    fn keyed_draw_is_pure_and_keyed() {
        // FALSIFY-SDC-006 / SDC-INV-001
        let base = keyed_draw(1, 2, 3, 4);
        assert_eq!(base, keyed_draw(1, 2, 3, 4));
        assert_ne!(base, keyed_draw(9, 2, 3, 4));
        assert_ne!(base, keyed_draw(1, 9, 3, 4));
        assert_ne!(base, keyed_draw(1, 2, 9, 4));
        assert_ne!(base, keyed_draw(1, 2, 3, 9));
        // keys are not interchangeable: swapping two components changes the value
        assert_ne!(keyed_draw(1, 2, 3, 4), keyed_draw(1, 3, 2, 4));
    }

    #[test]
    fn same_spec_and_seed_twice_gives_same_hash() {
        let a = run(&spec(1)).expect("run");
        let b = run(&spec(1)).expect("run");
        assert_eq!(a.len(), 200);
        assert_eq!(a, b);
    }

    #[test]
    fn shuffled_evaluation_order_gives_same_hash() {
        let base = run(&spec(1)).expect("run");
        for salt in 0..8u64 {
            let got = run_with(&spec(1), None, |k, n| shuffled(salt, k, n)).expect("run");
            assert_eq!(base, got, "salt {salt}");
        }
        // the shuffle must actually permute, or the test proves nothing
        assert_ne!(shuffled(0, 0, 8), (0..8).collect::<Vec<_>>());
    }

    #[test]
    fn different_seed_gives_different_hash() {
        let a = run(&spec(1)).expect("run");
        let b = run(&spec(2)).expect("run");
        assert_ne!(a[0], b[0]);
        assert_ne!(last(&a), last(&b));
    }

    #[test]
    fn chain_links_every_step_to_the_previous() {
        let base = run(&spec(1)).expect("run");
        let t = Tamper {
            step: 3,
            index: 2,
            bit: 5,
        };
        let bad = run_with(&spec(1), Some(t), |_, n| (0..n).collect()).expect("run");
        assert_eq!(&base[..3], &bad[..3], "no earlier hash may change");
        for k in 3..base.len() {
            assert_ne!(base[k], bad[k], "h_{k} must change");
        }
    }

    #[test]
    fn refuses_zero_dt_and_duplicate_ids() {
        let mut s = spec(1);
        s.dt_ns = 0;
        assert_eq!(run(&s), Err(SpecError::ZeroDt));
        let mut s = spec(1);
        s.entities[1].id = s.entities[0].id;
        assert_eq!(run(&s), Err(SpecError::DuplicateEntityId));
    }

    #[test]
    fn golden_final_hash_is_pinned() {
        let h = hex(&last(&run(&spec(1)).expect("run")));
        assert_eq!(
            h,
            "0d55ad4ffbc99c6c601e8b5d7865c93ddd47a802ce81c10a43f8904538e6986b"
        );
    }
}
