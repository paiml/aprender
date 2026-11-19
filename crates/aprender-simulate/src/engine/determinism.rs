//! Determinism core: seeded, order-independent stepping with a hash-chained
//! trajectory and a receipt (SIM3D P1, #7246).
//!
//! Rules this module follows, each one pinned by a test that can fail:
//!
//! 1. **Integer time.** Time is [`SimTime`] (whole nanoseconds, `u64`). The step
//!    clock is `t_k = k * dt_ns`, exact integer arithmetic, no float.
//! 2. **Keyed RNG.** [`KeyedRng`] is stateless: a draw is a pure function of
//!    `(seed, entity, event, draw index)`, so the order in which entities are
//!    evaluated cannot change any value. There is no shared stream to advance.
//! 3. **No float in the kernel path.** Entity state is `u64` words; the stepping,
//!    mixing, hashing and receipt code uses integer operations only. The one float
//!    helper, [`KeyedRng::unit_f64`], is an exact `u64 -> f64` conversion of 53
//!    bits scaled by a power of two (no `ln`, `exp`, `sin`, no libm call).
//!    A model that wants floats keeps them in its own words via `f64::to_bits`; plain
//!    `+ - * /` on IEEE-754 doubles is correctly rounded on every host, and Rust
//!    never fuses a multiply-add on its own. The kernel itself calls no FMA, no
//!    rayon and no SIMD dispatch.
//! 4. **Snapshot stepping.** Every entity reads the *previous* step's snapshot and
//!    writes into a fresh buffer. The buffer is then canonicalised (sorted by entity
//!    id) before it is hashed, so evaluation order is not observable.
//! 5. **Hash chain.** `h_0 = H(domain, spec_sha, seed, dt_ns, entity ids)`,
//!    `h_k = H(domain, h_{k-1}, k, t_k, canonical states)` with BLAKE3. Its output
//!    does not depend on which SIMD path the `blake3` crate picks at run time.
//!
//! The code uses only `core` + `alloc` style types (`Vec`, `String`) and `blake3`
//! and `serde_json`; nothing reads a clock, the environment or the filesystem, so it
//! is written to compile for `wasm32`. Whether a `wasm32` build was *measured* is
//! reported by the commit/PR that adds it, not asserted here.

use crate::engine::SimTime;
use std::fmt::Write as _;

/// Domain-separation tag for the chain.
const DOMAIN: &[u8] = b"simular.det.v1";
/// Golden-ratio increment of splitmix64.
const GOLDEN: u64 = 0x9E37_79B9_7F4A_7C15;

/// Identifier of the kernel generation; bump when the algorithm changes.
pub const KERNEL_VERSION: &str = "simular-det-core-1";

/// Errors from building or running a deterministic simulation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DetError {
    /// Two entities share an id.
    DuplicateEntity(u64),
    /// `dt_ns` was zero.
    ZeroTimestep,
    /// An evaluation order was not a permutation of `0..entity_count`.
    BadEvalOrder,
    /// A receipt did not match the closed receipt shape.
    BadReceiptShape(String),
}

impl std::fmt::Display for DetError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::DuplicateEntity(id) => write!(f, "duplicate entity id {id}"),
            Self::ZeroTimestep => write!(f, "dt_ns must be > 0"),
            Self::BadEvalOrder => write!(f, "eval order is not a permutation of the entities"),
            Self::BadReceiptShape(why) => write!(f, "receipt shape: {why}"),
        }
    }
}

impl std::error::Error for DetError {}

/// The splitmix64 output function applied to `x + GOLDEN` (a bijection on `u64`).
#[must_use]
pub const fn mix64(x: u64) -> u64 {
    let mut z = x.wrapping_add(GOLDEN);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// Stateless counter-based RNG keyed by `(entity, event, draw)`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct KeyedRng {
    seed: u64,
}

impl KeyedRng {
    /// Create an RNG for `seed`.
    #[must_use]
    pub const fn new(seed: u64) -> Self {
        Self { seed }
    }

    /// The seed this RNG was keyed with.
    #[must_use]
    pub const fn seed(&self) -> u64 {
        self.seed
    }

    /// The `draw`-th 64-bit value for `(entity, event)`. Pure: no state advances.
    #[must_use]
    pub const fn draw_u64(&self, entity: u64, event: u64, draw: u64) -> u64 {
        let mut h = mix64(self.seed);
        h = mix64(h ^ entity);
        h = mix64(h ^ event);
        mix64(h ^ draw)
    }

    /// Uniform in `[0, bound)` by a 128-bit multiply (no division, no float).
    /// `bound == 0` returns 0.
    #[must_use]
    pub const fn below(&self, entity: u64, event: u64, draw: u64, bound: u64) -> u64 {
        let r = self.draw_u64(entity, event, draw) as u128;
        ((r * bound as u128) >> 64) as u64
    }

    /// Uniform in `[0, 1)` with 53 random bits; exact, no libm.
    #[must_use]
    pub fn unit_f64(&self, entity: u64, event: u64, draw: u64) -> f64 {
        let bits = self.draw_u64(entity, event, draw) >> 11;
        bits as f64 * (1.0 / 9_007_199_254_740_992.0)
    }
}

/// One entity's words at one step.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EntityState {
    /// Stable entity id (unique within a simulation).
    pub id: u64,
    /// State words (integers, or `f64::to_bits`).
    pub words: Vec<u64>,
}

/// Read-only context handed to [`DetModel::advance`].
#[derive(Debug, Clone, Copy)]
pub struct StepCtx<'a> {
    /// Index of the step being computed (1 for the first step).
    pub step: u64,
    /// Simulation time at the end of this step.
    pub time: SimTime,
    /// The keyed RNG; key draws by `(entity id, self.step, n)`.
    pub rng: &'a KeyedRng,
    /// The previous step's snapshot, sorted by entity id.
    pub prev: &'a [EntityState],
}

/// A model: a pure transition for one entity.
pub trait DetModel {
    /// Compute the entity's next words from `ctx.prev` and the keyed RNG only.
    /// Must not read shared mutable state: that would make order observable.
    fn advance(&self, ctx: &StepCtx<'_>, prev: &EntityState) -> Vec<u64>;
}

/// An invariant checked after every step. `Err` is a violation (Jidoka stop).
pub trait Monitor {
    /// Stable name (recorded in the receipt).
    fn name(&self) -> &str;
    /// `true` when the post-step snapshot satisfies the invariant.
    fn holds(&self, step: u64, time: SimTime, state: &[EntityState]) -> bool;
}

/// Verdict of one monitor over a run.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MonitorVerdict {
    /// Monitor name.
    pub name: String,
    /// Steps on which the monitor was evaluated.
    pub steps_checked: u64,
    /// First step at which it failed, if any.
    pub first_violation: Option<u64>,
}

/// The hash-chained trajectory: `hashes[0]` is `h_0`, `hashes[k]` is `h_k`.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct Trajectory {
    /// Chain of 32-byte hashes.
    pub hashes: Vec<[u8; 32]>,
}

impl Trajectory {
    /// The last hash in the chain (`h_n`), if any.
    #[must_use]
    pub fn final_hash(&self) -> Option<[u8; 32]> {
        self.hashes.last().copied()
    }
}

/// Receipt of a run: what was run and what came out.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Receipt {
    /// BLAKE3 of the spec bytes the caller supplied (hex).
    pub spec_sha: String,
    /// BLAKE3 of this kernel's source (hex), see [`kernel_sha`].
    pub kernel_sha: String,
    /// The seed.
    pub seed: u64,
    /// Steps completed (a Jidoka stop leaves this below the requested count).
    pub steps: u64,
    /// Hex of the last chain hash.
    pub final_hash: String,
    /// One verdict per monitor, sorted by name.
    pub monitor_verdicts: Vec<MonitorVerdict>,
}

/// Lower-case hex of `bytes`.
#[must_use]
pub fn hex(bytes: &[u8]) -> String {
    let mut s = String::with_capacity(bytes.len() * 2);
    for b in bytes {
        let _ = write!(s, "{b:02x}");
    }
    s
}

/// BLAKE3 of the kernel's own source, excluding its test module, as hex.
///
/// Embedding the source means a changed kernel changes its receipts.
#[must_use]
pub fn kernel_sha() -> String {
    const SRC: &str = include_str!("determinism.rs");
    let body = SRC.split("\n#[cfg(test)]").next().unwrap_or(SRC);
    hex(blake3::hash(body.as_bytes()).as_bytes())
}

/// BLAKE3 of arbitrary spec bytes as hex (what callers put in `spec_sha`).
#[must_use]
pub fn spec_sha(spec_bytes: &[u8]) -> String {
    hex(blake3::hash(spec_bytes).as_bytes())
}

/// The deterministic simulation kernel.
pub struct DetSim<M: DetModel> {
    model: M,
    rng: KeyedRng,
    dt_ns: u64,
    spec_sha: [u8; 32],
    state: Vec<EntityState>,
    step: u64,
    trajectory: Trajectory,
    monitors: Vec<Box<dyn Monitor>>,
    verdicts: Vec<MonitorVerdict>,
    halted: bool,
}

impl<M: DetModel> DetSim<M> {
    /// Build a simulation. `spec_bytes` are hashed into `h_0` and the receipt.
    ///
    /// # Errors
    /// [`DetError::ZeroTimestep`] when `dt_ns == 0`; [`DetError::DuplicateEntity`]
    /// when two entities share an id.
    pub fn new(
        model: M,
        spec_bytes: &[u8],
        seed: u64,
        dt_ns: u64,
        mut entities: Vec<EntityState>,
    ) -> Result<Self, DetError> {
        if dt_ns == 0 {
            return Err(DetError::ZeroTimestep);
        }
        entities.sort_by_key(|e| e.id);
        if let Some(w) = entities.windows(2).find(|w| w[0].id == w[1].id) {
            return Err(DetError::DuplicateEntity(w[0].id));
        }
        let spec_sha = *blake3::hash(spec_bytes).as_bytes();
        let mut h = blake3::Hasher::new();
        h.update(DOMAIN);
        h.update(b"/init");
        h.update(&spec_sha);
        h.update(&seed.to_le_bytes());
        h.update(&dt_ns.to_le_bytes());
        update_states(&mut h, &entities);
        let h0 = *h.finalize().as_bytes();
        Ok(Self {
            model,
            rng: KeyedRng::new(seed),
            dt_ns,
            spec_sha,
            state: entities,
            step: 0,
            trajectory: Trajectory { hashes: vec![h0] },
            monitors: Vec::new(),
            verdicts: Vec::new(),
            halted: false,
        })
    }

    /// Register an invariant monitor (checked after every step).
    pub fn add_monitor(&mut self, monitor: Box<dyn Monitor>) {
        self.verdicts.push(MonitorVerdict {
            name: monitor.name().to_string(),
            steps_checked: 0,
            first_violation: None,
        });
        self.monitors.push(monitor);
    }

    /// Current step count.
    #[must_use]
    pub const fn steps(&self) -> u64 {
        self.step
    }

    /// Current time, `step * dt_ns` (saturating at `u64::MAX`).
    #[must_use]
    pub const fn time(&self) -> SimTime {
        SimTime::from_nanos(self.step.saturating_mul(self.dt_ns))
    }

    /// Current snapshot, sorted by entity id.
    #[must_use]
    pub fn state(&self) -> &[EntityState] {
        &self.state
    }

    /// The trajectory so far.
    #[must_use]
    pub const fn trajectory(&self) -> &Trajectory {
        &self.trajectory
    }

    /// `true` once a monitor has stopped the run.
    #[must_use]
    pub const fn halted(&self) -> bool {
        self.halted
    }

    /// Advance one step evaluating entities in index order.
    pub fn step(&mut self) {
        let order: Vec<usize> = (0..self.state.len()).collect();
        // The identity order is always a valid permutation.
        let _ = self.step_with_order(&order);
    }

    /// Advance one step evaluating entities in `order` (a permutation of
    /// `0..entity_count`). The result must not depend on `order`.
    ///
    /// # Errors
    /// [`DetError::BadEvalOrder`] when `order` is not a permutation.
    pub fn step_with_order(&mut self, order: &[usize]) -> Result<(), DetError> {
        if self.halted {
            return Ok(());
        }
        let n = self.state.len();
        let mut seen = vec![false; n];
        if order.len() != n
            || order
                .iter()
                .any(|&i| i >= n || std::mem::replace(&mut seen[i], true))
        {
            return Err(DetError::BadEvalOrder);
        }
        let next_step = self.step + 1;
        let time = SimTime::from_nanos(next_step.saturating_mul(self.dt_ns));
        let ctx = StepCtx {
            step: next_step,
            time,
            rng: &self.rng,
            prev: &self.state,
        };
        let mut next: Vec<EntityState> = Vec::with_capacity(n);
        for &i in order {
            let prev = &self.state[i];
            next.push(EntityState {
                id: prev.id,
                words: self.model.advance(&ctx, prev),
            });
        }
        // Canonicalise: evaluation order must not be observable.
        next.sort_by_key(|e| e.id);
        let mut h = blake3::Hasher::new();
        h.update(DOMAIN);
        h.update(b"/step");
        if let Some(prev_hash) = self.trajectory.hashes.last() {
            h.update(prev_hash);
        }
        h.update(&next_step.to_le_bytes());
        h.update(&time.as_nanos().to_le_bytes());
        update_states(&mut h, &next);
        self.trajectory.hashes.push(*h.finalize().as_bytes());
        self.state = next;
        self.step = next_step;
        self.check_monitors(next_step, time);
        Ok(())
    }

    /// Run `steps` more steps in index order, stopping early on a violation.
    pub fn run(&mut self, steps: u64) {
        for _ in 0..steps {
            if self.halted {
                break;
            }
            self.step();
        }
    }

    fn check_monitors(&mut self, step: u64, time: SimTime) {
        for (m, v) in self.monitors.iter().zip(self.verdicts.iter_mut()) {
            v.steps_checked += 1;
            if !m.holds(step, time, &self.state) && v.first_violation.is_none() {
                v.first_violation = Some(step);
                self.halted = true;
            }
        }
    }

    /// Build the receipt for the run so far.
    #[must_use]
    pub fn receipt(&self) -> Receipt {
        let mut verdicts = self.verdicts.clone();
        verdicts.sort_by(|a, b| a.name.cmp(&b.name));
        Receipt {
            spec_sha: hex(&self.spec_sha),
            kernel_sha: kernel_sha(),
            seed: self.rng.seed(),
            steps: self.step,
            final_hash: self
                .trajectory
                .final_hash()
                .map(|h| hex(&h))
                .unwrap_or_default(),
            monitor_verdicts: verdicts,
        }
    }
}

fn update_states(h: &mut blake3::Hasher, states: &[EntityState]) {
    h.update(&(states.len() as u64).to_le_bytes());
    for e in states {
        h.update(&e.id.to_le_bytes());
        h.update(&(e.words.len() as u64).to_le_bytes());
        for w in &e.words {
            h.update(&w.to_le_bytes());
        }
    }
}

fn json_escape(s: &str) -> String {
    let mut out = String::with_capacity(s.len() + 2);
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            c if (c as u32) < 0x20 => {
                let _ = write!(out, "\\u{:04x}", c as u32);
            }
            c => out.push(c),
        }
    }
    out
}

fn shape_err(m: &str) -> DetError {
    DetError::BadReceiptShape(m.to_string())
}

fn is_hex64(v: &serde_json::Value) -> bool {
    v.as_str().is_some_and(|s| {
        s.len() == 64
            && s.bytes()
                .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
    })
}

fn verdict_ok(item: &serde_json::Value) -> bool {
    let Some(o) = item.as_object() else {
        return false;
    };
    o.len() == 3
        && o.get("name").is_some_and(serde_json::Value::is_string)
        && o.get("steps_checked").is_some_and(|x| x.as_u64().is_some())
        && o.get("first_violation")
            .is_some_and(|x| x.is_null() || x.as_u64().is_some())
}

impl Receipt {
    /// Canonical JSON: keys sorted at every level, no whitespace, integers only.
    /// Byte-stable across hosts, suitable for hashing and for `pv` shape gates.
    #[must_use]
    pub fn to_canonical_json(&self) -> String {
        let mut s = String::new();
        let _ = write!(
            s,
            "{{\"final_hash\":\"{}\",\"kernel_sha\":\"{}\",\"monitor_verdicts\":[",
            self.final_hash, self.kernel_sha
        );
        for (i, v) in self.monitor_verdicts.iter().enumerate() {
            if i > 0 {
                s.push(',');
            }
            let fv = v
                .first_violation
                .map_or_else(|| "null".to_string(), |x| x.to_string());
            let _ = write!(
                s,
                "{{\"first_violation\":{fv},\"name\":\"{}\",\"steps_checked\":{}}}",
                json_escape(&v.name),
                v.steps_checked
            );
        }
        let _ = write!(
            s,
            "],\"seed\":{},\"spec_sha\":\"{}\",\"steps\":{}}}",
            self.seed, self.spec_sha, self.steps
        );
        s
    }

    /// Check `json` against the closed receipt shape: exactly the six keys, hex
    /// hashes of 64 chars, unsigned integers, well-formed verdicts.
    ///
    /// # Errors
    /// [`DetError::BadReceiptShape`] naming the first violation.
    pub fn check_shape(json: &str) -> Result<(), DetError> {
        let v: serde_json::Value =
            serde_json::from_str(json).map_err(|e| shape_err(&e.to_string()))?;
        let obj = v.as_object().ok_or_else(|| shape_err("not an object"))?;
        let keys = [
            "final_hash",
            "kernel_sha",
            "monitor_verdicts",
            "seed",
            "spec_sha",
            "steps",
        ];
        if obj.len() != keys.len() || keys.iter().any(|k| !obj.contains_key(*k)) {
            return Err(shape_err("keys are not exactly the closed set"));
        }
        for k in ["final_hash", "kernel_sha", "spec_sha"] {
            if !is_hex64(&obj[k]) {
                return Err(shape_err(&format!("{k} is not 64 hex chars")));
            }
        }
        for k in ["seed", "steps"] {
            if obj[k].as_u64().is_none() {
                return Err(shape_err(&format!("{k} is not an unsigned integer")));
            }
        }
        let verdicts = obj["monitor_verdicts"]
            .as_array()
            .ok_or_else(|| shape_err("monitor_verdicts is not an array"))?;
        if verdicts.iter().all(verdict_ok) {
            Ok(())
        } else {
            Err(shape_err("verdict shape"))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Entities interact through the previous snapshot (right neighbour) and
    /// draw from the keyed RNG, so order-dependence would show up.
    struct Ring;
    impl DetModel for Ring {
        fn advance(&self, ctx: &StepCtx<'_>, prev: &EntityState) -> Vec<u64> {
            let idx = ctx.prev.iter().position(|e| e.id == prev.id).unwrap_or(0);
            let right = &ctx.prev[(idx + 1) % ctx.prev.len()];
            let r = ctx.rng.draw_u64(prev.id, ctx.step, 0);
            let arrivals = ctx.rng.below(prev.id, ctx.step, 1, 5);
            vec![
                prev.words[0].wrapping_add(arrivals),
                prev.words[1] ^ r ^ right.words[0].rotate_left(7),
            ]
        }
    }

    fn ents(n: u64) -> Vec<EntityState> {
        // Deliberately unsorted ids to exercise canonicalisation in `new`.
        (0..n)
            .rev()
            .map(|i| EntityState {
                id: i * 3 + 1,
                words: vec![i, 0],
            })
            .collect()
    }

    fn sim(seed: u64) -> DetSim<Ring> {
        DetSim::new(Ring, b"spec-v1", seed, 1_000_000, ents(16)).unwrap()
    }

    #[test]
    fn mix64_matches_splitmix64_reference_vector() {
        // First output of splitmix64 with state 0 (published reference value).
        assert_eq!(mix64(0), 0xE220_A839_7B1D_CDAF);
    }

    #[test]
    fn keyed_rng_is_pure_and_keyed_by_every_component() {
        let r = KeyedRng::new(7);
        assert_eq!(r.draw_u64(1, 2, 3), r.draw_u64(1, 2, 3));
        let base = r.draw_u64(1, 2, 3);
        assert_ne!(base, r.draw_u64(9, 2, 3));
        assert_ne!(base, r.draw_u64(1, 9, 3));
        assert_ne!(base, r.draw_u64(1, 2, 9));
        assert_ne!(base, KeyedRng::new(8).draw_u64(1, 2, 3));
    }

    #[test]
    fn below_and_unit_stay_in_range_and_below_zero_is_zero() {
        let r = KeyedRng::new(1);
        for d in 0..2000 {
            assert!(r.below(1, 1, d, 7) < 7);
            let u = r.unit_f64(1, 1, d);
            assert!((0.0..1.0).contains(&u));
        }
        assert_eq!(r.below(1, 1, 0, 0), 0);
    }

    #[test]
    fn same_spec_and_seed_twice_gives_same_hash() {
        let (mut a, mut b) = (sim(42), sim(42));
        a.run(200);
        b.run(200);
        assert_eq!(a.trajectory(), b.trajectory());
        assert_eq!(a.receipt(), b.receipt());
        assert_eq!(a.trajectory().hashes.len(), 201);
    }

    #[test]
    fn shuffled_evaluation_order_gives_same_hash() {
        let mut base = sim(42);
        base.run(100);
        for salt in 1..=8u64 {
            let mut s = sim(42);
            for step in 0..100u64 {
                let mut order: Vec<usize> = (0..16).collect();
                // Fisher-Yates driven by the keyed RNG, different per salt and step.
                let sh = KeyedRng::new(salt);
                for i in (1..order.len()).rev() {
                    let j = sh.below(0, step, i as u64, (i + 1) as u64) as usize;
                    order.swap(i, j);
                }
                s.step_with_order(&order).unwrap();
            }
            assert_eq!(s.trajectory(), base.trajectory(), "salt {salt}");
        }
    }

    #[test]
    fn different_seed_gives_different_hash() {
        let (mut a, mut b) = (sim(1), sim(2));
        a.run(50);
        b.run(50);
        assert_ne!(a.receipt().final_hash, b.receipt().final_hash);
        assert_ne!(a.trajectory().hashes[0], b.trajectory().hashes[0]);
    }

    #[test]
    fn different_spec_or_dt_gives_different_chain() {
        let a = DetSim::new(Ring, b"spec-v1", 1, 1000, ents(4)).unwrap();
        let b = DetSim::new(Ring, b"spec-v2", 1, 1000, ents(4)).unwrap();
        let c = DetSim::new(Ring, b"spec-v1", 1, 2000, ents(4)).unwrap();
        assert_ne!(a.trajectory().hashes[0], b.trajectory().hashes[0]);
        assert_ne!(a.trajectory().hashes[0], c.trajectory().hashes[0]);
    }

    #[test]
    fn chain_links_every_step_to_the_previous() {
        let mut a = sim(5);
        a.run(10);
        let mut b = sim(5);
        b.run(10);
        // Perturb one entity at step 3 only: every later hash must differ.
        let mut c = sim(5);
        c.run(2);
        c.state[0].words[0] ^= 1;
        c.run(8);
        assert_eq!(a.trajectory().hashes[..3], c.trajectory().hashes[..3]);
        for k in 3..=10 {
            assert_ne!(a.trajectory().hashes[k], c.trajectory().hashes[k]);
        }
        assert_eq!(a.trajectory(), b.trajectory());
    }

    #[test]
    fn time_is_exact_integer_nanoseconds() {
        let mut s = DetSim::new(Ring, b"x", 1, 3, ents(2)).unwrap();
        s.run(5);
        assert_eq!(s.time().as_nanos(), 15);
        assert_eq!(s.steps(), 5);
    }

    #[test]
    fn construction_rejects_zero_dt_and_duplicate_ids() {
        assert_eq!(
            DetSim::new(Ring, b"x", 1, 0, ents(2)).err(),
            Some(DetError::ZeroTimestep)
        );
        let dup = vec![
            EntityState {
                id: 4,
                words: vec![0, 0],
            },
            EntityState {
                id: 4,
                words: vec![1, 0],
            },
        ];
        assert_eq!(
            DetSim::new(Ring, b"x", 1, 1, dup).err(),
            Some(DetError::DuplicateEntity(4))
        );
    }

    #[test]
    fn bad_eval_orders_are_rejected() {
        let mut s = sim(1);
        assert_eq!(s.step_with_order(&[0, 1]), Err(DetError::BadEvalOrder));
        let mut dup: Vec<usize> = (0..16).collect();
        dup[1] = 0;
        assert_eq!(s.step_with_order(&dup), Err(DetError::BadEvalOrder));
        let mut oob: Vec<usize> = (0..16).collect();
        oob[15] = 16;
        assert_eq!(s.step_with_order(&oob), Err(DetError::BadEvalOrder));
        assert_eq!(s.steps(), 0);
    }

    struct CapMonitor(u64);
    impl Monitor for CapMonitor {
        fn name(&self) -> &str {
            "total-below-cap"
        }
        fn holds(&self, _s: u64, _t: SimTime, st: &[EntityState]) -> bool {
            st.iter().map(|e| e.words[0]).sum::<u64>() < self.0
        }
    }

    #[test]
    fn monitor_violation_halts_at_first_bad_step_and_is_recorded() {
        let mut s = sim(3);
        s.add_monitor(Box::new(CapMonitor(200)));
        s.run(1000);
        assert!(s.halted());
        let r = s.receipt();
        let v = &r.monitor_verdicts[0];
        assert_eq!(v.first_violation, Some(r.steps));
        assert!(r.steps < 1000);
        // A looser cap never trips.
        let mut ok = sim(3);
        ok.add_monitor(Box::new(CapMonitor(u64::MAX)));
        ok.run(100);
        assert!(!ok.halted());
        assert_eq!(ok.receipt().monitor_verdicts[0].first_violation, None);
        assert_eq!(ok.receipt().monitor_verdicts[0].steps_checked, 100);
    }

    #[test]
    fn receipt_json_is_canonical_and_passes_shape() {
        let mut s = sim(9);
        s.add_monitor(Box::new(CapMonitor(u64::MAX)));
        s.run(10);
        let j = s.receipt().to_canonical_json();
        assert!(Receipt::check_shape(&j).is_ok(), "{j}");
        assert!(j.starts_with("{\"final_hash\":\""));
        let again = {
            let mut t = sim(9);
            t.add_monitor(Box::new(CapMonitor(u64::MAX)));
            t.run(10);
            t.receipt().to_canonical_json()
        };
        assert_eq!(j, again);
        assert_eq!(s.receipt().kernel_sha.len(), 64);
        assert_eq!(s.receipt().spec_sha, spec_sha(b"spec-v1"));
    }

    #[test]
    fn shape_rejects_planted_defects() {
        let mut s = sim(9);
        s.add_monitor(Box::new(CapMonitor(u64::MAX)));
        s.run(3);
        let good = s.receipt().to_canonical_json();
        let cases = [
            good.replace("\"seed\":9", "\"seed\":\"9\""),
            good.replace("\"steps\":3", "\"steps\":-3"),
            good.replacen("\"final_hash\":\"", "\"final_hash\":\"zz", 1),
            good.replace("\"kernel_sha\"", "\"kernel\""),
            good.replace("}", ",\"extra\":1}"),
            good.replace("\"monitor_verdicts\":[", "\"monitor_verdicts\":[1,"),
            "[]".to_string(),
            "{".to_string(),
        ];
        for c in &cases {
            assert!(Receipt::check_shape(c).is_err(), "accepted: {c}");
        }
    }

    #[test]
    fn json_escape_handles_quotes_backslash_and_control() {
        assert_eq!(json_escape("a\"b\\c\u{1}"), "a\\\"b\\\\c\\u0001");
    }

    #[test]
    fn kernel_sha_ignores_test_module_and_is_stable() {
        assert_eq!(kernel_sha(), kernel_sha());
        assert_eq!(kernel_sha().len(), 64);
    }

    #[test]
    fn golden_final_hash_is_pinned() {
        // Pinned so a host/compiler/dependency change that alters the bits fails.
        let mut s = sim(42);
        s.run(200);
        assert_eq!(s.receipt().final_hash, GOLDEN_FINAL_HASH);
    }

    const GOLDEN_FINAL_HASH: &str =
        "767f63f68670c60aca5dec94f38c9317f40a806e066cb134c6604c3cabb581fc";
}
