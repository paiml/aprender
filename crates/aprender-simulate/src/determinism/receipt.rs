//! Receipt of a determinism-kernel run (PMAT-7288): what was run and what came out.
//!
//! A [`Receipt`] names the spec bytes (SHA-256, what `sha256sum <spec>` prints), the kernel
//! source (BLAKE3), the seed, the step count, the last hash of the chain and one verdict per
//! invariant [`Monitor`]. [`Receipt::to_canonical_json`] writes it in the closed shape of
//! `contracts/sim3d-receipt-v1.yaml`; pv grades that output in
//! `crates/aprender-contracts-cli/tests/sim3d_kernel_receipt.rs`.
//!
//! The step itself is not defined here: every run goes through [`super::run_observed`].

use super::{run_observed, Entity, Spec, SpecError};
use std::fmt::Write as _;

/// An invariant checked on the canonical snapshot after every step.
pub trait Monitor {
    /// Stable name, recorded in the receipt. The pv shape admits only `INV-nnn`.
    fn name(&self) -> &str;
    /// `true` when the snapshot of step `step` (time `t_ns`, sorted by entity id) satisfies it.
    fn holds(&self, step: u64, t_ns: u64, state: &[Entity]) -> bool;
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

/// The hash-chained trajectory: `hashes[k]` is `h_k`.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct Trajectory {
    /// Chain of 32-byte hashes, one per step.
    pub hashes: Vec<[u8; 32]>,
}

impl Trajectory {
    /// The last hash in the chain, if any.
    #[must_use]
    pub fn final_hash(&self) -> Option<[u8; 32]> {
        self.hashes.last().copied()
    }
}

/// Receipt of a run: what was run and what came out.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Receipt {
    /// SHA-256 of the spec bytes the caller supplied (hex); `sha256sum <spec>` prints the same.
    pub spec_sha256: String,
    /// BLAKE3 of this kernel's source (hex, 64 chars), see [`kernel_sha`].
    pub kernel_sha: String,
    /// The seed.
    pub seed: u64,
    /// Steps run.
    pub steps: u64,
    /// Hex of the last chain hash.
    pub final_hash: String,
    /// One verdict per monitor, sorted by name.
    pub verdicts: Vec<MonitorVerdict>,
}

/// Why no receipt was written.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReceiptError {
    /// The kernel refused the spec.
    Spec(SpecError),
    /// `steps == 0`: there is no final hash to state.
    ZeroSteps,
    /// No monitor: a run nothing judged proves nothing (SIM-RCPT-INV-002).
    NoMonitors,
    /// Two monitors share a name, so a verdict would not name one invariant.
    DuplicateMonitor,
}

impl std::fmt::Display for ReceiptError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Spec(e) => write!(f, "spec refused by the kernel: {e:?}"),
            Self::ZeroSteps => f.write_str("zero steps: no final hash to state"),
            Self::NoMonitors => f.write_str("no monitor: a receipt needs at least one verdict"),
            Self::DuplicateMonitor => f.write_str("two monitors share a name"),
        }
    }
}

impl std::error::Error for ReceiptError {}

/// Lower-case hex of `bytes`.
#[must_use]
pub fn hex(bytes: &[u8]) -> String {
    let mut s = String::with_capacity(bytes.len() * 2);
    for b in bytes {
        let _ = write!(s, "{b:02x}");
    }
    s
}

/// SHA-256 of the spec bytes as hex; equals what `sha256sum` prints for the same bytes.
#[must_use]
pub fn spec_sha256(spec_bytes: &[u8]) -> String {
    use sha2::{Digest, Sha256};
    hex(&Sha256::digest(spec_bytes))
}

/// BLAKE3 of the kernel's own source (the step in `determinism.rs` and this file, each without
/// its test module) as hex. Embedding the source means a changed kernel changes its receipts.
#[must_use]
pub fn kernel_sha() -> String {
    fn body(src: &str) -> &str {
        src.split("\n#[cfg(test)]").next().unwrap_or(src)
    }
    let mut h = blake3::Hasher::new();
    h.update(body(include_str!("../determinism.rs")).as_bytes());
    h.update(body(include_str!("receipt.rs")).as_bytes());
    hex(h.finalize().as_bytes())
}

/// Run `spec` under `monitors` with evaluation order `order` and state the receipt.
/// `spec_bytes` are the bytes of the spec document the run answers.
///
/// # Errors
/// [`ReceiptError`] when the kernel refuses the spec, or when the run could not state a receipt
/// the shape accepts (no steps, no monitor, a repeated monitor name).
pub fn run_receipt_with(
    spec_bytes: &[u8],
    spec: &Spec,
    monitors: &[&dyn Monitor],
    order: impl FnMut(u64, usize) -> Vec<usize>,
) -> Result<(Trajectory, Receipt), ReceiptError> {
    if spec.steps == 0 {
        return Err(ReceiptError::ZeroSteps);
    }
    if monitors.is_empty() {
        return Err(ReceiptError::NoMonitors);
    }
    let mut verdicts: Vec<MonitorVerdict> = monitors
        .iter()
        .map(|m| MonitorVerdict {
            name: m.name().to_owned(),
            steps_checked: 0,
            first_violation: None,
        })
        .collect();
    let hashes = run_observed(spec, None, order, |k, t_ns, state| {
        for (m, v) in monitors.iter().zip(verdicts.iter_mut()) {
            v.steps_checked += 1;
            if v.first_violation.is_none() && !m.holds(k, t_ns, state) {
                v.first_violation = Some(k);
            }
        }
    })
    .map_err(ReceiptError::Spec)?;
    verdicts.sort_by(|a, b| a.name.cmp(&b.name));
    if verdicts.windows(2).any(|w| w[0].name == w[1].name) {
        return Err(ReceiptError::DuplicateMonitor);
    }
    let trajectory = Trajectory { hashes };
    let receipt = Receipt {
        spec_sha256: spec_sha256(spec_bytes),
        kernel_sha: kernel_sha(),
        seed: spec.seed,
        steps: spec.steps,
        final_hash: trajectory.final_hash().map(|h| hex(&h)).unwrap_or_default(),
        verdicts,
    };
    Ok((trajectory, receipt))
}

/// [`run_receipt_with`] in natural order.
///
/// # Errors
/// As [`run_receipt_with`].
pub fn run_receipt(
    spec_bytes: &[u8],
    spec: &Spec,
    monitors: &[&dyn Monitor],
) -> Result<(Trajectory, Receipt), ReceiptError> {
    run_receipt_with(spec_bytes, spec, monitors, |_, n| (0..n).collect())
}

fn json_escape(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            c if u32::from(c) < 0x20 => {
                let _ = write!(out, "\\u{:04x}", u32::from(c));
            }
            c => out.push(c),
        }
    }
    out
}

impl Receipt {
    /// Canonical JSON in the shape of `contracts/sim3d-receipt-v1.yaml`: keys sorted at every
    /// level, no whitespace, integers only. `kernel_sha` is the first 40 hex chars of the
    /// kernel's BLAKE3 (the shape's 40-hex width); the full digest stays in
    /// [`Receipt::kernel_sha`]. Byte-stable across hosts.
    #[must_use]
    pub fn to_canonical_json(&self) -> String {
        let mut s = String::new();
        let kernel40 = self.kernel_sha.get(..40).unwrap_or(&self.kernel_sha);
        let _ = write!(
            s,
            "{{\"final_hash\":\"{}\",\"kernel_sha\":\"{kernel40}\",\"schema\":\"sim3d-receipt/v1\",\"seed\":{},\"spec_sha256\":\"{}\",\"steps\":{},\"verdicts\":[",
            self.final_hash, self.seed, self.spec_sha256, self.steps
        );
        for (i, v) in self.verdicts.iter().enumerate() {
            if i > 0 {
                s.push(',');
            }
            let name = json_escape(&v.name);
            let _ = match v.first_violation {
                Some(step) => write!(
                    s,
                    "{{\"first_bad_step\":{step},\"holds\":false,\"invariant\":\"{name}\"}}"
                ),
                None => write!(s, "{{\"holds\":true,\"invariant\":\"{name}\"}}"),
            };
        }
        s.push_str("]}");
        s
    }
}

#[cfg(test)]
mod tests {
    use super::super::{keyed_draw, mix, Entity, Spec};
    use super::*;

    const SPEC_BYTES: &[u8] = b"schema: sim3d-spec/v1\nname: unit\n";

    fn spec(seed: u64) -> Spec {
        Spec {
            seed,
            dt_ns: 1_000_000,
            steps: 64,
            entities: (0..8)
                .map(|i| Entity {
                    id: 100 + i,
                    state: mix(i),
                })
                .collect(),
        }
    }

    /// Holds while the wrapping sum of all state words is at most the cap.
    struct Cap(&'static str, u64);
    impl Monitor for Cap {
        fn name(&self) -> &str {
            self.0
        }
        fn holds(&self, _step: u64, _t_ns: u64, state: &[Entity]) -> bool {
            state.iter().fold(0u64, |a, e| a.wrapping_add(e.state)) <= self.1
        }
    }

    /// Fails from step 3 on, and records the time it was shown.
    struct FromStep3;
    impl Monitor for FromStep3 {
        fn name(&self) -> &str {
            "INV-003"
        }
        fn holds(&self, step: u64, t_ns: u64, _state: &[Entity]) -> bool {
            assert_eq!(t_ns, step * 1_000_000, "integer step clock");
            step < 3
        }
    }

    fn shuffled(salt: u64, k: u64, n: usize) -> Vec<usize> {
        let mut v: Vec<usize> = (0..n).collect();
        for i in (1..n).rev() {
            let j = (keyed_draw(salt, i as u64, 7, k) % (i as u64 + 1)) as usize;
            v.swap(i, j);
        }
        v
    }

    #[test]
    fn spec_sha256_is_a_real_sha256() {
        // FIPS 180-2 vectors: what `printf abc | sha256sum` and `sha256sum /dev/null` print.
        assert_eq!(
            spec_sha256(b"abc"),
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
        );
        assert_eq!(
            spec_sha256(b""),
            "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
        );
        let ok = Cap("INV-001", u64::MAX);
        let (_, r) = run_receipt(b"abc", &spec(1), &[&ok]).expect("run");
        assert_eq!(
            r.spec_sha256,
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
        );
    }

    #[test]
    fn same_spec_and_seed_twice_gives_same_receipt() {
        let ok = Cap("INV-001", u64::MAX);
        let (ta, a) = run_receipt(SPEC_BYTES, &spec(1), &[&ok]).expect("run");
        let (tb, b) = run_receipt(SPEC_BYTES, &spec(1), &[&ok]).expect("run");
        assert_eq!(a, b);
        assert_eq!(ta, tb);
        assert_eq!(a.to_canonical_json(), b.to_canonical_json());
        // the receipt's final hash IS the kernel's last chain hash, not a second computation
        let chain = super::super::run(&spec(1)).expect("run");
        assert_eq!(ta.hashes, chain);
        assert_eq!(a.final_hash, hex(chain.last().expect("non-empty")));
        assert_eq!(ta.final_hash(), chain.last().copied());
        assert_eq!(a.steps, 64);
        assert_eq!(a.seed, 1);
    }

    #[test]
    fn shuffled_entity_order_gives_same_receipt() {
        let ok = Cap("INV-001", u64::MAX);
        let (_, base) = run_receipt(SPEC_BYTES, &spec(1), &[&ok]).expect("run");
        for salt in 0..8u64 {
            // shuffle the evaluation order of every step
            let (_, got) =
                run_receipt_with(SPEC_BYTES, &spec(1), &[&ok], |k, n| shuffled(salt, k, n))
                    .expect("run");
            assert_eq!(base, got, "salt {salt}");
            // and the order the entities are listed in the spec
            let mut s = spec(1);
            let perm = shuffled(salt, 99, s.entities.len());
            s.entities = perm.iter().map(|&i| spec(1).entities[i]).collect();
            let (_, listed) = run_receipt(SPEC_BYTES, &s, &[&ok]).expect("run");
            assert_eq!(base.final_hash, listed.final_hash, "listing salt {salt}");
        }
        assert_ne!(shuffled(0, 99, 8), (0..8).collect::<Vec<_>>());
    }

    #[test]
    fn different_seed_gives_different_final_hash() {
        let ok = Cap("INV-001", u64::MAX);
        let (_, a) = run_receipt(SPEC_BYTES, &spec(1), &[&ok]).expect("run");
        let (_, b) = run_receipt(SPEC_BYTES, &spec(2), &[&ok]).expect("run");
        assert_ne!(a.final_hash, b.final_hash);
        assert_eq!(a.spec_sha256, b.spec_sha256);
        assert_eq!((a.seed, b.seed), (1, 2));
    }

    #[test]
    fn verdicts_are_sorted_and_name_the_first_bad_step() {
        let ok = Cap("INV-001", u64::MAX);
        let never = Cap("INV-002", 0);
        let (_, r) = run_receipt(SPEC_BYTES, &spec(1), &[&FromStep3, &never, &ok]).expect("run");
        let got: Vec<_> = r
            .verdicts
            .iter()
            .map(|v| (v.name.as_str(), v.steps_checked, v.first_violation))
            .collect();
        assert_eq!(
            got,
            [
                ("INV-001", 64, None),
                ("INV-002", 64, Some(0)),
                ("INV-003", 64, Some(3))
            ]
        );
    }

    #[test]
    fn canonical_json_is_pinned_byte_for_byte() {
        let ok = Cap("INV-001", u64::MAX);
        let (_, r) = run_receipt(b"abc", &spec(1), &[&FromStep3, &ok]).expect("run");
        let k = kernel_sha();
        assert_eq!(k.len(), 64);
        assert_eq!(r.kernel_sha, k);
        assert_eq!(
            r.to_canonical_json(),
            format!(
                "{{\"final_hash\":\"{}\",\"kernel_sha\":\"{}\",\"schema\":\"sim3d-receipt/v1\",\"seed\":1,\
                 \"spec_sha256\":\"ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad\",\
                 \"steps\":64,\"verdicts\":[{{\"holds\":true,\"invariant\":\"INV-001\"}},\
                 {{\"first_bad_step\":3,\"holds\":false,\"invariant\":\"INV-003\"}}]}}",
                r.final_hash,
                &k[..40]
            )
        );
        // it parses, and a monitor name cannot break out of its string
        let odd = Cap("a\"b\\c\n", u64::MAX);
        let (_, r) = run_receipt(b"abc", &spec(1), &[&odd]).expect("run");
        let v: serde_json::Value = serde_json::from_str(&r.to_canonical_json()).expect("json");
        assert_eq!(v["verdicts"][0]["invariant"].as_str(), Some("a\"b\\c\n"));
    }

    #[test]
    fn refuses_a_run_that_cannot_state_a_receipt() {
        let ok = Cap("INV-001", u64::MAX);
        assert_eq!(
            run_receipt(SPEC_BYTES, &spec(1), &[]).map(|_| ()),
            Err(ReceiptError::NoMonitors)
        );
        let mut s = spec(1);
        s.steps = 0;
        assert_eq!(
            run_receipt(SPEC_BYTES, &s, &[&ok]).map(|_| ()),
            Err(ReceiptError::ZeroSteps)
        );
        let mut s = spec(1);
        s.dt_ns = 0;
        assert_eq!(
            run_receipt(SPEC_BYTES, &s, &[&ok]).map(|_| ()),
            Err(ReceiptError::Spec(super::super::SpecError::ZeroDt))
        );
        let dup = Cap("INV-001", 0);
        assert_eq!(
            run_receipt(SPEC_BYTES, &spec(1), &[&ok, &dup]).map(|_| ()),
            Err(ReceiptError::DuplicateMonitor)
        );
        for e in [
            ReceiptError::NoMonitors,
            ReceiptError::ZeroSteps,
            ReceiptError::DuplicateMonitor,
            ReceiptError::Spec(super::super::SpecError::ZeroDt),
        ] {
            assert!(!e.to_string().is_empty());
        }
    }
}
