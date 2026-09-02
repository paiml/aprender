//! PP-LLAMA-001 v3.0 §10 / PP-32 — the engine track's `AbRecord`.
//!
//! # What this is for, and what it must be unable to say
//!
//! Engine work — a kernel change, a flag, a scheduler fix — proceeds from today
//! and needs no comparator, no matrix run and no §12 row. What it must never do
//! is announce a parity ratio (PP-12): "the fused kernel is N times faster" is a
//! claim about two builds of `apr`, and it acquires a comparator only by
//! sleight of hand.
//!
//! So PP-32 is a *shape* rule: this record **has no field able to hold a
//! comparator, a second runtime name, or a parity verdict**, and
//! `deny_unknown_fields` means one cannot be added by a producer either. A JSON
//! document carrying `comparator`, `runtime`, `llama_agg` or `parity` does not
//! parse. That is the whole guard — there is nothing to remember to check.
//!
//! # What it must say
//!
//! - `delta_kind`: `config` (one binary, one flag) or `code` (two binaries, two
//!   commits). Both arms' effective configs are diffed and **any difference
//!   outside the declared delta is a hard error** — a `code` arm pair that also
//!   moved a flag measured two changes and attributed them to one.
//! - Two arms, each with its own `commit` and `sha256`. For a `config` delta
//!   they are the same build twice, and the record still says so explicitly.
//! - `interleaved: true` and a strictly alternating `order`. §4.3's reasoning
//!   is identical here: thermal and warm-cache state drift across a sweep.
//! - `prediction`, written **before** the run. A prediction recorded afterwards
//!   is a description.
//! - `interval`, the §4.3 replicate estimator over the per-replicate `agg`. It
//!   is stored on the wire *and* re-derived by [`AbRecord::validate`], so a
//!   stated interval its own replicates do not produce is refused — the same
//!   rule `bench_receipt.py` applies to ratios.

use serde::{Deserialize, Serialize};
use serde_json::Value;

use super::join::Ratio;
use super::receipt::RunId;
use super::replicate::{log_ratio_bound_or_point, ArmOrder, ReplicatePair};

/// Which of the two arms. There are exactly two, and neither is "the
/// comparator": both are `apr`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ArmId {
    /// The control: the tree as it stands.
    A,
    /// The change under test.
    B,
}

/// §10 — what differs between the arms.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", deny_unknown_fields)]
pub enum DeltaKind {
    /// One binary, one flag. The flag name is the entire declared delta.
    Config {
        /// The single effective-config key the arms are allowed to differ on.
        flag: String,
    },
    /// Two binaries, two commits. The declared delta is the code; the arms'
    /// effective configs must be **identical**.
    Code,
}

impl DeltaKind {
    /// The effective-config keys the arms are permitted to differ on.
    #[must_use]
    pub fn declared_keys(&self) -> Vec<&str> {
        match self {
            Self::Config { flag } => vec![flag.as_str()],
            Self::Code => Vec::new(),
        }
    }
}

/// One arm's identity and the configuration it actually resolved to.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Arm {
    /// Which arm.
    pub id: ArmId,
    /// The commit the binary was built from.
    pub commit: String,
    /// The binary's digest. 64 lowercase hex characters.
    pub sha256: String,
    /// `GET /v1/effective-config`, verbatim.
    pub effective_config: Value,
}

/// One key on which the two arms' effective configs differ.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ConfigDiff {
    /// The effective-config key.
    pub key: String,
    /// Arm A's value.
    pub a: Value,
    /// Arm B's value.
    pub b: Value,
}

/// One replicate of one arm.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AbReplicate {
    /// Which arm ran.
    pub arm: ArmId,
    /// Aggregate throughput for this replicate.
    pub agg: f64,
    /// Median per-request decode, when the run streamed.
    pub dec: Option<f64>,
    /// Server-reported prefill, when the server timed it.
    pub prefill: Option<f64>,
}

/// PP-32 — the engine track's record. Two arms of `apr`, interleaved, with a
/// prediction written before the run.
///
/// There is deliberately no `comparator`, no `runtime`, no `baseline` and no
/// `verdict` field, and `deny_unknown_fields` stops one being smuggled in.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AbRecord {
    /// The harness invocation both arms ran inside.
    pub run_id: RunId,
    /// RFC3339 UTC start instant.
    pub started_utc: String,
    /// Which host.
    pub host: String,
    /// What differs between the arms.
    pub delta_kind: DeltaKind,
    /// What the change was predicted to do, written before the run.
    pub prediction: String,
    /// Exactly two arms.
    pub arms: [Arm; 2],
    /// Must be `true`; §4.3's reasoning applies unchanged.
    pub interleaved: bool,
    /// The execution sequence, strictly alternating.
    pub order: Vec<ArmId>,
    /// The keys the two effective configs actually differ on.
    pub effective_config_diff: Vec<ConfigDiff>,
    /// One entry per element of `order`, in the same sequence.
    pub replicates: Vec<AbReplicate>,
    /// §4.3 — the exponentiated one-sided bound on `ln(B/A)` over the paired
    /// replicates. `None` when the design cannot support one.
    pub interval: Option<Ratio>,
}

impl AbRecord {
    /// Parse a record, refusing any key this type does not know — including,
    /// by construction, `comparator`.
    ///
    /// # Errors
    /// On malformed JSON, a missing required field, or an unknown one.
    pub fn parse(text: &str) -> Result<Self, String> {
        serde_json::from_str(text).map_err(|e| format!("parsing AbRecord: {e}"))
    }

    /// §10 — every rule the record must satisfy.
    ///
    /// # Errors
    /// When the arms are not `A` and `B`, when `interleaved` is false, when
    /// `order` does not strictly alternate, when the replicates do not match
    /// the order, when an effective-config difference lies outside the declared
    /// delta, when a `code` delta has identical shas on both arms, when the
    /// prediction is empty, or when the stored `interval` is not the one the
    /// replicates produce.
    pub fn validate(&self) -> Result<(), String> {
        self.validate_arms()?;
        if !self.interleaved {
            return Err(
                "PP-32: interleaved=false — the two arms must alternate within one harness \
                 invocation. Thermal state, warm caches and free VRAM drift across a sweep, and \
                 a block of A followed by a block of B measures the drift as well as the change"
                    .to_string(),
            );
        }
        self.validate_order()?;
        self.validate_config_diff()?;
        if self.prediction.trim().is_empty() {
            return Err(
                "PP-32: prediction is empty — §10 is 'predict, then verify', and a prediction \
                 recorded after the run is a description"
                    .to_string(),
            );
        }
        self.validate_interval()
    }

    /// §4.3 — the interval the replicates themselves produce.
    ///
    /// `subject` is arm B (the change) and `comparator` is arm A (the control),
    /// so a ratio above 1 means the change was faster.
    ///
    /// # The order is READ, not assumed
    ///
    /// A replicate is a **pair of adjacent runs** in this record's own `order`,
    /// and its [`ArmOrder`] is which of the two came first. The previous
    /// spelling paired the k-th `A` with the k-th `B` — however far apart they
    /// ran — and computed the order from `k % 2` and the first entry of
    /// `order`, which alternates by construction whatever `order` says.
    /// `log_ratio_lcb`'s counterbalancing refusal therefore could not fire from
    /// here at all: a blocked `A,A,A,B,B,B` sweep, the exact design §4.3's
    /// interleaving exists to reject, produced a **bound** as though it had
    /// alternated. Now a chunk that is not one of each arm yields `None`, and
    /// the pair order is the one the record recorded.
    ///
    /// Note the two disciplines this exposes, which are not the same rule:
    /// [`Self::validate_order`] requires the RUN sequence to alternate
    /// (`A,B,A,B,…`), while `log_ratio_lcb` requires the PAIR order to
    /// counterbalance (`AB, BA, AB, …`). An `A,B,A,B,…` record runs A first
    /// every time, so a first-run order effect is confounded with the arm and
    /// §4.3 gives it a point estimate and no bound. That is the honest reading
    /// of the design it recorded; it is not something this function can fix.
    #[must_use]
    pub fn derived_interval(&self) -> Option<Ratio> {
        log_ratio_bound_or_point(&self.replicate_pairs()?)
    }

    /// The interleaved pairs this record's `order` actually describes, or
    /// `None` when it describes none.
    fn replicate_pairs(&self) -> Option<Vec<ReplicatePair>> {
        if self.order.len() < 2 || self.replicates.len() != self.order.len() {
            return None;
        }
        let mut pairs = Vec::with_capacity(self.order.len() / 2);
        for (k, chunk) in self.order.chunks_exact(2).enumerate() {
            let (first, second) = (chunk[0], chunk[1]);
            if first == second {
                // Two runs of the same arm back to back: this is a block, not
                // an interleaved replicate, and there is no pair to form.
                return None;
            }
            let (a, b) = (&self.replicates[2 * k], &self.replicates[2 * k + 1]);
            if a.arm != first || b.arm != second {
                // The numbers are not in the sequence `order` claims, so which
                // ran first is unknown. Refused rather than guessed.
                return None;
            }
            let (control, change) = if first == ArmId::A { (a, b) } else { (b, a) };
            pairs.push(ReplicatePair {
                subject: change.agg,
                comparator: control.agg,
                order: if first == ArmId::A {
                    ArmOrder::ComparatorFirst
                } else {
                    ArmOrder::SubjectFirst
                },
            });
        }
        Some(pairs)
    }

    fn validate_arms(&self) -> Result<(), String> {
        if self.arms[0].id != ArmId::A || self.arms[1].id != ArmId::B {
            return Err(format!(
                "PP-32: arms are [{:?}, {:?}], expected [A, B]",
                self.arms[0].id, self.arms[1].id
            ));
        }
        for arm in &self.arms {
            if arm.sha256.len() != 64
                || !arm
                    .sha256
                    .bytes()
                    .all(|b| b.is_ascii_hexdigit() && !b.is_ascii_uppercase())
            {
                return Err(format!(
                    "PP-32: arm {:?} sha256 {:?} is not 64 lowercase hex characters",
                    arm.id, arm.sha256
                ));
            }
        }
        if self.delta_kind == DeltaKind::Code && self.arms[0].sha256 == self.arms[1].sha256 {
            return Err(
                "PP-32: delta_kind=code but both arms carry the same sha256 — a code delta needs \
                 two binaries, and one binary run twice measures noise"
                    .to_string(),
            );
        }
        Ok(())
    }

    fn validate_order(&self) -> Result<(), String> {
        if self.order.len() < 2 {
            return Err(format!(
                "PP-32: order has {} entries — an A/B record needs at least one of each",
                self.order.len()
            ));
        }
        if let Some(i) = self.order.windows(2).position(|w| w[0] == w[1]) {
            return Err(format!(
                "PP-32: order is not strictly alternating — entries {i} and {} are both {:?}",
                i + 1,
                self.order[i]
            ));
        }
        if self.replicates.len() != self.order.len() {
            return Err(format!(
                "PP-32: {} replicates against {} order entries — every run in the sequence must \
                 carry its numbers",
                self.replicates.len(),
                self.order.len()
            ));
        }
        for (i, (want, got)) in self.order.iter().zip(self.replicates.iter()).enumerate() {
            if *want != got.arm {
                return Err(format!(
                    "PP-32: replicate {i} is arm {:?} but order says {want:?}",
                    got.arm
                ));
            }
        }
        Ok(())
    }

    fn validate_config_diff(&self) -> Result<(), String> {
        let declared = self.delta_kind.declared_keys();
        let outside: Vec<&str> = self
            .effective_config_diff
            .iter()
            .map(|d| d.key.as_str())
            .filter(|k| !declared.contains(k))
            .collect();
        if outside.is_empty() {
            return Ok(());
        }
        Err(format!(
            "PP-32: the arms' effective configs differ on {outside:?}, outside the declared delta \
             {declared:?} — the run measured more than one change and attributed it to one"
        ))
    }

    fn validate_interval(&self) -> Result<(), String> {
        let derived = self.derived_interval();
        if intervals_agree(self.interval.as_ref(), derived.as_ref()) {
            return Ok(());
        }
        Err(format!(
            "PP-32: the stated interval {:?} is not the one these replicates produce ({derived:?}) \
             — a stated bound its own data does not reproduce is a fabricated measurement",
            self.interval
        ))
    }
}

/// Does a stated interval match the derived one?
///
/// `point` and `lcb95` are compared to a relative tolerance rather than bit for
/// bit: a record that has been through JSON can differ from the recomputed
/// value by an ulp, and refusing a record for that would be a rule about
/// `serde_json`'s float formatting rather than about the measurement. `method`
/// and `n` are compared exactly — those are claims, not measurements.
fn intervals_agree(stated: Option<&Ratio>, derived: Option<&Ratio>) -> bool {
    match (stated, derived) {
        (None, None) => true,
        (Some(a), Some(b)) => {
            a.method == b.method
                && a.n == b.n
                && close(Some(a.point), Some(b.point))
                && close(a.lcb95, b.lcb95)
        }
        _ => false,
    }
}

fn close(a: Option<f64>, b: Option<f64>) -> bool {
    match (a, b) {
        (None, None) => true,
        (Some(x), Some(y)) => (x - y).abs() <= 1e-9 * x.abs().max(y.abs()).max(1.0),
        _ => false,
    }
}

#[cfg(test)]
mod tests {
    // The `<selftest-name>__<sentence>` spelling is load-bearing: PP-29's
    // `scripts/spec_conformance.sh` joins the §6 invariant table to the test
    // list on the prefix before the double underscore, so renaming these to
    // single-underscore snake case would silently unjoin the rows they arm.
    #![allow(non_snake_case)]
    use super::*;
    use serde_json::json;

    fn arm(id: ArmId, commit: &str, sha: char, config: Value) -> Arm {
        Arm {
            id,
            commit: commit.to_string(),
            sha256: std::iter::repeat_n(sha, 64).collect(),
            effective_config: config,
        }
    }

    fn record(delta_kind: DeltaKind, arms: [Arm; 2]) -> AbRecord {
        let order = vec![ArmId::A, ArmId::B, ArmId::A, ArmId::B, ArmId::A, ArmId::B];
        let replicates: Vec<AbReplicate> = order
            .iter()
            .enumerate()
            .map(|(i, arm)| AbReplicate {
                arm: *arm,
                agg: if *arm == ArmId::A {
                    100.0 + i as f64
                } else {
                    130.0 + i as f64
                },
                dec: Some(40.0),
                prefill: None,
            })
            .collect();
        let mut r = AbRecord {
            run_id: RunId::derive("2026-09-02T10:11:12.345Z", "lambda", &"c".repeat(64), 7),
            started_utc: "2026-09-02T10:11:12.345Z".to_string(),
            host: "lambda".to_string(),
            delta_kind,
            prediction: "batched decode <= 3.5 ms/tok; agg(2) > 1.0x one client".to_string(),
            arms,
            interleaved: true,
            order,
            effective_config_diff: Vec::new(),
            replicates,
            interval: None,
        };
        r.interval = r.derived_interval();
        r
    }

    fn code_record() -> AbRecord {
        record(
            DeltaKind::Code,
            [
                arm(ArmId::A, "119f61738", 'a', json!({"max_batch": 11})),
                arm(ArmId::B, "2f0c9d114", 'b', json!({"max_batch": 11})),
            ],
        )
    }

    /// PP-32's must-not-fire: a `code` delta with two shas parses and validates.
    #[test]
    fn abrecord_ok__a_code_delta_with_two_shas_parses() {
        let r = code_record();
        let text = serde_json::to_string(&r).expect("serialises");
        let back = AbRecord::parse(&text).expect("round-trips");
        back.validate().expect("a conformant record validates");
        assert_eq!(back.arms, r.arms);
        assert_eq!(back.order, r.order);
        assert_eq!(back.replicates, r.replicates);
        assert_eq!(back.delta_kind, r.delta_kind);
        assert_eq!(back.run_id, r.run_id);
        assert!(
            intervals_agree(back.interval.as_ref(), r.interval.as_ref()),
            "{:?} vs {:?}",
            back.interval,
            r.interval
        );
        assert_eq!(back.arms[0].id, ArmId::A);
        assert_ne!(back.arms[0].sha256, back.arms[1].sha256);
        assert!(back.interval.expect("interval").point > 1.0);
    }

    /// The tolerance is a tolerance, not a hole: a bound that is actually
    /// different is still refused.
    #[test]
    fn the_interval_tolerance_admits_an_ulp_and_nothing_more() {
        let derived = code_record().interval.expect("interval");
        let one_ulp = Ratio {
            point: f64::from_bits(derived.point.to_bits() + 1),
            ..derived.clone()
        };
        assert!(intervals_agree(Some(&one_ulp), Some(&derived)));
        let moved = Ratio {
            point: derived.point * 1.000_01,
            ..derived.clone()
        };
        assert!(!intervals_agree(Some(&moved), Some(&derived)));
        assert!(!intervals_agree(None, Some(&derived)));
        assert!(!intervals_agree(Some(&derived), None));
    }

    /// PP-32's must-fire: the record has no field able to hold a comparator, and
    /// `deny_unknown_fields` means a producer cannot add one either.
    #[test]
    fn abrecord_comparator__a_comparator_field_does_not_parse() {
        let mut value = serde_json::to_value(code_record()).expect("serialises");
        value
            .as_object_mut()
            .expect("object")
            .insert("comparator".to_string(), json!({"runtime": "llama.cpp"}));
        let err = AbRecord::parse(&value.to_string()).expect_err("comparator must not parse");
        assert!(err.contains("comparator"), "{err}");

        // The same for every other spelling of a parity claim.
        for smuggled in ["runtime", "baseline", "parity", "agg_ratio", "llama_agg"] {
            let mut v = serde_json::to_value(code_record()).expect("serialises");
            v.as_object_mut()
                .expect("object")
                .insert(smuggled.to_string(), json!("llama.cpp"));
            assert!(
                AbRecord::parse(&v.to_string()).is_err(),
                "{smuggled} must not parse"
            );
        }
    }

    /// A block of A followed by a block of B measures the drift as well as the
    /// change.
    #[test]
    fn non_interleaved_ab_is_refused() {
        let mut r = code_record();
        r.interleaved = false;
        let err = r.validate().expect_err("interleaved=false");
        assert!(err.contains("alternate"), "{err}");

        // And the flag cannot lie about the sequence either.
        let mut blocked = code_record();
        blocked.order = vec![ArmId::A, ArmId::A, ArmId::B, ArmId::B];
        blocked.replicates = blocked
            .order
            .iter()
            .map(|arm| AbReplicate {
                arm: *arm,
                agg: 100.0,
                dec: None,
                prefill: None,
            })
            .collect();
        blocked.interval = blocked.derived_interval();
        let err = blocked.validate().expect_err("order does not alternate");
        assert!(err.contains("strictly alternating"), "{err}");
    }

    /// §10 — any effective-config difference outside the declared delta is a
    /// hard error, because the run then measured two changes.
    #[test]
    fn a_config_diff_outside_the_declared_delta_is_refused() {
        let mut r = record(
            DeltaKind::Config {
                flag: "FUSED_GATE_UP".to_string(),
            },
            [
                arm(ArmId::A, "119f61738", 'a', json!({"fused_gate_up": false})),
                arm(ArmId::A, "119f61738", 'a', json!({"fused_gate_up": true})),
            ],
        );
        r.arms[1].id = ArmId::B;
        r.effective_config_diff = vec![ConfigDiff {
            key: "FUSED_GATE_UP".to_string(),
            a: json!(false),
            b: json!(true),
        }];
        r.validate()
            .expect("the declared flag is allowed to differ");

        r.effective_config_diff.push(ConfigDiff {
            key: "max_batch".to_string(),
            a: json!(11),
            b: json!(16),
        });
        let err = r.validate().expect_err("max_batch is outside the delta");
        assert!(err.contains("max_batch"), "{err}");
        assert!(err.contains("FUSED_GATE_UP"), "{err}");

        // A `code` delta declares no config keys at all, so ANY diff is outside.
        let mut c = code_record();
        c.effective_config_diff = vec![ConfigDiff {
            key: "max_batch".to_string(),
            a: json!(11),
            b: json!(16),
        }];
        assert!(c.validate().is_err(), "a code delta permits no config diff");
    }

    /// Each arm's digest is checked, not just its presence.
    #[test]
    fn an_arm_digest_that_is_not_64_lowercase_hex_is_refused() {
        for bad in ["short", &"A".repeat(64), &"z".repeat(64), &"a".repeat(63)] {
            let mut r = code_record();
            r.arms[1].sha256 = (*bad).to_string();
            let err = r.validate().expect_err("{bad} must be refused");
            assert!(err.contains("64 lowercase hex"), "{bad}: {err}");
        }
        let mut swapped = code_record();
        swapped.arms.swap(0, 1);
        let err = swapped.validate().expect_err("arms out of order");
        assert!(err.contains("expected [A, B]"), "{err}");
    }

    /// A `code` delta needs two binaries; one binary run twice measures noise.
    #[test]
    fn a_code_delta_with_one_binary_is_refused() {
        let mut r = code_record();
        r.arms[1].sha256 = r.arms[0].sha256.clone();
        let err = r.validate().expect_err("one binary");
        assert!(err.contains("two binaries"), "{err}");
    }

    /// The stored interval must be the one the replicates produce — the rule
    /// `bench_receipt.py` applies to ratios, applied here.
    #[test]
    fn a_stated_interval_its_replicates_do_not_produce_is_refused() {
        let mut r = code_record();
        let mut fake = r.interval.clone().expect("interval");
        fake.lcb95 = Some(9.99);
        r.interval = Some(fake);
        let err = r.validate().expect_err("fabricated interval");
        assert!(err.contains("fabricated"), "{err}");
    }

    /// The prediction is written before the run; an empty one is a description.
    #[test]
    fn an_empty_prediction_is_refused() {
        let mut r = code_record();
        r.prediction = "   ".to_string();
        let err = r.validate().expect_err("no prediction");
        assert!(err.contains("predict, then verify"), "{err}");
    }

    /// The order and the replicates must describe the same sequence.
    #[test]
    fn replicates_must_match_the_declared_order() {
        let mut r = code_record();
        r.replicates.pop();
        let err = r.validate().expect_err("counts differ");
        assert!(err.contains("order entries"), "{err}");

        let mut swapped = code_record();
        swapped.replicates[0].arm = ArmId::B;
        let err = swapped.validate().expect_err("arm disagrees with order");
        assert!(err.contains("order says"), "{err}");
    }

    /// §4.3 MUST-FIRE, through `derived_interval`: a BLOCKED run —
    /// `A,A,A,B,B,B` rather than `A,B,A,B,A,B` — yields no interval at all.
    ///
    /// `derived_interval` used to pair the k-th `A` with the k-th `B` however
    /// far apart they ran, and to synthesise each pair's `ArmOrder` from
    /// `k % 2`, which alternates by construction. So a blocked sweep — the
    /// exact design §4.3's interleaving exists to reject — produced a bound as
    /// though it had alternated, and the counterbalancing refusal inside
    /// `log_ratio_lcb` was unreachable from here.
    #[test]
    fn abrecord_blocked__a_blocked_order_produces_no_interval() {
        let with_order = |order: Vec<ArmId>| -> AbRecord {
            let replicates: Vec<AbReplicate> = order
                .iter()
                .enumerate()
                .map(|(i, arm)| AbReplicate {
                    arm: *arm,
                    agg: if *arm == ArmId::A {
                        100.0 + i as f64
                    } else {
                        130.0 + i as f64
                    },
                    dec: Some(40.0),
                    prefill: None,
                })
                .collect();
            AbRecord {
                order,
                replicates,
                ..code_record()
            }
        };

        // MUST-NOT-FIRE: an interleaved sequence pairs, run by adjacent run.
        let interleaved = with_order(vec![
            ArmId::A,
            ArmId::B,
            ArmId::A,
            ArmId::B,
            ArmId::A,
            ArmId::B,
        ])
        .derived_interval()
        .expect("three adjacent pairs");
        assert_eq!(interleaved.n, 3);

        // MUST-FIRE: the same six runs, blocked, describe no interleaved pair.
        let blocked = with_order(vec![
            ArmId::A,
            ArmId::A,
            ArmId::A,
            ArmId::B,
            ArmId::B,
            ArmId::B,
        ]);
        assert!(
            blocked.derived_interval().is_none(),
            "a blocked sweep measures the drift as well as the change, and pairing its k-th A \
             with its k-th B pairs two runs minutes apart as though they had alternated"
        );
        // …and a record that STATES an interval its blocked replicates cannot
        // produce is refused through exactly this function.
        let stated = AbRecord {
            interval: Some(Ratio::reporting_only(
                1.3,
                crate::perf_gate::join::RatioMethod::ReplicateTLower,
                3,
            )),
            ..blocked
        };
        let err = stated
            .validate()
            .expect_err("a stated interval over a blocked sweep");
        assert!(err.contains("PP-32"), "{err}");
    }

    /// A counterbalanced sequence — `AB, BA, AB, BA, AB` — is the design
    /// `log_ratio_lcb` will bound: the arm that runs first flips every
    /// replicate, so a first-run order effect cancels instead of loading onto
    /// one arm.
    ///
    /// It is recorded here as the shape a bound REQUIRES. `validate_order`
    /// asks for the run sequence to alternate (`A,B,A,B,…`), which runs A first
    /// every time and therefore earns a point estimate and no bound — two
    /// different disciplines under one word, and a §10 question for the spec
    /// owner rather than something this function may decide.
    #[test]
    fn only_a_counterbalanced_sequence_earns_a_bound() {
        let order = vec![
            ArmId::A,
            ArmId::B,
            ArmId::B,
            ArmId::A,
            ArmId::A,
            ArmId::B,
            ArmId::B,
            ArmId::A,
            ArmId::A,
            ArmId::B,
        ];
        let replicates: Vec<AbReplicate> = order
            .iter()
            .enumerate()
            .map(|(i, arm)| AbReplicate {
                arm: *arm,
                agg: if *arm == ArmId::A {
                    100.0 + (i % 3) as f64
                } else {
                    130.0 + (i % 3) as f64
                },
                dec: None,
                prefill: None,
            })
            .collect();
        let counterbalanced = AbRecord {
            order,
            replicates,
            ..code_record()
        };
        let i = counterbalanced
            .derived_interval()
            .expect("five adjacent pairs");
        assert_eq!(i.n, 5);
        assert!(
            i.lcb95.is_some(),
            "five counterbalanced pairs support a bound: {i:?}"
        );

        // The must-not-fire's mirror: the SAME ten runs in the A,B,A,B,… order
        // `validate_order` asks for run A first every time, and get no bound.
        let abab: Vec<ArmId> = (0..10)
            .map(|i| if i % 2 == 0 { ArmId::A } else { ArmId::B })
            .collect();
        let replicates: Vec<AbReplicate> = abab
            .iter()
            .enumerate()
            .map(|(i, arm)| AbReplicate {
                arm: *arm,
                agg: if *arm == ArmId::A {
                    100.0 + (i % 3) as f64
                } else {
                    130.0 + (i % 3) as f64
                },
                dec: None,
                prefill: None,
            })
            .collect();
        let never_flips = AbRecord {
            order: abab,
            replicates,
            ..code_record()
        };
        let j = never_flips.derived_interval().expect("five adjacent pairs");
        assert_eq!(j.n, 5);
        assert!(
            j.lcb95.is_none(),
            "A always first is not counterbalanced, so §4.3 gives it no bound: {j:?}"
        );
    }

    /// The interval is B over A: a faster change is a ratio above 1.
    #[test]
    fn the_interval_is_the_change_over_the_control() {
        let r = code_record();
        let i = r.derived_interval().expect("three pairs");
        assert!(i.point > 1.2, "{i:?}");
        assert!(
            i.lcb95.is_none(),
            "three replicate pairs bound no variance (§4.3)"
        );
    }
}
