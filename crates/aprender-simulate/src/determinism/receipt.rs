//! Receipt of a determinism-kernel run (PMAT-7288): what was run and what came out.

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
