//! registry_case_table — the hermetic case table of PP-066 R-0a (#2904, PMAT-989):
//! `trueno::registry::BackendRegistry` probes → enumerates → prints, and every
//! backend kind is a LINE the user reads, never a silence (REG-1, REG-4, REG-7,
//! REG-8, REG-9, REG-11, REG-12; design quorum 2026-09-06: object-safe factories,
//! `device_uid` dedup across APIs, reserve against free memory).
//!
//! RED first: this file was committed before `trueno::registry` existed.
use trueno::registry::{
    Api, BackendEntry, BackendFactory, BackendKind, BackendRegistry, MemKind, MockBackendFactory,
    Reason, Source, Status,
};

fn ready(kind: BackendKind, api: Api, idx: u32, name: &str, uid: &str, free: u64) -> BackendEntry {
    BackendEntry {
        kind,
        api,
        device_index: Some(idx),
        device_uid: Some(uid.to_string()),
        device_name: name.to_string(),
        vendor: "NVIDIA".to_string(),
        vendor_id: Some(0x10de),
        device_type: "discrete-gpu".to_string(),
        mem_total: Some(24 << 30),
        mem_free: Some(free),
        mem_kind: MemKind::Discrete,
        compute_class: Some("sm_89".to_string()),
        caps: vec!["async".to_string()],
        source: Source::Dlopen("/usr/lib/libcuda.so.1".to_string()),
        status: Status::Ready,
        transport: None,
    }
}

/// REG-11 + invariant (i): `cpu` is always Ready and every kind of the fixed
/// list prints a line, on any machine, with no fixture at all.
#[test]
fn cpu_is_always_ready_and_every_kind_prints_a_line() {
    let reg = BackendRegistry::discover();
    let cpu =
        reg.entries.iter().find(|e| e.kind == BackendKind::Cpu).expect("a cpu entry always exists");
    assert_eq!(cpu.status, Status::Ready, "cpu ∈ Ready always (invariant i)");
    for kind in BackendKind::ALL {
        assert!(
            reg.entries.iter().any(|e| e.kind == kind),
            "kind {kind:?} must be an explicit line, Ready or Unavailable(reason) — absence is the silence REG-11 forbids"
        );
    }
    let block = reg.render_block("test");
    for kind in BackendKind::ALL {
        assert!(
            block.lines().any(|l| l.starts_with(&format!("backend: {:<6}", kind.as_str()))),
            "printed block must carry a `backend: {}` line:\n{block}",
            kind.as_str()
        );
    }
    assert!(
        block.lines().any(|l| l.starts_with("selected: ")),
        "REG-8: the selection is always printed"
    );
}

/// Invariant (iii): every entry carries a source and, if not Ready, a non-empty
/// reason. A kind with no factory is Unavailable, never missing.
#[test]
fn a_kind_without_a_factory_is_unavailable_with_a_reason_not_missing() {
    let cpu_only: Vec<Box<dyn BackendFactory>> = vec![];
    let reg = BackendRegistry::discover_with(&cpu_only, None);
    for kind in [BackendKind::Cuda, BackendKind::Wgpu, BackendKind::Metal, BackendKind::Hip] {
        let e = reg.entries.iter().find(|e| e.kind == kind).expect("every kind is a line");
        match &e.status {
            Status::Unavailable(reason) => {
                assert!(!reason.text().is_empty(), "{kind:?} reason must be non-empty")
            }
            Status::Ready => panic!("{kind:?} cannot be Ready with no factory"),
        }
    }
    for e in &reg.entries {
        assert!(!e.source.text().is_empty(), "every entry names its source");
    }
}

/// REG-9: two devices → two entries, one selected (`--device`, default 0).
#[test]
fn two_devices_are_two_entries_and_the_default_is_device_zero() {
    let cuda = MockBackendFactory::new(
        BackendKind::Cuda,
        vec![
            ready(BackendKind::Cuda, Api::CudaDriver, 0, "GPU A", "nvidia-gpu-a", 20 << 30),
            ready(BackendKind::Cuda, Api::CudaDriver, 1, "GPU B", "nvidia-gpu-b", 20 << 30),
        ],
    );
    let f: Vec<Box<dyn BackendFactory>> = vec![Box::new(cuda)];
    let reg = BackendRegistry::discover_with(&f, None);
    let cuda_entries: Vec<_> = reg.entries.iter().filter(|e| e.kind == BackendKind::Cuda).collect();
    assert_eq!(cuda_entries.len(), 2);
    let sel = reg.select_default();
    assert_eq!(sel.kind, BackendKind::Cuda);
    assert_eq!(sel.device_index, Some(0));
    assert_eq!(reg.distinct_devices(), 2);
}

/// REG-4: one vendor's failure never changes another's entry (Ollama #11849).
#[test]
fn one_vendors_failure_does_not_touch_anothers_entry() {
    let hip = MockBackendFactory::new(
        BackendKind::Hip,
        vec![BackendEntry {
            status: Status::Unavailable(Reason::NoBackend { vendor: "AMD".to_string() }),
            vendor: "AMD".to_string(),
            vendor_id: Some(0x1002),
            ..ready(BackendKind::Hip, Api::Hip, 0, "Radeon iGPU", "amd-igpu", 0)
        }],
    );
    let cuda = MockBackendFactory::new(
        BackendKind::Cuda,
        vec![ready(BackendKind::Cuda, Api::CudaDriver, 0, "GPU A", "nvidia-gpu-a", 20 << 30)],
    );
    let f: Vec<Box<dyn BackendFactory>> = vec![Box::new(hip), Box::new(cuda)];
    let reg = BackendRegistry::discover_with(&f, None);
    let hip_e = reg.entries.iter().find(|e| e.kind == BackendKind::Hip).expect("hip line");
    assert!(
        matches!(hip_e.status, Status::Unavailable(Reason::NoBackend { .. })),
        "AMD with no backend is LISTED, not ignored"
    );
    let sel = reg.select_default();
    assert_eq!(sel.kind, BackendKind::Cuda, "the working NVIDIA card is still selected");
}

/// Design quorum lane 2: one physical GPU reachable through two APIs (cuda
/// driver and wgpu-Vulkan/Metal) is two entries sharing a `device_uid` and ONE
/// device for selection — never double-counted.
#[test]
fn the_same_device_through_two_apis_shares_a_uid_and_counts_once() {
    let cuda = MockBackendFactory::new(
        BackendKind::Cuda,
        vec![ready(BackendKind::Cuda, Api::CudaDriver, 0, "GPU A", "nvidia-gpu-a", 20 << 30)],
    );
    let wgpu = MockBackendFactory::new(
        BackendKind::Wgpu,
        vec![BackendEntry {
            transport: Some("vulkan".to_string()),
            ..ready(BackendKind::Wgpu, Api::Wgpu, 0, "GPU A", "nvidia-gpu-a", 20 << 30)
        }],
    );
    let f: Vec<Box<dyn BackendFactory>> = vec![Box::new(cuda), Box::new(wgpu)];
    let reg = BackendRegistry::discover_with(&f, None);
    assert_eq!(
        reg.entries
            .iter()
            .filter(|e| e.status == Status::Ready && e.kind != BackendKind::Cpu)
            .count(),
        2
    );
    assert_eq!(reg.distinct_devices(), 1, "two entries, one device");
    assert_eq!(
        reg.select_default().kind,
        BackendKind::Cuda,
        "REG-8: first Ready non-cpu entry wins; the wgpu twin is the same device"
    );
}

/// REG-7 / FX-7: a reserve that exceeds free memory is Unavailable(ReserveExceedsFree)
/// with both numbers, and the selection falls to cpu WITH the reason printed —
/// never a silent CPU run.
#[test]
fn a_reserve_exceeding_free_memory_is_a_named_refusal_never_a_silent_cpu_run() {
    let cuda = MockBackendFactory::new(
        BackendKind::Cuda,
        vec![ready(BackendKind::Cuda, Api::CudaDriver, 0, "GPU A", "nvidia-gpu-a", 1 << 30)],
    );
    let f: Vec<Box<dyn BackendFactory>> = vec![Box::new(cuda)];
    let reg = BackendRegistry::discover_with(&f, Some(999 << 30));
    let e = reg.entries.iter().find(|e| e.kind == BackendKind::Cuda).expect("cuda line");
    match &e.status {
        Status::Unavailable(Reason::ReserveExceedsFree { reserve_bytes, free_bytes }) => {
            assert_eq!(*reserve_bytes, 999 << 30);
            assert_eq!(*free_bytes, 1 << 30);
        }
        other => panic!("expected ReserveExceedsFree, got {other:?}"),
    }
    let sel = reg.select_default();
    assert_eq!(sel.kind, BackendKind::Cpu);
    assert!(sel.reason.contains("reserve"), "the selection names why: {}", sel.reason);
    let block = reg.render_block("test");
    assert!(block.contains("ReserveExceedsFree"), "{block}");
    assert!(block.contains("reserve="), "REG-7: the reserve is printed with its basis: {block}");
}

/// REG-12: nothing is persisted — a fixture round-trips through JSON, and a
/// registry built from a fixture says so in `source`.
#[test]
fn fixtures_round_trip_through_json_and_name_their_source() {
    let cuda = MockBackendFactory::new(
        BackendKind::Cuda,
        vec![ready(BackendKind::Cuda, Api::CudaDriver, 0, "GPU A", "nvidia-gpu-a", 20 << 30)],
    );
    let f: Vec<Box<dyn BackendFactory>> = vec![Box::new(cuda)];
    let reg = BackendRegistry::discover_with(&f, None);
    let json = reg.to_json().expect("serialize");
    let back = BackendRegistry::from_fixture_json(&json, "tests/fixtures/registry/roundtrip.json")
        .expect("fixture parses");
    assert_eq!(back.entries, reg.entries);
    assert!(
        back.source.starts_with("fixture("),
        "a fixture-built registry is loud about it: {}",
        back.source
    );
    assert_eq!(reg.source, "machine");
    assert!(BackendRegistry::from_fixture_json("{not json", "x").is_err());
}

/// The JSON shape the `apr devices --json` schema contract pins.
#[test]
fn json_carries_the_schema_top_level_keys() {
    let reg = BackendRegistry::discover();
    let v: serde_json::Value =
        serde_json::from_str(&reg.to_json().expect("serialize")).expect("valid json");
    for key in [
        "schema",
        "discovered_at_unix",
        "source",
        "reserve_bytes",
        "reserve_basis",
        "entries",
        "selected",
    ] {
        assert!(v.get(key).is_some(), "missing top-level key {key}: {v}");
    }
    assert_eq!(v["schema"], "apr-devices-v1");
    assert!(
        v["entries"].as_array().map(|a| a.len() >= 5).unwrap_or(false),
        "at least the five kind lines"
    );
}

/// FX-7 through two APIs (found by the CLI catalogue on the one-cuda fixture):
/// the cuda-driver entry was refused for memory, but its wgpu twin reports no
/// free memory and stayed Ready — so the selection slid onto the very card that
/// was just refused. A refusal propagates to every entry sharing the
/// `device_uid`, with the figure the sibling measured.
#[test]
fn a_reserve_refusal_propagates_to_the_devices_other_api_entries() {
    let cuda = MockBackendFactory::new(
        BackendKind::Cuda,
        vec![ready(BackendKind::Cuda, Api::CudaDriver, 0, "GPU A", "nvidia-gpu-a", 1 << 30)],
    );
    let wgpu = MockBackendFactory::new(
        BackendKind::Wgpu,
        vec![BackendEntry {
            mem_free: None,
            mem_total: None,
            transport: Some("vulkan".to_string()),
            ..ready(BackendKind::Wgpu, Api::Wgpu, 0, "GPU A", "nvidia-gpu-a", 0)
        }],
    );
    let f: Vec<Box<dyn BackendFactory>> = vec![Box::new(cuda), Box::new(wgpu)];
    let reg = BackendRegistry::discover_with(&f, Some(999 << 30));
    let wgpu_e = reg.entries.iter().find(|e| e.kind == BackendKind::Wgpu).expect("wgpu line");
    assert!(
        matches!(wgpu_e.status, Status::Unavailable(Reason::ReserveExceedsFree { free_bytes, .. }) if free_bytes == 1 << 30),
        "the twin carries the sibling's measured free memory: {:?}",
        wgpu_e.status
    );
    assert_eq!(
        reg.select_default().kind,
        BackendKind::Cpu,
        "the selection must not slide onto the refused card"
    );
}

/// Found on intel (two AMD W5700X, both "AMD Unknown (RADV NAVI10)" through
/// wgpu): two DIFFERENT cards with one name collapsed into one device_uid and
/// REG-9 counted one device. Same-named entries within one API are
/// disambiguated by ordinal; distinct cards stay distinct.
#[test]
fn two_different_cards_with_one_name_stay_two_devices() {
    let wgpu = MockBackendFactory::new(
        BackendKind::Wgpu,
        vec![
            BackendEntry {
                transport: Some("vulkan".to_string()),
                ..ready(
                    BackendKind::Wgpu,
                    Api::Wgpu,
                    0,
                    "AMD Unknown (RADV NAVI10)",
                    "amd:amd-unknown-radv-navi10",
                    8 << 30,
                )
            },
            BackendEntry {
                transport: Some("vulkan".to_string()),
                ..ready(
                    BackendKind::Wgpu,
                    Api::Wgpu,
                    1,
                    "AMD Unknown (RADV NAVI10)",
                    "amd:amd-unknown-radv-navi10",
                    8 << 30,
                )
            },
        ],
    );
    let f: Vec<Box<dyn BackendFactory>> = vec![Box::new(wgpu)];
    let reg = BackendRegistry::discover_with(&f, None);
    assert_eq!(reg.distinct_devices(), 2, "two cards, two devices");
    let uids: Vec<_> = reg
        .entries
        .iter()
        .filter(|e| e.kind == BackendKind::Wgpu)
        .map(|e| e.device_uid.clone())
        .collect();
    assert_ne!(uids[0], uids[1], "{uids:?}");
    assert_eq!(reg.select_default().device_index, Some(0));
}
