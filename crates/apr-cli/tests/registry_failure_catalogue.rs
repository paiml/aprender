//! registry_failure_catalogue — PP-066 R-0a (#2904, PMAT-989): the hermetic
//! rows of the REG/FX case table, run against the `apr` binary with fixture
//! registries (`APR_REGISTRY_FIXTURE`) so the table is the same on every host.
//! The non-hermetic fixtures (FX-2/4/5/6/8/9: real drivers, real cards) are
//! host-dogfood rows recorded as receipts, not CI claims (design quorum
//! 2026-09-06). Every hermetic row has its must-RED twin: a defective fixture
//! under `tests/fixtures/registry/defective/` that the same check refuses.
use std::path::PathBuf;
use std::process::{Command, Output};

fn fixture(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/registry")
        .join(name)
}

fn apr_devices(envs: &[(&str, &str)], json: bool) -> Output {
    let mut cmd = Command::new(env!("CARGO_BIN_EXE_apr"));
    cmd.arg("devices");
    if json {
        cmd.arg("--json");
    }
    cmd.env_remove("APR_RESERVE_BYTES")
        .env_remove("APR_REGISTRY_FIXTURE");
    for (k, v) in envs {
        cmd.env(k, v);
    }
    cmd.output().expect("spawn apr")
}

fn stdout(o: &Output) -> String {
    String::from_utf8_lossy(&o.stdout).into_owned()
}

fn kind_lines(text: &str) -> Vec<&str> {
    ["cpu", "cuda", "wgpu", "metal", "hip"]
        .into_iter()
        .filter(|k| {
            text.lines()
                .any(|l| l.starts_with(&format!("backend: {k:<6}")))
        })
        .collect()
}

/// FX-11 / REG-11: on a CPU-only machine every kind is still a line, and the
/// selection is cpu with its reason. Exit 0 — discovery is never a failure (REG-1).
#[test]
fn fx11_cpu_only_prints_all_five_kinds_and_selects_cpu_with_a_reason() {
    let out = apr_devices(
        &[(
            "APR_REGISTRY_FIXTURE",
            fixture("cpu-only.json").to_str().expect("utf8"),
        )],
        false,
    );
    assert!(
        out.status.success(),
        "exit 0 on a CPU-only box; got {:?}\n{}",
        out.status,
        String::from_utf8_lossy(&out.stderr)
    );
    let text = stdout(&out);
    assert_eq!(
        kind_lines(&text).len(),
        5,
        "all five kinds must print:\n{text}"
    );
    assert!(
        text.contains("reason=DriverNotFound(libcuda.so.1)"),
        "{text}"
    );
    assert!(
        text.lines()
            .any(|l| l.starts_with("selected: cpu") && l.contains("no ready gpu")),
        "{text}"
    );
    assert!(
        text.contains("override: APR_REGISTRY_FIXTURE="),
        "REG-8: an override is loud:\n{text}"
    );
    assert!(
        text.contains("source=fixture("),
        "a fixture registry says so:\n{text}"
    );
}

/// The must-RED twin of FX-11: a defective fixture that drops the `metal` line
/// makes the five-kinds check FAIL — the check discriminates.
#[test]
fn fx11_twin_a_fixture_missing_a_kind_line_is_caught() {
    let out = apr_devices(
        &[(
            "APR_REGISTRY_FIXTURE",
            fixture("defective/missing-metal-line.json")
                .to_str()
                .expect("utf8"),
        )],
        false,
    );
    assert!(out.status.success());
    let text = stdout(&out);
    assert_eq!(
        kind_lines(&text).len(),
        4,
        "the defective fixture must be caught by the five-kinds check:\n{text}"
    );
}

/// FX-7 / REG-7: a reserve that exceeds free memory is `ReserveExceedsFree`,
/// printed with both numbers, the override is printed, and the selection
/// falls to cpu WITH the reason — never a silent CPU run.
#[test]
fn fx7_reserve_exceeding_free_memory_is_a_named_refusal() {
    let out = apr_devices(
        &[
            (
                "APR_REGISTRY_FIXTURE",
                fixture("one-cuda.json").to_str().expect("utf8"),
            ),
            ("APR_RESERVE_BYTES", "999G"),
        ],
        false,
    );
    assert!(out.status.success());
    let text = stdout(&out);
    assert!(
        text.contains("override: APR_RESERVE_BYTES=1072668082176"),
        "{text}"
    );
    assert!(
        text.contains("ReserveExceedsFree{reserve=1072668082176, free=21474836480}"),
        "{text}"
    );
    assert!(
        text.lines()
            .any(|l| l.starts_with("selected: cpu") && l.contains("reserve")),
        "{text}"
    );
}

/// FX-7 twin: the same fixture with the default reserve selects the cuda device.
#[test]
fn fx7_twin_the_default_reserve_fits_and_cuda_is_selected() {
    let out = apr_devices(
        &[(
            "APR_REGISTRY_FIXTURE",
            fixture("one-cuda.json").to_str().expect("utf8"),
        )],
        false,
    );
    let text = stdout(&out);
    assert!(
        text.lines()
            .any(|l| l.starts_with("selected: cuda device[0]")),
        "{text}"
    );
}

/// REG-4 / REG-9 / lane 2: an AMD device with no backend is LISTED as
/// NoBackend(AMD) and does not change the NVIDIA entry; a device reachable
/// through two APIs is two entries and one selection.
#[test]
fn reg4_a_vendor_without_a_backend_is_listed_not_ignored() {
    let out = apr_devices(
        &[(
            "APR_REGISTRY_FIXTURE",
            fixture("two-vendors.json").to_str().expect("utf8"),
        )],
        false,
    );
    let text = stdout(&out);
    assert!(
        text.lines()
            .any(|l| l.starts_with("backend: hip") && l.contains("NoBackend(AMD)")),
        "{text}"
    );
    assert!(
        text.lines()
            .any(|l| l.starts_with("selected: cuda device[0]")),
        "{text}"
    );
    let out = apr_devices(
        &[(
            "APR_REGISTRY_FIXTURE",
            fixture("one-cuda.json").to_str().expect("utf8"),
        )],
        true,
    );
    let v: serde_json::Value = serde_json::from_str(&stdout(&out)).expect("json");
    let ready_gpu: Vec<_> = v["entries"]
        .as_array()
        .expect("entries")
        .iter()
        .filter(|e| e["status"]["state"] == "ready" && e["kind"] != "cpu")
        .collect();
    assert_eq!(
        ready_gpu.len(),
        2,
        "cuda-driver and wgpu twins are both listed"
    );
    assert_eq!(
        ready_gpu[0]["device_uid"], ready_gpu[1]["device_uid"],
        "one physical device"
    );
    assert!(
        v["selected"]["reason"]
            .as_str()
            .expect("reason")
            .contains("1 physical device"),
        "{}",
        v["selected"]
    );
}

/// REG-12: nothing is cached — two different fixtures give two different
/// outputs in the same process environment, and the source names the file.
#[test]
fn reg12_nothing_is_persisted_between_discoveries() {
    let a = stdout(&apr_devices(
        &[(
            "APR_REGISTRY_FIXTURE",
            fixture("cpu-only.json").to_str().expect("utf8"),
        )],
        false,
    ));
    let b = stdout(&apr_devices(
        &[(
            "APR_REGISTRY_FIXTURE",
            fixture("one-cuda.json").to_str().expect("utf8"),
        )],
        false,
    ));
    assert_ne!(a, b);
    assert!(a.contains("cpu-only.json") && b.contains("one-cuda.json"));
}

/// `--json` validates against `contracts/schemas/apr-devices-v1.schema.json`
/// for every fixture AND for this machine (the schema is the contract
/// `contracts/apr-devices-schema-v1.yaml` binds).
#[test]
fn json_output_validates_against_the_schema_on_fixtures_and_on_this_machine() {
    let schema_text = std::fs::read_to_string(
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../../contracts/schemas/apr-devices-v1.schema.json"),
    )
    .expect("schema file");
    let schema: serde_json::Value = serde_json::from_str(&schema_text).expect("schema json");
    let validator = jsonschema::validator_for(&schema).expect("schema compiles");
    let mut runs: Vec<(String, Output)> = ["cpu-only.json", "one-cuda.json", "two-vendors.json"]
        .into_iter()
        .map(|f| {
            (
                f.to_string(),
                apr_devices(
                    &[("APR_REGISTRY_FIXTURE", fixture(f).to_str().expect("utf8"))],
                    true,
                ),
            )
        })
        .collect();
    runs.push(("this machine".to_string(), apr_devices(&[], true)));
    for (name, out) in runs {
        assert!(out.status.success(), "{name}: exit {:?}", out.status);
        let v: serde_json::Value =
            serde_json::from_str(&stdout(&out)).unwrap_or_else(|e| panic!("{name}: {e}"));
        let errors: Vec<String> = validator
            .iter_errors(&v)
            .map(|e| format!("{} at {}", e, e.instance_path))
            .collect();
        assert!(errors.is_empty(), "{name}: schema violations: {errors:?}");
        assert!(
            v["entries"]
                .as_array()
                .map(|a| a.len() >= 5)
                .unwrap_or(false),
            "{name}: five kind lines"
        );
    }
}

/// Schema twin: the defective fixture (four entries) is REFUSED by the schema
/// (`minItems: 5`) — the schema discriminates, it is not decoration.
#[test]
fn schema_twin_the_defective_fixture_is_refused_by_the_schema() {
    let schema_text = std::fs::read_to_string(
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../../contracts/schemas/apr-devices-v1.schema.json"),
    )
    .expect("schema file");
    let schema: serde_json::Value = serde_json::from_str(&schema_text).expect("schema json");
    let validator = jsonschema::validator_for(&schema).expect("schema compiles");
    let out = apr_devices(
        &[(
            "APR_REGISTRY_FIXTURE",
            fixture("defective/missing-metal-line.json")
                .to_str()
                .expect("utf8"),
        )],
        true,
    );
    let v: serde_json::Value = serde_json::from_str(&stdout(&out)).expect("json");
    assert!(
        !validator.is_valid(&v),
        "a four-entry document must not validate"
    );
}

/// A malformed override is exit code 4 (InvalidInput) with the name of the
/// variable, never a silent default.
#[test]
fn a_malformed_reserve_override_is_refused_by_name() {
    let out = apr_devices(&[("APR_RESERVE_BYTES", "lots")], false);
    assert_eq!(out.status.code(), Some(4), "{:?}", out.status);
    assert!(String::from_utf8_lossy(&out.stderr).contains("APR_RESERVE_BYTES"));
}
