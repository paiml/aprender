//! build.rs reads `contracts/entrenar/binding.yaml` through serde-saphyr (#5326).
//! If the parse silently failed, build.rs would emit `CONTRACT_BINDING_SOURCE=none`
//! and zero totals; these tests fail on that, so the YAML reader swap cannot go dark.

#[test]
fn build_rs_parsed_binding_yaml() {
    assert_eq!(
        option_env!("CONTRACT_BINDING_SOURCE"),
        Some("binding.yaml"),
        "build.rs did not parse binding.yaml (source=none means the YAML reader failed)"
    );
}

#[test]
fn build_rs_binding_totals_are_consistent() {
    let num = |v: Option<&str>| -> u32 {
        v.expect("build.rs set the CONTRACT_* total").parse().expect("numeric total")
    };
    let total = num(option_env!("CONTRACT_TOTAL"));
    let implemented = num(option_env!("CONTRACT_IMPLEMENTED"));
    let partial = num(option_env!("CONTRACT_PARTIAL"));
    let gaps = num(option_env!("CONTRACT_GAPS"));
    assert!(total > 0, "binding.yaml parsed to zero bindings");
    assert_eq!(total, implemented + partial + gaps);
    assert_eq!(gaps, 0, "AllImplemented policy: no not_implemented bindings");
}
