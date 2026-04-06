    use super::*;
    use crate::binding::parse_binding_str;
    use crate::schema::parse_contract_str;

    #[test]
    fn wired_test_for_implemented_binding() {
        let contract_yaml = r#"
metadata:
  version: "1.0.0"
  description: "Softmax"
  references: ["Bridle 1990"]
equations:
  softmax:
    formula: "f(x) = exp(x_i)/sum"
proof_obligations:
  - type: invariant
    property: "Output sums to 1"
    tolerance: 1.0e-6
    applies_to: all
falsification_tests: []
"#;
        let binding_yaml = r#"
version: "1.0.0"
target_crate: aprender
bindings:
  - contract: softmax-kernel-v1.yaml
    equation: softmax
    module_path: "aprender::nn::functional::softmax"
    function: softmax
    status: implemented
"#;
        let contract = parse_contract_str(contract_yaml).unwrap();
        let binding = parse_binding_str(binding_yaml).unwrap();
        let code = generate_wired_probar_tests(&contract, "softmax-kernel-v1.yaml", &binding);

        assert!(code.contains("CONTRACT: softmax-kernel-v1.yaml"));
        assert!(code.contains("use proptest::prelude::*"));
        assert!(code.contains("use aprender::nn::functional::softmax"));
        assert!(code.contains("proptest!"));
        assert!(code.contains("prop_output_sums_to_1"));
        assert!(code.contains("softmax(&x, -1)"));
    }

    #[test]
    fn wired_test_skips_simd_obligations() {
        let contract_yaml = r#"
metadata:
  version: "1.0.0"
  description: "Test"
  references: ["Paper"]
equations:
  f:
    formula: "f(x) = x"
proof_obligations:
  - type: equivalence
    property: "SIMD matches scalar"
    tolerance: 8.0
    applies_to: simd
falsification_tests: []
"#;
        let binding_yaml = r#"
version: "1.0.0"
target_crate: test
bindings:
  - contract: test.yaml
    equation: f
    module_path: "test::f"
    function: f
    status: implemented
"#;
        let contract = parse_contract_str(contract_yaml).unwrap();
        let binding = parse_binding_str(binding_yaml).unwrap();
        let code = generate_wired_probar_tests(&contract, "test.yaml", &binding);
        assert!(code.contains("#[ignore"));
        assert!(code.contains("SIMD equivalence"));
    }

    #[test]
    fn wired_test_for_no_bindings() {
        let contract_yaml = r#"
metadata:
  version: "1.0.0"
  description: "Test"
  references: ["Paper"]
equations:
  f:
    formula: "f(x) = x"
falsification_tests: []
"#;
        let binding_yaml = r#"
version: "1.0.0"
target_crate: test
bindings: []
"#;
        let contract = parse_contract_str(contract_yaml).unwrap();
        let binding = parse_binding_str(binding_yaml).unwrap();
        let code = generate_wired_probar_tests(&contract, "test.yaml", &binding);
        assert!(code.contains("No wired tests"));
    }

    #[test]
    fn wired_test_includes_contract_hash() {
        let contract_yaml = r#"
metadata:
  version: "1.0.0"
  description: "Test"
  references: ["Paper"]
equations:
  f:
    formula: "f(x) = x"
proof_obligations:
  - type: invariant
    property: "output finite"
falsification_tests: []
"#;
        let binding_yaml = r#"
version: "1.0.0"
target_crate: test
bindings:
  - contract: test.yaml
    equation: f
    module_path: "test::f"
    function: f
    status: implemented
"#;
        let contract = parse_contract_str(contract_yaml).unwrap();
        let binding = parse_binding_str(binding_yaml).unwrap();
        let code = generate_wired_probar_tests(&contract, "test.yaml", &binding);
        assert!(code.contains("HASH: sha256:"));
    }

    #[test]
    fn simple_hash_deterministic() {
        let h1 = simple_hash("softmax-kernel-v1.yaml");
        let h2 = simple_hash("softmax-kernel-v1.yaml");
        assert_eq!(h1, h2);
        let h3 = simple_hash("rmsnorm-kernel-v1.yaml");
        assert_ne!(h1, h3);
    }

    #[test]
    fn wired_bound_obligation() {
        let contract_yaml = r#"
metadata:
  version: "1.0.0"
  description: "Bound test"
  references: ["Paper"]
equations:
  f:
    formula: "f(x) = x"
proof_obligations:
  - type: bound
    property: "Output bounded in 0 1"
    applies_to: all
falsification_tests: []
"#;
        let binding_yaml = r#"
version: "1.0.0"
target_crate: test
bindings:
  - contract: test.yaml
    equation: f
    module_path: "test::f"
    function: myfn
    status: implemented
"#;
        let contract = parse_contract_str(contract_yaml).unwrap();
        let binding = parse_binding_str(binding_yaml).unwrap();
        let code = generate_wired_probar_tests(&contract, "test.yaml", &binding);
        assert!(code.contains("Bound:"));
        assert!(code.contains("myfn(&x)"));
    }

    #[test]
    fn wired_monotonicity_obligation() {
        let contract_yaml = r#"
metadata:
  version: "1.0.0"
  description: "Mono test"
  references: ["Paper"]
equations:
  f:
    formula: "f(x) = x"
proof_obligations:
  - type: monotonicity
    property: "Order preservation"
    applies_to: all
falsification_tests: []
"#;
        let binding_yaml = r#"
version: "1.0.0"
target_crate: test
bindings:
  - contract: test.yaml
    equation: f
    module_path: "test::f"
    function: myfn
    status: implemented
"#;
        let contract = parse_contract_str(contract_yaml).unwrap();
        let binding = parse_binding_str(binding_yaml).unwrap();
        let code = generate_wired_probar_tests(&contract, "test.yaml", &binding);
        assert!(code.contains("Monotonicity:"));
        assert!(code.contains("monotonicity violated"));
    }

    #[test]
    fn wired_idempotency_obligation() {
        let contract_yaml = r#"
metadata:
  version: "1.0.0"
  description: "Idemp test"
  references: ["Paper"]
equations:
  f:
    formula: "f(x) = x"
proof_obligations:
  - type: idempotency
    property: "Double application stable"
    tolerance: 1.0e-5
    applies_to: all
falsification_tests: []
"#;
        let binding_yaml = r#"
version: "1.0.0"
target_crate: test
bindings:
  - contract: test.yaml
    equation: f
    module_path: "test::f"
    function: softmax
    status: implemented
"#;
        let contract = parse_contract_str(contract_yaml).unwrap();
        let binding = parse_binding_str(binding_yaml).unwrap();
        let code = generate_wired_probar_tests(&contract, "test.yaml", &binding);
        assert!(code.contains("Idempotency:"));
        assert!(code.contains("softmax(&x, -1)"));
        assert!(code.contains("softmax(&once, -1)"));
    }

    #[test]
    fn wired_equivalence_non_simd() {
        let contract_yaml = r#"
metadata:
  version: "1.0.0"
  description: "Equiv test"
  references: ["Paper"]
equations:
  f:
    formula: "f(x) = x"
proof_obligations:
  - type: equivalence
    property: "Implementations match"
    tolerance: 1.0e-4
    applies_to: all
falsification_tests: []
"#;
        let binding_yaml = r#"
version: "1.0.0"
target_crate: test
bindings:
  - contract: test.yaml
    equation: f
    module_path: "test::f"
    function: f
    status: implemented
"#;
        let contract = parse_contract_str(contract_yaml).unwrap();
        let binding = parse_binding_str(binding_yaml).unwrap();
        let code = generate_wired_probar_tests(&contract, "test.yaml", &binding);
        assert!(code.contains("Equivalence:"));
        assert!(code.contains("ULP tolerance"));
    }

    #[test]
    fn wired_generic_linearity() {
        let contract_yaml = r#"
metadata:
  version: "1.0.0"
  description: "Lin test"
  references: ["Paper"]
equations:
  f:
    formula: "f(x) = x"
proof_obligations:
  - type: linearity
    property: "Linear scaling"
    applies_to: all
falsification_tests: []
"#;
        let binding_yaml = r#"
version: "1.0.0"
target_crate: test
bindings:
  - contract: test.yaml
    equation: f
    module_path: "test::f"
    function: f
    status: implemented
"#;
        let contract = parse_contract_str(contract_yaml).unwrap();
        let binding = parse_binding_str(binding_yaml).unwrap();
        let code = generate_wired_probar_tests(&contract, "test.yaml", &binding);
        assert!(code.contains("linearity:"));
        assert!(code.contains("TODO: wire up"));
    }

    #[test]
    fn wired_generic_conservation() {
        let contract_yaml = r#"
metadata:
  version: "1.0.0"
  description: "Cons test"
  references: ["Paper"]
equations:
  f:
    formula: "f(x) = x"
proof_obligations:
  - type: conservation
    property: "Mass conservation"
    applies_to: all
falsification_tests: []
"#;
        let binding_yaml = r#"
version: "1.0.0"
target_crate: test
bindings:
  - contract: test.yaml
    equation: f
    module_path: "test::f"
    function: f
    status: implemented
"#;
        let contract = parse_contract_str(contract_yaml).unwrap();
        let binding = parse_binding_str(binding_yaml).unwrap();
        let code = generate_wired_probar_tests(&contract, "test.yaml", &binding);
        assert!(code.contains("conservation:"));
    }

    #[test]
    fn wired_struct_method_binding() {
        let contract_yaml = r#"
metadata:
  version: "1.0.0"
  description: "Struct method"
  references: ["Paper"]
equations:
  f:
    formula: "f(x) = x"
proof_obligations:
  - type: invariant
    property: "Output finite"
    applies_to: all
falsification_tests: []
"#;
        let binding_yaml = r#"
version: "1.0.0"
target_crate: test
bindings:
  - contract: test.yaml
    equation: f
    module_path: "test::MyStruct"
    function: "MyStruct::forward"
    status: implemented
"#;
        let contract = parse_contract_str(contract_yaml).unwrap();
        let binding = parse_binding_str(binding_yaml).unwrap();
        let code = generate_wired_probar_tests(&contract, "test.yaml", &binding);
        assert!(code.contains("TODO: wire up struct"));
    }

    #[test]
    fn wired_tensor_method_binding() {
        let contract_yaml = r#"
metadata:
  version: "1.0.0"
  description: "Tensor method"
  references: ["Paper"]
equations:
  f:
    formula: "f(x) = x"
proof_obligations:
  - type: invariant
    property: "Output finite"
    applies_to: all
falsification_tests: []
"#;
        let binding_yaml = r#"
version: "1.0.0"
target_crate: test
bindings:
  - contract: test.yaml
    equation: f
    module_path: "test::Tensor"
    function: "Tensor::matmul"
    status: implemented
"#;
        let contract = parse_contract_str(contract_yaml).unwrap();
        let binding = parse_binding_str(binding_yaml).unwrap();
        let code = generate_wired_probar_tests(&contract, "test.yaml", &binding);
        assert!(code.contains("x.matmul(&x)"));
    }

    #[test]
    fn wired_no_obligations_with_bindings() {
        let contract_yaml = r#"
metadata:
  version: "1.0.0"
  description: "No obls"
  references: ["Paper"]
equations:
  f:
    formula: "f(x) = x"
falsification_tests: []
"#;
        let binding_yaml = r#"
version: "1.0.0"
target_crate: test
bindings:
  - contract: test.yaml
    equation: f
    module_path: "test::f"
    function: f
    status: implemented
"#;
        let contract = parse_contract_str(contract_yaml).unwrap();
        let binding = parse_binding_str(binding_yaml).unwrap();
        let code = generate_wired_probar_tests(&contract, "test.yaml", &binding);
        assert!(code.contains("No proof obligations in this contract"));
    }
