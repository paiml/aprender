    /// Verify wired test generation handles empty property names gracefully
    #[test]
    fn wired_empty_property_name() {
        let contract_yaml = r#"
metadata:
  version: "1.0.0"
  description: "Empty prop"
  references: ["Paper"]
equations:
  f:
    formula: "f(x) = x"
proof_obligations:
  - type: invariant
    property: ""
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
        assert!(code.contains("prop_obligation_0"));
    }

    /// Verify code generation for mixed obligation types (invariant, bound, equivalence, monotonicity)
    #[test]
    fn wired_multiple_obligations_mixed() {
        let contract_yaml = r#"
metadata:
  version: "1.0.0"
  description: "Mixed"
  references: ["Paper"]
equations:
  f:
    formula: "f(x) = x"
proof_obligations:
  - type: invariant
    property: "Output finite"
    tolerance: 1.0e-6
    applies_to: all
  - type: bound
    property: "Output bounded"
    applies_to: all
  - type: equivalence
    property: "SIMD matches"
    tolerance: 8.0
    applies_to: simd
  - type: monotonicity
    property: "Order preserved"
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
        assert!(code.contains("Invariant:"));
        assert!(code.contains("Bound:"));
        assert!(code.contains("SIMD equivalence"));
        assert!(code.contains("Monotonicity:"));
    }

    /// Verify missing binding generates an ignored test with "no binding available"
    #[test]
    fn wired_no_binding_generates_ignored() {
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
  - contract: OTHER.yaml
    equation: f
    module_path: "test::f"
    function: f
    status: implemented
"#;
        let contract = parse_contract_str(contract_yaml).unwrap();
        let binding = parse_binding_str(binding_yaml).unwrap();
        let code = generate_wired_probar_tests(&contract, "test.yaml", &binding);
        assert!(code.contains("no binding available"));
    }

    /// Verify idempotency code generation for a non-softmax free function
    #[test]
    fn wired_idempotency_non_softmax_free_fn() {
        let contract_yaml = r#"
metadata:
  version: "1.0.0"
  description: "Idemp non-softmax"
  references: ["Paper"]
equations:
  f:
    formula: "f(x) = x"
proof_obligations:
  - type: idempotency
    property: "Idemp check"
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
    function: relu
    status: implemented
"#;
        let contract = parse_contract_str(contract_yaml).unwrap();
        let binding = parse_binding_str(binding_yaml).unwrap();
        let code = generate_wired_probar_tests(&contract, "test.yaml", &binding);
        assert!(code.contains("relu(&x)"));
        assert!(code.contains("relu(&once)"));
    }

    /// Verify idempotency code generation for a struct method binding
    #[test]
    fn wired_idempotency_struct_method() {
        let contract_yaml = r#"
metadata:
  version: "1.0.0"
  description: "Idemp struct"
  references: ["Paper"]
equations:
  f:
    formula: "f(x) = x"
proof_obligations:
  - type: idempotency
    property: "Idemp struct check"
    applies_to: all
falsification_tests: []
"#;
        let binding_yaml = r#"
version: "1.0.0"
target_crate: test
bindings:
  - contract: test.yaml
    equation: f
    module_path: "test::S"
    function: "S::forward"
    status: implemented
"#;
        let contract = parse_contract_str(contract_yaml).unwrap();
        let binding = parse_binding_str(binding_yaml).unwrap();
        let code = generate_wired_probar_tests(&contract, "test.yaml", &binding);
        assert!(code.contains("once = x.clone()"));
        assert!(code.contains("twice = once.clone()"));
    }

    /// Verify bound obligation generates a TODO stub for struct method binding
    #[test]
    fn wired_bound_struct_method() {
        let contract_yaml = r#"
metadata:
  version: "1.0.0"
  description: "Bound struct"
  references: ["Paper"]
equations:
  f:
    formula: "f(x) = x"
proof_obligations:
  - type: bound
    property: "Bounded output"
    applies_to: all
falsification_tests: []
"#;
        let binding_yaml = r#"
version: "1.0.0"
target_crate: test
bindings:
  - contract: test.yaml
    equation: f
    module_path: "test::S"
    function: "S::forward"
    status: implemented
"#;
        let contract = parse_contract_str(contract_yaml).unwrap();
        let binding = parse_binding_str(binding_yaml).unwrap();
        let code = generate_wired_probar_tests(&contract, "test.yaml", &binding);
        assert!(code.contains("Bound:"));
        assert!(code.contains("Deferred (PMAT-753): wire up struct"));
    }

    /// Verify monotonicity code generation for a tensor method binding
    #[test]
    fn wired_monotonicity_tensor_method() {
        let contract_yaml = r#"
metadata:
  version: "1.0.0"
  description: "Mono tensor"
  references: ["Paper"]
equations:
  f:
    formula: "f(x) = x"
proof_obligations:
  - type: monotonicity
    property: "Order preserved"
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
        assert!(code.contains("Monotonicity:"));
        assert!(code.contains("x.matmul(&x)"));
    }

    /// Verify invariant code generation includes the formal specification string
    #[test]
    fn wired_invariant_with_formal() {
        let contract_yaml = r#"
metadata:
  version: "1.0.0"
  description: "Formal"
  references: ["Paper"]
equations:
  f:
    formula: "f(x) = x"
proof_obligations:
  - type: invariant
    property: "Sum is one"
    formal: "|sum - 1| < eps"
    tolerance: 1.0e-6
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
        assert!(code.contains("Formal: |sum - 1| < eps"));
    }

    /// Verify symmetry obligation generates code with "symmetry:" marker
    #[test]
    fn wired_symmetry_obligation() {
        let contract_yaml = r#"
metadata:
  version: "1.0.0"
  description: "Sym test"
  references: ["Paper"]
equations:
  f:
    formula: "f(x) = x"
proof_obligations:
  - type: symmetry
    property: "Permutation invariant"
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
        assert!(code.contains("symmetry:"));
    }

    /// Verify associativity obligation generates code with "associativity:" marker
    #[test]
    fn wired_associativity_obligation() {
        let contract_yaml = r#"
metadata:
  version: "1.0.0"
  description: "Assoc test"
  references: ["Paper"]
equations:
  f:
    formula: "f(x) = x"
proof_obligations:
  - type: associativity
    property: "Grouping invariant"
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
        assert!(code.contains("associativity:"));
    }

    /// Verify invariant code generation for tensor method uses self-call syntax
    #[test]
    fn wired_invariant_tensor_method() {
        let contract_yaml = r#"
metadata:
  version: "1.0.0"
  description: "Inv tensor"
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
        assert!(code.contains("Invariant:"));
    }

    /// Verify default tolerance of 1e-6 is used when no tolerance is specified
    #[test]
    fn wired_default_tolerance() {
        let contract_yaml = r#"
metadata:
  version: "1.0.0"
  description: "Default tol"
  references: ["Paper"]
equations:
  f:
    formula: "f(x) = x"
proof_obligations:
  - type: invariant
    property: "Finite output"
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
        assert!(code.contains("1e-6"));
    }

    /// Verify `emit_imports` deduplicates identical module paths
    #[test]
    fn emit_imports_deduplicates() {
        let binding_yaml = r#"
version: "1.0.0"
target_crate: test
bindings:
  - contract: test.yaml
    equation: f
    module_path: "test::mod1"
    function: f
    status: implemented
  - contract: test.yaml
    equation: g
    module_path: "test::mod1"
    function: g
    status: implemented
"#;
        let binding = parse_binding_str(binding_yaml).unwrap();
        let refs: Vec<&KernelBinding> = binding.bindings.iter().collect();
        let mut out = String::new();
        emit_imports(&mut out, &refs);
        assert_eq!(out.matches("use test::mod1;").count(), 1);
    }
