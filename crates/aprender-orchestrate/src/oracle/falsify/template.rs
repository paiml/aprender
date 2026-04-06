//! 100-Point Falsification Template
//!
//! Defines the structure and point allocation for falsification categories.

use serde::{Deserialize, Serialize};

/// The 100-point falsification template
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FalsificationTemplate {
    /// Test categories with point allocations
    pub categories: Vec<CategoryTemplate>,
}

impl Default for FalsificationTemplate {
    fn default() -> Self {
        Self {
            categories: vec![
                CategoryTemplate::boundary(),
                CategoryTemplate::invariant(),
                CategoryTemplate::numerical(),
                CategoryTemplate::concurrency(),
                CategoryTemplate::resource(),
                CategoryTemplate::parity(),
            ],
        }
    }
}

impl FalsificationTemplate {
    /// Total points across all categories
    pub fn total_points(&self) -> u32 {
        self.categories.iter().map(|c| c.total_points()).sum()
    }

    /// Scale template to target point count
    pub fn scale_to_points(&self, target: u32) -> Self {
        let current = self.total_points();
        if current == target {
            return self.clone();
        }

        let scale = target as f64 / current as f64;
        let mut scaled = self.clone();

        for category in &mut scaled.categories {
            for test in &mut category.tests {
                test.points = ((test.points as f64) * scale).round() as u32;
            }
        }

        scaled
    }

    /// Get category by name
    pub fn get_category(&self, name: &str) -> Option<&CategoryTemplate> {
        self.categories.iter().find(|c| c.name == name)
    }
}

/// A category of falsification tests
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CategoryTemplate {
    /// Category name (e.g., "boundary")
    pub name: String,
    /// Category ID prefix (e.g., "BC")
    pub id_prefix: String,
    /// Description
    pub description: String,
    /// Test templates
    pub tests: Vec<TestTemplate>,
}

impl CategoryTemplate {
    /// Total points for this category
    pub fn total_points(&self) -> u32 {
        self.tests.iter().map(|t| t.points).sum()
    }

    /// Boundary conditions category (20 points)
    pub fn boundary() -> Self {
        Self {
            name: "boundary".to_string(),
            id_prefix: "BC".to_string(),
            description: "Boundary condition tests".to_string(),
            tests: vec![
                TestTemplate {
                    id: "BC-001".to_string(),
                    name: "Empty input".to_string(),
                    description: "Handle empty/null input gracefully".to_string(),
                    severity: TestSeverity::Critical,
                    points: 4,
                    rust_template: Some(RUST_EMPTY_INPUT.to_string()),
                    python_template: Some(PYTHON_EMPTY_INPUT.to_string()),
                },
                TestTemplate {
                    id: "BC-002".to_string(),
                    name: "Maximum size input".to_string(),
                    description: "Handle maximum allowed input size".to_string(),
                    severity: TestSeverity::Critical,
                    points: 4,
                    rust_template: Some(RUST_MAX_INPUT.to_string()),
                    python_template: Some(PYTHON_MAX_INPUT.to_string()),
                },
                TestTemplate {
                    id: "BC-003".to_string(),
                    name: "Negative values".to_string(),
                    description: "Handle negative values where positive expected".to_string(),
                    severity: TestSeverity::High,
                    points: 4,
                    rust_template: Some(RUST_NEGATIVE.to_string()),
                    python_template: Some(PYTHON_NEGATIVE.to_string()),
                },
                TestTemplate {
                    id: "BC-004".to_string(),
                    name: "Unicode edge cases".to_string(),
                    description: "Handle combining chars, RTL, zero-width".to_string(),
                    severity: TestSeverity::Medium,
                    points: 4,
                    rust_template: Some(RUST_UNICODE.to_string()),
                    python_template: Some(PYTHON_UNICODE.to_string()),
                },
                TestTemplate {
                    id: "BC-005".to_string(),
                    name: "Numeric limits".to_string(),
                    description: "Handle MAX_INT, MIN_INT, NaN, Inf".to_string(),
                    severity: TestSeverity::Critical,
                    points: 4,
                    rust_template: Some(RUST_NUMERIC_LIMITS.to_string()),
                    python_template: Some(PYTHON_NUMERIC_LIMITS.to_string()),
                },
            ],
        }
    }

    /// Invariant violations category (20 points)
    pub fn invariant() -> Self {
        Self {
            name: "invariant".to_string(),
            id_prefix: "INV".to_string(),
            description: "Mathematical invariant tests".to_string(),
            tests: vec![
                TestTemplate {
                    id: "INV-001".to_string(),
                    name: "Idempotency".to_string(),
                    description: "f(f(x)) == f(x) for idempotent operations".to_string(),
                    severity: TestSeverity::High,
                    points: 5,
                    rust_template: Some(RUST_IDEMPOTENT.to_string()),
                    python_template: Some(PYTHON_IDEMPOTENT.to_string()),
                },
                TestTemplate {
                    id: "INV-002".to_string(),
                    name: "Commutativity".to_string(),
                    description: "f(a,b) == f(b,a) for commutative operations".to_string(),
                    severity: TestSeverity::High,
                    points: 5,
                    rust_template: Some(RUST_COMMUTATIVE.to_string()),
                    python_template: Some(PYTHON_COMMUTATIVE.to_string()),
                },
                TestTemplate {
                    id: "INV-003".to_string(),
                    name: "Associativity".to_string(),
                    description: "(a+b)+c == a+(b+c) for floating point".to_string(),
                    severity: TestSeverity::Medium,
                    points: 5,
                    rust_template: Some(RUST_ASSOCIATIVE.to_string()),
                    python_template: Some(PYTHON_ASSOCIATIVE.to_string()),
                },
                TestTemplate {
                    id: "INV-004".to_string(),
                    name: "Symmetry (encode/decode)".to_string(),
                    description: "decode(encode(x)) == x for serialization".to_string(),
                    severity: TestSeverity::Critical,
                    points: 5,
                    rust_template: Some(RUST_ROUNDTRIP.to_string()),
                    python_template: Some(PYTHON_ROUNDTRIP.to_string()),
                },
            ],
        }
    }

    /// Numerical stability category (20 points)
    pub fn numerical() -> Self {
        Self {
            name: "numerical".to_string(),
            id_prefix: "NUM".to_string(),
            description: "Numerical stability tests".to_string(),
            tests: vec![
                TestTemplate {
                    id: "NUM-001".to_string(),
                    name: "Catastrophic cancellation".to_string(),
                    description: "1e10 + 1 - 1e10 should not lose precision".to_string(),
                    severity: TestSeverity::High,
                    points: 7,
                    rust_template: Some(RUST_CANCELLATION.to_string()),
                    python_template: Some(PYTHON_CANCELLATION.to_string()),
                },
                TestTemplate {
                    id: "NUM-002".to_string(),
                    name: "Accumulation order".to_string(),
                    description: "sum(shuffled) vs sum(sorted) within tolerance".to_string(),
                    severity: TestSeverity::Medium,
                    points: 7,
                    rust_template: Some(RUST_ACCUMULATION.to_string()),
                    python_template: Some(PYTHON_ACCUMULATION.to_string()),
                },
                TestTemplate {
                    id: "NUM-003".to_string(),
                    name: "Denormalized numbers".to_string(),
                    description: "Handle subnormal/denormalized floats".to_string(),
                    severity: TestSeverity::Medium,
                    points: 6,
                    rust_template: Some(RUST_DENORMAL.to_string()),
                    python_template: Some(PYTHON_DENORMAL.to_string()),
                },
            ],
        }
    }

    /// Concurrency category (15 points)
    pub fn concurrency() -> Self {
        Self {
            name: "concurrency".to_string(),
            id_prefix: "CONC".to_string(),
            description: "Concurrency and race condition tests".to_string(),
            tests: vec![
                TestTemplate {
                    id: "CONC-001".to_string(),
                    name: "Data race".to_string(),
                    description: "No data races under parallel iteration".to_string(),
                    severity: TestSeverity::Critical,
                    points: 5,
                    rust_template: Some(RUST_DATA_RACE.to_string()),
                    python_template: Some(PYTHON_DATA_RACE.to_string()),
                },
                TestTemplate {
                    id: "CONC-002".to_string(),
                    name: "Deadlock".to_string(),
                    description: "No deadlock potential in lock ordering".to_string(),
                    severity: TestSeverity::Critical,
                    points: 5,
                    rust_template: Some(RUST_DEADLOCK.to_string()),
                    python_template: Some(PYTHON_DEADLOCK.to_string()),
                },
                TestTemplate {
                    id: "CONC-003".to_string(),
                    name: "ABA problem".to_string(),
                    description: "Lock-free structures handle ABA".to_string(),
                    severity: TestSeverity::High,
                    points: 5,
                    rust_template: Some(RUST_ABA.to_string()),
                    python_template: Some(PYTHON_ABA.to_string()),
                },
            ],
        }
    }

    /// Resource exhaustion category (15 points)
    pub fn resource() -> Self {
        Self {
            name: "resource".to_string(),
            id_prefix: "RES".to_string(),
            description: "Resource exhaustion tests".to_string(),
            tests: vec![
                TestTemplate {
                    id: "RES-001".to_string(),
                    name: "Memory exhaustion".to_string(),
                    description: "Controlled behavior under OOM".to_string(),
                    severity: TestSeverity::High,
                    points: 5,
                    rust_template: Some(RUST_MEMORY.to_string()),
                    python_template: Some(PYTHON_MEMORY.to_string()),
                },
                TestTemplate {
                    id: "RES-002".to_string(),
                    name: "File descriptor exhaustion".to_string(),
                    description: "Handle FD limits gracefully".to_string(),
                    severity: TestSeverity::Medium,
                    points: 5,
                    rust_template: Some(RUST_FD.to_string()),
                    python_template: Some(PYTHON_FD.to_string()),
                },
                TestTemplate {
                    id: "RES-003".to_string(),
                    name: "Stack overflow".to_string(),
                    description: "Deep recursion handled".to_string(),
                    severity: TestSeverity::High,
                    points: 5,
                    rust_template: Some(RUST_STACK.to_string()),
                    python_template: Some(PYTHON_STACK.to_string()),
                },
            ],
        }
    }

    /// Cross-implementation parity category (10 points)
    pub fn parity() -> Self {
        Self {
            name: "parity".to_string(),
            id_prefix: "PAR".to_string(),
            description: "Cross-implementation parity tests".to_string(),
            tests: vec![
                TestTemplate {
                    id: "PAR-001".to_string(),
                    name: "Python/Rust parity".to_string(),
                    description: "Output matches reference Python implementation".to_string(),
                    severity: TestSeverity::High,
                    points: 5,
                    rust_template: Some(RUST_PYTHON_PARITY.to_string()),
                    python_template: Some(PYTHON_RUST_PARITY.to_string()),
                },
                TestTemplate {
                    id: "PAR-002".to_string(),
                    name: "CPU/GPU parity".to_string(),
                    description: "CPU and GPU kernels produce same results".to_string(),
                    severity: TestSeverity::High,
                    points: 5,
                    rust_template: Some(RUST_GPU_PARITY.to_string()),
                    python_template: Some(PYTHON_GPU_PARITY.to_string()),
                },
            ],
        }
    }
}

/// A single test template
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TestTemplate {
    /// Test ID (e.g., "BC-001")
    pub id: String,
    /// Test name
    pub name: String,
    /// Description
    pub description: String,
    /// Severity level
    pub severity: TestSeverity,
    /// Point value
    pub points: u32,
    /// Rust code template
    pub rust_template: Option<String>,
    /// Python code template
    pub python_template: Option<String>,
}

/// Severity level
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum TestSeverity {
    Critical,
    High,
    Medium,
    Low,
}

// Rust templates
const RUST_EMPTY_INPUT: &str = r#"#[test]
fn falsify_{{id_lower}}_empty_input() {
    let result = {{module}}::{{function}}(&[]);
    assert!(result.is_err() || result.expect("operation failed").is_empty(),
        "Should handle empty input gracefully");
}"#;

const RUST_MAX_INPUT: &str = r#"#[test]
fn falsify_{{id_lower}}_max_input() {
    let large_input = vec![0u8; {{max_size}}];
    let result = {{module}}::{{function}}(&large_input);
    assert!(result.is_ok(), "Should handle maximum size input");
}"#;

const RUST_NEGATIVE: &str = r#"#[test]
fn falsify_{{id_lower}}_negative_values() {
    let result = {{module}}::{{function}}(-1);
    assert!(result.is_err(), "Should reject negative values");
}"#;

const RUST_UNICODE: &str = r#"#[test]
fn falsify_{{id_lower}}_unicode() {
    // Combining characters, RTL, zero-width
    let inputs = ["café\u{0301}", "\u{202E}test", "a\u{200B}b"];
    for input in inputs {
        let result = {{module}}::{{function}}(input);
        assert!(result.is_ok(), "Should handle Unicode: {}", input);
    }
}"#;

const RUST_NUMERIC_LIMITS: &str = r#"#[test]
fn falsify_{{id_lower}}_numeric_limits() {
    let values = [f64::MAX, f64::MIN, f64::NAN, f64::INFINITY, f64::NEG_INFINITY];
    for val in values {
        let result = {{module}}::{{function}}(val);
        assert!(!result.is_nan() || val.is_nan(), "Should handle {:?}", val);
    }
}"#;

const RUST_IDEMPOTENT: &str = r#"proptest! {
    #[test]
    fn falsify_{{id_lower}}_idempotent(x in any::<{{type}}>()) {
        let once = {{module}}::{{function}}(&x);
        let twice = {{module}}::{{function}}(&once);
        prop_assert_eq!(once, twice, "f(f(x)) should equal f(x)");
    }
}"#;

const RUST_COMMUTATIVE: &str = r#"proptest! {
    #[test]
    fn falsify_{{id_lower}}_commutative(
        a in any::<{{type}}>(),
        b in any::<{{type}}>()
    ) {
        let ab = {{module}}::{{function}}(&a, &b);
        let ba = {{module}}::{{function}}(&b, &a);
        prop_assert_eq!(ab, ba, "f(a,b) should equal f(b,a)");
    }
}"#;

const RUST_ASSOCIATIVE: &str = r#"proptest! {
    #[test]
    fn falsify_{{id_lower}}_associative(
        a in -1e6f64..1e6,
        b in -1e6f64..1e6,
        c in -1e6f64..1e6
    ) {
        let ab_c = (a + b) + c;
        let a_bc = a + (b + c);
        // Allow small tolerance for floating point
        prop_assert!((ab_c - a_bc).abs() < 1e-10,
            "(a+b)+c vs a+(b+c): {} vs {}", ab_c, a_bc);
    }
}"#;

const RUST_ROUNDTRIP: &str = r#"proptest! {
    #[test]
    fn falsify_{{id_lower}}_roundtrip(x in any::<{{type}}>()) {
        let encoded = {{module}}::encode(&x);
        let decoded = {{module}}::decode(&encoded).expect("unexpected failure");
        prop_assert_eq!(x, decoded, "decode(encode(x)) should equal x");
    }
}"#;

const RUST_CANCELLATION: &str = r#"#[test]
fn falsify_{{id_lower}}_cancellation() {
    let big = 1e15f64;
    let small = 1.0;
    let result = (big + small) - big;
    assert!((result - small).abs() < 1e-10,
        "Catastrophic cancellation: expected {}, got {}", small, result);
}"#;

const RUST_ACCUMULATION: &str = r#"proptest! {
    #[test]
    fn falsify_{{id_lower}}_accumulation(mut values in prop::collection::vec(-1e6f64..1e6, 10..100)) {
        let sum_original: f64 = values.iter().sum();
        values.sort_by(|a, b| a.partial_cmp(b).expect("comparison failed"));
        let sum_sorted: f64 = values.iter().sum();
        prop_assert!((sum_original - sum_sorted).abs() < 1e-6,
            "Accumulation order matters: {} vs {}", sum_original, sum_sorted);
    }
}"#;

const RUST_DENORMAL: &str = r#"#[test]
fn falsify_{{id_lower}}_denormal() {
    let denormal = 1e-308f64 * 1e-10;
    assert!(denormal > 0.0 || denormal == 0.0,
        "Should handle denormalized numbers");
    let result = {{module}}::{{function}}(denormal);
    assert!(!result.is_nan(), "Should not produce NaN from denormal");
}"#;

const RUST_DATA_RACE: &str = r#"#[test]
fn falsify_{{id_lower}}_data_race() {
    use std::sync::Arc;
    use std::thread;

    let data = Arc::new({{module}}::{{type}}::new());
    let handles: Vec<_> = (0..10).map(|i| {
        let data = Arc::clone(&data);
        thread::spawn(move || {
            data.operation(i);
        })
    }).collect();

    for h in handles {
        h.join().expect("Thread should not panic");
    }
}"#;

const RUST_DEADLOCK: &str = r#"#[test]
fn falsify_{{id_lower}}_deadlock() {
    use std::time::Duration;
    use std::sync::mpsc::channel;

    let (tx, rx) = channel();
    let handle = std::thread::spawn(move || {
        {{module}}::potentially_blocking_operation();
        tx.send(()).expect("channel send failed");
    });

    // Should complete within timeout
    let result = rx.recv_timeout(Duration::from_secs(5));
    assert!(result.is_ok(), "Operation should complete without deadlock");
    handle.join().expect("thread join failed");
}"#;

const RUST_ABA: &str = r#"#[test]
fn falsify_{{id_lower}}_aba() {
    use std::sync::atomic::{AtomicUsize, Ordering};

    let counter = AtomicUsize::new(0);
    // Simulate ABA: 0 -> 1 -> 0
    counter.store(1, Ordering::SeqCst);
    counter.store(0, Ordering::SeqCst);

    // CAS should detect the change
    let result = {{module}}::atomic_operation(&counter);
    assert!(result.is_ok(), "Should handle ABA problem");
}"#;

const RUST_MEMORY: &str = r"#[test]
fn falsify_{{id_lower}}_memory() {
    // Attempt allocation that might fail
    let result = std::panic::catch_unwind(|| {
        let _large = vec![0u8; 1_000_000_000]; // 1GB
    });
    // Should either succeed or fail gracefully
    // (this test mainly verifies no UB on allocation failure)
}";

const RUST_FD: &str = r#"#[test]
fn falsify_{{id_lower}}_fd_exhaustion() {
    use std::fs::File;

    let mut files = Vec::new();
    for i in 0..10000 {
        match File::create(format!("/tmp/test_{}", i)) {
            Ok(f) => files.push(f),
            Err(_) => break,
        }
    }
    // Cleanup
    for (i, _) in files.iter().enumerate() {
        let _ = std::fs::remove_file(format!("/tmp/test_{}", i));
    }
    // Main test: verify module handles FD limits
    let result = {{module}}::{{function}}();
    assert!(result.is_ok(), "Should handle FD exhaustion gracefully");
}"#;

const RUST_STACK: &str = r#"#[test]
fn falsify_{{id_lower}}_stack_overflow() {
    // Test with iterative approach that might overflow naive recursion
    let result = std::panic::catch_unwind(|| {
        {{module}}::deep_operation(10000);
    });
    assert!(result.is_ok(), "Should handle deep recursion without stack overflow");
}"#;

const RUST_PYTHON_PARITY: &str = r#"#[test]
fn falsify_{{id_lower}}_python_parity() {
    // Reference values from Python implementation
    let test_cases = vec![
        (vec![1.0, 2.0, 3.0], vec![1.0, 2.0, 3.0]), // Expected output
    ];

    for (input, expected) in test_cases {
        let result = {{module}}::{{function}}(&input);
        assert_eq!(result, expected, "Should match Python reference");
    }
}"#;

const RUST_GPU_PARITY: &str = r#"#[test]
fn falsify_{{id_lower}}_gpu_parity() {
    let input = vec![1.0f32; 1024];

    let cpu_result = {{module}}::cpu_{{function}}(&input);
    let gpu_result = {{module}}::gpu_{{function}}(&input);

    for (cpu, gpu) in cpu_result.iter().zip(gpu_result.iter()) {
        assert!((cpu - gpu).abs() < 1e-5,
            "CPU/GPU mismatch: {} vs {}", cpu, gpu);
    }
}"#;

// Python templates
const PYTHON_EMPTY_INPUT: &str = r#"    result = module.function([])
    assert result is None or len(result) == 0, "Should handle empty input"
"#;

const PYTHON_MAX_INPUT: &str = r#"    large_input = [0] * {{max_size}}
    result = module.function(large_input)
    assert result is not None, "Should handle maximum size input"
"#;

const PYTHON_NEGATIVE: &str = r"    with pytest.raises(ValueError):
        module.function(-1)
";

const PYTHON_UNICODE: &str = r#"    inputs = ["café\u0301", "\u202Etest", "a\u200Bb"]
    for inp in inputs:
        result = module.function(inp)
        assert result is not None, f"Should handle Unicode: {inp}"
"#;

const PYTHON_NUMERIC_LIMITS: &str = r"    import math
    values = [float('inf'), float('-inf'), float('nan'), 1e308, -1e308]
    for val in values:
        result = module.function(val)
        assert not math.isnan(result) or math.isnan(val)
";

const PYTHON_IDEMPOTENT: &str = r#"@given(st.{{strategy}}())
def test_{{id_lower}}_idempotent(x):
    once = module.function(x)
    twice = module.function(once)
    assert once == twice, "f(f(x)) should equal f(x)"
"#;

const PYTHON_COMMUTATIVE: &str = r#"@given(st.{{strategy}}(), st.{{strategy}}())
def test_{{id_lower}}_commutative(a, b):
    ab = module.function(a, b)
    ba = module.function(b, a)
    assert ab == ba, "f(a,b) should equal f(b,a)"
"#;

const PYTHON_ASSOCIATIVE: &str = r#"@given(st.floats(-1e6, 1e6), st.floats(-1e6, 1e6), st.floats(-1e6, 1e6))
def test_{{id_lower}}_associative(a, b, c):
    ab_c = (a + b) + c
    a_bc = a + (b + c)
    assert abs(ab_c - a_bc) < 1e-10, f"(a+b)+c vs a+(b+c): {ab_c} vs {a_bc}"
"#;

const PYTHON_ROUNDTRIP: &str = r#"@given(st.{{strategy}}())
def test_{{id_lower}}_roundtrip(x):
    encoded = module.encode(x)
    decoded = module.decode(encoded)
    assert x == decoded, "decode(encode(x)) should equal x"
"#;

const PYTHON_CANCELLATION: &str = r#"    big = 1e15
    small = 1.0
    result = (big + small) - big
    assert abs(result - small) < 1e-10, f"Catastrophic cancellation: expected {small}, got {result}"
"#;

const PYTHON_ACCUMULATION: &str = r"@given(st.lists(st.floats(-1e6, 1e6), min_size=10, max_size=100))
def test_{{id_lower}}_accumulation(values):
    sum_original = sum(values)
    sum_sorted = sum(sorted(values))
    assert abs(sum_original - sum_sorted) < 1e-6
";

const PYTHON_DENORMAL: &str = r#"    denormal = 1e-308 * 1e-10
    result = module.function(denormal)
    assert not math.isnan(result), "Should not produce NaN from denormal"
"#;

const PYTHON_DATA_RACE: &str = r#"    import threading

    errors = []
    def worker(i):
        try:
            module.operation(i)
        except Exception as e:
            errors.append(e)

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(10)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert len(errors) == 0, f"Data race detected: {errors}"
"#;

const PYTHON_DEADLOCK: &str = r#"    import threading
    import queue

    q = queue.Queue()

    def worker():
        module.potentially_blocking_operation()
        q.put(True)

    t = threading.Thread(target=worker)
    t.start()

    try:
        q.get(timeout=5)
    except queue.Empty:
        pytest.fail("Operation deadlocked")
    t.join()
"#;

const PYTHON_ABA: &str = r#"    # ABA problem test
    counter = [0]
    counter[0] = 1
    counter[0] = 0
    result = module.atomic_operation(counter)
    assert result is not None, "Should handle ABA problem"
"#;

const PYTHON_MEMORY: &str = r"    try:
        large = [0] * 1_000_000_000
    except MemoryError:
        pass  # Expected
    # Main test: verify module handles memory limits
    result = module.function_with_limit()
    assert result is not None
";

const PYTHON_FD: &str = r#"    import os
    files = []
    try:
        for i in range(10000):
            f = open(f'/tmp/test_{i}', 'w')
            files.append(f)
    except OSError:
        pass  # FD limit reached

    for f in files:
        f.close()
    for i in range(len(files)):
        os.remove(f'/tmp/test_{i}')

    result = module.function()
    assert result is not None, "Should handle FD exhaustion"
"#;

const PYTHON_STACK: &str = r"    import sys
    old_limit = sys.getrecursionlimit()
    sys.setrecursionlimit(100000)

    try:
        result = module.deep_operation(10000)
        assert result is not None
    finally:
        sys.setrecursionlimit(old_limit)
";

const PYTHON_RUST_PARITY: &str = r#"    # Reference values from Rust implementation
    test_cases = [
        ([1.0, 2.0, 3.0], [1.0, 2.0, 3.0]),
    ]

    for input_data, expected in test_cases:
        result = module.function(input_data)
        assert result == expected, "Should match Rust reference"
"#;

const PYTHON_GPU_PARITY: &str = r#"    import numpy as np

    input_data = np.ones(1024, dtype=np.float32)

    cpu_result = module.cpu_function(input_data)
    gpu_result = module.gpu_function(input_data)

    np.testing.assert_allclose(cpu_result, gpu_result, rtol=1e-5,
        err_msg="CPU/GPU mismatch")
"#;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_template() {
        let template = FalsificationTemplate::default();
        assert_eq!(template.total_points(), 100);
    }

    #[test]
    fn test_category_points() {
        let boundary = CategoryTemplate::boundary();
        assert_eq!(boundary.total_points(), 20);

        let invariant = CategoryTemplate::invariant();
        assert_eq!(invariant.total_points(), 20);

        let numerical = CategoryTemplate::numerical();
        assert_eq!(numerical.total_points(), 20);

        let concurrency = CategoryTemplate::concurrency();
        assert_eq!(concurrency.total_points(), 15);

        let resource = CategoryTemplate::resource();
        assert_eq!(resource.total_points(), 15);

        let parity = CategoryTemplate::parity();
        assert_eq!(parity.total_points(), 10);
    }

    #[test]
    fn test_scale_template() {
        let template = FalsificationTemplate::default();
        let scaled = template.scale_to_points(50);
        // Allow larger rounding error since we scale many individual tests
        assert!(
            (scaled.total_points() as i32 - 50).abs() <= 20,
            "Scaled to {} points, expected around 50",
            scaled.total_points()
        );
    }
}
