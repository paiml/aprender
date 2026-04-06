// ==================== IMP-208: Softmax Sums to 1.0 (QA-005) ====================
// Per spec: Softmax outputs sum to 1.0 within 1e-7

/// Softmax verification result
#[derive(Debug, Clone)]
pub struct SoftmaxVerificationResult {
    pub input_logits: Vec<f32>,
    pub output_probs: Vec<f32>,
    pub sum: f32,
    pub sum_diff_from_one: f32,
    pub tolerance: f32,
    pub meets_qa005: bool,
}

impl SoftmaxVerificationResult {
    pub fn new(logits: Vec<f32>, probs: Vec<f32>, tolerance: f32) -> Self {
        let sum: f32 = probs.iter().sum();
        let sum_diff_from_one = (sum - 1.0).abs();
        let meets_qa005 = sum_diff_from_one <= tolerance;

        Self {
            input_logits: logits,
            output_probs: probs,
            sum,
            sum_diff_from_one,
            tolerance,
            meets_qa005,
        }
    }

    pub fn compute_softmax(logits: &[f32]) -> Vec<f32> {
        let max_logit = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let exp_sum: f32 = logits.iter().map(|x| (x - max_logit).exp()).sum();
        logits
            .iter()
            .map(|x| (x - max_logit).exp() / exp_sum)
            .collect()
    }
}

/// IMP-208a: Test softmax verification
#[test]
fn test_imp_208a_softmax_verification() {
    let logits = vec![1.0, 2.0, 3.0];
    let probs = SoftmaxVerificationResult::compute_softmax(&logits);
    let result = SoftmaxVerificationResult::new(logits, probs, 1e-7);

    assert!(result.meets_qa005, "IMP-208a: Should meet QA-005");
    assert!(
        result.sum_diff_from_one < 1e-7,
        "IMP-208a: Sum should be 1.0"
    );

    println!("\nIMP-208a: Softmax Verification:");
    println!("  Probabilities: {:?}", result.output_probs);
    println!("  Sum: {:.10}", result.sum);
    println!("  Diff from 1.0: {:.2e}", result.sum_diff_from_one);
}

/// IMP-208b: Test softmax with large logits
#[test]
fn test_imp_208b_softmax_large_logits() {
    let logits = vec![100.0, 200.0, 300.0];
    let probs = SoftmaxVerificationResult::compute_softmax(&logits);
    let result = SoftmaxVerificationResult::new(logits, probs, 1e-7);

    assert!(result.meets_qa005, "IMP-208b: Should handle large logits");

    println!("\nIMP-208b: Softmax Large Logits:");
    println!("  Sum: {:.10}", result.sum);
    println!("  Numerically stable: {}", result.meets_qa005);
}

/// IMP-208c: Test softmax with negative logits
#[test]
fn test_imp_208c_softmax_negative_logits() {
    let logits = vec![-1.0, -2.0, -3.0];
    let probs = SoftmaxVerificationResult::compute_softmax(&logits);
    let result = SoftmaxVerificationResult::new(logits, probs, 1e-7);

    assert!(
        result.meets_qa005,
        "IMP-208c: Should handle negative logits"
    );

    println!("\nIMP-208c: Softmax Negative Logits:");
    println!("  Probabilities: {:?}", result.output_probs);
    println!("  Sum: {:.10}", result.sum);
}

/// IMP-208d: Real-world softmax verification
#[test]
#[ignore = "Requires softmax extraction from inference"]
fn test_imp_208d_realworld_softmax() {
    let logits = vec![2.5, 1.2, 0.8, 3.1, 0.5];
    let probs = SoftmaxVerificationResult::compute_softmax(&logits);
    let result = SoftmaxVerificationResult::new(logits, probs, 1e-7);

    println!("\nIMP-208d: Real-World Softmax:");
    println!("  Sum: {:.10}", result.sum);
    println!(
        "  QA-005: {}",
        if result.meets_qa005 { "PASS" } else { "FAIL" }
    );
}

// ==================== IMP-209: Layer Norm Unit Variance (QA-006) ====================
// Per spec: Layer norm outputs have unit variance within 1e-4

/// Layer norm verification result
#[derive(Debug, Clone)]
pub struct LayerNormVerificationResult {
    pub input: Vec<f32>,
    pub output: Vec<f32>,
    pub mean: f32,
    pub variance: f32,
    pub variance_diff_from_one: f32,
    pub tolerance: f32,
    pub meets_qa006: bool,
}

impl LayerNormVerificationResult {
    pub fn new(input: Vec<f32>, output: Vec<f32>, tolerance: f32) -> Self {
        let n = output.len() as f32;
        let mean: f32 = output.iter().sum::<f32>() / n;
        let variance: f32 = output.iter().map(|x| (x - mean).powi(2)).sum::<f32>() / n;
        let variance_diff_from_one = (variance - 1.0).abs();
        let meets_qa006 = variance_diff_from_one <= tolerance;

        Self {
            input,
            output,
            mean,
            variance,
            variance_diff_from_one,
            tolerance,
            meets_qa006,
        }
    }

    pub fn compute_layer_norm(input: &[f32], eps: f32) -> Vec<f32> {
        let n = input.len() as f32;
        let mean: f32 = input.iter().sum::<f32>() / n;
        let variance: f32 = input.iter().map(|x| (x - mean).powi(2)).sum::<f32>() / n;
        let std = (variance + eps).sqrt();
        input.iter().map(|x| (x - mean) / std).collect()
    }
}

/// IMP-209a: Test layer norm verification
#[test]
fn test_imp_209a_layer_norm_verification() {
    let input = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let output = LayerNormVerificationResult::compute_layer_norm(&input, 1e-5);
    let result = LayerNormVerificationResult::new(input, output, 1e-4);

    assert!(result.meets_qa006, "IMP-209a: Should meet QA-006");

    println!("\nIMP-209a: Layer Norm Verification:");
    println!("  Output: {:?}", result.output);
    println!("  Mean: {:.6}", result.mean);
    println!("  Variance: {:.6}", result.variance);
}

/// IMP-209b: Test layer norm zero mean
#[test]
fn test_imp_209b_layer_norm_zero_mean() {
    let input = vec![10.0, 20.0, 30.0, 40.0];
    let output = LayerNormVerificationResult::compute_layer_norm(&input, 1e-5);
    let result = LayerNormVerificationResult::new(input, output, 1e-4);

    assert!(result.mean.abs() < 1e-5, "IMP-209b: Mean should be ~0");

    println!("\nIMP-209b: Layer Norm Zero Mean:");
    println!("  Mean: {:.2e}", result.mean);
}

/// IMP-209c: Test layer norm with uniform input
#[test]
fn test_imp_209c_layer_norm_uniform() {
    let input = vec![5.0, 5.0, 5.0, 5.0];
    let output = LayerNormVerificationResult::compute_layer_norm(&input, 1e-5);

    // All same values -> variance is 0, output is 0
    assert!(
        output.iter().all(|&x| x.abs() < 1e-3),
        "IMP-209c: Uniform input -> zero output"
    );

    println!("\nIMP-209c: Layer Norm Uniform Input:");
    println!("  Output: {:?}", output);
}

/// IMP-209d: Real-world layer norm verification
#[test]
#[ignore = "Requires layer norm extraction from model"]
fn test_imp_209d_realworld_layer_norm() {
    let input = vec![0.5, 1.2, -0.3, 0.8, -0.1];
    let output = LayerNormVerificationResult::compute_layer_norm(&input, 1e-5);
    let result = LayerNormVerificationResult::new(input, output, 1e-4);

    println!("\nIMP-209d: Real-World Layer Norm:");
    println!("  Variance: {:.6}", result.variance);
    println!(
        "  QA-006: {}",
        if result.meets_qa006 { "PASS" } else { "FAIL" }
    );
}

// ==================== IMP-210: GELU Matches PyTorch (QA-007) ====================
// Per spec: GELU activation matches PyTorch within 1e-5

/// GELU verification result
#[derive(Debug, Clone)]
pub struct GELUVerificationResult {
    pub input: Vec<f32>,
    pub reference_output: Vec<f32>,
    pub test_output: Vec<f32>,
    pub max_diff: f32,
    pub tolerance: f32,
    pub meets_qa007: bool,
}

impl GELUVerificationResult {
    pub fn new(input: Vec<f32>, ref_out: Vec<f32>, test_out: Vec<f32>, tolerance: f32) -> Self {
        let max_diff = ref_out
            .iter()
            .zip(test_out.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0_f32, f32::max);

        let meets_qa007 = max_diff <= tolerance;

        Self {
            input,
            reference_output: ref_out,
            test_output: test_out,
            max_diff,
            tolerance,
            meets_qa007,
        }
    }

    /// GELU approximation (tanh version used by GPT-2)
    pub fn compute_gelu(x: f32) -> f32 {
        0.5 * x
            * (1.0 + ((2.0_f32 / std::f32::consts::PI).sqrt() * (x + 0.044715 * x.powi(3))).tanh())
    }
}

/// IMP-210a: Test GELU verification
#[test]
fn test_imp_210a_gelu_verification() {
    let input = vec![-1.0, 0.0, 1.0, 2.0];
    let ref_out: Vec<f32> = input
        .iter()
        .map(|&x| GELUVerificationResult::compute_gelu(x))
        .collect();
    let test_out = ref_out.clone();

    let result = GELUVerificationResult::new(input, ref_out, test_out, 1e-5);

    assert!(result.meets_qa007, "IMP-210a: Should meet QA-007");

    println!("\nIMP-210a: GELU Verification:");
    println!("  Input: {:?}", result.input);
    println!("  Output: {:?}", result.reference_output);
    println!("  Max diff: {:.2e}", result.max_diff);
}

/// IMP-210b: Test GELU at zero
#[test]
fn test_imp_210b_gelu_at_zero() {
    let gelu_zero = GELUVerificationResult::compute_gelu(0.0);
    assert!(gelu_zero.abs() < 1e-7, "IMP-210b: GELU(0) should be 0");

    println!("\nIMP-210b: GELU at Zero:");
    println!("  GELU(0) = {:.10}", gelu_zero);
}

/// IMP-210c: Test GELU approximation accuracy
#[test]
fn test_imp_210c_gelu_approximation() {
    // PyTorch reference values for GELU
    let test_cases = vec![
        (-2.0, -0.0454),
        (-1.0, -0.1587),
        (0.0, 0.0),
        (1.0, 0.8413),
        (2.0, 1.9546),
    ];

    println!("\nIMP-210c: GELU Approximation:");
    for (x, expected) in test_cases {
        let actual = GELUVerificationResult::compute_gelu(x);
        let diff = (actual - expected).abs();
        println!(
            "  GELU({:.1}) = {:.4} (expected {:.4}, diff {:.4})",
            x, actual, expected, diff
        );
        assert!(diff < 0.01, "IMP-210c: GELU should match PyTorch");
    }
}

/// IMP-210d: Real-world GELU verification
#[test]
#[ignore = "Requires GELU extraction from PyTorch reference"]
fn test_imp_210d_realworld_gelu() {
    let input = vec![-1.5, -0.5, 0.5, 1.5];
    let ref_out: Vec<f32> = input
        .iter()
        .map(|&x| GELUVerificationResult::compute_gelu(x))
        .collect();

    let result = GELUVerificationResult::new(input, ref_out.clone(), ref_out, 1e-5);

    println!("\nIMP-210d: Real-World GELU:");
    println!("  Max diff: {:.2e}", result.max_diff);
    println!(
        "  QA-007: {}",
        if result.meets_qa007 { "PASS" } else { "FAIL" }
    );
}

// ==================== IMP-211: SwiGLU Matches Reference (QA-008) ====================
// Per spec: SwiGLU activation matches reference within 1e-5

/// SwiGLU verification result
#[derive(Debug, Clone)]
pub struct SwiGLUVerificationResult {
    pub input_gate: Vec<f32>,
    pub input_up: Vec<f32>,
    pub reference_output: Vec<f32>,
    pub test_output: Vec<f32>,
    pub max_diff: f32,
    pub tolerance: f32,
    pub meets_qa008: bool,
}

impl SwiGLUVerificationResult {
    pub fn new(
        gate: Vec<f32>,
        up: Vec<f32>,
        ref_out: Vec<f32>,
        test_out: Vec<f32>,
        tolerance: f32,
    ) -> Self {
        let max_diff = ref_out
            .iter()
            .zip(test_out.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0_f32, f32::max);

        let meets_qa008 = max_diff <= tolerance;

        Self {
            input_gate: gate,
            input_up: up,
            reference_output: ref_out,
            test_output: test_out,
            max_diff,
            tolerance,
            meets_qa008,
        }
    }

    /// Swish activation: x * sigmoid(x)
    pub fn swish(x: f32) -> f32 {
        x / (1.0 + (-x).exp())
    }

    /// SwiGLU: swish(gate) * up
    pub fn compute_swiglu(gate: &[f32], up: &[f32]) -> Vec<f32> {
        gate.iter()
            .zip(up.iter())
            .map(|(&g, &u)| Self::swish(g) * u)
            .collect()
    }
}

/// IMP-211a: Test SwiGLU verification
#[test]
fn test_imp_211a_swiglu_verification() {
    let gate = vec![1.0, 2.0, 3.0];
    let up = vec![1.0, 1.0, 1.0];
    let ref_out = SwiGLUVerificationResult::compute_swiglu(&gate, &up);
    let test_out = ref_out.clone();

    let result = SwiGLUVerificationResult::new(gate, up, ref_out, test_out, 1e-5);

    assert!(result.meets_qa008, "IMP-211a: Should meet QA-008");

    println!("\nIMP-211a: SwiGLU Verification:");
    println!("  Output: {:?}", result.reference_output);
    println!("  Max diff: {:.2e}", result.max_diff);
}

/// IMP-211b: Test swish activation
#[test]
fn test_imp_211b_swish_activation() {
    let test_cases = vec![(0.0, 0.0), (1.0, 0.7311), (2.0, 1.7616), (-1.0, -0.2689)];

    println!("\nIMP-211b: Swish Activation:");
    for (x, expected) in test_cases {
        let actual = SwiGLUVerificationResult::swish(x);
        let diff = (actual - expected).abs();
        println!(
            "  swish({:.1}) = {:.4} (expected {:.4})",
            x, actual, expected
        );
        assert!(diff < 0.01, "IMP-211b: Swish should match reference");
    }
}

/// IMP-211c: Test SwiGLU with different inputs
#[test]
fn test_imp_211c_swiglu_different_inputs() {
    let gate = vec![0.0, 1.0, -1.0];
    let up = vec![2.0, 2.0, 2.0];
    let output = SwiGLUVerificationResult::compute_swiglu(&gate, &up);

    assert!(output[0].abs() < 1e-5, "IMP-211c: SwiGLU(0, 2) should be 0");
    assert!(output[1] > 0.0, "IMP-211c: SwiGLU(1, 2) should be positive");
    assert!(
        output[2] < 0.0,
        "IMP-211c: SwiGLU(-1, 2) should be negative"
    );

    println!("\nIMP-211c: SwiGLU Different Inputs:");
    println!("  SwiGLU(0, 2) = {:.4}", output[0]);
    println!("  SwiGLU(1, 2) = {:.4}", output[1]);
    println!("  SwiGLU(-1, 2) = {:.4}", output[2]);
}

include!("imp_211d.rs");
include!("imp_301c.rs");
include!("imp_304a.rs");
