//! Reduction and softmax WGSL compute shaders.

/// Max reduction compute shader (WGSL)
///
/// Computes max(input) using parallel reduction
/// Used as first pass in softmax to ensure numerical stability
pub(crate) const MAX_REDUCTION_SHADER: &str = r#"
@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> result: array<f32>;

var<workgroup> partial_max: array<f32, 256>;

@compute @workgroup_size(256)
fn main(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>,
) {
    let idx = global_id.x;
    let local_idx = local_id.x;
    let len = arrayLength(&input);

    // Load value or negative infinity
    var max_val: f32 = -3.402823466e+38; // -FLT_MAX
    if (idx < len) {
        max_val = input[idx];
    }
    partial_max[local_idx] = max_val;

    workgroupBarrier();

    // Parallel reduction within workgroup (find max)
    var stride: u32 = 128u;
    while (stride > 0u) {
        if (local_idx < stride) {
            partial_max[local_idx] = max(partial_max[local_idx], partial_max[local_idx + stride]);
        }
        stride = stride / 2u;
        workgroupBarrier();
    }

    // First thread writes workgroup result
    if (local_idx == 0u) {
        result[global_id.x / 256u] = partial_max[0];
    }
}
"#;

/// Sum reduction compute shader (WGSL)
///
/// Computes sum(input) using parallel reduction
/// Used in softmax to sum exp values
pub(crate) const SUM_REDUCTION_SHADER: &str = r#"
@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> result: array<f32>;

var<workgroup> partial_sums: array<f32, 256>;

@compute @workgroup_size(256)
fn main(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>,
) {
    let idx = global_id.x;
    let local_idx = local_id.x;
    let len = arrayLength(&input);

    // Load value
    var sum: f32 = 0.0;
    if (idx < len) {
        sum = input[idx];
    }
    partial_sums[local_idx] = sum;

    workgroupBarrier();

    // Parallel reduction within workgroup
    var stride: u32 = 128u;
    while (stride > 0u) {
        if (local_idx < stride) {
            partial_sums[local_idx] = partial_sums[local_idx] + partial_sums[local_idx + stride];
        }
        stride = stride / 2u;
        workgroupBarrier();
    }

    // First thread writes workgroup result
    if (local_idx == 0u) {
        result[global_id.x / 256u] = partial_sums[0];
    }
}
"#;

/// Softmax exp-subtract compute shader (WGSL)
///
/// Computes exp(input[i] - max_val) for each element
/// Second pass in softmax: numerically stable exp computation
pub(crate) const SOFTMAX_EXP_SHADER: &str = r#"
@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;

struct MaxValue {
    max_val: f32,
}

@group(0) @binding(2) var<uniform> params: MaxValue;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    let len = arrayLength(&input);

    if (idx < len) {
        // exp(x - max) for numerical stability
        output[idx] = exp(input[idx] - params.max_val);
    }
}
"#;

/// Softmax normalize compute shader (WGSL)
///
/// Computes output[i] = input[i] / sum_val for each element
/// Fourth pass in softmax: normalize by sum of exp values
pub(crate) const SOFTMAX_NORMALIZE_SHADER: &str = r#"
@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;

struct SumValue {
    sum_val: f32,
}

@group(0) @binding(2) var<uniform> params: SumValue;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    let len = arrayLength(&input);

    if (idx < len) {
        // Normalize by sum
        output[idx] = input[idx] / params.sum_val;
    }
}
"#;

/// Log-softmax compute shader (WGSL)
///
/// Computes log_softmax[i] = input[i] - max_val - log(sum_val) for each element
/// Numerically stable log-softmax in single pass after reductions
pub(crate) const LOG_SOFTMAX_SHADER: &str = r#"
@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;

struct LogSoftmaxParams {
    max_val: f32,
    log_sum_exp: f32,
}

@group(0) @binding(2) var<uniform> params: LogSoftmaxParams;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    let len = arrayLength(&input);

    if (idx < len) {
        // log_softmax(x)[i] = x[i] - max - log(sum(exp(x - max)))
        output[idx] = input[idx] - params.max_val - params.log_sum_exp;
    }
}
"#;
