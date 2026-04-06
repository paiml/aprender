//! PageRank iteration kernel.
//!
//! Matches `pagerank-kernel-v1.yaml`.
//! r' = d * M * r + (1 - d) / n
//!
//! Where M is a row-stochastic transition matrix, d is the damping factor,
//! and n is the number of nodes.

/// Scalar reference implementation of a single PageRank iteration.
///
/// Computes `output[i] = damping * sum_j(transition[i*n+j] * rank[j]) + (1 - damping) / n`.
///
/// - `transition`: flattened `n x n` row-stochastic transition matrix
/// - `rank`: current rank vector, length `n`
/// - `output`: new rank vector, length `n`
///
/// # Panics
///
/// Panics if dimensions are inconsistent or `n` is zero.
pub fn pagerank_iterate_scalar(
    transition: &[f32],
    rank: &[f32],
    n: usize,
    damping: f32,
    output: &mut [f32],
) {
    assert!(n > 0, "n must be > 0");
    assert_eq!(transition.len(), n * n, "transition length mismatch");
    assert_eq!(rank.len(), n, "rank length mismatch");
    assert_eq!(output.len(), n, "output length mismatch");

    let teleport = (1.0 - damping) / n as f32;

    for i in 0..n {
        let mut sum = 0.0_f32;
        for j in 0..n {
            sum += transition[i * n + j] * rank[j];
        }
        output[i] = damping * sum + teleport;
    }
}

/// AVX2 implementation of a single PageRank iteration.
///
/// Delegates to scalar (matrix-vector multiply benefits more from
/// blocking strategies than simple SIMD for small graphs).
///
/// # Safety
///
/// Requires AVX2 support on the target CPU.
///
/// # Panics
///
/// Same as [`pagerank_iterate_scalar`].
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
pub unsafe fn pagerank_iterate_avx2(
    transition: &[f32],
    rank: &[f32],
    n: usize,
    damping: f32,
    output: &mut [f32],
) {
    pagerank_iterate_scalar(transition, rank, n, damping, output);
}

/// PTX assembly for the PageRank iteration kernel.
///
/// 1 block per node, reduction for dot product of transition row with rank vector.
pub fn pagerank_iterate_ptx() -> &'static str {
    r#".version 8.5
.target sm_90
.address_size 64

// PageRank iteration kernel: 1 block per node.
// Each block computes the dot product of one transition row with the rank vector.
// Params: transition_ptr, rank_ptr, output_ptr, n, damping
.visible .entry pagerank_kernel(
    .param .u64 transition_ptr,
    .param .u64 rank_ptr,
    .param .u64 output_ptr,
    .param .u32 n,
    .param .f32 damping
)
{
    .reg .u32 %tid, %bid, %n, %j, %stride;
    .reg .u64 %t_base, %r_base, %o_base, %addr;
    .reg .u32 %row_off, %tmp, %partner;
    .reg .f32 %sum, %tval, %rval, %damp, %teleport, %result;
    .reg .f32 %k_one, %n_f, %other;
    .reg .pred %p;
    .shared .f32 smem[256];

    mov.u32 %bid, %ctaid.x;
    mov.u32 %tid, %tid.x;
    ld.param.u32 %n, [n];
    ld.param.f32 %damp, [damping];
    ld.param.u64 %t_base, [transition_ptr];
    ld.param.u64 %r_base, [rank_ptr];
    ld.param.u64 %o_base, [output_ptr];

    // row_off = bid * n
    mul.lo.u32 %row_off, %bid, %n;

    // Partial dot product: each thread handles elements tid, tid+256, ...
    mov.f32 %sum, 0f00000000;
    mov.u32 %j, %tid;
DOT_LOOP:
    setp.ge.u32 %p, %j, %n;
    @%p bra DOT_DONE;

    // Load transition[bid*n + j]
    add.u32 %tmp, %row_off, %j;
    cvt.u64.u32 %addr, %tmp;
    shl.b64 %addr, %addr, 2;
    add.u64 %addr, %t_base, %addr;
    ld.global.f32 %tval, [%addr];

    // Load rank[j]
    cvt.u64.u32 %addr, %j;
    shl.b64 %addr, %addr, 2;
    add.u64 %addr, %r_base, %addr;
    ld.global.f32 %rval, [%addr];

    fma.rn.f32 %sum, %tval, %rval, %sum;

    add.u32 %j, %j, 256;
    bra DOT_LOOP;
DOT_DONE:

    // Store partial sum to shared memory
    st.shared.f32 [smem + %tid * 4], %sum;
    bar.sync 0;

    // Tree reduction in shared memory
    mov.u32 %stride, 128;
REDUCE_LOOP:
    setp.lt.u32 %p, %tid, %stride;
    @!%p bra REDUCE_SKIP;
    add.u32 %partner, %tid, %stride;
    ld.shared.f32 %other, [smem + %partner * 4];
    ld.shared.f32 %sum, [smem + %tid * 4];
    add.f32 %sum, %sum, %other;
    st.shared.f32 [smem + %tid * 4], %sum;
REDUCE_SKIP:
    bar.sync 0;
    shr.b32 %stride, %stride, 1;
    setp.ge.u32 %p, %stride, 1;
    @%p bra REDUCE_LOOP;

    // Thread 0 writes the final result
    setp.ne.u32 %p, %tid, 0;
    @%p bra DONE;

    ld.shared.f32 %sum, [smem];
    // output[bid] = damping * sum + (1 - damping) / n
    mov.f32 %k_one, 0f3F800000;
    sub.f32 %teleport, %k_one, %damp;
    cvt.rn.f32.u32 %n_f, %n;
    div.approx.f32 %teleport, %teleport, %n_f;
    fma.rn.f32 %result, %damp, %sum, %teleport;

    cvt.u64.u32 %addr, %bid;
    shl.b64 %addr, %addr, 2;
    add.u64 %addr, %o_base, %addr;
    st.global.f32 [%addr], %result;

DONE:
    ret;
}
"#
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;

    // ---------------------------------------------------------------
    // Scalar tests
    // ---------------------------------------------------------------

    #[test]
    fn test_pagerank_uniform_initial() {
        // 3-node graph, uniform transition, uniform rank
        let n = 3;
        let transition = [
            1.0 / 3.0,
            1.0 / 3.0,
            1.0 / 3.0,
            1.0 / 3.0,
            1.0 / 3.0,
            1.0 / 3.0,
            1.0 / 3.0,
            1.0 / 3.0,
            1.0 / 3.0,
        ];
        let rank = [1.0 / 3.0_f32; 3];
        let mut output = [0.0_f32; 3];

        pagerank_iterate_scalar(&transition, &rank, n, 0.85, &mut output);

        // Uniform transition + uniform rank = uniform output
        let expected = 1.0 / 3.0;
        for (i, &o) in output.iter().enumerate() {
            assert!(
                (o - expected).abs() < 1e-5,
                "output[{i}] = {o}, expected ~{expected}"
            );
        }
    }

    #[test]
    fn test_pagerank_known_2node() {
        // 2-node graph: node 0 -> node 1, node 1 -> node 0
        let n = 2;
        let transition = [0.0_f32, 1.0, 1.0, 0.0];
        let rank = [0.5_f32, 0.5];
        let mut output = [0.0_f32; 2];
        let damping = 0.85;

        pagerank_iterate_scalar(&transition, &rank, n, damping, &mut output);

        // output[0] = 0.85 * (0*0.5 + 1*0.5) + 0.15/2 = 0.85*0.5 + 0.075 = 0.5
        // output[1] = 0.85 * (1*0.5 + 0*0.5) + 0.15/2 = 0.85*0.5 + 0.075 = 0.5
        assert!((output[0] - 0.5).abs() < 1e-6, "output[0] = {}", output[0]);
        assert!((output[1] - 0.5).abs() < 1e-6, "output[1] = {}", output[1]);
    }

    #[test]
    fn test_pagerank_convergence() {
        // After many iterations with a valid transition matrix, ranks should
        // converge to a stationary distribution
        let n = 3;
        let transition = [0.0, 0.5, 0.5, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0];
        let mut rank = vec![1.0 / 3.0_f32; n];
        let mut output = vec![0.0_f32; n];
        let damping = 0.85;

        for _ in 0..100 {
            pagerank_iterate_scalar(&transition, &rank, n, damping, &mut output);
            rank.copy_from_slice(&output);
        }

        let sum: f32 = output.iter().sum();
        assert!(
            (sum - 1.0).abs() < 1e-4,
            "rank sum should be ~1.0, got {sum}"
        );
    }

    #[test]
    #[should_panic(expected = "n must be > 0")]
    fn test_pagerank_zero_n() {
        let mut output: [f32; 0] = [];
        pagerank_iterate_scalar(&[], &[], 0, 0.85, &mut output);
    }

    proptest! {
        #[test]
        fn prop_pagerank_output_sums_to_one(n in 2_usize..8) {
            // Build a row-stochastic transition matrix
            let mut transition = vec![0.0_f32; n * n];
            for i in 0..n {
                let val = 1.0 / n as f32;
                for j in 0..n {
                    transition[i * n + j] = val;
                }
            }
            // Uniform initial rank that sums to 1
            let rank = vec![1.0 / n as f32; n];
            let mut output = vec![0.0_f32; n];

            pagerank_iterate_scalar(&transition, &rank, n, 0.85, &mut output);

            let sum: f32 = output.iter().sum();
            prop_assert!(
                (sum - 1.0).abs() < 1e-4,
                "PageRank output should sum to ~1.0, got {sum}"
            );
        }
    }

    // ---------------------------------------------------------------
    // AVX2 tests
    // ---------------------------------------------------------------

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn test_pagerank_avx2_parity() {
        if !is_x86_feature_detected!("avx2") {
            return;
        }
        let n = 4;
        let transition = [
            0.25_f32, 0.25, 0.25, 0.25, 0.0, 0.0, 0.5, 0.5, 0.5, 0.5, 0.0, 0.0, 0.1, 0.2, 0.3, 0.4,
        ];
        let rank = [0.25_f32; 4];
        let mut scalar_out = [0.0_f32; 4];
        let mut avx2_out = [0.0_f32; 4];

        pagerank_iterate_scalar(&transition, &rank, n, 0.85, &mut scalar_out);
        unsafe {
            pagerank_iterate_avx2(&transition, &rank, n, 0.85, &mut avx2_out);
        }

        assert_eq!(scalar_out, avx2_out);
    }

    // ---------------------------------------------------------------
    // PTX structural tests
    // ---------------------------------------------------------------

    #[test]
    fn test_pagerank_ptx_version() {
        let ptx = pagerank_iterate_ptx();
        assert!(
            ptx.contains(".version 8.5"),
            "PTX must declare .version 8.5"
        );
    }

    #[test]
    fn test_pagerank_ptx_target() {
        let ptx = pagerank_iterate_ptx();
        assert!(ptx.contains(".target sm_90"), "PTX must target sm_90");
    }

    #[test]
    fn test_pagerank_ptx_entry() {
        let ptx = pagerank_iterate_ptx();
        assert!(
            ptx.contains(".entry pagerank_kernel"),
            "PTX must have .entry"
        );
    }

    #[test]
    fn test_pagerank_ptx_ret() {
        let ptx = pagerank_iterate_ptx();
        assert!(ptx.contains("ret;"), "PTX must have ret;");
    }

    #[test]
    fn test_pagerank_ptx_balanced_braces() {
        let ptx = pagerank_iterate_ptx();
        let opens = ptx.chars().filter(|&c| c == '{').count();
        let closes = ptx.chars().filter(|&c| c == '}').count();
        assert_eq!(
            opens, closes,
            "PTX must have balanced braces: {opens} opens vs {closes} closes"
        );
    }

    #[test]
    fn test_pagerank_ptx_shared_memory() {
        let ptx = pagerank_iterate_ptx();
        assert!(
            ptx.contains(".shared"),
            "PTX must use shared memory for reduction"
        );
    }

    #[test]
    fn test_pagerank_ptx_bar_sync() {
        let ptx = pagerank_iterate_ptx();
        assert!(
            ptx.contains("bar.sync"),
            "PTX must have bar.sync for synchronization"
        );
    }
}
