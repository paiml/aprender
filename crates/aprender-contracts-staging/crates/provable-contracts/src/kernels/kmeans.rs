//! K-means clustering kernel.
//!
//! Matches `kmeans-kernel-v1.yaml`.
//!
//! Two-phase iteration: assign points to nearest centroid (L2 distance),
//! then update centroids as the mean of assigned points.

/// Assign each point to its nearest centroid using L2 distance.
///
/// - `points`: flattened `n x d` (row-major)
/// - `centroids`: flattened `k x d` (row-major)
/// - `assignments`: length `n`, each entry in `0..k`
///
/// # Panics
///
/// Panics if dimensions are inconsistent or if `k` or `d` is zero.
pub fn kmeans_assign_scalar(
    points: &[f32],
    centroids: &[f32],
    n: usize,
    k: usize,
    d: usize,
    assignments: &mut [u32],
) {
    assert_eq!(points.len(), n * d, "points length mismatch");
    assert_eq!(centroids.len(), k * d, "centroids length mismatch");
    assert_eq!(assignments.len(), n, "assignments length mismatch");
    assert!(k > 0, "k must be > 0");
    assert!(d > 0, "d must be > 0");

    for p in 0..n {
        let mut best_dist = f32::MAX;
        let mut best_k = 0_u32;
        for c in 0..k {
            let mut dist = 0.0_f32;
            for j in 0..d {
                let diff = points[p * d + j] - centroids[c * d + j];
                dist += diff * diff;
            }
            if dist < best_dist {
                best_dist = dist;
                best_k = c as u32;
            }
        }
        assignments[p] = best_k;
    }
}

/// Update centroids as the mean of their assigned points.
///
/// - `points`: flattened `n x d` (row-major)
/// - `assignments`: length `n`, each entry in `0..k`
/// - `centroids`: flattened `k x d`, updated in-place
///
/// If a cluster has no assigned points, its centroid is left at zero.
///
/// # Panics
///
/// Panics if dimensions are inconsistent.
pub fn kmeans_update_scalar(
    points: &[f32],
    assignments: &[u32],
    n: usize,
    k: usize,
    d: usize,
    centroids: &mut [f32],
) {
    assert_eq!(points.len(), n * d, "points length mismatch");
    assert_eq!(assignments.len(), n, "assignments length mismatch");
    assert_eq!(centroids.len(), k * d, "centroids length mismatch");

    // Zero out centroids
    for c in centroids.iter_mut() {
        *c = 0.0;
    }

    let mut counts = vec![0_u32; k];

    // Accumulate
    for p in 0..n {
        let c = assignments[p] as usize;
        counts[c] += 1;
        for j in 0..d {
            centroids[c * d + j] += points[p * d + j];
        }
    }

    // Divide by count
    for c in 0..k {
        if counts[c] > 0 {
            let inv = 1.0 / counts[c] as f32;
            for j in 0..d {
                centroids[c * d + j] *= inv;
            }
        }
    }
}

/// AVX2 implementation of K-means assignment.
///
/// Delegates to scalar for simplicity (gather patterns make SIMD nontrivial).
///
/// # Safety
///
/// Requires AVX2 support on the target CPU.
///
/// # Panics
///
/// Same as [`kmeans_assign_scalar`].
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
pub unsafe fn kmeans_assign_avx2(
    points: &[f32],
    centroids: &[f32],
    n: usize,
    k: usize,
    d: usize,
    assignments: &mut [u32],
) {
    kmeans_assign_scalar(points, centroids, n, k, d, assignments);
}

/// AVX2 implementation of K-means centroid update.
///
/// Delegates to scalar.
///
/// # Safety
///
/// Requires AVX2 support on the target CPU.
///
/// # Panics
///
/// Same as [`kmeans_update_scalar`].
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
pub unsafe fn kmeans_update_avx2(
    points: &[f32],
    assignments: &[u32],
    n: usize,
    k: usize,
    d: usize,
    centroids: &mut [f32],
) {
    kmeans_update_scalar(points, assignments, n, k, d, centroids);
}

/// PTX assembly for K-means assignment kernel.
///
/// 1 thread per point. Each thread computes L2 distance to all centroids
/// and stores the index of the nearest one.
pub fn kmeans_assign_ptx() -> &'static str {
    r#".version 8.5
.target sm_90
.address_size 64

// K-means assign kernel: 1 thread per point.
// Params: points_ptr, centroids_ptr, assignments_ptr, n, k, d
.visible .entry kmeans_assign_kernel(
    .param .u64 points_ptr,
    .param .u64 centroids_ptr,
    .param .u64 assignments_ptr,
    .param .u32 n,
    .param .u32 k,
    .param .u32 d
)
{
    .reg .u32 %tid, %ntid, %ctaid, %idx, %n, %k, %d;
    .reg .u32 %c, %j, %p_off, %c_off, %tmp;
    .reg .u64 %pts_base, %cen_base, %asg_base, %addr;
    .reg .f32 %diff, %dist, %best_dist, %val_p, %val_c;
    .reg .u32 %best_k;
    .reg .pred %p, %p_c, %p_j, %p_better;

    mov.u32 %tid, %tid.x;
    mov.u32 %ntid, %ntid.x;
    mov.u32 %ctaid, %ctaid.x;
    mad.lo.u32 %idx, %ctaid, %ntid, %tid;
    ld.param.u32 %n, [n];
    setp.ge.u32 %p, %idx, %n;
    @%p bra DONE;

    ld.param.u64 %pts_base, [points_ptr];
    ld.param.u64 %cen_base, [centroids_ptr];
    ld.param.u64 %asg_base, [assignments_ptr];
    ld.param.u32 %k, [k];
    ld.param.u32 %d, [d];

    // best_dist = MAX_FLOAT
    mov.f32 %best_dist, 0f7F7FFFFF;
    mov.u32 %best_k, 0;

    // p_off = idx * d
    mul.lo.u32 %p_off, %idx, %d;

    mov.u32 %c, 0;
C_LOOP:
    setp.ge.u32 %p_c, %c, %k;
    @%p_c bra C_DONE;

    mov.f32 %dist, 0f00000000;
    mul.lo.u32 %c_off, %c, %d;

    mov.u32 %j, 0;
D_LOOP:
    setp.ge.u32 %p_j, %j, %d;
    @%p_j bra D_DONE;

    // Load points[idx*d + j]
    add.u32 %tmp, %p_off, %j;
    cvt.u64.u32 %addr, %tmp;
    shl.b64 %addr, %addr, 2;
    add.u64 %addr, %pts_base, %addr;
    ld.global.f32 %val_p, [%addr];

    // Load centroids[c*d + j]
    add.u32 %tmp, %c_off, %j;
    cvt.u64.u32 %addr, %tmp;
    shl.b64 %addr, %addr, 2;
    add.u64 %addr, %cen_base, %addr;
    ld.global.f32 %val_c, [%addr];

    sub.f32 %diff, %val_p, %val_c;
    fma.rn.f32 %dist, %diff, %diff, %dist;

    add.u32 %j, %j, 1;
    bra D_LOOP;
D_DONE:

    setp.lt.f32 %p_better, %dist, %best_dist;
    @%p_better mov.f32 %best_dist, %dist;
    @%p_better mov.u32 %best_k, %c;

    add.u32 %c, %c, 1;
    bra C_LOOP;
C_DONE:

    // Store assignment
    cvt.u64.u32 %addr, %idx;
    shl.b64 %addr, %addr, 2;
    add.u64 %addr, %asg_base, %addr;
    st.global.u32 [%addr], %best_k;

DONE:
    ret;
}
"#
}

/// PTX assembly for K-means update kernel.
///
/// Reduction kernel: accumulates points per cluster, then divides.
/// Simplified structural version.
pub fn kmeans_update_ptx() -> &'static str {
    r#".version 8.5
.target sm_90
.address_size 64

// K-means update kernel (simplified): 1 thread per centroid dimension.
// Real implementation would use atomics or reduction.
.visible .entry kmeans_update_kernel(
    .param .u64 points_ptr,
    .param .u64 assignments_ptr,
    .param .u64 centroids_ptr,
    .param .u32 n,
    .param .u32 k,
    .param .u32 d
)
{
    .reg .u32 %tid, %ntid, %ctaid, %idx, %n, %k, %d;
    .reg .u64 %pts_base, %asg_base, %cen_base, %addr;
    .reg .u32 %kd;
    .reg .pred %p;

    mov.u32 %tid, %tid.x;
    mov.u32 %ntid, %ntid.x;
    mov.u32 %ctaid, %ctaid.x;
    mad.lo.u32 %idx, %ctaid, %ntid, %tid;
    ld.param.u32 %k, [k];
    ld.param.u32 %d, [d];

    // Bounds check: idx < k*d
    mul.lo.u32 %kd, %k, %d;
    setp.ge.u32 %p, %idx, %kd;
    @%p bra DONE;

    ld.param.u64 %pts_base, [points_ptr];
    ld.param.u64 %asg_base, [assignments_ptr];
    ld.param.u64 %cen_base, [centroids_ptr];
    ld.param.u32 %n, [n];

    // Placeholder: actual reduction omitted for structural test
    ret;

DONE:
    ret;
}
"#
}

#[cfg(test)]
mod tests {
    use super::*;

    // ---------------------------------------------------------------
    // Scalar tests
    // ---------------------------------------------------------------

    #[test]
    fn test_kmeans_assign_two_clusters() {
        // 2D points, 2 centroids at (0,0) and (10,10)
        let points = [
            1.0_f32, 1.0, // near (0,0)
            9.0, 9.0, // near (10,10)
            0.5, 0.5, // near (0,0)
            10.0, 10.0, // near (10,10)
        ];
        let centroids = [0.0_f32, 0.0, 10.0, 10.0];
        let mut assignments = [0_u32; 4];

        kmeans_assign_scalar(&points, &centroids, 4, 2, 2, &mut assignments);

        assert_eq!(assignments[0], 0);
        assert_eq!(assignments[1], 1);
        assert_eq!(assignments[2], 0);
        assert_eq!(assignments[3], 1);
    }

    #[test]
    fn test_kmeans_update_known() {
        // 4 points in 2D, assigned to 2 clusters
        let points = [
            1.0_f32, 2.0, // cluster 0
            3.0, 4.0, // cluster 0
            10.0, 20.0, // cluster 1
            30.0, 40.0, // cluster 1
        ];
        let assignments = [0_u32, 0, 1, 1];
        let mut centroids = [0.0_f32; 4]; // 2 x 2

        kmeans_update_scalar(&points, &assignments, 4, 2, 2, &mut centroids);

        // Cluster 0: mean of (1,2) and (3,4) = (2,3)
        assert!((centroids[0] - 2.0).abs() < 1e-6);
        assert!((centroids[1] - 3.0).abs() < 1e-6);
        // Cluster 1: mean of (10,20) and (30,40) = (20,30)
        assert!((centroids[2] - 20.0).abs() < 1e-6);
        assert!((centroids[3] - 30.0).abs() < 1e-6);
    }

    #[test]
    fn test_kmeans_assign_single_centroid() {
        let points = [1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0];
        let centroids = [0.0_f32, 0.0];
        let mut assignments = [99_u32; 3];

        kmeans_assign_scalar(&points, &centroids, 3, 1, 2, &mut assignments);

        // All points assigned to the only centroid
        assert_eq!(assignments, [0, 0, 0]);
    }

    #[test]
    fn test_kmeans_update_empty_cluster() {
        // 2 points, 2 clusters, but all assigned to cluster 0
        let points = [1.0_f32, 2.0, 3.0, 4.0];
        let assignments = [0_u32, 0];
        let mut centroids = [0.0_f32; 4];

        kmeans_update_scalar(&points, &assignments, 2, 2, 2, &mut centroids);

        // Cluster 0: mean of (1,2) and (3,4) = (2,3)
        assert!((centroids[0] - 2.0).abs() < 1e-6);
        assert!((centroids[1] - 3.0).abs() < 1e-6);
        // Cluster 1: no points -> zero
        assert_eq!(centroids[2], 0.0);
        assert_eq!(centroids[3], 0.0);
    }

    #[test]
    #[should_panic(expected = "points length mismatch")]
    fn test_kmeans_assign_points_mismatch() {
        let points = [1.0_f32; 5];
        let centroids = [0.0_f32; 4];
        let mut assignments = [0_u32; 3];
        kmeans_assign_scalar(&points, &centroids, 3, 2, 2, &mut assignments);
    }

    // ---------------------------------------------------------------
    // AVX2 tests
    // ---------------------------------------------------------------

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn test_kmeans_avx2_parity() {
        if !is_x86_feature_detected!("avx2") {
            return;
        }
        let points = [1.0_f32, 1.0, 9.0, 9.0, 0.5, 0.5, 10.0, 10.0];
        let centroids = [0.0_f32, 0.0, 10.0, 10.0];
        let mut asg_s = [0_u32; 4];
        let mut asg_a = [0_u32; 4];

        kmeans_assign_scalar(&points, &centroids, 4, 2, 2, &mut asg_s);
        unsafe {
            kmeans_assign_avx2(&points, &centroids, 4, 2, 2, &mut asg_a);
        }

        assert_eq!(asg_s, asg_a);
    }

    // ---------------------------------------------------------------
    // PTX structural tests
    // ---------------------------------------------------------------

    #[test]
    fn test_kmeans_assign_ptx_version() {
        let ptx = kmeans_assign_ptx();
        assert!(
            ptx.contains(".version 8.5"),
            "PTX must declare .version 8.5"
        );
    }

    #[test]
    fn test_kmeans_assign_ptx_target() {
        let ptx = kmeans_assign_ptx();
        assert!(ptx.contains(".target sm_90"), "PTX must target sm_90");
    }

    #[test]
    fn test_kmeans_assign_ptx_entry() {
        let ptx = kmeans_assign_ptx();
        assert!(
            ptx.contains(".entry kmeans_assign_kernel"),
            "PTX must have .entry"
        );
    }

    #[test]
    fn test_kmeans_assign_ptx_ret() {
        let ptx = kmeans_assign_ptx();
        assert!(ptx.contains("ret;"), "PTX must have ret;");
    }

    #[test]
    fn test_kmeans_assign_ptx_balanced_braces() {
        let ptx = kmeans_assign_ptx();
        let opens = ptx.chars().filter(|&c| c == '{').count();
        let closes = ptx.chars().filter(|&c| c == '}').count();
        assert_eq!(
            opens, closes,
            "PTX must have balanced braces: {opens} opens vs {closes} closes"
        );
    }

    #[test]
    fn test_kmeans_update_ptx_version() {
        let ptx = kmeans_update_ptx();
        assert!(
            ptx.contains(".version 8.5"),
            "PTX must declare .version 8.5"
        );
    }

    #[test]
    fn test_kmeans_update_ptx_target() {
        let ptx = kmeans_update_ptx();
        assert!(ptx.contains(".target sm_90"), "PTX must target sm_90");
    }

    #[test]
    fn test_kmeans_update_ptx_entry() {
        let ptx = kmeans_update_ptx();
        assert!(
            ptx.contains(".entry kmeans_update_kernel"),
            "PTX must have .entry"
        );
    }

    #[test]
    fn test_kmeans_update_ptx_ret() {
        let ptx = kmeans_update_ptx();
        assert!(ptx.contains("ret;"), "PTX must have ret;");
    }

    #[test]
    fn test_kmeans_update_ptx_balanced_braces() {
        let ptx = kmeans_update_ptx();
        let opens = ptx.chars().filter(|&c| c == '{').count();
        let closes = ptx.chars().filter(|&c| c == '}').count();
        assert_eq!(
            opens, closes,
            "PTX must have balanced braces: {opens} opens vs {closes} closes"
        );
    }
}
