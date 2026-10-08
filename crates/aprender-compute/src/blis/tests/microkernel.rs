use super::super::microkernels::*;
use super::super::*;

// ========================================================================
// Phase 2: Microkernel Tests
// ========================================================================

#[test]
fn test_microkernel_scalar_single_k() {
    // MR=8, NR=6, K=1
    let a = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]; // 8x1
    let b = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]; // 1x6
    let mut c = vec![0.0; MR * NR]; // 8x6 column-major

    microkernel_scalar(1, &a, &b, &mut c, MR);

    // c[j,i] = a[i] * b[j]
    for j in 0..NR {
        for i in 0..MR {
            let expected = a[i] * b[j];
            assert!(
                (c[j * MR + i] - expected).abs() < 1e-6,
                "Mismatch at ({}, {}): {} vs {}",
                i,
                j,
                c[j * MR + i],
                expected
            );
        }
    }
}

#[test]
fn test_microkernel_scalar_accumulation() {
    let a = vec![1.0; MR * 4]; // 8x4
    let b = vec![1.0; 4 * NR]; // 4x6
    let mut c = vec![0.0; MR * NR];

    microkernel_scalar(4, &a, &b, &mut c, MR);

    // Each output should be 4.0 (sum of 4 ones)
    for val in &c {
        assert!((val - 4.0).abs() < 1e-6);
    }
}

#[test]
#[cfg(target_arch = "x86_64")]
fn test_microkernel_avx2_matches_scalar() {
    if !is_x86_feature_detected!("avx2") || !is_x86_feature_detected!("fma") {
        return;
    }

    let k = 64;
    let a: Vec<f32> = (0..MR * k).map(|i| (i as f32) * 0.1).collect();
    let b: Vec<f32> = (0..k * NR).map(|i| (i as f32) * 0.01).collect();

    let mut c_scalar = vec![0.0; MR * NR];
    let mut c_avx2 = vec![0.0; MR * NR];

    microkernel_scalar(k, &a, &b, &mut c_scalar, MR);

    // SAFETY: test-only usage with controlled inputs
    unsafe {
        microkernel_8x6_avx2(k, a.as_ptr(), b.as_ptr(), c_avx2.as_mut_ptr(), MR);
    }

    for i in 0..MR * NR {
        let diff = (c_scalar[i] - c_avx2[i]).abs();
        let rel_diff = diff / c_scalar[i].abs().max(1e-10);
        assert!(
            rel_diff < 1e-5,
            "Mismatch at {}: scalar={}, avx2={}, rel_diff={}",
            i,
            c_scalar[i],
            c_avx2[i],
            rel_diff
        );
    }
}

// ========================================================================
// AVX2 ASM Microkernel Tests (coverage for microkernel_8x6_avx2_asm)
// ========================================================================

#[test]
#[cfg(target_arch = "x86_64")]
fn test_microkernel_avx2_asm_matches_scalar() {
    if !is_x86_feature_detected!("avx2") || !is_x86_feature_detected!("fma") {
        return;
    }

    let k = 64;
    let a: Vec<f32> = (0..MR * k).map(|i| (i as f32) * 0.1).collect();
    let b: Vec<f32> = (0..k * NR).map(|i| (i as f32) * 0.01).collect();

    let mut c_scalar = vec![0.0; MR * NR];
    let mut c_asm = vec![0.0; MR * NR];

    microkernel_scalar(k, &a, &b, &mut c_scalar, MR);

    // SAFETY: test-only usage with controlled inputs
    unsafe {
        microkernel_8x6_avx2_asm(k, a.as_ptr(), b.as_ptr(), c_asm.as_mut_ptr(), MR);
    }

    for i in 0..MR * NR {
        let diff = (c_scalar[i] - c_asm[i]).abs();
        let rel_diff = diff / c_scalar[i].abs().max(1e-10);
        assert!(
            rel_diff < 1e-4,
            "avx2_asm mismatch at {}: scalar={}, asm={}, rel_diff={}",
            i,
            c_scalar[i],
            c_asm[i],
            rel_diff
        );
    }
}

#[test]
#[cfg(target_arch = "x86_64")]
fn test_microkernel_avx2_asm_k_less_than_4() {
    if !is_x86_feature_detected!("avx2") || !is_x86_feature_detected!("fma") {
        return;
    }

    for k in [1, 2, 3] {
        let a: Vec<f32> = (0..MR * k).map(|i| (i as f32 + 1.0) * 0.5).collect();
        let b: Vec<f32> = (0..k * NR).map(|i| (i as f32 + 1.0) * 0.3).collect();

        let mut c_scalar = vec![0.0; MR * NR];
        let mut c_asm = vec![0.0; MR * NR];

        microkernel_scalar(k, &a, &b, &mut c_scalar, MR);

        // SAFETY: test-only usage with controlled inputs
        unsafe {
            microkernel_8x6_avx2_asm(k, a.as_ptr(), b.as_ptr(), c_asm.as_mut_ptr(), MR);
        }

        for i in 0..MR * NR {
            let diff = (c_scalar[i] - c_asm[i]).abs();
            assert!(
                diff < 1e-4,
                "avx2_asm k={} mismatch at {}: scalar={}, asm={}, diff={}",
                k,
                i,
                c_scalar[i],
                c_asm[i],
                diff
            );
        }
    }
}

#[test]
#[cfg(target_arch = "x86_64")]
fn test_microkernel_avx2_asm_k_with_remainder() {
    if !is_x86_feature_detected!("avx2") || !is_x86_feature_detected!("fma") {
        return;
    }

    // k=5 → 1 unrolled iteration + 1 remainder
    // k=7 → 1 unrolled iteration + 3 remainder
    for k in [5, 6, 7, 9, 13, 17] {
        let a: Vec<f32> = (0..MR * k).map(|i| ((i % 10) as f32) * 0.1).collect();
        let b: Vec<f32> = (0..k * NR).map(|i| ((i % 7) as f32) * 0.2).collect();

        let mut c_scalar = vec![0.0; MR * NR];
        let mut c_asm = vec![0.0; MR * NR];

        microkernel_scalar(k, &a, &b, &mut c_scalar, MR);

        // SAFETY: test-only usage with controlled inputs
        unsafe {
            microkernel_8x6_avx2_asm(k, a.as_ptr(), b.as_ptr(), c_asm.as_mut_ptr(), MR);
        }

        for i in 0..MR * NR {
            let diff = (c_scalar[i] - c_asm[i]).abs();
            let rel_diff = diff / c_scalar[i].abs().max(1e-10);
            assert!(
                rel_diff < 1e-4,
                "avx2_asm k={} mismatch at {}: scalar={}, asm={}, rel_diff={}",
                k,
                i,
                c_scalar[i],
                c_asm[i],
                rel_diff
            );
        }
    }
}

#[test]
#[cfg(target_arch = "x86_64")]
fn test_microkernel_avx2_asm_k_exact_multiple_of_4() {
    if !is_x86_feature_detected!("avx2") || !is_x86_feature_detected!("fma") {
        return;
    }

    for k in [4, 8, 16, 32] {
        let a: Vec<f32> = (0..MR * k).map(|i| (i as f32) * 0.05).collect();
        let b: Vec<f32> = (0..k * NR).map(|i| (i as f32) * 0.02).collect();

        let mut c_scalar = vec![0.0; MR * NR];
        let mut c_asm = vec![0.0; MR * NR];

        microkernel_scalar(k, &a, &b, &mut c_scalar, MR);

        // SAFETY: test-only usage with controlled inputs
        unsafe {
            microkernel_8x6_avx2_asm(k, a.as_ptr(), b.as_ptr(), c_asm.as_mut_ptr(), MR);
        }

        for i in 0..MR * NR {
            let diff = (c_scalar[i] - c_asm[i]).abs();
            let rel_diff = diff / c_scalar[i].abs().max(1e-10);
            assert!(
                rel_diff < 1e-4,
                "avx2_asm k={} mismatch at {}: scalar={}, asm={}",
                k,
                i,
                c_scalar[i],
                c_asm[i]
            );
        }
    }
}

// ========================================================================
// True ASM Microkernel Tests (coverage for microkernel_8x6_true_asm)
// ========================================================================

#[test]
#[cfg(target_arch = "x86_64")]
fn test_microkernel_true_asm_matches_scalar() {
    if !is_x86_feature_detected!("avx2") || !is_x86_feature_detected!("fma") {
        return;
    }

    let k = 64;
    let a: Vec<f32> = (0..MR * k).map(|i| (i as f32) * 0.1).collect();
    let b: Vec<f32> = (0..k * NR).map(|i| (i as f32) * 0.01).collect();

    let mut c_scalar = vec![0.0; MR * NR];
    let mut c_asm = vec![0.0; MR * NR];

    microkernel_scalar(k, &a, &b, &mut c_scalar, MR);

    // SAFETY: test-only usage with controlled inputs
    unsafe {
        microkernel_8x6_true_asm(k, a.as_ptr(), b.as_ptr(), c_asm.as_mut_ptr(), MR);
    }

    for i in 0..MR * NR {
        let diff = (c_scalar[i] - c_asm[i]).abs();
        let rel_diff = diff / c_scalar[i].abs().max(1e-10);
        assert!(
            rel_diff < 1e-4,
            "true_asm mismatch at {}: scalar={}, asm={}, rel_diff={}",
            i,
            c_scalar[i],
            c_asm[i],
            rel_diff
        );
    }
}

#[test]
#[cfg(target_arch = "x86_64")]
fn test_microkernel_true_asm_k_less_than_4() {
    if !is_x86_feature_detected!("avx2") || !is_x86_feature_detected!("fma") {
        return;
    }

    for k in [1, 2, 3] {
        let a: Vec<f32> = (0..MR * k).map(|i| (i as f32 + 1.0) * 0.5).collect();
        let b: Vec<f32> = (0..k * NR).map(|i| (i as f32 + 1.0) * 0.3).collect();

        let mut c_scalar = vec![0.0; MR * NR];
        let mut c_asm = vec![0.0; MR * NR];

        microkernel_scalar(k, &a, &b, &mut c_scalar, MR);

        // SAFETY: test-only usage with controlled inputs
        unsafe {
            microkernel_8x6_true_asm(k, a.as_ptr(), b.as_ptr(), c_asm.as_mut_ptr(), MR);
        }

        for i in 0..MR * NR {
            let diff = (c_scalar[i] - c_asm[i]).abs();
            assert!(
                diff < 1e-4,
                "true_asm k={} mismatch at {}: scalar={}, asm={}, diff={}",
                k,
                i,
                c_scalar[i],
                c_asm[i],
                diff
            );
        }
    }
}

#[test]
#[cfg(target_arch = "x86_64")]
fn test_microkernel_true_asm_k_with_remainder() {
    if !is_x86_feature_detected!("avx2") || !is_x86_feature_detected!("fma") {
        return;
    }

    // Test k values that exercise both main loop and remainder
    for k in [5, 6, 7, 9, 11, 15] {
        let a: Vec<f32> = (0..MR * k).map(|i| ((i % 10) as f32) * 0.1).collect();
        let b: Vec<f32> = (0..k * NR).map(|i| ((i % 7) as f32) * 0.2).collect();

        let mut c_scalar = vec![0.0; MR * NR];
        let mut c_asm = vec![0.0; MR * NR];

        microkernel_scalar(k, &a, &b, &mut c_scalar, MR);

        // SAFETY: test-only usage with controlled inputs
        unsafe {
            microkernel_8x6_true_asm(k, a.as_ptr(), b.as_ptr(), c_asm.as_mut_ptr(), MR);
        }

        for i in 0..MR * NR {
            let diff = (c_scalar[i] - c_asm[i]).abs();
            let rel_diff = diff / c_scalar[i].abs().max(1e-10);
            assert!(
                rel_diff < 1e-4,
                "true_asm k={} mismatch at {}: scalar={}, asm={}, rel_diff={}",
                k,
                i,
                c_scalar[i],
                c_asm[i],
                rel_diff
            );
        }
    }
}

#[test]
#[cfg(target_arch = "x86_64")]
fn test_microkernel_true_asm_k_exact_multiple_of_4() {
    if !is_x86_feature_detected!("avx2") || !is_x86_feature_detected!("fma") {
        return;
    }

    for k in [4, 8, 12, 16, 32, 64] {
        let a: Vec<f32> = (0..MR * k).map(|i| (i as f32) * 0.05).collect();
        let b: Vec<f32> = (0..k * NR).map(|i| (i as f32) * 0.02).collect();

        let mut c_scalar = vec![0.0; MR * NR];
        let mut c_asm = vec![0.0; MR * NR];

        microkernel_scalar(k, &a, &b, &mut c_scalar, MR);

        // SAFETY: test-only usage with controlled inputs
        unsafe {
            microkernel_8x6_true_asm(k, a.as_ptr(), b.as_ptr(), c_asm.as_mut_ptr(), MR);
        }

        for i in 0..MR * NR {
            let diff = (c_scalar[i] - c_asm[i]).abs();
            let rel_diff = diff / c_scalar[i].abs().max(1e-10);
            assert!(
                rel_diff < 1e-4,
                "true_asm k={} mismatch at {}: scalar={}, asm={}",
                k,
                i,
                c_scalar[i],
                c_asm[i]
            );
        }
    }
}

#[test]
#[cfg(target_arch = "x86_64")]
fn test_microkernel_true_asm_accumulates_into_c() {
    if !is_x86_feature_detected!("avx2") || !is_x86_feature_detected!("fma") {
        return;
    }

    let k = 8;
    let a: Vec<f32> = vec![1.0; MR * k];
    let b: Vec<f32> = vec![1.0; k * NR];

    // Pre-fill C with non-zero values
    let mut c = vec![10.0f32; MR * NR];

    // SAFETY: test-only usage with controlled inputs
    unsafe {
        microkernel_8x6_true_asm(k, a.as_ptr(), b.as_ptr(), c.as_mut_ptr(), MR);
    }

    // Each output should be 10.0 + k = 18.0
    for (i, val) in c.iter().enumerate() {
        assert!(
            (*val - 18.0).abs() < 1e-3,
            "true_asm accumulation at {}: expected 18.0, got {}",
            i,
            val
        );
    }
}

// ========================================================================
// FALSIFY-BLIS-ASM-OOB: the asm kernel must not read past its A or B panel
// ========================================================================
//
// The last ir panel of a packed A buffer ends exactly at the end of its
// allocation. A kernel that reads even one byte past the panel can cross
// into an unmapped page and fault, and only when the allocator happens to
// put the buffer there. These tests put the panel end on a PROT_NONE page,
// so any read past it faults on every run instead of one run in many.

#[cfg(all(target_arch = "x86_64", target_os = "linux"))]
mod guard_page {
    use std::ffi::c_void;

    const PROT_NONE: i32 = 0;
    const PROT_READ: i32 = 1;
    const PROT_WRITE: i32 = 2;
    const MAP_PRIVATE: i32 = 0x02;
    const MAP_ANONYMOUS: i32 = 0x20;
    const SC_PAGESIZE: i32 = 30;

    extern "C" {
        fn mmap(
            addr: *mut c_void,
            len: usize,
            prot: i32,
            flags: i32,
            fd: i32,
            off: i64,
        ) -> *mut c_void;
        fn mprotect(addr: *mut c_void, len: usize, prot: i32) -> i32;
        fn munmap(addr: *mut c_void, len: usize) -> i32;
        fn sysconf(name: i32) -> i64;
    }

    /// `len` f32s whose last element is the last readable byte before a
    /// PROT_NONE page.
    pub struct GuardedF32 {
        base: *mut c_void,
        map_len: usize,
        data: *mut f32,
        len: usize,
    }

    impl GuardedF32 {
        pub fn new(values: &[f32]) -> Self {
            // SAFETY: plain libc calls; every result is checked before use.
            unsafe {
                let page = usize::try_from(sysconf(SC_PAGESIZE)).expect("page size");
                let bytes = values.len() * 4;
                let data_pages = bytes.div_ceil(page).max(1);
                let map_len = (data_pages + 1) * page;
                let base = mmap(
                    std::ptr::null_mut(),
                    map_len,
                    PROT_READ | PROT_WRITE,
                    MAP_PRIVATE | MAP_ANONYMOUS,
                    -1,
                    0,
                );
                assert!(base as isize != -1, "mmap failed");
                let guard = base.cast::<u8>().add(data_pages * page);
                assert_eq!(mprotect(guard.cast(), page, PROT_NONE), 0, "mprotect failed");
                let data = guard.sub(bytes).cast::<f32>();
                std::ptr::copy_nonoverlapping(values.as_ptr(), data, values.len());
                Self { base, map_len, data, len: values.len() }
            }
        }

        pub fn as_ptr(&self) -> *const f32 {
            self.data
        }

        pub fn len(&self) -> usize {
            self.len
        }
    }

    impl Drop for GuardedF32 {
        fn drop(&mut self) {
            // SAFETY: base/map_len are exactly what mmap returned.
            unsafe {
                munmap(self.base, self.map_len);
            }
        }
    }
}

#[test]
#[cfg(all(target_arch = "x86_64", target_os = "linux"))]
fn falsify_blis_asm_oob_001_true_asm_stays_inside_panels() {
    if !is_x86_feature_detected!("avx2") || !is_x86_feature_detected!("fma") {
        return;
    }

    // Multiples of 4 (the unroll), each remainder, and the KC block size.
    for k in [4usize, 5, 6, 7, 8, 12, 64, 255, 256] {
        let a_vals: Vec<f32> = (0..MR * k).map(|i| ((i % 17) as f32) * 0.25 - 2.0).collect();
        let b_vals: Vec<f32> = (0..k * NR).map(|i| ((i % 13) as f32) * 0.5 - 3.0).collect();
        let a = guard_page::GuardedF32::new(&a_vals);
        let b = guard_page::GuardedF32::new(&b_vals);
        assert_eq!(a.len(), MR * k);
        assert_eq!(b.len(), k * NR);

        let mut c_ref = vec![1.0f32; MR * NR];
        let mut c_asm = vec![1.0f32; MR * NR];
        microkernel_scalar(k, &a_vals, &b_vals, &mut c_ref, MR);

        // SAFETY: A holds MR*k and B holds k*NR f32s; C holds MR*NR with ldc = MR.
        unsafe {
            microkernel_8x6_true_asm(k, a.as_ptr(), b.as_ptr(), c_asm.as_mut_ptr(), MR);
        }

        for i in 0..MR * NR {
            let tol = 1e-4 * c_ref[i].abs().max(1.0);
            assert!(
                (c_ref[i] - c_asm[i]).abs() <= tol,
                "FALSIFY-BLIS-ASM-OOB-001 k={k} i={i}: scalar {} vs asm {}",
                c_ref[i],
                c_asm[i]
            );
        }
    }
}
