use super::*;
use proptest::prelude::*;

proptest! {
    /// Falsifiable claim A-009: Backend selection is deterministic
    #[test]
    fn prop_backend_selection_deterministic(size in 0usize..1_000_000) {
        let selector = BackendSelector::default();

        let result1 = selector.select_for_size(size, true);
        let result2 = selector.select_for_size(size, true);

        prop_assert_eq!(result1, result2);
    }

    /// Falsifiable claim: compute_seed is deterministic
    #[test]
    fn prop_compute_seed_deterministic(
        backend_idx in 0u8..8,
        size in 0usize..1_000_000,
        cycle in 0u32..100,
        master_seed in any::<u64>()
    ) {
        let backend = match backend_idx % 8 {
            0 => Backend::Scalar,
            1 => Backend::SSE2,
            2 => Backend::AVX,
            3 => Backend::AVX2,
            4 => Backend::AVX512,
            5 => Backend::NEON,
            6 => Backend::WasmSIMD,
            7 => Backend::GPU,
            // SAFETY: unreachable — modulo 8 produces 0..=7
            _ => unreachable!(),
        };

        let seed1 = compute_seed(backend, size, cycle, master_seed);
        let seed2 = compute_seed(backend, size, cycle, master_seed);

        prop_assert_eq!(seed1, seed2);
    }
}
