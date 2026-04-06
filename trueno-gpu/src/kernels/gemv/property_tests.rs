use super::*;
use crate::kernels::Kernel;
use proptest::prelude::*;

proptest! {
    #[test]
    fn gemv_always_valid(k in 32u32..8192, n in 32u32..65536) {
        let kernel = GemvKernel::new(k, n);
        let ptx = kernel.emit_ptx();

        prop_assert!(ptx.contains(".version"), "Missing PTX version");
        prop_assert!(ptx.contains(".entry"), "Missing entry point");
        prop_assert!(ptx.contains("gemv"), "Missing kernel name");
    }
}
