use super::*;

#[test]
fn test_cuda_error_string_success() {
    assert_eq!(cuda_error_string(CUDA_SUCCESS), "CUDA_SUCCESS");
}

#[test]
fn test_cuda_error_string_oom() {
    assert_eq!(
        cuda_error_string(CUDA_ERROR_OUT_OF_MEMORY),
        "CUDA_ERROR_OUT_OF_MEMORY"
    );
}

#[test]
fn test_cuda_error_string_unknown() {
    assert_eq!(cuda_error_string(99999), "CUDA_ERROR_UNKNOWN");
}

#[test]
fn test_cuda_constants() {
    // Verify constants match CUDA header
    assert_eq!(CUDA_SUCCESS, 0);
    assert_eq!(CUDA_ERROR_NO_DEVICE, 100);
    assert_eq!(CUDA_ERROR_INVALID_PTX, 218);
}

#[test]
fn test_custream_flags() {
    assert_eq!(CU_STREAM_DEFAULT, 0);
    assert_eq!(CU_STREAM_NON_BLOCKING, 1);
}

#[test]
#[cfg(not(feature = "cuda"))]
fn test_driver_load_without_feature() {
    assert!(CudaDriver::load().is_none());
}

#[test]
#[cfg(not(feature = "cuda"))]
fn test_check_without_feature() {
    let result = CudaDriver::check(CUDA_SUCCESS);
    assert!(result.is_err());
}

#[test]
fn test_all_error_strings() {
    // Test all known error codes have proper strings
    assert_eq!(
        cuda_error_string(CUDA_ERROR_INVALID_VALUE),
        "CUDA_ERROR_INVALID_VALUE"
    );
    assert_eq!(
        cuda_error_string(CUDA_ERROR_NOT_INITIALIZED),
        "CUDA_ERROR_NOT_INITIALIZED"
    );
    assert_eq!(
        cuda_error_string(CUDA_ERROR_DEINITIALIZED),
        "CUDA_ERROR_DEINITIALIZED"
    );
    assert_eq!(
        cuda_error_string(CUDA_ERROR_INVALID_DEVICE),
        "CUDA_ERROR_INVALID_DEVICE"
    );
    assert_eq!(
        cuda_error_string(CUDA_ERROR_NOT_FOUND),
        "CUDA_ERROR_NOT_FOUND"
    );
}

#[test]
fn test_error_codes_are_distinct() {
    // All error codes should be distinct
    let codes = [
        CUDA_SUCCESS,
        CUDA_ERROR_INVALID_VALUE,
        CUDA_ERROR_OUT_OF_MEMORY,
        CUDA_ERROR_NOT_INITIALIZED,
        CUDA_ERROR_DEINITIALIZED,
        CUDA_ERROR_NO_DEVICE,
        CUDA_ERROR_INVALID_DEVICE,
        CUDA_ERROR_INVALID_PTX,
        CUDA_ERROR_NOT_FOUND,
    ];
    for i in 0..codes.len() {
        for j in (i + 1)..codes.len() {
            assert_ne!(
                codes[i], codes[j],
                "Error codes at {} and {} are equal",
                i, j
            );
        }
    }
}

#[test]
fn test_type_sizes() {
    // Verify FFI types have expected sizes
    assert_eq!(std::mem::size_of::<CUresult>(), std::mem::size_of::<i32>());
    assert_eq!(std::mem::size_of::<CUdevice>(), std::mem::size_of::<i32>());
    assert_eq!(
        std::mem::size_of::<CUdeviceptr>(),
        std::mem::size_of::<u64>()
    );
    // Opaque pointers are pointer-sized
    assert_eq!(
        std::mem::size_of::<CUcontext>(),
        std::mem::size_of::<*mut ()>()
    );
    assert_eq!(
        std::mem::size_of::<CUmodule>(),
        std::mem::size_of::<*mut ()>()
    );
    assert_eq!(
        std::mem::size_of::<CUfunction>(),
        std::mem::size_of::<*mut ()>()
    );
    assert_eq!(
        std::mem::size_of::<CUstream>(),
        std::mem::size_of::<*mut ()>()
    );
}

#[test]
fn test_null_pointers() {
    use std::ptr;
    // Null pointers are valid for CUDA types
    let ctx: CUcontext = ptr::null_mut();
    let module: CUmodule = ptr::null_mut();
    let func: CUfunction = ptr::null_mut();
    let stream: CUstream = ptr::null_mut();

    assert!(ctx.is_null());
    assert!(module.is_null());
    assert!(func.is_null());
    assert!(stream.is_null());
}
