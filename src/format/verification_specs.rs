//! Formal verification specifications for APR format operations.
//!
//! These specifications use Verus-compatible `requires`/`ensures` contracts
//! to express preconditions and postconditions for critical format operations.
//! They serve as machine-checkable documentation of safety invariants.

/// Validate that a tensor shape is well-formed for APR format.
///
/// # Specification
///
/// #[requires(dims.len() <= 4)]
/// #[ensures(result == true ==> dims.iter().all(|&d| d > 0))]
/// #[ensures(result == true ==> dims.iter().map(|&d| d as u64).product::<u64>() <= MAX_ELEMENTS)]
pub fn is_valid_tensor_shape(dims: &[usize]) -> bool {
    const MAX_ELEMENTS: u64 = 1 << 40; // 1 trillion elements max
    if dims.is_empty() || dims.len() > 4 {
        return false;
    }
    if dims.iter().any(|&d| d == 0) {
        return false;
    }
    let total: u64 = dims.iter().map(|&d| d as u64).product();
    total <= MAX_ELEMENTS
}

/// Validate APR header magic bytes.
///
/// #[requires(header.len() >= 4)]
/// #[ensures(result.is_some() ==> result.unwrap() == 1 || result.unwrap() == 2)]
pub fn validate_magic(header: &[u8]) -> Option<u8> {
    if header.len() < 4 {
        return None;
    }
    match &header[..4] {
        b"APR\0" => Some(2),
        b"APRN" => Some(1),
        _ => None,
    }
}

/// Compute aligned offset for tensor data placement.
///
/// #[requires(alignment > 0 && alignment.is_power_of_two())]
/// #[ensures(result >= offset)]
/// #[ensures(result % alignment as u64 == 0)]
/// #[ensures(result - offset < alignment as u64)]
pub fn align_offset(offset: u64, alignment: usize) -> u64 {
    debug_assert!(alignment > 0 && alignment.is_power_of_two());
    let align = alignment as u64;
    let remainder = offset % align;
    if remainder == 0 {
        offset
    } else {
        offset + (align - remainder)
    }
}

/// Validate GGUF dimension swap for row-major import.
///
/// #[requires(gguf_shape.len() == 2)]
/// #[requires(gguf_shape[0] > 0 && gguf_shape[1] > 0)]
/// #[ensures(result[0] == gguf_shape[1])]
/// #[ensures(result[1] == gguf_shape[0])]
/// #[invariant(result[0] * result[1] == gguf_shape[0] * gguf_shape[1])]
pub fn swap_dims_for_row_major(gguf_shape: [usize; 2]) -> [usize; 2] {
    [gguf_shape[1], gguf_shape[0]]
}

/// Validate quantization block size for K-quant formats.
///
/// #[requires(block_size > 0)]
/// #[ensures(result == true ==> block_size.is_power_of_two())]
/// #[ensures(result == true ==> block_size >= 32 && block_size <= 256)]
pub fn is_valid_quant_block_size(block_size: usize) -> bool {
    block_size.is_power_of_two() && block_size >= 32 && block_size <= 256
}

/// Compute the number of quantization blocks for a tensor.
///
/// #[requires(elements > 0 && block_size > 0)]
/// #[requires(elements % block_size == 0)]
/// #[ensures(result * block_size == elements)]
pub fn quant_block_count(elements: usize, block_size: usize) -> usize {
    debug_assert!(block_size > 0 && elements % block_size == 0);
    elements / block_size
}

/// Validate metadata key length per APR spec.
///
/// #[requires(!key.is_empty())]
/// #[ensures(result == true ==> key.len() <= 256)]
/// #[ensures(result == true ==> key.is_ascii())]
pub fn is_valid_metadata_key(key: &str) -> bool {
    !key.is_empty() && key.len() <= 256 && key.is_ascii()
}

/// Validate tensor name per APR naming convention.
///
/// #[requires(!name.is_empty())]
/// #[ensures(result == true ==> !name.contains(' '))]
/// #[ensures(result == true ==> name.len() <= 512)]
/// #[decreases(name.len())]
pub fn is_valid_tensor_name(name: &str) -> bool {
    !name.is_empty() && name.len() <= 512 && !name.contains(' ') && name.is_ascii()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_valid_shapes() {
        assert!(is_valid_tensor_shape(&[768, 768]));
        assert!(is_valid_tensor_shape(&[32000, 4096]));
        assert!(!is_valid_tensor_shape(&[]));
        assert!(!is_valid_tensor_shape(&[0, 768]));
        assert!(!is_valid_tensor_shape(&[1, 2, 3, 4, 5])); // >4 dims
    }

    #[test]
    fn test_validate_magic() {
        assert_eq!(validate_magic(b"APR\0rest"), Some(2));
        assert_eq!(validate_magic(b"APRNrest"), Some(1));
        assert_eq!(validate_magic(b"GGUF"), None);
        assert_eq!(validate_magic(b"AP"), None);
    }

    #[test]
    fn test_align_offset() {
        assert_eq!(align_offset(0, 64), 0);
        assert_eq!(align_offset(1, 64), 64);
        assert_eq!(align_offset(60, 64), 64);
        assert_eq!(align_offset(64, 64), 64);
        assert_eq!(align_offset(128, 64), 128);
    }

    #[test]
    fn test_swap_dims() {
        assert_eq!(swap_dims_for_row_major([4096, 32000]), [32000, 4096]);
        assert_eq!(swap_dims_for_row_major([1, 1]), [1, 1]);
    }
}
