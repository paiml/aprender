use super::*;

// cuda-tile-behavior.md: Falsification test #31
#[test]
fn test_tensor_view_creation() {
    let view = TensorView::<f32>::new([2, 3, 4, 5]);
    assert_eq!(view.shape(), &[2, 3, 4, 5]);
    assert_eq!(view.numel(), 120);
    assert_eq!(view.ndim(), 4);
    assert!(view.is_contiguous());
}

#[test]
fn test_tensor_view_1d() {
    let view = TensorView::<f32>::new_1d(100);
    assert_eq!(view.shape(), &[100, 1, 1, 1]);
    assert_eq!(view.numel(), 100);
    assert_eq!(view.ndim(), 1);
}

#[test]
fn test_tensor_view_2d() {
    let view = TensorView::<f32>::new_2d(10, 20);
    assert_eq!(view.shape(), &[10, 20, 1, 1]);
    assert_eq!(view.numel(), 200);
    assert_eq!(view.ndim(), 2);
}

#[test]
fn test_tensor_view_strides() {
    let view = TensorView::<f32>::new([2, 3, 4, 5]);
    // Row-major: strides[i] = product of shape[i+1..]
    assert_eq!(view.strides(), &[60, 20, 5, 1]);
}

// cuda-tile-behavior.md: Falsification test #32
#[test]
fn test_tensor_view_slice() {
    let view = TensorView::<f32>::new([10, 20, 1, 1]);
    let sliced = view.slice(2..7);

    assert_eq!(sliced.shape(), &[5, 20, 1, 1]);
    assert_eq!(sliced.offset(), 40); // 2 * 20
    assert_eq!(sliced.numel(), 100);
}

#[test]
fn test_tensor_view_slice_dim() {
    let view = TensorView::<f32>::new([10, 20, 30, 1]);
    let sliced = view.slice_dim(1, 5..15);

    assert_eq!(sliced.shape(), &[10, 10, 30, 1]);
    assert_eq!(sliced.offset(), 5 * 30); // offset by 5 in dim 1
}

// cuda-tile-behavior.md: Falsification test #33
#[test]
fn test_tensor_view_transpose() {
    let view = TensorView::<f32>::new([2, 3, 1, 1]);
    let transposed = view.transpose(0, 1);

    assert_eq!(transposed.shape(), &[3, 2, 1, 1]);
    assert_eq!(transposed.strides(), &[1, 3, 1, 1]); // Swapped strides
    assert!(!transposed.is_contiguous()); // Non-contiguous after transpose
}

#[test]
fn test_tensor_view_reshape() {
    let view = TensorView::<f32>::new([2, 3, 4, 1]);
    let reshaped = view.reshape([6, 4, 1, 1]).unwrap();

    assert_eq!(reshaped.shape(), &[6, 4, 1, 1]);
    assert_eq!(reshaped.numel(), 24);
}

#[test]
fn test_tensor_view_reshape_invalid() {
    let view = TensorView::<f32>::new([2, 3, 4, 1]);
    let result = view.reshape([5, 5, 1, 1]); // 25 != 24
    assert!(result.is_none());
}

#[test]
fn test_tensor_view_squeeze() {
    let view = TensorView::<f32>::new([1, 3, 1, 4]);
    let squeezed = view.squeeze();

    assert_eq!(squeezed.shape()[0], 3);
    assert_eq!(squeezed.shape()[1], 4);
    assert_eq!(squeezed.ndim(), 2);
}

#[test]
fn test_tensor_view_unsqueeze() {
    let view = TensorView::<f32>::new_2d(3, 4);
    let unsqueezed = view.unsqueeze(0).unwrap();

    assert_eq!(unsqueezed.shape(), &[1, 3, 4, 1]);
    assert_eq!(unsqueezed.ndim(), 3);
}

#[test]
fn test_tensor_view_linear_index() {
    let view = TensorView::<f32>::new([2, 3, 4, 1]);

    // First element
    assert_eq!(view.linear_index([0, 0, 0, 0]), 0);

    // Element at [1, 0, 0, 0]
    assert_eq!(view.linear_index([1, 0, 0, 0]), 12); // 1 * 12

    // Element at [0, 1, 0, 0]
    assert_eq!(view.linear_index([0, 1, 0, 0]), 4); // 1 * 4

    // Element at [1, 2, 3, 0]
    assert_eq!(view.linear_index([1, 2, 3, 0]), 12 + 8 + 3); // 23
}

#[test]
fn test_tensor_view_is_empty() {
    let empty = TensorView::<f32>::new([0, 1, 1, 1]);
    assert!(empty.is_empty());

    let non_empty = TensorView::<f32>::new([1, 1, 1, 1]);
    assert!(!non_empty.is_empty());
}

#[test]
fn test_tensor_view_with_strides() {
    let view = TensorView::<f32>::with_strides([2, 3, 1, 1], [6, 2, 1, 1]);
    assert_eq!(view.strides(), &[6, 2, 1, 1]);
    assert!(!view.is_contiguous()); // Custom strides
}

#[test]
fn test_tensor_view_default() {
    let view = TensorView::<f32>::default();
    assert_eq!(view.numel(), 1);
    assert_eq!(view.ndim(), 1);
}

#[test]
fn test_memory_layout() {
    let view =
        TensorView::<f32>::new([4, 4, 1, 1]).with_layout(MemoryLayout::Tiled { tile_size: [2, 2] });

    assert!(matches!(view.layout(), MemoryLayout::Tiled { tile_size: [2, 2] }));
}

#[test]
fn test_tensor_view_clone() {
    let view = TensorView::<f32>::new([2, 3, 4, 5]);
    let cloned = view.clone();

    assert_eq!(view.shape(), cloned.shape());
    assert_eq!(view.strides(), cloned.strides());
}

// cuda-tile-behavior.md: Falsification test #34 - Dimension accessors
#[test]
fn test_tensor_view_dim_accessors() {
    let view = TensorView::<f32>::new([2, 3, 4, 5]);

    assert_eq!(view.dim(0), 2);
    assert_eq!(view.dim(1), 3);
    assert_eq!(view.dim(2), 4);
    assert_eq!(view.dim(3), 5);

    assert_eq!(view.stride(0), 60);
    assert_eq!(view.stride(1), 20);
    assert_eq!(view.stride(2), 5);
    assert_eq!(view.stride(3), 1);
}

// cuda-tile-behavior.md: Falsification test #35 - Contiguity detection
#[test]
fn test_contiguity_after_operations() {
    let view = TensorView::<f32>::new([4, 4, 1, 1]);
    assert!(view.is_contiguous());

    // Slice preserves contiguity
    let sliced = view.slice(1..3);
    assert!(sliced.is_contiguous());

    // Transpose breaks contiguity
    let transposed = view.transpose(0, 1);
    assert!(!transposed.is_contiguous());
}
