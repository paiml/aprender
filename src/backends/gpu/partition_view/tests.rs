use super::*;

// cuda-tile-behavior.md: Falsification test #36
#[test]
fn test_tile_count_exact_fit() {
    let tensor = TensorView::<f32>::new([16, 32, 1, 1]);
    let partition = PartitionView::new(tensor, [4, 8, 1, 1]);

    assert_eq!(partition.tile_count(), [4, 4, 1, 1]);
    assert_eq!(partition.total_tiles(), 16);
}

#[test]
fn test_tile_count_with_remainder() {
    let tensor = TensorView::<f32>::new([17, 33, 1, 1]);
    let partition = PartitionView::new(tensor, [4, 8, 1, 1]);

    // 17/4 = 5 (rounded up), 33/8 = 5 (rounded up)
    assert_eq!(partition.tile_count(), [5, 5, 1, 1]);
    assert_eq!(partition.total_tiles(), 25);
}

// cuda-tile-behavior.md: Falsification test #37
#[test]
fn test_tile_iteration_covers_all() {
    let tensor = TensorView::<f32>::new([8, 8, 1, 1]);
    let partition = PartitionView::new(tensor, [4, 4, 1, 1]);

    let tiles: Vec<_> = partition.iter_tiles().collect();
    assert_eq!(tiles.len(), 4);

    // Verify all tiles
    assert_eq!(tiles[0].tile_idx, [0, 0, 0, 0]);
    assert_eq!(tiles[1].tile_idx, [0, 1, 0, 0]);
    assert_eq!(tiles[2].tile_idx, [1, 0, 0, 0]);
    assert_eq!(tiles[3].tile_idx, [1, 1, 0, 0]);
}

// cuda-tile-behavior.md: Falsification test #38
#[test]
fn test_edge_tiles() {
    let tensor = TensorView::<f32>::new([10, 10, 1, 1]);
    let partition = PartitionView::new(tensor, [8, 8, 1, 1]);

    // First tile: full size
    let tile_0 = partition.get_tile([0, 0, 0, 0]).unwrap();
    assert_eq!(tile_0.size, [8, 8, 1, 1]);
    assert!(!tile_0.is_edge);

    // Edge tile: partial size
    let tile_1 = partition.get_tile([1, 1, 0, 0]).unwrap();
    assert_eq!(tile_1.size, [2, 2, 1, 1]); // 10 - 8 = 2
    assert!(tile_1.is_edge);
}

#[test]
fn test_get_tile_view() {
    let tensor = TensorView::<f32>::new([16, 16, 1, 1]);
    let partition = PartitionView::new(tensor, [8, 8, 1, 1]);

    let tile_view = partition.get_tile_view([1, 1, 0, 0]).unwrap();
    assert_eq!(tile_view.shape()[0], 8);
    assert_eq!(tile_view.shape()[1], 8);
    assert_eq!(tile_view.offset(), 8 * 16 + 8); // Row 8, Col 8
}

#[test]
fn test_power_of_two_tiles() {
    let tensor = TensorView::<f32>::new([256, 256, 1, 1]);
    let partition = PartitionView::new_power_of_two(tensor, [4, 4, 0, 0]);

    assert_eq!(partition.tile_shape(), &[16, 16, 1, 1]);
    assert!(partition.is_power_of_two_tiles());
}

#[test]
fn test_non_power_of_two_detection() {
    let tensor = TensorView::<f32>::new([100, 100, 1, 1]);
    let partition = PartitionView::new(tensor, [12, 12, 1, 1]);

    assert!(!partition.is_power_of_two_tiles());
}

#[test]
fn test_2d_partition() {
    let tensor = TensorView::<f32>::new_2d(100, 200);
    let partition = PartitionView::new_2d(tensor, 16, 32);

    assert_eq!(partition.tile_shape(), &[16, 32, 1, 1]);
    assert_eq!(partition.tile_count(), [7, 7, 1, 1]); // ceil(100/16), ceil(200/32)
}

#[test]
fn test_elements_per_tile() {
    let tensor = TensorView::<f32>::new([64, 64, 1, 1]);
    let partition = PartitionView::new(tensor, [8, 8, 1, 1]);

    assert_eq!(partition.elements_per_tile(), 64);
}

#[test]
fn test_workgroup_size_2d() {
    let tensor = TensorView::<f32>::new([64, 64, 1, 1]);
    let partition = PartitionView::new(tensor, [16, 16, 1, 1]);

    let (x, y, z) = partition.recommended_workgroup_size();
    assert_eq!((x, y, z), (16, 16, 1));
}

#[test]
fn test_workgroup_size_1d() {
    let tensor = TensorView::<f32>::new_1d(1024);
    let partition = PartitionView::new(tensor, [256, 1, 1, 1]);

    let (x, y, z) = partition.recommended_workgroup_size();
    assert_eq!((x, y, z), (256, 1, 1));
}

#[test]
fn test_invalid_tile_index() {
    let tensor = TensorView::<f32>::new([8, 8, 1, 1]);
    let partition = PartitionView::new(tensor, [4, 4, 1, 1]);

    assert!(partition.get_tile([5, 0, 0, 0]).is_none());
    assert!(partition.get_tile([0, 5, 0, 0]).is_none());
}

#[test]
fn test_iterator_size_hint() {
    let tensor = TensorView::<f32>::new([16, 16, 1, 1]);
    let partition = PartitionView::new(tensor, [4, 4, 1, 1]);

    let iter = partition.iter_tiles();
    assert_eq!(iter.size_hint(), (16, Some(16)));
    assert_eq!(iter.len(), 16);
}

#[test]
fn test_tile_info_start_positions() {
    let tensor = TensorView::<f32>::new([20, 20, 1, 1]);
    let partition = PartitionView::new(tensor, [8, 8, 1, 1]);

    let tile_00 = partition.get_tile([0, 0, 0, 0]).unwrap();
    assert_eq!(tile_00.start, [0, 0, 0, 0]);

    let tile_11 = partition.get_tile([1, 1, 0, 0]).unwrap();
    assert_eq!(tile_11.start, [8, 8, 0, 0]);

    let tile_22 = partition.get_tile([2, 2, 0, 0]).unwrap();
    assert_eq!(tile_22.start, [16, 16, 0, 0]);
}

// cuda-tile-behavior.md: Falsification test #39 - Tile coverage completeness
#[test]
fn test_complete_coverage() {
    let tensor = TensorView::<f32>::new([15, 17, 1, 1]);
    let partition = PartitionView::new(tensor, [4, 4, 1, 1]);

    // Count all elements covered by tiles
    let mut total_elements = 0;
    for tile in partition.iter_tiles() {
        total_elements += tile.size[0] * tile.size[1];
    }

    assert_eq!(total_elements, 15 * 17);
}

// cuda-tile-behavior.md: Falsification test #40 - Clone behavior
#[test]
fn test_partition_clone() {
    let tensor = TensorView::<f32>::new([32, 32, 1, 1]);
    let partition = PartitionView::new(tensor, [8, 8, 1, 1]);
    let cloned = partition.clone();

    assert_eq!(partition.tile_shape(), cloned.tile_shape());
    assert_eq!(partition.tile_count(), cloned.tile_count());
}

#[test]
#[should_panic(expected = "Tile dimensions must be non-zero")]
fn test_zero_tile_dimension_panics() {
    let tensor = TensorView::<f32>::new([16, 16, 1, 1]);
    let _partition = PartitionView::new(tensor, [0, 8, 1, 1]);
}

#[test]
fn test_single_tile() {
    let tensor = TensorView::<f32>::new([8, 8, 1, 1]);
    let partition = PartitionView::new(tensor, [16, 16, 1, 1]); // Tile larger than tensor

    assert_eq!(partition.tile_count(), [1, 1, 1, 1]);
    assert_eq!(partition.total_tiles(), 1);

    let tile = partition.get_tile([0, 0, 0, 0]).unwrap();
    assert_eq!(tile.size, [8, 8, 1, 1]); // Clamped to tensor size
    assert!(tile.is_edge); // Smaller than full tile
}
