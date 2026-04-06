// Simplified GPU BFS - Level-Synchronous (easier to implement than frontier-based)
//
// Algorithm:
// 1. Initialize all distances to u32::MAX except source (0)
// 2. For each level, process all nodes at current distance
// 3. Update unvisited neighbors to distance + 1
// 4. Repeat until no updates

struct BfsParams {
    num_nodes: u32,
    current_level: u32,
    source_node: u32,
    _padding: u32,
}

@group(0) @binding(0) var<uniform> params: BfsParams;
@group(0) @binding(1) var<storage, read> row_offsets: array<u32>;
@group(0) @binding(2) var<storage, read> col_indices: array<u32>;
@group(0) @binding(3) var<storage, read_write> distances: array<atomic<u32>>;
@group(0) @binding(4) var<storage, read_write> updated: atomic<u32>;

// Process one node per thread
@compute @workgroup_size(256)
fn bfs_level(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let node_id = global_id.x;

    // Bounds check
    if (node_id >= params.num_nodes) {
        return;
    }

    // Only process nodes at current level
    let dist = atomicLoad(&distances[node_id]);
    if (dist != params.current_level) {
        return;
    }

    // CB-001: bounds-check row_offsets access
    let ro_len = arrayLength(&row_offsets);
    if (node_id >= ro_len || (node_id + 1u) >= ro_len) {
        return;
    }
    let start = row_offsets[node_id];
    let end = row_offsets[node_id + 1u];

    let ci_len = arrayLength(&col_indices);
    for (var i = start; i < end; i++) {
        // CB-001: bounds-check col_indices access
        if (i >= ci_len) {
            break;
        }
        let neighbor = col_indices[i];

        // CB-001: bounds-check distances access
        if (neighbor >= params.num_nodes) {
            continue;
        }

        // Try to update neighbor's distance (atomic compare-exchange)
        let old_dist = atomicMin(&distances[neighbor], params.current_level + 1u);

        // If we updated a neighbor, mark that work was done
        if (old_dist > params.current_level + 1u) {
            atomicStore(&updated, 1u);
        }
    }
}
