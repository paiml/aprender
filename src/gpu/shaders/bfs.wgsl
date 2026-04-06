// GPU BFS Compute Shader
// Based on Gunrock (Wang et al., ACM ToPC 2017) frontier-based traversal
//
// Algorithm:
// 1. Start with source node in frontier
// 2. For each vertex in frontier:
//    - Mark all unvisited neighbors as visited
//    - Add neighbors to next frontier
// 3. Swap frontiers and repeat until empty

struct GraphData {
    num_nodes: u32,
    num_edges: u32,
}

@group(0) @binding(0) var<storage, read> graph_meta: GraphData;
@group(0) @binding(1) var<storage, read> row_offsets: array<u32>;
@group(0) @binding(2) var<storage, read> col_indices: array<u32>;
@group(0) @binding(3) var<storage, read_write> visited: array<atomic<u32>>;
@group(0) @binding(4) var<storage, read_write> distances: array<u32>;
@group(0) @binding(5) var<storage, read> frontier: array<u32>;
@group(0) @binding(6) var<storage, read_write> next_frontier: array<u32>;
@group(0) @binding(7) var<storage, read_write> next_frontier_size: atomic<u32>;

// BFS kernel - process one frontier vertex per thread
@compute @workgroup_size(256)
fn bfs_kernel(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let tid = global_id.x;
    let frontier_size = arrayLength(&frontier);

    // Bounds check
    if (tid >= frontier_size) {
        return;
    }

    // Get current vertex from frontier
    let vertex = frontier[tid];
    if (vertex >= graph_meta.num_nodes) {
        return;
    }

    // CB-001: bounds-check row_offsets and distances access
    let ro_len = arrayLength(&row_offsets);
    let dist_len = arrayLength(&distances);
    if (vertex >= ro_len || (vertex + 1u) >= ro_len || vertex >= dist_len) {
        return;
    }

    let start = row_offsets[vertex];
    let end = row_offsets[vertex + 1u];
    if (vertex >= arrayLength(&distances)) {
        return;
    }
    let current_dist = distances[vertex];

    let ci_len = arrayLength(&col_indices);
    let nf_len = arrayLength(&next_frontier);
    let vis_len = arrayLength(&visited);
    for (var i = start; i < end; i++) {
        if (i >= ci_len) {
            break;
        }
        let neighbor = col_indices[i];

        if (neighbor >= graph_meta.num_nodes || neighbor >= vis_len) {
            continue;
        }

        if (neighbor >= arrayLength(&visited)) {
            continue;
        }
        let was_visited = atomicExchange(&visited[neighbor], 1u);

        if (was_visited == 0u) {
            if (neighbor < dist_len) {
                distances[neighbor] = current_dist + 1u;
            }

            // Add to next frontier (atomic counter)
            let next_idx = atomicAdd(&next_frontier_size, 1u);
            if (next_idx < nf_len) {
                next_frontier[next_idx] = neighbor;
            }
        }
    }
}
