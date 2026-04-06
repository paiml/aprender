// GPU PageRank - Sparse Matrix-Vector Multiplication (SpMV)
//
// Algorithm:
// 1. Initialize all scores to 1/N
// 2. For each iteration:
//    - new_pr(v) = (1-d)/N + d * sum(pr(u) / out_degree(u)) for all u -> v
//    - Handle dangling nodes (out_degree = 0) by redistributing to all nodes
// 3. Repeat until convergence (20 iterations typical)
//
// Based on Page et al. (1999) and GraphBLAST (Yang et al., ACM ToMS 2022)

struct PageRankParams {
    num_nodes: u32,
    damping: f32,        // Typically 0.85
    iteration: u32,
    dangling_sum: f32,   // Sum of ranks from dangling nodes (computed in pass 1)
}

@group(0) @binding(0) var<uniform> params: PageRankParams;
@group(0) @binding(1) var<storage, read> row_offsets: array<u32>;
@group(0) @binding(2) var<storage, read> col_indices: array<u32>;
@group(0) @binding(3) var<storage, read> current_scores: array<f32>;
@group(0) @binding(4) var<storage, read_write> next_scores: array<f32>;
@group(0) @binding(5) var<storage, read> out_degrees: array<u32>;

// Process one node per thread
@compute @workgroup_size(256)
fn pagerank_iteration(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let node_id = global_id.x;

    // Bounds check
    if (node_id >= params.num_nodes) {
        return;
    }

    // Base score (teleport probability + dangling contribution)
    let base_score = (1.0 - params.damping) / f32(params.num_nodes);
    let dangling_contrib = params.damping * params.dangling_sum / f32(params.num_nodes);

    // Accumulate contributions from incoming neighbors
    // For each u where u -> node_id: contribution += scores(u) / degree(u)
    var contribution_sum = 0.0;

    let ro_len = arrayLength(&row_offsets);
    let ci_len = arrayLength(&col_indices);
    let od_len = arrayLength(&out_degrees);
    let cs_len = arrayLength(&current_scores);

    // Note: Iterating over all nodes to find incoming edges is O(N*M)
    // Production implementation would use reverse CSR for O(M) complexity
    for (var u = 0u; u < params.num_nodes; u++) {
        // CB-001: bounds-check out_degrees access
        if (u >= od_len) {
            break;
        }
        let out_degree = out_degrees[u];

        // Skip dangling nodes (handled separately via dangling_sum)
        if (out_degree == 0u) {
            continue;
        }

        // CB-001: bounds-check row_offsets access
        if (u >= ro_len || (u + 1u) >= ro_len) {
            continue;
        }
        let start = row_offsets[u];
        let end = row_offsets[u + 1u];

        // Check if u has edge to node_id
        for (var i = start; i < end; i++) {
            // CB-001: bounds-check col_indices access
            if (i >= ci_len) {
                break;
            }
            if (col_indices[i] == node_id) {
                // u -> node_id edge found
                // CB-001: bounds-check current_scores access
                if (u < cs_len) {
                    contribution_sum += current_scores[u] / f32(out_degree);
                }
                break; // Found edge, no need to check more
            }
        }
    }

    // Update score: (1-d)/N + d*dangling/N + d * sum(PR(u) / out_degree(u))
    // CB-001: bounds-check next_scores access
    let ns_len = arrayLength(&next_scores);
    if (node_id < ns_len) {
        next_scores[node_id] = base_score + dangling_contrib + params.damping * contribution_sum;
    }
}
