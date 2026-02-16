use crate::autograd::Tensor;

/// Replace negative infinity values with zero.
pub(crate) fn replace_neg_infinity(values: &mut [f32]) {
    for v in values {
        if *v == f32::NEG_INFINITY {
            *v = 0.0;
        }
    }
}

/// Accumulate node features for batched graphs using a reducer.
fn accumulate_batched<F>(
    x_data: &[f32],
    batch_indices: &[usize],
    num_nodes: usize,
    num_features: usize,
    num_graphs: usize,
    init: f32,
    mut reducer: F,
) -> Vec<f32>
where
    F: FnMut(f32, f32) -> f32,
{
    let mut result = vec![init; num_graphs * num_features];
    for i in 0..num_nodes {
        let graph_id = batch_indices[i];
        for f in 0..num_features {
            let idx = graph_id * num_features + f;
            result[idx] = reducer(result[idx], x_data[i * num_features + f]);
        }
    }
    result
}

/// Accumulate node features for a single graph using a reducer.
fn accumulate_single<F>(
    x_data: &[f32],
    num_nodes: usize,
    num_features: usize,
    init: f32,
    mut reducer: F,
) -> Vec<f32>
where
    F: FnMut(f32, f32) -> f32,
{
    let mut result = vec![init; num_features];
    for i in 0..num_nodes {
        for f in 0..num_features {
            result[f] = reducer(result[f], x_data[i * num_features + f]);
        }
    }
    result
}

/// Accumulate and compute mean for batched data.
fn accumulate_mean_batched(
    x_data: &[f32],
    batch_indices: &[usize],
    num_nodes: usize,
    num_features: usize,
    num_graphs: usize,
) -> Vec<f32> {
    let mut counts = vec![0usize; num_graphs];
    let sums = accumulate_batched(
        x_data,
        batch_indices,
        num_nodes,
        num_features,
        num_graphs,
        0.0,
        |a, b| a + b,
    );

    // Count nodes per graph
    for &graph_id in batch_indices.iter().take(num_nodes) {
        counts[graph_id] += 1;
    }

    // Convert sums to means
    let mut means = sums;
    for g in 0..num_graphs {
        let count = counts[g].max(1) as f32;
        for f in 0..num_features {
            means[g * num_features + f] /= count;
        }
    }
    means
}

/// Accumulate and compute mean for single graph data.
fn accumulate_mean_single(x_data: &[f32], num_nodes: usize, num_features: usize) -> Vec<f32> {
    let mut mean = accumulate_single(x_data, num_nodes, num_features, 0.0, |a, b| a + b);
    let divisor = num_nodes.max(1) as f32;
    for m in &mut mean {
        *m /= divisor;
    }
    mean
}

/// Global mean pooling for graph-level predictions.
///
/// Aggregates all node features into a single graph representation
/// by computing the mean across nodes.
#[must_use]
pub fn global_mean_pool(x: &Tensor, batch: Option<&[usize]>) -> Tensor {
    let num_nodes = x.shape()[0];
    let num_features = x.shape()[1];
    let x_data = x.data();

    if let Some(batch_indices) = batch {
        let num_graphs = batch_indices.iter().max().map_or(0, |&m| m + 1);
        let means =
            accumulate_mean_batched(x_data, batch_indices, num_nodes, num_features, num_graphs);
        Tensor::new(&means, &[num_graphs, num_features])
    } else {
        let mean = accumulate_mean_single(x_data, num_nodes, num_features);
        Tensor::new(&mean, &[1, num_features])
    }
}

/// Global sum pooling for graph-level predictions.
#[must_use]
pub fn global_sum_pool(x: &Tensor, batch: Option<&[usize]>) -> Tensor {
    let num_nodes = x.shape()[0];
    let num_features = x.shape()[1];
    let x_data = x.data();

    if let Some(batch_indices) = batch {
        let num_graphs = batch_indices.iter().max().map_or(0, |&m| m + 1);
        let sums = accumulate_batched(
            x_data,
            batch_indices,
            num_nodes,
            num_features,
            num_graphs,
            0.0,
            |a, b| a + b,
        );
        Tensor::new(&sums, &[num_graphs, num_features])
    } else {
        let sum = accumulate_single(x_data, num_nodes, num_features, 0.0, |a, b| a + b);
        Tensor::new(&sum, &[1, num_features])
    }
}

/// Global max pooling for graph-level predictions.
pub fn global_max_pool(x: &Tensor, batch: Option<&[usize]>) -> Tensor {
    let num_nodes = x.shape()[0];
    let num_features = x.shape()[1];
    let x_data = x.data();

    if let Some(batch_indices) = batch {
        let num_graphs = batch_indices.iter().max().map_or(0, |&m| m + 1);
        let mut maxs = accumulate_batched(
            x_data,
            batch_indices,
            num_nodes,
            num_features,
            num_graphs,
            f32::NEG_INFINITY,
            f32::max,
        );
        replace_neg_infinity(&mut maxs);
        Tensor::new(&maxs, &[num_graphs, num_features])
    } else {
        let mut maxs =
            accumulate_single(x_data, num_nodes, num_features, f32::NEG_INFINITY, f32::max);
        replace_neg_infinity(&mut maxs);
        Tensor::new(&maxs, &[1, num_features])
    }
}
