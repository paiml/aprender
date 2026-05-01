// SHIP-TWO-001 — `graph-centrality-v1` algorithm-level PARTIAL
// discharge for FALSIFY-GC-001..008 (closes 8/8 sweep).
//
// Contract: `contracts/graph-centrality-v1.yaml`.

// ===========================================================================
// Reference graph (undirected, simple, adjacency-list)
// ===========================================================================

#[derive(Debug, Clone)]
pub struct UndirectedGraph {
    pub n: usize,
    pub adj: Vec<Vec<usize>>,
}

impl UndirectedGraph {
    #[must_use]
    pub fn empty(n: usize) -> Self {
        Self { n, adj: vec![Vec::new(); n] }
    }

    pub fn add_edge(&mut self, u: usize, v: usize) {
        if u >= self.n || v >= self.n || u == v { return; }
        if !self.adj[u].contains(&v) { self.adj[u].push(v); }
        if !self.adj[v].contains(&u) { self.adj[v].push(u); }
    }

    #[must_use]
    pub fn complete(n: usize) -> Self {
        let mut g = Self::empty(n);
        for u in 0..n { for v in (u + 1)..n { g.add_edge(u, v); } }
        g
    }

    #[must_use]
    pub fn star(leaves: usize) -> Self {
        // Center = 0, leaves = 1..=leaves.
        let n = leaves + 1;
        let mut g = Self::empty(n);
        for i in 1..n { g.add_edge(0, i); }
        g
    }
}

#[must_use]
pub fn degree_centrality(g: &UndirectedGraph) -> Vec<f64> {
    if g.n <= 1 { return vec![0.0; g.n]; }
    let denom = (g.n - 1) as f64;
    g.adj.iter().map(|nbrs| nbrs.len() as f64 / denom).collect()
}

/// BFS shortest distances from `src`. Disconnected nodes get u32::MAX.
#[must_use]
pub fn bfs_distances(g: &UndirectedGraph, src: usize) -> Vec<u32> {
    let mut dist = vec![u32::MAX; g.n];
    if src >= g.n { return dist; }
    dist[src] = 0;
    let mut q = std::collections::VecDeque::new();
    q.push_back(src);
    while let Some(u) = q.pop_front() {
        for &v in &g.adj[u] {
            if dist[v] == u32::MAX {
                dist[v] = dist[u] + 1;
                q.push_back(v);
            }
        }
    }
    dist
}

#[must_use]
pub fn closeness_centrality(g: &UndirectedGraph) -> Vec<f64> {
    let mut out = vec![0.0_f64; g.n];
    for v in 0..g.n {
        let dist = bfs_distances(g, v);
        let mut sum = 0_u64;
        let mut reachable = 0_u64;
        for d in &dist {
            if *d != u32::MAX && *d > 0 {
                sum += *d as u64;
                reachable += 1;
            }
        }
        out[v] = if reachable == 0 || sum == 0 { 0.0 } else { reachable as f64 / sum as f64 };
    }
    out
}

#[must_use]
pub fn harmonic_centrality(g: &UndirectedGraph) -> Vec<f64> {
    if g.n <= 1 { return vec![0.0; g.n]; }
    let denom = (g.n - 1) as f64;
    let mut out = vec![0.0_f64; g.n];
    for v in 0..g.n {
        let dist = bfs_distances(g, v);
        let mut sum = 0.0;
        for (u, d) in dist.iter().enumerate() {
            if u == v || *d == u32::MAX { continue; }
            sum += 1.0 / (*d as f64);
        }
        out[v] = sum / denom;
    }
    out
}

/// Brandes algorithm for unweighted betweenness (unnormalized).
#[must_use]
pub fn betweenness_centrality(g: &UndirectedGraph) -> Vec<f64> {
    let n = g.n;
    let mut cb = vec![0.0_f64; n];
    for s in 0..n {
        let mut stack = Vec::new();
        let mut preds: Vec<Vec<usize>> = vec![Vec::new(); n];
        let mut sigma = vec![0_u64; n];
        sigma[s] = 1;
        let mut dist = vec![-1_i64; n];
        dist[s] = 0;
        let mut q = std::collections::VecDeque::new();
        q.push_back(s);
        while let Some(v) = q.pop_front() {
            stack.push(v);
            for &w in &g.adj[v] {
                if dist[w] < 0 {
                    dist[w] = dist[v] + 1;
                    q.push_back(w);
                }
                if dist[w] == dist[v] + 1 {
                    sigma[w] += sigma[v];
                    preds[w].push(v);
                }
            }
        }
        let mut delta = vec![0.0_f64; n];
        while let Some(w) = stack.pop() {
            for &v in &preds[w] {
                let frac = (sigma[v] as f64 / sigma[w] as f64) * (1.0 + delta[w]);
                delta[v] += frac;
            }
            if w != s { cb[w] += delta[w]; }
        }
    }
    // Undirected: each pair counted twice, divide by 2.
    for x in &mut cb { *x /= 2.0; }
    cb
}

/// Power iteration for principal eigenvector centrality.
#[must_use]
pub fn eigenvector_centrality(g: &UndirectedGraph, max_iter: usize, tol: f64) -> Vec<f64> {
    if g.n == 0 { return vec![]; }
    let mut x = vec![1.0_f64 / (g.n as f64).sqrt(); g.n];
    for _ in 0..max_iter {
        let mut y = vec![0.0_f64; g.n];
        for u in 0..g.n {
            for &v in &g.adj[u] { y[u] += x[v]; }
        }
        let norm: f64 = y.iter().map(|v| v * v).sum::<f64>().sqrt();
        if norm == 0.0 { return vec![0.0; g.n]; }
        let new: Vec<f64> = y.iter().map(|v| v / norm).collect();
        let diff: f64 = new.iter().zip(x.iter()).map(|(a, b)| (a - b).abs()).sum();
        x = new;
        if diff < tol { break; }
    }
    // Ensure non-negative orientation (flip if predominantly negative).
    if x.iter().sum::<f64>() < 0.0 {
        for v in &mut x { *v = -*v; }
    }
    x
}

/// Katz centrality with attenuation `alpha` and base `beta`.
/// `(I - alpha * A)` x = `beta * 1`. We solve via fixed-point iteration.
#[must_use]
pub fn katz_centrality(g: &UndirectedGraph, alpha: f64, beta: f64, max_iter: usize, tol: f64) -> Option<Vec<f64>> {
    if g.n == 0 || beta <= 0.0 || alpha <= 0.0 { return None; }
    let mut x = vec![beta; g.n];
    for _ in 0..max_iter {
        let mut y = vec![beta; g.n];
        for u in 0..g.n {
            for &v in &g.adj[u] { y[u] += alpha * x[v]; }
        }
        let diff: f64 = y.iter().zip(x.iter()).map(|(a, b)| (a - b).abs()).sum();
        x = y;
        if diff < tol { break; }
        if x.iter().any(|v| !v.is_finite()) { return None; }
    }
    Some(x)
}

// ===========================================================================
// GC-001 — Degree centrality bounded in [0, 1]
// ===========================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Gc001Verdict { Pass, Fail }

#[must_use]
pub fn verdict_from_degree_bounded(g: &UndirectedGraph) -> Gc001Verdict {
    if g.n < 2 { return Gc001Verdict::Fail; }
    for c in degree_centrality(g) {
        if !(0.0..=1.0).contains(&c) { return Gc001Verdict::Fail; }
    }
    Gc001Verdict::Pass
}

// ===========================================================================
// GC-002 — Betweenness non-negative
// ===========================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Gc002Verdict { Pass, Fail }

#[must_use]
pub fn verdict_from_betweenness_nonnegative(g: &UndirectedGraph) -> Gc002Verdict {
    if g.n == 0 { return Gc002Verdict::Fail; }
    for c in betweenness_centrality(g) {
        if !c.is_finite() || c < -1e-9 { return Gc002Verdict::Fail; }
    }
    Gc002Verdict::Pass
}

// ===========================================================================
// GC-003 — Eigenvector non-negativity
// ===========================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Gc003Verdict { Pass, Fail }

#[must_use]
pub fn verdict_from_eigenvector_nonnegative(g: &UndirectedGraph) -> Gc003Verdict {
    if g.n == 0 { return Gc003Verdict::Fail; }
    for c in eigenvector_centrality(g, 1000, 1e-9) {
        if !c.is_finite() || c < -1e-6 { return Gc003Verdict::Fail; }
    }
    Gc003Verdict::Pass
}

// ===========================================================================
// GC-004 — Complete graph: all nodes equal centrality
// ===========================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Gc004Verdict { Pass, Fail }

#[must_use]
pub fn verdict_from_complete_symmetry(n: usize) -> Gc004Verdict {
    if n < 2 { return Gc004Verdict::Fail; }
    let g = UndirectedGraph::complete(n);
    let dc = degree_centrality(&g);
    let target = dc[0];
    for v in &dc {
        if (v - target).abs() > 1e-12 { return Gc004Verdict::Fail; }
    }
    if (target - 1.0).abs() > 1e-12 { return Gc004Verdict::Fail; }
    Gc004Verdict::Pass
}

// ===========================================================================
// GC-005 — Star graph: center degree centrality == 1
// ===========================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Gc005Verdict { Pass, Fail }

#[must_use]
pub fn verdict_from_star_center_degree(leaves: usize) -> Gc005Verdict {
    if leaves < 2 { return Gc005Verdict::Fail; }
    let g = UndirectedGraph::star(leaves);
    let dc = degree_centrality(&g);
    if (dc[0] - 1.0).abs() > 1e-12 { return Gc005Verdict::Fail; }
    // Leaves should have degree centrality = 1/(n-1) = 1/leaves.
    let leaf_expected = 1.0 / (leaves as f64);
    for v in dc.iter().skip(1) {
        if (*v - leaf_expected).abs() > 1e-12 { return Gc005Verdict::Fail; }
    }
    Gc005Verdict::Pass
}

// ===========================================================================
// GC-006 — Harmonic centrality bounded in [0, 1]
// ===========================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Gc006Verdict { Pass, Fail }

#[must_use]
pub fn verdict_from_harmonic_bounded(g: &UndirectedGraph) -> Gc006Verdict {
    if g.n < 2 { return Gc006Verdict::Fail; }
    for c in harmonic_centrality(g) {
        if !c.is_finite() || !(0.0..=1.0).contains(&c) { return Gc006Verdict::Fail; }
    }
    Gc006Verdict::Pass
}

// ===========================================================================
// GC-007 — Closeness > 0 for all nodes in connected graphs
// ===========================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Gc007Verdict { Pass, Fail }

#[must_use]
pub fn verdict_from_closeness_positive(g: &UndirectedGraph) -> Gc007Verdict {
    if g.n < 2 { return Gc007Verdict::Fail; }
    for c in closeness_centrality(g) {
        if !c.is_finite() || c <= 0.0 { return Gc007Verdict::Fail; }
    }
    Gc007Verdict::Pass
}

// ===========================================================================
// GC-008 — Katz strictly positive when alpha < 1/lambda_max and beta > 0
// ===========================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Gc008Verdict { Pass, Fail }

#[must_use]
pub fn verdict_from_katz_positive(g: &UndirectedGraph, alpha: f64, beta: f64) -> Gc008Verdict {
    if alpha <= 0.0 || beta <= 0.0 { return Gc008Verdict::Fail; }
    let x = match katz_centrality(g, alpha, beta, 1000, 1e-9) {
        Some(v) => v, None => return Gc008Verdict::Fail,
    };
    for v in x {
        if !v.is_finite() || v <= 0.0 { return Gc008Verdict::Fail; }
    }
    Gc008Verdict::Pass
}

#[cfg(test)]
mod tests {
    use super::*;

    fn path_graph(n: usize) -> UndirectedGraph {
        let mut g = UndirectedGraph::empty(n);
        for i in 0..n.saturating_sub(1) { g.add_edge(i, i + 1); }
        g
    }

    // Reference impl checks
    #[test] fn ref_complete_3_degree() {
        let g = UndirectedGraph::complete(3);
        let dc = degree_centrality(&g);
        for v in &dc { assert!((v - 1.0).abs() < 1e-12); }
    }

    #[test] fn ref_star_5_degree() {
        let g = UndirectedGraph::star(5);
        let dc = degree_centrality(&g);
        assert!((dc[0] - 1.0).abs() < 1e-12);
        for v in dc.iter().skip(1) { assert!((v - 0.2).abs() < 1e-12); }
    }

    // GC-001
    #[test] fn gc001_pass_complete() {
        assert_eq!(verdict_from_degree_bounded(&UndirectedGraph::complete(5)), Gc001Verdict::Pass);
    }
    #[test] fn gc001_pass_star() {
        assert_eq!(verdict_from_degree_bounded(&UndirectedGraph::star(8)), Gc001Verdict::Pass);
    }
    #[test] fn gc001_pass_path() {
        assert_eq!(verdict_from_degree_bounded(&path_graph(10)), Gc001Verdict::Pass);
    }
    #[test] fn gc001_fail_n_lt_2() {
        assert_eq!(verdict_from_degree_bounded(&UndirectedGraph::empty(1)), Gc001Verdict::Fail);
    }

    // GC-002
    #[test] fn gc002_pass_complete() {
        assert_eq!(verdict_from_betweenness_nonnegative(&UndirectedGraph::complete(5)), Gc002Verdict::Pass);
    }
    #[test] fn gc002_pass_star() {
        assert_eq!(verdict_from_betweenness_nonnegative(&UndirectedGraph::star(8)), Gc002Verdict::Pass);
    }
    #[test] fn gc002_pass_path() {
        assert_eq!(verdict_from_betweenness_nonnegative(&path_graph(10)), Gc002Verdict::Pass);
    }

    // GC-003
    #[test] fn gc003_pass_complete() {
        assert_eq!(verdict_from_eigenvector_nonnegative(&UndirectedGraph::complete(5)), Gc003Verdict::Pass);
    }
    #[test] fn gc003_pass_star() {
        assert_eq!(verdict_from_eigenvector_nonnegative(&UndirectedGraph::star(6)), Gc003Verdict::Pass);
    }

    // GC-004
    #[test] fn gc004_pass_n3() { assert_eq!(verdict_from_complete_symmetry(3), Gc004Verdict::Pass); }
    #[test] fn gc004_pass_n10() { assert_eq!(verdict_from_complete_symmetry(10), Gc004Verdict::Pass); }
    #[test] fn gc004_fail_n_lt_2() { assert_eq!(verdict_from_complete_symmetry(1), Gc004Verdict::Fail); }

    // GC-005
    #[test] fn gc005_pass_5_leaves() { assert_eq!(verdict_from_star_center_degree(5), Gc005Verdict::Pass); }
    #[test] fn gc005_pass_10_leaves() { assert_eq!(verdict_from_star_center_degree(10), Gc005Verdict::Pass); }
    #[test] fn gc005_fail_too_few() { assert_eq!(verdict_from_star_center_degree(1), Gc005Verdict::Fail); }

    // GC-006
    #[test] fn gc006_pass_complete() {
        assert_eq!(verdict_from_harmonic_bounded(&UndirectedGraph::complete(5)), Gc006Verdict::Pass);
    }
    #[test] fn gc006_pass_path() {
        assert_eq!(verdict_from_harmonic_bounded(&path_graph(10)), Gc006Verdict::Pass);
    }
    #[test] fn gc006_pass_disconnected() {
        // Two isolated edges: harmonic should still be in [0, 1] (small).
        let mut g = UndirectedGraph::empty(4);
        g.add_edge(0, 1); g.add_edge(2, 3);
        assert_eq!(verdict_from_harmonic_bounded(&g), Gc006Verdict::Pass);
    }

    // GC-007
    #[test] fn gc007_pass_complete() {
        assert_eq!(verdict_from_closeness_positive(&UndirectedGraph::complete(5)), Gc007Verdict::Pass);
    }
    #[test] fn gc007_pass_path() {
        assert_eq!(verdict_from_closeness_positive(&path_graph(10)), Gc007Verdict::Pass);
    }

    // GC-008
    #[test] fn gc008_pass_small_alpha() {
        // For K_5, λ_max = n-1 = 4, so 1/4 = 0.25; alpha < 0.25 → Pass.
        let g = UndirectedGraph::complete(5);
        assert_eq!(verdict_from_katz_positive(&g, 0.1, 1.0), Gc008Verdict::Pass);
    }
    #[test] fn gc008_fail_zero_beta() {
        let g = UndirectedGraph::complete(5);
        assert_eq!(verdict_from_katz_positive(&g, 0.1, 0.0), Gc008Verdict::Fail);
    }
    #[test] fn gc008_fail_zero_alpha() {
        let g = UndirectedGraph::complete(5);
        assert_eq!(verdict_from_katz_positive(&g, 0.0, 1.0), Gc008Verdict::Fail);
    }
}
