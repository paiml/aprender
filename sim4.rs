fn majority_vote(neighbors: &[(f32, usize)]) -> usize {
    let mut class_counts = std::collections::HashMap::new();
    for (_dist, label) in neighbors {
        *class_counts.entry(*label).or_insert(0) += 1;
    }
    *class_counts.iter().max_by_key(|(_, count)| *count).map(|(l, _)| l).expect("ne")
}
fn weighted_vote(neighbors: &[(f32, usize)]) -> usize {
    let mut class_weights = std::collections::HashMap::new();
    for (dist, label) in neighbors {
        let weight = if *dist < 1e-10 { 1.0 } else { 1.0 / dist };
        *class_weights.entry(*label).or_insert(0.0) += weight;
    }
    *class_weights.iter().max_by(|(_, a), (_, b)| f32::total_cmp(a, b)).map(|(l, _)| l).expect("ne")
}
fn main() {
    // 4-way tie over {0,1,2,3}, one neighbor each
    let n4 = vec![(1.0f32,0usize),(1.0f32,1usize),(1.0f32,2usize),(1.0f32,3usize)];
    // weighted: equal distance => equal weight => tie over {2,5}
    let nw = vec![(2.0f32,5usize),(2.0f32,2usize)];
    println!("{} {}", majority_vote(&n4), weighted_vote(&nw));
}
