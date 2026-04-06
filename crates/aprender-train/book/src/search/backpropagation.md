# Backpropagation

After simulation, propagate the result back up the tree.

## Standard Backpropagation

```rust
fn backpropagate(&mut self, leaf_id: NodeId, reward: f32) {
    let mut current = Some(leaf_id);

    while let Some(node_id) = current {
        let node = self.tree.get_mut(node_id);
        node.visits += 1;
        node.total_reward += reward;
        current = node.parent;
    }
}
```

## Reward Normalization

Normalize rewards to [0, 1] for consistent UCB calculations:

```rust
fn backpropagate_normalized(&mut self, leaf_id: NodeId, reward: f32) {
    // Track min/max for normalization
    self.stats.update_bounds(reward);
    let normalized = (reward - self.stats.min) / (self.stats.max - self.stats.min);

    self.backpropagate(leaf_id, normalized);
}
```

## RAVE (Rapid Action Value Estimation)

Share statistics across sibling nodes:

```rust
fn backpropagate_rave(&mut self, path: &[NodeId], reward: f32) {
    let actions_taken: HashSet<_> = path.iter()
        .filter_map(|&id| self.tree.get(id).action.clone())
        .collect();

    for &node_id in path {
        let node = self.tree.get_mut(node_id);
        node.visits += 1;
        node.total_reward += reward;

        // Update AMAF statistics for siblings
        for sibling in self.tree.siblings(node_id) {
            if actions_taken.contains(&sibling.action) {
                sibling.amaf_visits += 1;
                sibling.amaf_reward += reward;
            }
        }
    }
}
```

## Virtual Loss

For parallel MCTS, prevent thread collision:

```rust
fn apply_virtual_loss(&mut self, path: &[NodeId]) {
    for &node_id in path {
        let node = self.tree.get_mut(node_id);
        node.virtual_loss += 1;
    }
}

fn remove_virtual_loss(&mut self, path: &[NodeId], reward: f32) {
    for &node_id in path {
        let node = self.tree.get_mut(node_id);
        node.virtual_loss -= 1;
        node.visits += 1;
        node.total_reward += reward;
    }
}
```
