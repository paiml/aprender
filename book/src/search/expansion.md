# Expansion and Simulation

After selection, MCTS expands the tree and simulates to estimate value.

## Expansion

When a leaf node is reached, expand by adding child nodes:

```rust
fn expand(&mut self, node_id: NodeId) {
    let state = self.tree.get_state(node_id);
    let actions = self.state_space.actions(&state);

    for action in actions {
        let child_state = action.apply(&state);
        let prior = self.policy.predict(&state, &action);
        self.tree.add_child(node_id, action, child_state, prior);
    }
}
```

### Lazy Expansion

For large action spaces, expand lazily:

```rust
fn expand_one(&mut self, node_id: NodeId) -> Option<NodeId> {
    let unexpanded = self.tree.unexpanded_actions(node_id);
    if let Some(action) = unexpanded.first() {
        return Some(self.tree.add_child(node_id, action));
    }
    None
}
```

## Simulation (Rollout)

Simulate from current state to terminal:

```rust
fn simulate(&self, state: &State) -> f32 {
    let mut current = state.clone();

    while !current.is_terminal() {
        // Random policy for simulation
        let actions = self.state_space.actions(&current);
        let action = actions.choose(&mut self.rng).unwrap();
        current = action.apply(&current);
    }

    current.reward().unwrap_or(0.0)
}
```

### Guided Simulation

Use policy network for better rollouts:

```rust
fn guided_simulate(&self, state: &State) -> f32 {
    let mut current = state.clone();

    while !current.is_terminal() {
        let probs = self.policy.predict(&current);
        let action = sample_from_distribution(&probs);
        current = action.apply(&current);
    }

    current.reward().unwrap_or(0.0)
}
```
