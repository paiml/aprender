# UCB1/PUCT Selection

Selection policies balance exploration of new paths with exploitation of known good paths.

## UCB1 (Upper Confidence Bound 1)

The classic selection policy for MCTS:

```rust
fn ucb1(node: &Node, parent_visits: u32, c: f32) -> f32 {
    if node.visits == 0 {
        return f32::INFINITY;  // Always explore unvisited
    }

    let exploitation = node.total_reward / node.visits as f32;
    let exploration = c * (parent_visits as f32).ln().sqrt()
                        / (node.visits as f32).sqrt();

    exploitation + exploration
}
```

### Exploration Constant

- `c = 1.414` (sqrt(2)): Theoretically optimal for [0,1] rewards
- `c > 1.414`: More exploration, good for sparse rewards
- `c < 1.414`: More exploitation, good for dense rewards

## PUCT (Predictor + UCT)

Used in AlphaGo/AlphaZero, incorporates policy network priors:

```rust
fn puct(node: &Node, parent_visits: u32, c: f32) -> f32 {
    let exploitation = node.total_reward / (1 + node.visits) as f32;
    let exploration = c * node.prior
                        * (parent_visits as f32).sqrt()
                        / (1 + node.visits) as f32;

    exploitation + exploration
}
```

### Prior Probabilities

The `prior` comes from a policy network that predicts action probabilities:

```rust
impl PolicyNetwork for MyNetwork {
    fn predict(&self, state: &State) -> Vec<(Action, f32)> {
        // Return (action, probability) pairs
        self.forward(state.features())
            .softmax()
            .into_actions()
    }
}
```
