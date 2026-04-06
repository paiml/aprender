# Policy Network Integration

Guide MCTS with learned prior probabilities.

## PolicyNetwork Trait

```rust
pub trait PolicyNetwork<S: State, A: Action> {
    /// Predict action probabilities for a state
    fn predict(&self, state: &S) -> Vec<(A, f32)>;

    /// Optional: Predict state value
    fn value(&self, state: &S) -> f32 { 0.5 }
}
```

## Integration with MCTS

```rust
let mut mcts = MctsSearch::new(config, state_space, action_space)
    .with_policy(policy_network);

// Policy priors guide node selection
let result = mcts.search(&initial_state);
```

## Training the Policy Network

Use MCTS visit counts as training targets:

```rust
fn generate_training_data(mcts: &MctsSearch) -> TrainingData {
    let root = mcts.tree.root();

    // Visit counts become policy targets
    let policy_target: Vec<f32> = root.children.iter()
        .map(|c| c.visits as f32 / root.visits as f32)
        .collect();

    // Final outcome becomes value target
    let value_target = mcts.result.reward;

    TrainingData {
        state: root.state.features(),
        policy: policy_target,
        value: value_target,
    }
}
```

## AlphaZero-Style Training Loop

```rust
fn self_play_training(mut policy: PolicyNetwork, iterations: usize) {
    for _ in 0..iterations {
        // Self-play with current policy
        let mut mcts = MctsSearch::new(config).with_policy(&policy);
        let game_data = play_game(&mut mcts);

        // Train on game outcomes
        let training_data = generate_training_data(&game_data);
        policy.train(&training_data);
    }
}
```

## Temperature-Based Action Selection

During self-play, use temperature to control exploration:

```rust
fn select_action(mcts: &MctsSearch, temperature: f32) -> Action {
    let visits: Vec<f32> = mcts.root_children()
        .map(|c| (c.visits as f32).powf(1.0 / temperature))
        .collect();

    let sum: f32 = visits.iter().sum();
    let probs: Vec<f32> = visits.iter().map(|v| v / sum).collect();

    sample_from_distribution(&probs)
}
```
