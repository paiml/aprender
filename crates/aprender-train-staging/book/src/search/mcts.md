# Monte Carlo Tree Search (MCTS)

MCTS is a best-first search algorithm that uses random simulations to evaluate positions.

## Algorithm Overview

MCTS proceeds in four phases:

1. **Selection**: Traverse tree using UCB1/PUCT to select promising nodes
2. **Expansion**: Add new child nodes for unexplored actions
3. **Simulation**: Random rollout to terminal state
4. **Backpropagation**: Update statistics along the path

## Core Types

```rust
use entrenar::search::{
    State, Action, StateSpace, ActionSpace,
    MctsSearch, MctsConfig, SearchTree,
};

// Define your state and action types
struct CodeState { /* partial AST */ }
struct CodeAction { /* transformation rule */ }

// Implement required traits
impl State for CodeState {
    fn is_terminal(&self) -> bool { /* check if complete */ }
}

impl Action for CodeAction {
    fn apply(&self, state: &CodeState) -> CodeState { /* apply transform */ }
}
```

## Configuration

```rust
let config = MctsConfig {
    exploration_constant: 1.414,  // UCB1 exploration term
    max_iterations: 1000,         // Search budget
    max_depth: 50,                // Maximum tree depth
    use_puct: true,               // Use PUCT instead of UCB1
    seed: Some(42),               // Reproducible search
};
```

## Selection Policies

### UCB1 (Upper Confidence Bound)

```
UCB1(n) = Q(n) / N(n) + c * sqrt(ln(N_parent) / N(n))
```

- `Q(n)`: Total reward through node
- `N(n)`: Visit count
- `c`: Exploration constant

### PUCT (Predictor + UCT)

```
PUCT(n) = Q(n) / N(n) + c * P(n) * sqrt(N_parent) / (1 + N(n))
```

- `P(n)`: Prior probability from policy network
