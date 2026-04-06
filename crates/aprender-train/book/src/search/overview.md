# Search Algorithms

Entrenar provides search algorithms for exploring code generation spaces, particularly useful for program synthesis
tasks.

## Available Algorithms

### Monte Carlo Tree Search (MCTS)

MCTS is a heuristic search algorithm that combines tree search with random sampling. In the context of code generation:

- **State**: Partial AST being constructed
- **Action**: AST transformation rules
- **Reward**: Compilation success, test passage, or semantic correctness

```rust
use entrenar::search::{MctsSearch, MctsConfig};

let config = MctsConfig {
    exploration_constant: 1.414,
    max_iterations: 1000,
    max_depth: 50,
    ..Default::default()
};

let mut mcts = MctsSearch::new(config, state_space, action_space);
let result = mcts.search(&initial_state);
```

## Key Features

- **UCB1/PUCT Selection**: Balance exploration vs exploitation
- **Policy Network Integration**: Guide search with learned priors
- **Transposition Tables**: Avoid redundant state exploration
- **Configurable Depth Limits**: Control search complexity

## Use Cases

1. **Python-to-Rust Translation**: Search for valid Rust AST constructions
2. **Code Completion**: Find likely next tokens
3. **Bug Fixing**: Search for patches that fix failing tests
4. **Optimization**: Find performance-improving transformations
