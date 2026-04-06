# State and Action Spaces

Define the search space through state and action traits.

## State Trait

```rust
pub trait State: Clone + Hash + Eq {
    /// Check if this is a terminal state
    fn is_terminal(&self) -> bool;

    /// Optional: Get reward for terminal states
    fn reward(&self) -> Option<f32> { None }
}
```

## Action Trait

```rust
pub trait Action: Clone {
    /// Apply this action to a state
    fn apply(&self, state: &impl State) -> impl State;
}
```

## StateSpace Trait

```rust
pub trait StateSpace<S: State, A: Action> {
    /// Get valid actions from a state
    fn actions(&self, state: &S) -> Vec<A>;

    /// Simulate from state to terminal
    fn simulate(&self, state: &S) -> f32;
}
```

## Example: Code Generation

```rust
#[derive(Clone, Hash, Eq, PartialEq)]
struct PartialAst {
    tokens: Vec<Token>,
    open_scopes: usize,
}

impl State for PartialAst {
    fn is_terminal(&self) -> bool {
        self.open_scopes == 0 && self.is_syntactically_complete()
    }
}

enum AstAction {
    AddToken(Token),
    OpenScope,
    CloseScope,
}

impl Action for AstAction {
    fn apply(&self, state: &PartialAst) -> PartialAst {
        let mut new_state = state.clone();
        match self {
            AstAction::AddToken(t) => new_state.tokens.push(t.clone()),
            AstAction::OpenScope => new_state.open_scopes += 1,
            AstAction::CloseScope => new_state.open_scopes -= 1,
        }
        new_state
    }
}
```
