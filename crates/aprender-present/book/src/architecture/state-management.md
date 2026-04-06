# State Management

Manage application state with message passing.

## Pattern

```
Event → Message → State Update → UI Rebuild
```

## State Struct

```rust
#[derive(Default)]
struct AppState {
    counter: i32,
    username: String,
    items: Vec<String>,
}
```

## Messages

```rust
enum Message {
    Increment,
    Decrement,
    SetUsername(String),
    AddItem(String),
}
```

## Update Function

```rust
impl AppState {
    fn update(&mut self, msg: Message) {
        match msg {
            Message::Increment => self.counter += 1,
            Message::Decrement => self.counter -= 1,
            Message::SetUsername(name) => self.username = name,
            Message::AddItem(item) => self.items.push(item),
        }
    }
}
```

## Connecting to Widgets

```rust
// Widget emits message
if let Some(msg) = button.event(&event) {
    if msg.downcast_ref::<ButtonClicked>().is_some() {
        state.update(Message::Increment);
    }
}

// Rebuild UI from state
let ui = Column::new()
    .child(Text::new(format!("Count: {}", state.counter)));
```

## Immutability

State should be the single source of truth:

```rust
// GOOD: State owns data
struct State { items: Vec<Item> }

// BAD: Widget owns data
struct List { items: Vec<Item> }  // Where's the source of truth?
```

## Derived State

Compute from base state:

```rust
impl AppState {
    fn total(&self) -> i32 {
        self.items.len() as i32
    }

    fn is_empty(&self) -> bool {
        self.items.is_empty()
    }
}
```

## Verified Test

```rust
#[test]
fn test_state_management() {
    struct State { count: i32 }
    enum Msg { Inc, Dec }

    impl State {
        fn update(&mut self, msg: Msg) {
            match msg {
                Msg::Inc => self.count += 1,
                Msg::Dec => self.count -= 1,
            }
        }
    }

    let mut state = State { count: 0 };
    state.update(Msg::Inc);
    assert_eq!(state.count, 1);
}
```
