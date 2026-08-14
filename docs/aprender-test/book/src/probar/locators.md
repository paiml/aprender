# Locators

![Locator Coverage](../assets/coverage_magma.png)

Probar provides Playwright-style locators for finding game elements with full Playwright parity.

## Locator Strategy

```
┌─────────────────────────────────────────────────────────────────┐
│                      LOCATOR STRATEGIES                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                   │
│  ┌─────────────┐   ┌─────────────┐   ┌─────────────┐            │
│  │    CSS      │   │   TestID    │   │    Text     │            │
│  │  Selector   │   │  Selector   │   │  Selector   │            │
│  │ "button.x"  │   │ "submit-btn"│   │ "Click me"  │            │
│  └──────┬──────┘   └──────┬──────┘   └──────┬──────┘            │
│         │                 │                 │                    │
│         └────────────┬────┴────────────────┘                    │
│                      ▼                                           │
│              ┌──────────────┐                                    │
│              │   Locator    │                                    │
│              │   Chain      │                                    │
│              └──────┬───────┘                                    │
│                     │                                            │
│         ┌──────────┼──────────┐                                 │
│         ▼          ▼          ▼                                 │
│    ┌────────┐ ┌────────┐ ┌────────┐                             │
│    │ filter │ │  and   │ │   or   │                             │
│    │ (opts) │ │ (loc)  │ │ (loc)  │                             │
│    └────────┘ └────────┘ └────────┘                             │
│                                                                   │
│  SEMANTIC: role, label, placeholder, alt_text                    │
│  SPATIAL:  within_radius, in_bounds, nearest_to                  │
│  ECS:      has_component, component_matches                      │
│                                                                   │
└─────────────────────────────────────────────────────────────────┘
```

## Basic Locators

```rust
use probar::{Locator, Selector};

// CSS selector
let button = Locator::new("button.primary");

// Test ID selector (recommended for stability)
let submit = Locator::by_test_id("submit-button");

// Text content
let start = Locator::by_text("Start Game");

// Entity selector (WASM games)
let player = Locator::from_selector(Selector::entity("player"));
```

## Semantic Locators (PMAT-001)

Probar supports Playwright's semantic locators for accessible testing:

```rust
use probar::{Locator, Selector};

// Role selector (ARIA roles)
let button = Locator::by_role("button");
let link = Locator::by_role("link");
let textbox = Locator::by_role("textbox");

// Role with name filter (like Playwright's { name: 'Submit' })
let submit = Locator::by_role_with_name("button", "Submit");

// Label selector (form elements by label text)
let username = Locator::by_label("Username");
let password = Locator::by_label("Password");

// Placeholder selector
let search = Locator::by_placeholder("Search...");
let email = Locator::by_placeholder("Enter email");

// Alt text selector (images)
let logo = Locator::by_alt_text("Company Logo");
let avatar = Locator::by_alt_text("Player Avatar");
```

### Selector Variants

```rust
use probar::Selector;

// All selector types
let css = Selector::css("button.primary");
let xpath = Selector::XPath("//button[@id='submit']".into());
let text = Selector::text("Click me");
let test_id = Selector::test_id("submit-btn");
let entity = Selector::entity("hero");

// Semantic selectors
let role = Selector::role("button");
let role_named = Selector::role_with_name("button", "Submit");
let label = Selector::label("Username");
let placeholder = Selector::placeholder("Search");
let alt = Selector::alt_text("Logo");

// Combined with text filter
let css_text = Selector::CssWithText {
    css: "button".into(),
    text: "Submit".into(),
};
```

## Entity Queries

```rust
let platform = WebPlatform::new_for_test(config);

// Find single entity
let player = platform.locate(Locator::id("player"))?;
let pos = platform.get_position(player);

// Find all matching
let coins: Vec<Entity> = platform.locate_all(Locator::tag("coin"));
assert_eq!(coins.len(), 5);

// First matching
let first_enemy = platform.locate_first(Locator::tag("enemy"));
```

## Locator Operations (PMAT-002)

Probar supports Playwright's locator composition operations:

### Filter

```rust
use probar::{Locator, FilterOptions};

// Filter with hasText
let active_buttons = Locator::new("button")
    .filter(FilterOptions::new().has_text("Active"));

// Filter with hasNotText
let enabled = Locator::new("button")
    .filter(FilterOptions::new().has_not_text("Disabled"));

// Filter with child locator
let with_icon = Locator::new("button")
    .filter(FilterOptions::new().has(Locator::new(".icon")));

// Combined filters
let opts = FilterOptions::new()
    .has_text("Submit")
    .has_not_text("Cancel");
```

### And/Or Composition

```rust
use probar::Locator;

// AND - both conditions must match (intersection)
let active_button = Locator::new("button")
    .and(Locator::new(".active"));
// Produces: "button.active"

// OR - either condition can match (union)
let clickable = Locator::new("button")
    .or(Locator::new("a.btn"));
// Produces: "button, a.btn"

// Chain multiple ORs
let any_interactive = Locator::new("button")
    .or(Locator::new("a"))
    .or(Locator::new("[role='button']"));
```

### Index Operations

```rust
use probar::Locator;

// Get first element
let first_item = Locator::new("li.menu-item").first();

// Get last element
let last_item = Locator::new("li.menu-item").last();

// Get nth element (0-indexed)
let third_item = Locator::new("li.menu-item").nth(2);

// Chained operations
let second_active = Locator::new("button")
    .and(Locator::new(".active"))
    .nth(1);
```

## Compound Locators

```rust
// AND - must match all
let armed_enemy = Locator::new(".enemy")
    .and(Locator::new(".armed"));

// OR - match any
let interactable = Locator::new(".door")
    .or(Locator::new(".chest"));

// Combined with index
let first_enemy = Locator::new(".enemy").first();
```

## Spatial Locators

```rust
// Within radius
let nearby = Locator::within_radius(player_pos, 100.0);

// In bounds
let visible = Locator::in_bounds(screen_bounds);

// Nearest to point
let closest_enemy = Locator::nearest_to(player_pos)
    .with_filter(Locator::tag("enemy"));
```

## Component-Based Locators

```rust
// Has specific component
let physics_entities = Locator::has_component::<RigidBody>();

// Component matches predicate
let low_health = Locator::component_matches::<Health>(|h| h.value < 20);

// Has all components
let complete_entities = Locator::has_all_components::<(
    Position,
    Velocity,
    Sprite,
)>();
```

## Type-Safe Locators (with derive)

Using `jugar-probar-derive` for compile-time checked selectors:

```rust
use jugar_probar_derive::Entity;

#[derive(Entity)]
#[entity(id = "player")]
struct Player;

#[derive(Entity)]
#[entity(tag = "enemy")]
struct Enemy;

// Compile-time verified
let player = platform.locate::<Player>()?;
let enemies = platform.locate_all::<Enemy>();
```

## Waiting for Elements

```rust
// Wait for entity to exist
let boss = platform.wait_for(
    Locator::id("boss"),
    Duration::from_secs(5),
)?;

// Wait for condition
platform.wait_until(
    || platform.locate(Locator::id("door")).is_some(),
    Duration::from_secs(2),
)?;
```

## Locator Chains

```rust
// Find children
let player_weapon = Locator::id("player")
    .child(Locator::tag("weapon"));

// Find parent
let weapon_owner = Locator::id("sword")
    .parent();

// Find siblings
let adjacent_tiles = Locator::id("current_tile")
    .siblings();
```

## Actions on Located Elements

```rust
let button = platform.locate(Locator::id("start_button"))?;

// Get info
let pos = platform.get_position(button);
let bounds = platform.get_bounds(button);
let visible = platform.is_visible(button);

// Interact
platform.click(button);
platform.hover(button);

// Check state
let enabled = platform.is_enabled(button);
let focused = platform.is_focused(button);
```

## Example Test

```rust
#[test]
fn test_coin_collection() {
    let mut platform = WebPlatform::new_for_test(config);

    // Count initial coins
    let initial_coins = platform.locate_all(Locator::tag("coin")).len();
    assert_eq!(initial_coins, 5);

    // Move player to first coin
    let first_coin = platform.locate_first(Locator::tag("coin")).unwrap();
    let coin_pos = platform.get_position(first_coin);

    // Simulate movement
    move_player_to(&mut platform, coin_pos);

    // Coin should be collected
    let remaining_coins = platform.locate_all(Locator::tag("coin")).len();
    assert_eq!(remaining_coins, 4);

    // Score should increase
    let score_display = platform.locate(Locator::id("score")).unwrap();
    let score_text = platform.get_text(score_display);
    assert!(score_text.contains("10"));
}
```
