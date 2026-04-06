use super::*;

// =========================================================================
// PropertyPath Tests
// =========================================================================

#[test]
fn test_property_path_new() {
    let path = PropertyPath::new("user.profile.name");
    assert_eq!(path.segments(), &["user", "profile", "name"]);
}

#[test]
fn test_property_path_root() {
    let path = PropertyPath::root();
    assert!(path.is_root());
    assert!(path.is_empty());
}

#[test]
fn test_property_path_len() {
    let path = PropertyPath::new("a.b.c");
    assert_eq!(path.len(), 3);
}

#[test]
fn test_property_path_join() {
    let path = PropertyPath::new("user");
    let joined = path.join("name");
    assert_eq!(joined.to_string_path(), "user.name");
}

#[test]
fn test_property_path_parent() {
    let path = PropertyPath::new("user.profile.name");
    let parent = path.parent().unwrap();
    assert_eq!(parent.to_string_path(), "user.profile");
}

#[test]
fn test_property_path_leaf() {
    let path = PropertyPath::new("user.profile.name");
    assert_eq!(path.leaf(), Some("name"));
}

#[test]
fn test_property_path_display() {
    let path = PropertyPath::new("a.b.c");
    assert_eq!(format!("{path}"), "a.b.c");
}

#[test]
fn test_property_path_from_str() {
    let path: PropertyPath = "user.name".into();
    assert_eq!(path.segments(), &["user", "name"]);
}

// =========================================================================
// BindingConfig Tests
// =========================================================================

#[test]
fn test_binding_config_one_way() {
    let config = BindingConfig::one_way("count", "value");
    assert_eq!(config.source.to_string_path(), "count");
    assert_eq!(config.target, "value");
    assert_eq!(config.direction, BindingDirection::OneWay);
}

#[test]
fn test_binding_config_two_way() {
    let config = BindingConfig::two_way("user.name", "text");
    assert_eq!(config.direction, BindingDirection::TwoWay);
}

#[test]
fn test_binding_config_transform() {
    let config = BindingConfig::one_way("count", "label").transform("toString");
    assert_eq!(config.transform, Some("toString".to_string()));
}

#[test]
fn test_binding_config_fallback() {
    let config = BindingConfig::one_way("user.name", "text").fallback("Anonymous");
    assert_eq!(config.fallback, Some("Anonymous".to_string()));
}

// =========================================================================
// ReactiveCell Tests
// =========================================================================

#[test]
fn test_reactive_cell_new() {
    let cell = ReactiveCell::new(42);
    assert_eq!(cell.get(), 42);
}

#[test]
fn test_reactive_cell_set() {
    let cell = ReactiveCell::new(0);
    cell.set(100);
    assert_eq!(cell.get(), 100);
}

#[test]
fn test_reactive_cell_update() {
    let cell = ReactiveCell::new(10);
    cell.update(|v| *v *= 2);
    assert_eq!(cell.get(), 20);
}

#[test]
fn test_reactive_cell_subscribe() {
    use std::sync::atomic::{AtomicI32, Ordering};

    let cell = ReactiveCell::new(0);
    let count = Arc::new(AtomicI32::new(0));
    let count_clone = count.clone();

    cell.subscribe(move |v| {
        count_clone.store(*v, Ordering::SeqCst);
    });

    cell.set(42);
    assert_eq!(count.load(Ordering::SeqCst), 42);
}

#[test]
fn test_reactive_cell_default() {
    let cell: ReactiveCell<i32> = ReactiveCell::default();
    assert_eq!(cell.get(), 0);
}

#[test]
fn test_reactive_cell_clone() {
    let cell1 = ReactiveCell::new(10);
    let cell2 = cell1.clone();

    cell1.set(20);
    assert_eq!(cell2.get(), 20); // Shares same underlying value
}

// =========================================================================
// Computed Tests
// =========================================================================

#[test]
fn test_computed_new() {
    let computed = Computed::new(|| 42);
    assert_eq!(computed.get(), 42);
}

#[test]
fn test_computed_caches() {
    use std::sync::atomic::{AtomicUsize, Ordering};

    let call_count = Arc::new(AtomicUsize::new(0));
    let call_count_clone = call_count.clone();

    let computed = Computed::new(move || {
        call_count_clone.fetch_add(1, Ordering::SeqCst);
        42
    });

    computed.get();
    computed.get();
    computed.get();

    assert_eq!(call_count.load(Ordering::SeqCst), 1); // Only computed once
}

#[test]
fn test_computed_invalidate() {
    use std::sync::atomic::{AtomicUsize, Ordering};

    let call_count = Arc::new(AtomicUsize::new(0));
    let call_count_clone = call_count.clone();

    let computed = Computed::new(move || {
        call_count_clone.fetch_add(1, Ordering::SeqCst);
        42
    });

    computed.get();
    computed.invalidate();
    computed.get();

    assert_eq!(call_count.load(Ordering::SeqCst), 2); // Computed twice
}

// =========================================================================
// BindingExpression Tests
// =========================================================================

#[test]
fn test_binding_expression_new() {
    let expr = BindingExpression::new("{{ user.name }}");
    assert_eq!(expr.dependencies.len(), 1);
    assert_eq!(expr.dependencies[0].to_string_path(), "user.name");
}

#[test]
fn test_binding_expression_property() {
    let expr = BindingExpression::property("count");
    assert!(expr.is_simple_property());
    assert_eq!(expr.as_property().unwrap().to_string_path(), "count");
}

#[test]
fn test_binding_expression_with_transform() {
    let expr = BindingExpression::new("{{ count | format }}");
    assert_eq!(expr.dependencies[0].to_string_path(), "count");
}

#[test]
fn test_binding_expression_multiple_deps() {
    let expr = BindingExpression::new("{{ first }} and {{ second }}");
    assert_eq!(expr.dependencies.len(), 2);
}

// =========================================================================
// EventBinding Tests
// =========================================================================

#[test]
fn test_event_binding_new() {
    let binding = EventBinding::new("click", ActionBinding::toggle("visible"));
    assert_eq!(binding.event, "click");
}

#[test]
fn test_event_binding_on_click() {
    let binding = EventBinding::on_click(ActionBinding::dispatch("submit"));
    assert_eq!(binding.event, "click");
}

#[test]
fn test_event_binding_on_change() {
    let binding = EventBinding::on_change(ActionBinding::set("value", "new"));
    assert_eq!(binding.event, "change");
}

// =========================================================================
// ActionBinding Tests
// =========================================================================

#[test]
fn test_action_binding_set() {
    let action = ActionBinding::set("user.name", "Alice");
    if let ActionBinding::SetProperty { path, value } = action {
        assert_eq!(path.to_string_path(), "user.name");
        assert_eq!(value, "Alice");
    } else {
        panic!("Expected SetProperty");
    }
}

#[test]
fn test_action_binding_toggle() {
    let action = ActionBinding::toggle("visible");
    if let ActionBinding::ToggleProperty { path } = action {
        assert_eq!(path.to_string_path(), "visible");
    } else {
        panic!("Expected ToggleProperty");
    }
}

#[test]
fn test_action_binding_increment() {
    let action = ActionBinding::increment("count");
    if let ActionBinding::IncrementProperty { path, amount } = action {
        assert_eq!(path.to_string_path(), "count");
        assert!(amount.is_none());
    } else {
        panic!("Expected IncrementProperty");
    }
}

#[test]
fn test_action_binding_increment_by() {
    let action = ActionBinding::increment_by("score", 10.0);
    if let ActionBinding::IncrementProperty { amount, .. } = action {
        assert_eq!(amount, Some(10.0));
    } else {
        panic!("Expected IncrementProperty");
    }
}

#[test]
fn test_action_binding_navigate() {
    let action = ActionBinding::navigate("/home");
    if let ActionBinding::Navigate { route } = action {
        assert_eq!(route, "/home");
    } else {
        panic!("Expected Navigate");
    }
}

#[test]
fn test_action_binding_dispatch() {
    let action = ActionBinding::dispatch("submit");
    if let ActionBinding::Dispatch { message, payload } = action {
        assert_eq!(message, "submit");
        assert!(payload.is_none());
    } else {
        panic!("Expected Dispatch");
    }
}

#[test]
fn test_action_binding_dispatch_with() {
    let action = ActionBinding::dispatch_with("submit", "form_data");
    if let ActionBinding::Dispatch { message, payload } = action {
        assert_eq!(message, "submit");
        assert_eq!(payload, Some("form_data".to_string()));
    } else {
        panic!("Expected Dispatch");
    }
}

#[test]
fn test_action_binding_batch() {
    let action = ActionBinding::batch([
        ActionBinding::increment("count"),
        ActionBinding::navigate("/next"),
    ]);
    if let ActionBinding::Batch { actions } = action {
        assert_eq!(actions.len(), 2);
    } else {
        panic!("Expected Batch");
    }
}

// =========================================================================
// BindingManager Tests
// =========================================================================

#[test]
fn test_binding_manager_new() {
    let manager = BindingManager::new();
    assert_eq!(manager.active_count(), 0);
}

#[test]
fn test_binding_manager_register() {
    let mut manager = BindingManager::new();
    let id = manager.register("widget1", BindingConfig::one_way("count", "text"));
    assert_eq!(id.0, 0);
    assert_eq!(manager.active_count(), 1);
}

#[test]
fn test_binding_manager_unregister() {
    let mut manager = BindingManager::new();
    let id = manager.register("widget1", BindingConfig::one_way("count", "text"));
    manager.unregister(id);
    assert_eq!(manager.active_count(), 0);
}

#[test]
fn test_binding_manager_bindings_for_widget() {
    let mut manager = BindingManager::new();
    manager.register("widget1", BindingConfig::one_way("count", "text"));
    manager.register("widget1", BindingConfig::one_way("name", "label"));
    manager.register("widget2", BindingConfig::one_way("other", "value"));

    let bindings = manager.bindings_for_widget("widget1");
    assert_eq!(bindings.len(), 2);
}

#[test]
fn test_binding_manager_bindings_for_path() {
    let mut manager = BindingManager::new();
    manager.register("widget1", BindingConfig::one_way("user.name", "text"));
    manager.register("widget2", BindingConfig::one_way("user.name", "label"));

    let path = PropertyPath::new("user.name");
    let bindings = manager.bindings_for_path(&path);
    assert_eq!(bindings.len(), 2);
}

#[test]
fn test_binding_manager_on_state_change() {
    let mut manager = BindingManager::new();
    manager.register("widget1", BindingConfig::one_way("count", "text"));
    manager.register("widget2", BindingConfig::one_way("count", "label"));

    let path = PropertyPath::new("count");
    let updates = manager.on_state_change(&path, "42");

    assert_eq!(updates.len(), 2);
    assert!(updates.iter().any(|u| u.widget_id == "widget1"));
    assert!(updates.iter().any(|u| u.widget_id == "widget2"));
}

#[test]
fn test_binding_manager_on_widget_change_two_way() {
    let mut manager = BindingManager::new();
    manager.register("input1", BindingConfig::two_way("user.name", "value"));

    let updates = manager.on_widget_change("input1", "value", "Alice");

    assert_eq!(updates.len(), 1);
    assert_eq!(updates[0].path.to_string_path(), "user.name");
    assert_eq!(updates[0].value, "Alice");
}

#[test]
fn test_binding_manager_on_widget_change_one_way_no_propagate() {
    let mut manager = BindingManager::new();
    manager.register("label1", BindingConfig::one_way("count", "text"));

    let updates = manager.on_widget_change("label1", "text", "new value");

    assert!(updates.is_empty()); // One-way doesn't propagate back
}

#[test]
fn test_binding_manager_with_debounce() {
    let manager = BindingManager::new().with_debounce(100);
    assert_eq!(manager.debounce_ms, Some(100));
}

#[test]
fn test_binding_manager_queue_and_flush() {
    let mut manager = BindingManager::new();
    manager.register("widget1", BindingConfig::one_way("count", "text"));

    manager.queue_update(
        UpdateSource::State,
        PropertyPath::new("count"),
        "42".to_string(),
    );

    let (widget_updates, _) = manager.flush();
    assert_eq!(widget_updates.len(), 1);
}

#[test]
fn test_binding_manager_clear() {
    let mut manager = BindingManager::new();
    manager.register("w1", BindingConfig::one_way("a", "b"));
    manager.register("w2", BindingConfig::one_way("c", "d"));
    manager.clear();
    assert_eq!(manager.active_count(), 0);
}

// =========================================================================
// ValueConverter Tests
// =========================================================================

#[test]
fn test_identity_converter() {
    let converter = IdentityConverter;
    assert_eq!(converter.convert("hello").unwrap(), "hello");
    assert_eq!(converter.convert_back("world").unwrap(), "world");
}

#[test]
fn test_bool_to_string_converter() {
    let converter = BoolToStringConverter::new();
    assert_eq!(converter.convert("true").unwrap(), "true");
    assert_eq!(converter.convert("false").unwrap(), "false");
    assert_eq!(converter.convert("1").unwrap(), "true");
    assert_eq!(converter.convert("0").unwrap(), "false");
}

#[test]
fn test_bool_to_string_converter_custom() {
    let converter = BoolToStringConverter::with_strings("Yes", "No");
    assert_eq!(converter.convert("true").unwrap(), "Yes");
    assert_eq!(converter.convert("false").unwrap(), "No");
    assert_eq!(converter.convert_back("Yes").unwrap(), "true");
    assert_eq!(converter.convert_back("No").unwrap(), "false");
}

#[test]
fn test_bool_to_string_converter_error() {
    let converter = BoolToStringConverter::new();
    assert!(converter.convert("invalid").is_err());
}

#[test]
fn test_number_format_converter() {
    let converter = NumberFormatConverter::new().decimals(2);
    assert_eq!(converter.convert("42").unwrap(), "42.00");
    assert_eq!(converter.convert("3.14159").unwrap(), "3.14");
}

#[test]
fn test_number_format_converter_with_prefix_suffix() {
    let converter = NumberFormatConverter::new()
        .decimals(2)
        .prefix("$")
        .suffix(" USD");
    assert_eq!(converter.convert("100").unwrap(), "$100.00 USD");
    assert_eq!(converter.convert_back("$100.00 USD").unwrap(), "100.00");
}

#[test]
fn test_number_format_converter_error() {
    let converter = NumberFormatConverter::new();
    assert!(converter.convert("not a number").is_err());
}

#[test]
fn test_conversion_error_display() {
    let err = ConversionError {
        message: "test error".to_string(),
    };
    assert!(err.to_string().contains("test error"));
}

// =========================================================================
// Widget/State Update Tests
// =========================================================================

#[test]
fn test_widget_update_struct() {
    let update = WidgetUpdate {
        widget_id: "input1".to_string(),
        property: "value".to_string(),
        value: "Hello".to_string(),
    };
    assert_eq!(update.widget_id, "input1");
}

#[test]
fn test_state_update_struct() {
    let update = StateUpdate {
        path: PropertyPath::new("user.name"),
        value: "Alice".to_string(),
    };
    assert_eq!(update.path.to_string_path(), "user.name");
}

#[test]
fn test_binding_id_default() {
    let id = BindingId::default();
    assert_eq!(id.0, 0);
}

#[test]
fn test_update_source_eq() {
    assert_eq!(UpdateSource::State, UpdateSource::State);
    assert_eq!(UpdateSource::Widget, UpdateSource::Widget);
    assert_ne!(UpdateSource::State, UpdateSource::Widget);
}

// =========================================================================
// PropertyPath Additional Tests
// =========================================================================

#[test]
fn test_property_path_empty_string() {
    let path = PropertyPath::new("");
    assert!(path.is_empty());
    assert!(path.is_root());
}

#[test]
fn test_property_path_trailing_dots() {
    let path = PropertyPath::new("user.name.");
    assert_eq!(path.segments(), &["user", "name"]);
}

#[test]
fn test_property_path_leading_dots() {
    let path = PropertyPath::new(".user.name");
    assert_eq!(path.segments(), &["user", "name"]);
}

#[test]
fn test_property_path_multiple_dots() {
    let path = PropertyPath::new("user..name");
    assert_eq!(path.segments(), &["user", "name"]);
}

#[test]
fn test_property_path_parent_of_root() {
    let path = PropertyPath::root();
    assert!(path.parent().is_none());
}

#[test]
fn test_property_path_leaf_of_root() {
    let path = PropertyPath::root();
    assert!(path.leaf().is_none());
}

#[test]
fn test_property_path_single_segment() {
    let path = PropertyPath::new("count");
    assert_eq!(path.len(), 1);
    assert_eq!(path.leaf(), Some("count"));
    let parent = path.parent().unwrap();
    assert!(parent.is_empty());
}

#[test]
fn test_property_path_hash() {
    use std::collections::HashSet;

    let mut set = HashSet::new();
    set.insert(PropertyPath::new("user.name"));
    set.insert(PropertyPath::new("user.email"));

    assert!(set.contains(&PropertyPath::new("user.name")));
    assert!(!set.contains(&PropertyPath::new("other")));
}

#[test]
fn test_property_path_clone() {
    let path = PropertyPath::new("a.b.c");
    let cloned = path.clone();
    assert_eq!(path, cloned);
}

#[test]
fn test_property_path_debug() {
    let path = PropertyPath::new("user.name");
    let debug = format!("{path:?}");
    assert!(debug.contains("PropertyPath"));
}

#[test]
fn test_property_path_serialize() {
    let path = PropertyPath::new("user.name");
    let json = serde_json::to_string(&path).unwrap();
    assert!(json.contains("user"));
    assert!(json.contains("name"));
}

#[test]
fn test_property_path_deserialize() {
    let json = r#"{"segments":["user","profile"]}"#;
    let path: PropertyPath = serde_json::from_str(json).unwrap();
    assert_eq!(path.to_string_path(), "user.profile");
}

// =========================================================================
// BindingDirection Additional Tests
// =========================================================================

#[test]
fn test_binding_direction_default() {
    assert_eq!(BindingDirection::default(), BindingDirection::OneWay);
}

#[test]
fn test_binding_direction_all_variants() {
    assert_eq!(BindingDirection::OneWay, BindingDirection::OneWay);
    assert_eq!(BindingDirection::TwoWay, BindingDirection::TwoWay);
    assert_eq!(BindingDirection::OneTime, BindingDirection::OneTime);
}

#[test]
fn test_binding_direction_clone() {
    let dir = BindingDirection::TwoWay;
    let cloned = dir;
    assert_eq!(dir, cloned);
}

#[test]
fn test_binding_direction_debug() {
    let dir = BindingDirection::OneTime;
    let debug = format!("{dir:?}");
    assert!(debug.contains("OneTime"));
}

// =========================================================================
// BindingConfig Additional Tests
// =========================================================================

#[test]
fn test_binding_config_chained_builders() {
    let config = BindingConfig::one_way("count", "label")
        .transform("toString")
        .fallback("N/A");

    assert_eq!(config.transform, Some("toString".to_string()));
    assert_eq!(config.fallback, Some("N/A".to_string()));
}

#[test]
fn test_binding_config_clone() {
    let config = BindingConfig::two_way("user.name", "value");
    let cloned = config;
    assert_eq!(cloned.direction, BindingDirection::TwoWay);
}

#[test]
fn test_binding_config_debug() {
    let config = BindingConfig::one_way("path", "prop");
    let debug = format!("{config:?}");
    assert!(debug.contains("BindingConfig"));
}

// =========================================================================
// ReactiveCell Additional Tests
// =========================================================================

#[test]
fn test_reactive_cell_multiple_subscribers() {
    use std::sync::atomic::{AtomicI32, Ordering};

    let cell = ReactiveCell::new(0);
    let count1 = Arc::new(AtomicI32::new(0));
    let count2 = Arc::new(AtomicI32::new(0));
    let c1 = count1.clone();
    let c2 = count2.clone();

    cell.subscribe(move |v| {
        c1.store(*v, Ordering::SeqCst);
    });
    cell.subscribe(move |v| {
        c2.store(*v * 2, Ordering::SeqCst);
    });

    cell.set(10);
    assert_eq!(count1.load(Ordering::SeqCst), 10);
    assert_eq!(count2.load(Ordering::SeqCst), 20);
}

#[test]
fn test_reactive_cell_debug() {
    let cell = ReactiveCell::new(42);
    let debug = format!("{cell:?}");
    assert!(debug.contains("ReactiveCell"));
    assert!(debug.contains("42"));
}

#[test]
fn test_reactive_cell_string() {
    let cell = ReactiveCell::new("hello".to_string());
    cell.set("world".to_string());
    assert_eq!(cell.get(), "world");
}

// =========================================================================
// Computed Additional Tests
// =========================================================================

#[test]
fn test_computed_with_closure_capture() {
    let base = 10;
    let computed = Computed::new(move || base * 2);
    assert_eq!(computed.get(), 20);
}

#[test]
fn test_computed_debug() {
    let computed = Computed::new(|| 42);
    computed.get(); // Populate cache
    let debug = format!("{computed:?}");
    assert!(debug.contains("Computed"));
}

#[test]
fn test_computed_invalidate_recomputes() {
    use std::sync::atomic::{AtomicI32, Ordering};

    let counter = Arc::new(AtomicI32::new(0));
    let counter_clone = counter;

    let computed = Computed::new(move || counter_clone.fetch_add(1, Ordering::SeqCst) + 1);

    assert_eq!(computed.get(), 1);
    assert_eq!(computed.get(), 1); // Cached

    computed.invalidate();
    assert_eq!(computed.get(), 2); // Recomputed
}

// =========================================================================
// BindingExpression Additional Tests
// =========================================================================

#[test]
fn test_binding_expression_no_deps() {
    let expr = BindingExpression::new("Hello World");
    assert!(expr.dependencies.is_empty());
}

#[test]
fn test_binding_expression_complex() {
    let expr = BindingExpression::new("{{ user.name | uppercase }} ({{ user.age }})");
    assert_eq!(expr.dependencies.len(), 2);
}

#[test]
fn test_binding_expression_not_simple() {
    let expr = BindingExpression::new("Hello {{ name }}!");
    assert!(!expr.is_simple_property());
    assert!(expr.as_property().is_none());
}

#[test]
fn test_binding_expression_clone() {
    let expr = BindingExpression::property("count");
    let cloned = expr.clone();
    assert_eq!(cloned.expression, expr.expression);
}

#[test]
fn test_binding_expression_debug() {
    let expr = BindingExpression::new("{{ test }}");
    let debug = format!("{expr:?}");
    assert!(debug.contains("BindingExpression"));
}

#[test]
fn test_binding_expression_serialize() {
    let expr = BindingExpression::property("count");
    let json = serde_json::to_string(&expr).unwrap();
    assert!(json.contains("expression"));
}

// =========================================================================
// EventBinding Additional Tests
// =========================================================================

#[test]
fn test_event_binding_clone() {
    let binding = EventBinding::on_click(ActionBinding::toggle("visible"));
    let cloned = binding;
    assert_eq!(cloned.event, "click");
}

#[test]
fn test_event_binding_debug() {
    let binding = EventBinding::new("submit", ActionBinding::dispatch("send"));
    let debug = format!("{binding:?}");
    assert!(debug.contains("EventBinding"));
}

#[test]
fn test_event_binding_serialize() {
    let binding = EventBinding::on_click(ActionBinding::toggle("flag"));
    let json = serde_json::to_string(&binding).unwrap();
    assert!(json.contains("click"));
}

// =========================================================================
// ActionBinding Additional Tests
// =========================================================================

#[test]
fn test_action_binding_empty_batch() {
    let action = ActionBinding::batch([]);
    if let ActionBinding::Batch { actions } = action {
        assert!(actions.is_empty());
    } else {
        panic!("Expected Batch");
    }
}

#[test]
fn test_action_binding_clone() {
    let action = ActionBinding::set("path", "value");
    let cloned = action;
    if let ActionBinding::SetProperty { path, .. } = cloned {
        assert_eq!(path.to_string_path(), "path");
    }
}

#[test]
fn test_action_binding_debug() {
    let action = ActionBinding::toggle("flag");
    let debug = format!("{action:?}");
    assert!(debug.contains("ToggleProperty"));
}

#[test]
fn test_action_binding_serialize() {
    let action = ActionBinding::increment("counter");
    let json = serde_json::to_string(&action).unwrap();
    assert!(json.contains("IncrementProperty"));
}

// =========================================================================
// BindingManager Additional Tests
// =========================================================================

#[test]
fn test_binding_manager_default() {
    let manager = BindingManager::default();
    assert_eq!(manager.active_count(), 0);
    assert!(manager.debounce_ms.is_none());
}

#[test]
fn test_binding_manager_multiple_registers() {
    let mut manager = BindingManager::new();
    let id1 = manager.register("w1", BindingConfig::one_way("a", "b"));
    let id2 = manager.register("w1", BindingConfig::one_way("c", "d"));
    assert_ne!(id1.0, id2.0);
    assert_eq!(manager.active_count(), 2);
}

#[test]
fn test_binding_manager_unregister_nonexistent() {
    let mut manager = BindingManager::new();
    manager.unregister(BindingId(999)); // Should not panic
}

#[test]
fn test_binding_manager_inactive_not_counted() {
    let mut manager = BindingManager::new();
    let id = manager.register("w1", BindingConfig::one_way("a", "b"));
    manager.register("w2", BindingConfig::one_way("c", "d"));
    manager.unregister(id);
    assert_eq!(manager.active_count(), 1);
}

#[test]
fn test_binding_manager_bindings_for_widget_empty() {
    let manager = BindingManager::new();
    assert!(manager.bindings_for_widget("nonexistent").is_empty());
}

#[test]
fn test_binding_manager_bindings_for_path_empty() {
    let manager = BindingManager::new();
    let path = PropertyPath::new("nonexistent");
    assert!(manager.bindings_for_path(&path).is_empty());
}

#[test]
fn test_binding_manager_on_state_change_nested_path() {
    let mut manager = BindingManager::new();
    manager.register("w1", BindingConfig::one_way("user", "data"));

    let path = PropertyPath::new("user.name");
    let updates = manager.on_state_change(&path, "Alice");

    // Should match because user.name starts with user
    assert_eq!(updates.len(), 1);
}

#[test]
fn test_binding_manager_on_state_change_inactive() {
    let mut manager = BindingManager::new();
    let id = manager.register("w1", BindingConfig::one_way("count", "text"));
    manager.unregister(id);

    let path = PropertyPath::new("count");
    let updates = manager.on_state_change(&path, "42");

    assert!(updates.is_empty());
}

#[test]
fn test_binding_manager_queue_widget_update() {
    let mut manager = BindingManager::new();
    manager.queue_update(
        UpdateSource::Widget,
        PropertyPath::new("field"),
        "value".to_string(),
    );

    let (widget_updates, state_updates) = manager.flush();
    assert!(widget_updates.is_empty());
    assert_eq!(state_updates.len(), 1);
}

#[test]
fn test_binding_manager_debug() {
    let manager = BindingManager::new();
    let debug = format!("{manager:?}");
    assert!(debug.contains("BindingManager"));
}

// =========================================================================
// ActiveBinding Tests
// =========================================================================

#[test]
fn test_active_binding_clone() {
    let binding = ActiveBinding {
        id: BindingId(1),
        widget_id: "widget".to_string(),
        config: BindingConfig::one_way("path", "prop"),
        current_value: Some("value".to_string()),
        active: true,
    };
    let cloned = binding;
    assert_eq!(cloned.id, BindingId(1));
}

#[test]
fn test_active_binding_debug() {
    let binding = ActiveBinding {
        id: BindingId(0),
        widget_id: "w".to_string(),
        config: BindingConfig::one_way("a", "b"),
        current_value: None,
        active: true,
    };
    let debug = format!("{binding:?}");
    assert!(debug.contains("ActiveBinding"));
}

// =========================================================================
// PendingUpdate Tests
// =========================================================================

#[test]
fn test_pending_update_clone() {
    let update = PendingUpdate {
        source: UpdateSource::State,
        path: PropertyPath::new("count"),
        value: "42".to_string(),
        timestamp: 12345,
    };
    let cloned = update;
    assert_eq!(cloned.timestamp, 12345);
}

#[test]
fn test_pending_update_debug() {
    let update = PendingUpdate {
        source: UpdateSource::Widget,
        path: PropertyPath::new("field"),
        value: "val".to_string(),
        timestamp: 0,
    };
    let debug = format!("{update:?}");
    assert!(debug.contains("PendingUpdate"));
}

// =========================================================================
// ConversionError Additional Tests
// =========================================================================

#[test]
fn test_conversion_error_debug() {
    let err = ConversionError {
        message: "test".to_string(),
    };
    let debug = format!("{err:?}");
    assert!(debug.contains("ConversionError"));
}

#[test]
fn test_conversion_error_clone() {
    let err = ConversionError {
        message: "original".to_string(),
    };
    let cloned = err;
    assert_eq!(cloned.message, "original");
}

// =========================================================================
// ValueConverter Additional Tests
// =========================================================================

#[test]
fn test_identity_converter_default() {
    let converter = IdentityConverter;
    assert_eq!(converter.convert("test").unwrap(), "test");
}

#[test]
fn test_identity_converter_debug() {
    let converter = IdentityConverter;
    let debug = format!("{converter:?}");
    assert!(debug.contains("IdentityConverter"));
}

#[test]
fn test_bool_converter_yes_no() {
    let converter = BoolToStringConverter::new();
    assert_eq!(converter.convert("yes").unwrap(), "true");
    assert_eq!(converter.convert("no").unwrap(), "false");
}

#[test]
fn test_bool_converter_default() {
    let converter = BoolToStringConverter::default();
    assert_eq!(converter.true_string, "");
    assert_eq!(converter.false_string, "");
}

#[test]
fn test_bool_converter_debug() {
    let converter = BoolToStringConverter::new();
    let debug = format!("{converter:?}");
    assert!(debug.contains("BoolToStringConverter"));
}

#[test]
fn test_bool_converter_convert_back_error() {
    let converter = BoolToStringConverter::with_strings("Y", "N");
    assert!(converter.convert_back("Maybe").is_err());
}

#[test]
fn test_number_format_default() {
    let converter = NumberFormatConverter::default();
    assert_eq!(converter.decimals, 0);
    assert!(converter.prefix.is_empty());
    assert!(converter.suffix.is_empty());
}

#[test]
fn test_number_format_debug() {
    let converter = NumberFormatConverter::new().decimals(2);
    let debug = format!("{converter:?}");
    assert!(debug.contains("NumberFormatConverter"));
}

#[test]
fn test_number_format_negative() {
    let converter = NumberFormatConverter::new().decimals(2);
    assert_eq!(converter.convert("-42.5").unwrap(), "-42.50");
}

#[test]
fn test_number_format_convert_back_error() {
    let converter = NumberFormatConverter::new();
    assert!(converter.convert_back("not-a-number").is_err());
}

#[test]
fn test_number_format_strip_partial() {
    let converter = NumberFormatConverter::new().prefix("$");
    // Value without prefix should still work
    assert_eq!(converter.convert_back("100").unwrap(), "100");
}
