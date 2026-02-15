use super::*;
use crate::nn::{Dropout, Linear, ReLU};

#[test]
fn test_sequential_empty() {
    let model = Sequential::new();
    assert!(model.is_empty());
    assert_eq!(model.len(), 0);

    let x = Tensor::ones(&[1, 10]);
    let y = model.forward(&x);
    assert_eq!(y.data(), x.data());
}

#[test]
fn test_sequential_single_layer() {
    let model = Sequential::new().add(Linear::with_seed(10, 5, Some(42)));

    assert_eq!(model.len(), 1);

    let x = Tensor::ones(&[2, 10]);
    let y = model.forward(&x);
    assert_eq!(y.shape(), &[2, 5]);
}

#[test]
fn test_sequential_multiple_layers() {
    let model = Sequential::new()
        .add(Linear::with_seed(10, 8, Some(42)))
        .add(ReLU::new())
        .add(Linear::with_seed(8, 5, Some(43)));

    assert_eq!(model.len(), 3);

    let x = Tensor::ones(&[4, 10]);
    let y = model.forward(&x);
    assert_eq!(y.shape(), &[4, 5]);
}

#[test]
fn test_sequential_parameters() {
    let model = Sequential::new()
        .add(Linear::new(10, 8))
        .add(ReLU::new())
        .add(Linear::new(8, 5));

    let params = model.parameters();
    // Linear(10, 8): weight + bias = 2
    // ReLU: 0
    // Linear(8, 5): weight + bias = 2
    // Total: 4
    assert_eq!(params.len(), 4);
}

#[test]
fn test_sequential_num_parameters() {
    let model = Sequential::new()
        .add(Linear::new(10, 8))
        .add(Linear::new(8, 5));

    // Linear(10, 8): 10*8 + 8 = 88
    // Linear(8, 5): 8*5 + 5 = 45
    // Total: 133
    assert_eq!(model.num_parameters(), 133);
}

#[test]
fn test_sequential_train_eval() {
    let mut model = Sequential::new()
        .add(Linear::new(10, 5))
        .add(Dropout::new(0.5));

    assert!(model.training());

    model.eval();
    assert!(!model.training());

    model.train();
    assert!(model.training());
}

#[test]
fn test_module_list_basic() {
    let layers = ModuleList::new()
        .add(Linear::new(10, 10))
        .add(Linear::new(10, 10));

    assert_eq!(layers.len(), 2);

    // Access by index
    let layer0 = layers.get(0).expect("layer 0 should exist");
    let x = Tensor::ones(&[1, 10]);
    let _ = layer0.forward(&x);
}

#[test]
fn test_module_list_parameters() {
    let layers = ModuleList::new()
        .add(Linear::new(10, 5))
        .add(Linear::new(5, 3));

    // Linear(10, 5): 2 params
    // Linear(5, 3): 2 params
    assert_eq!(layers.parameters().len(), 4);
}

#[test]
fn test_module_list_iterate() {
    let layers = ModuleList::new()
        .add(Linear::new(10, 10))
        .add(Linear::new(10, 10))
        .add(Linear::new(10, 10));

    let mut count = 0;
    for _ in layers.iter() {
        count += 1;
    }
    assert_eq!(count, 3);
}

// ModuleDict tests

#[test]
fn test_module_dict_empty() {
    let dict = ModuleDict::new();
    assert!(dict.is_empty());
    assert_eq!(dict.len(), 0);
}

#[test]
fn test_module_dict_insert_and_get() {
    let dict = ModuleDict::new()
        .insert("layer1", Linear::new(10, 5))
        .insert("layer2", Linear::new(5, 3));

    assert_eq!(dict.len(), 2);
    assert!(dict.contains("layer1"));
    assert!(dict.contains("layer2"));
    assert!(!dict.contains("layer3"));

    let layer1 = dict.get("layer1").expect("layer1 should exist");
    let x = Tensor::ones(&[2, 10]);
    let y = layer1.forward(&x);
    assert_eq!(y.shape(), &[2, 5]);
}

#[test]
fn test_module_dict_parameters() {
    let dict = ModuleDict::new()
        .insert("encoder", Linear::new(10, 5))
        .insert("decoder", Linear::new(5, 10));

    // 2 params per Linear (weight + bias) * 2 = 4
    assert_eq!(dict.parameters().len(), 4);
}

#[test]
fn test_module_dict_keys_order() {
    let dict = ModuleDict::new()
        .insert("first", Linear::new(10, 10))
        .insert("second", Linear::new(10, 10))
        .insert("third", Linear::new(10, 10));

    let keys: Vec<&str> = dict.keys().collect();
    assert_eq!(keys, vec!["first", "second", "third"]);
}

#[test]
fn test_module_dict_iterate() {
    let dict = ModuleDict::new()
        .insert("a", Linear::new(10, 10))
        .insert("b", Linear::new(10, 10));

    let mut names = Vec::new();
    for (name, _module) in dict.iter() {
        names.push(name);
    }
    assert_eq!(names, vec!["a", "b"]);
}

#[test]
fn test_module_dict_remove() {
    let mut dict = ModuleDict::new()
        .insert("keep", Linear::new(10, 5))
        .insert("remove", Linear::new(5, 3));

    assert_eq!(dict.len(), 2);

    let removed = dict.remove("remove");
    assert!(removed.is_some());
    assert_eq!(dict.len(), 1);
    assert!(!dict.contains("remove"));
    assert!(dict.contains("keep"));

    // Keys should also be updated
    let keys: Vec<&str> = dict.keys().collect();
    assert_eq!(keys, vec!["keep"]);
}

#[test]
fn test_module_dict_train_eval() {
    let mut dict = ModuleDict::new()
        .insert("layer", Linear::new(10, 5))
        .insert("dropout", Dropout::new(0.5));

    assert!(dict.training());

    dict.eval();
    assert!(!dict.training());

    dict.train();
    assert!(dict.training());
}

#[test]
fn test_module_dict_replace() {
    let dict = ModuleDict::new()
        .insert("layer", Linear::new(10, 5))
        .insert("layer", Linear::new(10, 10)); // Replace

    // Should still have only one key
    assert_eq!(dict.len(), 1);

    let layer = dict.get("layer").expect("layer should exist after replace");
    let x = Tensor::ones(&[2, 10]);
    let y = layer.forward(&x);
    assert_eq!(y.shape(), &[2, 10]); // From the second Linear
}

#[test]
fn test_sequential_default() {
    let model: Sequential = Sequential::default();
    assert!(model.is_empty());
}

#[test]
fn test_sequential_add_boxed() {
    let linear: Box<dyn Module> = Box::new(Linear::new(10, 5));
    let model = Sequential::new().add_boxed(linear);
    assert_eq!(model.len(), 1);
}

#[test]
fn test_module_list_default() {
    let list: ModuleList = ModuleList::default();
    assert!(list.is_empty());
}

#[test]
fn test_module_list_add_boxed() {
    let linear: Box<dyn Module> = Box::new(Linear::new(10, 5));
    let list = ModuleList::new().add_boxed(linear);
    assert_eq!(list.len(), 1);
}

#[test]
fn test_module_list_iter() {
    let list = ModuleList::new().add(Linear::new(10, 5)).add(ReLU::new());
    let count = list.iter().count();
    assert_eq!(count, 2);
}

#[test]
fn test_module_dict_default() {
    let dict: ModuleDict = ModuleDict::default();
    assert!(dict.is_empty());
}

#[test]
fn test_module_dict_insert_boxed() {
    let linear: Box<dyn Module> = Box::new(Linear::new(10, 5));
    let dict = ModuleDict::new().insert_boxed("layer", linear);
    assert!(dict.contains("layer"));
}

#[test]
fn test_module_dict_keys_iter() {
    let dict = ModuleDict::new()
        .insert("a", Linear::new(10, 5))
        .insert("b", ReLU::new());
    let keys: Vec<_> = dict.keys().collect();
    assert_eq!(keys.len(), 2);
    // Use iter() to verify both modules are accessible
    let modules: Vec<_> = dict.iter().collect();
    assert_eq!(modules.len(), 2);
}

// ==========================================================================
// Additional Coverage Tests
// ==========================================================================

#[test]
fn test_sequential_debug() {
    let model = Sequential::new().add(Linear::new(10, 5)).add(ReLU::new());
    let debug_str = format!("{:?}", model);
    assert!(debug_str.contains("Sequential"));
    assert!(debug_str.contains("num_modules"));
}

#[test]
fn test_sequential_parameters_mut() {
    let mut model = Sequential::new()
        .add(Linear::new(10, 8))
        .add(Linear::new(8, 5));

    let params = model.parameters_mut();
    assert_eq!(params.len(), 4);
}

#[test]
fn test_module_list_debug() {
    let list = ModuleList::new().add(Linear::new(10, 5)).add(ReLU::new());
    let debug_str = format!("{:?}", list);
    assert!(debug_str.contains("ModuleList"));
    assert!(debug_str.contains("num_modules"));
}

#[test]
fn test_module_list_get_mut() {
    let mut list = ModuleList::new()
        .add(Linear::new(10, 5))
        .add(Linear::new(5, 3));

    // Get mutable reference
    let layer = list.get_mut(0);
    assert!(layer.is_some());

    // Index out of bounds
    let missing = list.get_mut(10);
    assert!(missing.is_none());
}

#[test]
fn test_module_list_get_missing() {
    let list = ModuleList::new().add(Linear::new(10, 5));

    // Out of bounds
    let missing = list.get(5);
    assert!(missing.is_none());
}

#[test]
fn test_module_list_train_eval() {
    let mut list = ModuleList::new()
        .add(Linear::new(10, 5))
        .add(Dropout::new(0.5));

    assert!(list.training);

    list.eval();
    assert!(!list.training);

    list.train();
    assert!(list.training);
}

#[test]
fn test_module_list_parameters_mut() {
    let mut list = ModuleList::new()
        .add(Linear::new(10, 8))
        .add(Linear::new(8, 5));

    let params = list.parameters_mut();
    assert_eq!(params.len(), 4);
}

#[test]
fn test_module_dict_debug() {
    let dict = ModuleDict::new()
        .insert("layer1", Linear::new(10, 5))
        .insert("layer2", ReLU::new());
    let debug_str = format!("{:?}", dict);
    assert!(debug_str.contains("ModuleDict"));
    assert!(debug_str.contains("keys"));
}

#[test]
fn test_module_dict_get_mut() {
    let mut dict = ModuleDict::new().insert("layer", Linear::new(10, 5));

    // Get mutable reference
    let layer = dict.get_mut("layer");
    assert!(layer.is_some());

    // Key doesn't exist
    let missing = dict.get_mut("nonexistent");
    assert!(missing.is_none());
}

#[test]
fn test_module_dict_get_missing() {
    let dict = ModuleDict::new().insert("layer", Linear::new(10, 5));

    let missing = dict.get("nonexistent");
    assert!(missing.is_none());
}

#[test]
fn test_module_dict_parameters_mut() {
    let mut dict = ModuleDict::new()
        .insert("enc", Linear::new(10, 8))
        .insert("dec", Linear::new(8, 5));

    let params = dict.parameters_mut();
    assert_eq!(params.len(), 4);
}

#[test]
fn test_module_dict_remove_nonexistent() {
    let mut dict = ModuleDict::new().insert("keep", Linear::new(10, 5));

    let removed = dict.remove("nonexistent");
    assert!(removed.is_none());
    assert_eq!(dict.len(), 1); // Should be unchanged
}

#[test]
fn test_module_dict_insert_boxed_replace() {
    let linear1: Box<dyn Module> = Box::new(Linear::new(10, 5));
    let linear2: Box<dyn Module> = Box::new(Linear::new(10, 10));

    let dict = ModuleDict::new()
        .insert_boxed("layer", linear1)
        .insert_boxed("layer", linear2); // Replace

    assert_eq!(dict.len(), 1);
    let keys: Vec<_> = dict.keys().collect();
    assert_eq!(keys, vec!["layer"]);

    // Verify it's the second layer
    let layer = dict.get("layer").unwrap();
    let x = Tensor::ones(&[1, 10]);
    let y = layer.forward(&x);
    assert_eq!(y.shape(), &[1, 10]); // Output from second Linear
}

#[test]
fn test_sequential_empty_forward() {
    let model = Sequential::new();
    let x = Tensor::new(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
    let y = model.forward(&x);

    // Empty sequential should return input unchanged
    assert_eq!(y.data(), x.data());
    assert_eq!(y.shape(), x.shape());
}

include!("container_tests_part_02.rs");
