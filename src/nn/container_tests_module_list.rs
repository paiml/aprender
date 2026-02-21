use super::*;

#[test]
fn test_module_list_empty() {
    let list = ModuleList::new();
    assert!(list.is_empty());
    assert_eq!(list.len(), 0);

    let params = list.parameters();
    assert!(params.is_empty());
}

#[test]
fn test_sequential_train_propagates() {
    let mut model = Sequential::new()
        .add(Dropout::new(0.5))
        .add(Dropout::new(0.3));

    // Start in training mode
    assert!(model.training());

    // Switch to eval
    model.eval();
    assert!(!model.training());

    // Test forward still works in eval mode
    let x = Tensor::ones(&[4, 10]);
    let _ = model.forward(&x);
}

#[test]
fn test_module_list_multiple_get() {
    let list = ModuleList::new()
        .add(Linear::new(10, 8))
        .add(Linear::new(8, 5))
        .add(Linear::new(5, 3));

    // Test getting each module
    for i in 0..3 {
        let module = list.get(i);
        assert!(module.is_some(), "Module {} should exist", i);
    }
}

#[test]
fn test_module_dict_iter_forward() {
    let dict = ModuleDict::new()
        .insert("l1", Linear::new(10, 8))
        .insert("l2", Linear::new(8, 5))
        .insert("l3", Linear::new(5, 3));

    // Use iter to forward through each module in sequence
    let mut x = Tensor::ones(&[2, 10]);
    for (_name, module) in dict.iter() {
        x = module.forward(&x);
    }

    // Final shape should be from l3
    assert_eq!(x.shape(), &[2, 3]);
}

#[test]
fn test_sequential_deep_network() {
    // Build a deeper network
    let model = Sequential::new()
        .add(Linear::new(100, 64))
        .add(ReLU::new())
        .add(Linear::new(64, 32))
        .add(ReLU::new())
        .add(Linear::new(32, 16))
        .add(ReLU::new())
        .add(Linear::new(16, 10));

    assert_eq!(model.len(), 7);

    let x = Tensor::ones(&[8, 100]);
    let y = model.forward(&x);
    assert_eq!(y.shape(), &[8, 10]);
}
