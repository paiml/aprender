
#[test]
fn test_linear_svm_convergence() {
    let x = Matrix::from_vec(4, 2, vec![0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0])
        .expect("4x2 matrix with 8 values");
    let y = vec![0, 0, 1, 1];

    // With very few iterations, might not converge
    let mut svm_few_iter = LinearSVM::new().with_max_iter(10).with_learning_rate(0.01);
    svm_few_iter
        .fit(&x, &y)
        .expect("Training should succeed with valid data");

    // With many iterations, should converge better
    let mut svm_many_iter = LinearSVM::new().with_max_iter(2000).with_learning_rate(0.1);
    svm_many_iter
        .fit(&x, &y)
        .expect("Training should succeed with valid data");

    // Both should train successfully
    assert!(svm_few_iter.weights.is_some());
    assert!(svm_many_iter.weights.is_some());
}

#[test]
fn test_linear_svm_default() {
    let svm1 = LinearSVM::new();
    let svm2 = LinearSVM::default();

    assert_eq!(svm1.c, svm2.c);
    assert_eq!(svm1.learning_rate, svm2.learning_rate);
    assert_eq!(svm1.max_iter, svm2.max_iter);
}
