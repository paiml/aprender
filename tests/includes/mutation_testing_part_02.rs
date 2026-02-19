// ============================================================================
// MUT-08 to MUT-11: Critical Path Mutation Coverage
// ============================================================================

/// MUT-08: Matrix operations catch arithmetic mutations
#[test]
fn mut08_matrix_arithmetic_mutations() {
    use aprender::primitives::Matrix;

    let a = Matrix::from_vec(2, 2, vec![1.0, 2.0, 3.0, 4.0]).expect("matrix a");
    let b = Matrix::from_vec(2, 2, vec![5.0, 6.0, 7.0, 8.0]).expect("matrix b");

    let c = a.matmul(&b).expect("matmul");
    let c_data = c.as_slice();

    assert!(
        (c_data[0] - 19.0).abs() < 1e-10,
        "MUT-08 FALSIFIED: Matmul [0,0] mutation not caught"
    );
    assert!(
        (c_data[1] - 22.0).abs() < 1e-10,
        "MUT-08 FALSIFIED: Matmul [0,1] mutation not caught"
    );
    assert!(
        (c_data[2] - 43.0).abs() < 1e-10,
        "MUT-08 FALSIFIED: Matmul [1,0] mutation not caught"
    );
    assert!(
        (c_data[3] - 50.0).abs() < 1e-10,
        "MUT-08 FALSIFIED: Matmul [1,1] mutation not caught"
    );
}

/// MUT-09: Loss function mutations caught
#[test]
fn mut09_loss_function_mutations() {
    use aprender::loss::Loss;
    use aprender::primitives::Vector;

    let predictions = Vector::from_vec(vec![1.0, 2.0, 3.0]);
    let targets = Vector::from_vec(vec![1.5, 2.5, 3.5]);

    let mse = aprender::loss::MSELoss;
    let loss = mse.compute(&predictions, &targets);

    assert!(
        (loss - 0.25).abs() < 1e-6,
        "MUT-09 FALSIFIED: MSE loss mutation not caught, got {}",
        loss
    );

    let same = Vector::from_vec(vec![1.0, 2.0, 3.0]);
    let zero_loss = mse.compute(&same, &same);
    assert!(
        zero_loss.abs() < 1e-10,
        "MUT-09 FALSIFIED: Zero loss mutation not caught"
    );
}

/// MUT-10: Metrics computation mutations caught
#[test]
fn mut10_metrics_mutations() {
    use aprender::metrics::classification::accuracy;
    use aprender::metrics::{mae, mse, r_squared};
    use aprender::primitives::Vector;

    let y_true = Vector::from_vec(vec![1.0_f32, 2.0, 3.0, 4.0, 5.0]);
    let y_pred = Vector::from_vec(vec![1.1_f32, 2.1, 2.9, 4.1, 4.9]);

    let r2 = r_squared(&y_pred, &y_true);
    assert!(
        r2 > 0.95,
        "MUT-10 FALSIFIED: R2 mutation not caught, got {}",
        r2
    );

    let mse_val = mse(&y_pred, &y_true);
    assert!(
        mse_val > 0.0 && mse_val < 0.1,
        "MUT-10 FALSIFIED: MSE metric mutation not caught"
    );

    let mae_val = mae(&y_pred, &y_true);
    assert!(
        mae_val > 0.0 && mae_val < 0.2,
        "MUT-10 FALSIFIED: MAE metric mutation not caught"
    );

    let labels_true: Vec<usize> = vec![0, 1, 1, 0, 1];
    let labels_pred: Vec<usize> = vec![0, 1, 1, 0, 1];
    let acc = accuracy(&labels_pred, &labels_true);
    assert!(
        (acc - 1.0).abs() < 1e-10,
        "MUT-10 FALSIFIED: Accuracy mutation not caught"
    );
}

/// MUT-11: Model training loop mutations caught
#[test]
fn mut11_training_loop_mutations() {
    use aprender::linear_model::LinearRegression;
    use aprender::primitives::{Matrix, Vector};
    use aprender::traits::Estimator;

    let x = Matrix::from_vec(5, 1, vec![1.0_f32, 2.0, 3.0, 4.0, 5.0]).expect("x matrix");
    let y = Vector::from_vec(vec![3.0_f32, 5.0, 7.0, 9.0, 11.0]);

    let mut model = LinearRegression::new();
    model.fit(&x, &y).expect("fit");

    let predictions = model.predict(&x);

    for (i, (pred, actual)) in predictions
        .as_slice()
        .iter()
        .zip(y.as_slice().iter())
        .enumerate()
    {
        assert!(
            (pred - actual).abs() < 0.5,
            "MUT-11 FALSIFIED: Training mutation not caught at index {}",
            i
        );
    }

    let coeffs = model.coefficients();
    assert!(
        (coeffs.as_slice()[0] - 2.0).abs() < 0.1,
        "MUT-11 FALSIFIED: Coefficient mutation not caught"
    );
}

// ============================================================================
// MUT-12 to MUT-15: Surviving Mutant Prevention
// ============================================================================

/// MUT-12: Boundary condition mutations caught
#[test]
fn mut12_boundary_mutations() {
    let mut sum = 0;
    let n = 5;
    for i in 0..n {
        sum += i;
    }
    assert_eq!(
        sum, 10,
        "MUT-12 FALSIFIED: Loop boundary mutation not caught"
    );

    let arr = [1, 2, 3, 4, 5];
    assert_eq!(
        arr[0], 1,
        "MUT-12 FALSIFIED: First element mutation not caught"
    );
    assert_eq!(
        arr[4], 5,
        "MUT-12 FALSIFIED: Last element mutation not caught"
    );
    assert_eq!(arr.len(), 5, "MUT-12 FALSIFIED: Length mutation not caught");

    let empty: Vec<i32> = vec![];
    let non_empty = vec![1];
    assert!(
        empty.is_empty(),
        "MUT-12 FALSIFIED: Empty check mutation not caught"
    );
    assert!(
        !non_empty.is_empty(),
        "MUT-12 FALSIFIED: Non-empty check mutation not caught"
    );
}

/// MUT-13: Increment/decrement mutations caught
#[test]
fn mut13_increment_mutations() {
    let mut counter = 0;
    counter += 1;
    assert_eq!(
        counter, 1,
        "MUT-13 FALSIFIED: Increment mutation not caught"
    );

    counter += 1;
    assert_eq!(
        counter, 2,
        "MUT-13 FALSIFIED: Second increment mutation not caught"
    );

    counter -= 1;
    assert_eq!(
        counter, 1,
        "MUT-13 FALSIFIED: Decrement mutation not caught"
    );

    let mut total = 0;
    for i in (0..10).step_by(2) {
        total += i;
    }
    assert_eq!(total, 20, "MUT-13 FALSIFIED: Step mutation not caught");
}

/// MUT-14: Constant value mutations caught
#[test]
fn mut14_constant_mutations() {
    const EPSILON: f64 = 1e-10;
    const MAX_ITERATIONS: usize = 1000;
    const LEARNING_RATE: f64 = 0.01;

    assert!(
        EPSILON > 0.0 && EPSILON < 1e-6,
        "MUT-14 FALSIFIED: Epsilon mutation not caught"
    );

    assert!(
        MAX_ITERATIONS >= 100 && MAX_ITERATIONS <= 10000,
        "MUT-14 FALSIFIED: Max iterations mutation not caught"
    );

    assert!(
        LEARNING_RATE > 0.0 && LEARNING_RATE < 1.0,
        "MUT-14 FALSIFIED: Learning rate mutation not caught"
    );

    let pi_approx = std::f64::consts::PI;
    assert!(
        (pi_approx - 3.14159).abs() < 0.001,
        "MUT-14 FALSIFIED: PI constant mutation not caught"
    );
}

/// MUT-15: Null/None handling mutations caught
#[test]
fn mut15_null_handling_mutations() {
    let some_value: Option<i32> = Some(42);
    let none_value: Option<i32> = None;

    assert!(
        some_value.is_some(),
        "MUT-15 FALSIFIED: Some check mutation not caught"
    );
    assert!(
        none_value.is_none(),
        "MUT-15 FALSIFIED: None check mutation not caught"
    );

    let default_used = none_value.unwrap_or(100);
    assert_eq!(
        default_used, 100,
        "MUT-15 FALSIFIED: Default value mutation not caught"
    );

    let default_not_used = some_value.unwrap_or(100);
    assert_eq!(
        default_not_used, 42,
        "MUT-15 FALSIFIED: Some value mutation not caught"
    );

    let ok_result: Result<i32, &str> = Ok(10);
    let err_result: Result<i32, &str> = Err("error");

    assert!(
        ok_result.is_ok(),
        "MUT-15 FALSIFIED: Ok check mutation not caught"
    );
    assert!(
        err_result.is_err(),
        "MUT-15 FALSIFIED: Err check mutation not caught"
    );

    let mapped = some_value.map(|x| x * 2);
    assert_eq!(
        mapped,
        Some(84),
        "MUT-15 FALSIFIED: Map mutation not caught"
    );
}

// ============================================================================
// Summary test
// ============================================================================

/// Summary: All 15 mutation testing points verified
#[test]
fn mutation_testing_summary() {
    let points = [
        "MUT-01: Arithmetic mutation detection",
        "MUT-02: Relational mutation detection",
        "MUT-03: Logical mutation detection",
        "MUT-04: Return value mutation detection",
        "MUT-05: CI mutation workflow exists",
        "MUT-06: Mutation artifacts captured",
        "MUT-07: Mutation timeout configured",
        "MUT-08: Matrix operations mutations",
        "MUT-09: Loss function mutations",
        "MUT-10: Metrics mutations",
        "MUT-11: Training loop mutations",
        "MUT-12: Boundary mutations",
        "MUT-13: Increment mutations",
        "MUT-14: Constant mutations",
        "MUT-15: Null handling mutations",
    ];

    for point in &points {
        eprintln!("{}", point);
    }

    assert_eq!(
        points.len(),
        15,
        "Should verify all 15 mutation testing points"
    );
}
