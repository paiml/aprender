use super::*;

#[test]
fn test_arima_new() {
    let model = ARIMA::new(1, 1, 1);
    assert_eq!(model.order(), (1, 1, 1));
    assert!(model.ar_coefficients().is_none());
    assert!(model.ma_coefficients().is_none());
}

#[test]
fn test_difference() {
    let data = Vector::from_slice(&[1.0, 3.0, 6.0, 10.0]);
    let diff = ARIMA::difference(&data).expect("difference should succeed");
    assert_eq!(diff.as_slice(), &[2.0, 3.0, 4.0]);
}

#[test]
fn test_difference_insufficient_data() {
    let data = Vector::from_slice(&[1.0]);
    let result = ARIMA::difference(&data);
    assert!(result.is_err());
}

#[test]
fn test_integrate() {
    let diff = vec![2.0, 3.0, 4.0];
    let integrated = ARIMA::integrate(&diff, 1.0).expect("integrate should succeed");
    assert_eq!(integrated, vec![3.0, 6.0, 10.0]);
}

#[test]
fn test_arima_fit_basic() {
    let mut model = ARIMA::new(1, 0, 0);
    let data = Vector::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    model.fit(&data).expect("fit should succeed");

    assert!(model.ar_coefficients().is_some());
    assert_eq!(model.ar_coefficients().expect("should exist").len(), 1);
}

#[test]
fn test_arima_fit_insufficient_data() {
    let mut model = ARIMA::new(2, 1, 1);
    let data = Vector::from_slice(&[1.0, 2.0]);
    let result = model.fit(&data);
    assert!(result.is_err());
}

#[test]
fn test_arima_forecast_not_fitted() {
    let model = ARIMA::new(1, 1, 1);
    let result = model.forecast(3);
    assert!(result.is_err());
}

#[test]
fn test_arima_forecast_basic() {
    let mut model = ARIMA::new(1, 0, 0);
    let data = Vector::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    model.fit(&data).expect("fit should succeed");

    let forecast = model.forecast(3).expect("forecast should succeed");
    assert_eq!(forecast.len(), 3);

    // Forecasts should be reasonable (not NaN, not infinite)
    for &value in forecast.as_slice() {
        assert!(value.is_finite());
    }
}

#[test]
fn test_arima_with_differencing() {
    let mut model = ARIMA::new(1, 1, 0);
    let data = Vector::from_slice(&[10.0, 12.0, 15.0, 19.0, 24.0, 30.0]);
    model.fit(&data).expect("fit should succeed");

    let forecast = model.forecast(2).expect("forecast should succeed");
    assert_eq!(forecast.len(), 2);

    // Forecasts should be positive and finite
    for &value in forecast.as_slice() {
        assert!(value.is_finite());
        assert!(value > 0.0);
    }
}

#[test]
fn test_arima_order() {
    let model = ARIMA::new(2, 1, 3);
    assert_eq!(model.order(), (2, 1, 3));
}

#[test]
fn test_arima_intercept() {
    let mut model = ARIMA::new(1, 0, 0);
    let data = Vector::from_slice(&[2.0, 4.0, 6.0, 8.0, 10.0]);
    model.fit(&data).expect("fit should succeed");

    let intercept = model.intercept();
    assert!(intercept.is_finite());
}

#[test]
fn test_arima_with_ma_component() {
    // Test ARIMA(0, 0, 1) - pure MA model
    let mut model = ARIMA::new(0, 0, 1);
    let data = Vector::from_slice(&[1.0, 2.0, 1.5, 2.5, 2.0, 3.0, 2.5]);
    model.fit(&data).expect("fit should succeed");

    assert!(model.ma_coefficients().is_some());
    assert_eq!(model.ma_coefficients().unwrap().len(), 1);

    let forecast = model.forecast(2).expect("forecast should succeed");
    assert_eq!(forecast.len(), 2);
    for &value in forecast.as_slice() {
        assert!(value.is_finite());
    }
}

#[test]
fn test_arima_with_ar_and_ma() {
    // Test ARIMA(1, 0, 1) - both AR and MA
    let mut model = ARIMA::new(1, 0, 1);
    let data = Vector::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]);
    model.fit(&data).expect("fit should succeed");

    assert!(model.ar_coefficients().is_some());
    assert!(model.ma_coefficients().is_some());

    let forecast = model.forecast(3).expect("forecast should succeed");
    assert_eq!(forecast.len(), 3);
}

#[test]
fn test_arima_higher_order_differencing() {
    // Test ARIMA(1, 2, 0) - second order differencing
    let mut model = ARIMA::new(1, 2, 0);
    let data = Vector::from_slice(&[1.0, 4.0, 9.0, 16.0, 25.0, 36.0, 49.0, 64.0]);
    model.fit(&data).expect("fit should succeed");

    let forecast = model.forecast(2).expect("forecast should succeed");
    assert_eq!(forecast.len(), 2);
    for &value in forecast.as_slice() {
        assert!(value.is_finite());
    }
}

#[test]
fn test_arima_full_model() {
    // Test ARIMA(1, 1, 1) - full model
    let mut model = ARIMA::new(1, 1, 1);
    let data = Vector::from_slice(&[10.0, 12.0, 13.0, 15.0, 14.0, 16.0, 18.0, 17.0]);
    model.fit(&data).expect("fit should succeed");

    let forecast = model.forecast(3).expect("forecast should succeed");
    assert_eq!(forecast.len(), 3);

    // All forecasts should be finite
    for &value in forecast.as_slice() {
        assert!(value.is_finite());
    }
}

#[test]
fn test_arima_higher_order_ar() {
    // Test ARIMA(3, 0, 0) - AR(3) model
    let mut model = ARIMA::new(3, 0, 0);
    let data = Vector::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]);
    model.fit(&data).expect("fit should succeed");

    assert!(model.ar_coefficients().is_some());
    assert_eq!(model.ar_coefficients().unwrap().len(), 3);

    let forecast = model.forecast(2).expect("forecast should succeed");
    assert_eq!(forecast.len(), 2);
}

#[test]
fn test_arima_higher_order_ma() {
    // Test ARIMA(0, 0, 2) - MA(2) model
    let mut model = ARIMA::new(0, 0, 2);
    let data = Vector::from_slice(&[1.0, 2.0, 1.5, 2.5, 2.0, 3.0, 2.5, 3.5]);
    model.fit(&data).expect("fit should succeed");

    assert!(model.ma_coefficients().is_some());
    assert_eq!(model.ma_coefficients().unwrap().len(), 2);
}

#[test]
fn test_integrate_empty() {
    let diff: Vec<f64> = vec![];
    let integrated = ARIMA::integrate(&diff, 5.0).expect("integrate should succeed");
    assert!(integrated.is_empty());
}
