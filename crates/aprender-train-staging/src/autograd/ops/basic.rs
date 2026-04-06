//! Basic autograd operations: add, mul, scale, sum

use crate::autograd::{BackwardOp, Tensor};
use ndarray::Array1;
use std::cell::RefCell;
use std::rc::Rc;

/// Add two tensors
pub fn add(a: &Tensor, b: &Tensor) -> Tensor {
    let data = a.data() + b.data();
    let requires_grad = a.requires_grad() || b.requires_grad();

    let mut result = Tensor::new(data, requires_grad);

    if requires_grad {
        let a_clone = a.clone();
        let b_clone = b.clone();
        let backward_op =
            Rc::new(AddBackward { a: a_clone, b: b_clone, result_grad: result.grad_cell() });
        result.set_backward_op(backward_op);
    }

    result
}

struct AddBackward {
    a: Tensor,
    b: Tensor,
    result_grad: Rc<RefCell<Option<Array1<f32>>>>,
}

impl BackwardOp for AddBackward {
    fn backward(&self) {
        if let Some(grad) = self.result_grad.borrow().as_ref() {
            if self.a.requires_grad() {
                self.a.accumulate_grad(grad.clone());
            }
            if self.b.requires_grad() {
                self.b.accumulate_grad(grad.clone());
            }

            // Recursively call backward on inputs
            if let Some(op) = self.a.backward_op() {
                op.backward();
            }
            if let Some(op) = self.b.backward_op() {
                op.backward();
            }
        }
    }
}

/// Multiply two tensors element-wise
pub fn mul(a: &Tensor, b: &Tensor) -> Tensor {
    let data = a.data() * b.data();
    let requires_grad = a.requires_grad() || b.requires_grad();

    let mut result = Tensor::new(data, requires_grad);

    if requires_grad {
        let a_clone = a.clone();
        let b_clone = b.clone();
        let backward_op =
            Rc::new(MulBackward { a: a_clone, b: b_clone, result_grad: result.grad_cell() });
        result.set_backward_op(backward_op);
    }

    result
}

struct MulBackward {
    a: Tensor,
    b: Tensor,
    result_grad: Rc<RefCell<Option<Array1<f32>>>>,
}

impl BackwardOp for MulBackward {
    fn backward(&self) {
        if let Some(grad) = self.result_grad.borrow().as_ref() {
            if self.a.requires_grad() {
                // ∂L/∂a = ∂L/∂out * b
                let grad_a = grad * self.b.data();
                self.a.accumulate_grad(grad_a);
            }
            if self.b.requires_grad() {
                // ∂L/∂b = ∂L/∂out * a
                let grad_b = grad * self.a.data();
                self.b.accumulate_grad(grad_b);
            }

            // Recursively call backward on inputs
            if let Some(op) = self.a.backward_op() {
                op.backward();
            }
            if let Some(op) = self.b.backward_op() {
                op.backward();
            }
        }
    }
}

/// Scale tensor by a scalar
pub fn scale(a: &Tensor, factor: f32) -> Tensor {
    let data = a.data() * factor;
    let requires_grad = a.requires_grad();

    let mut result = Tensor::new(data, requires_grad);

    if requires_grad {
        let a_clone = a.clone();
        let backward_op =
            Rc::new(ScaleBackward { a: a_clone, factor, result_grad: result.grad_cell() });
        result.set_backward_op(backward_op);
    }

    result
}

struct ScaleBackward {
    a: Tensor,
    factor: f32,
    result_grad: Rc<RefCell<Option<Array1<f32>>>>,
}

impl BackwardOp for ScaleBackward {
    fn backward(&self) {
        if let Some(grad) = self.result_grad.borrow().as_ref() {
            if self.a.requires_grad() {
                // ∂L/∂a = ∂L/∂out * factor
                let grad_a = grad * self.factor;
                self.a.accumulate_grad(grad_a);
            }

            if let Some(op) = self.a.backward_op() {
                op.backward();
            }
        }
    }
}

/// Add tensor b scaled by a factor to tensor a: result = a + scale * b
///
/// This is useful for LoRA: y = Wx + scale * (BA)x
pub fn add_scaled(a: &Tensor, b: &Tensor, scale_factor: f32) -> Tensor {
    assert_eq!(a.len(), b.len(), "Tensors must have same length");

    // Compute a + scale * b
    let a_data = a.data();
    let b_data = b.data();
    let result_data: Vec<f32> = a_data
        .iter()
        .zip(b_data.iter())
        .map(|(&a_val, &b_val)| a_val + scale_factor * b_val)
        .collect();

    let requires_grad = a.requires_grad() || b.requires_grad();
    let mut result = Tensor::new(Array1::from(result_data), requires_grad);

    if requires_grad {
        let a_clone = a.clone();
        let b_clone = b.clone();
        let backward_op = Rc::new(AddScaledBackward {
            a: a_clone,
            b: b_clone,
            scale: scale_factor,
            result_grad: result.grad_cell(),
        });
        result.set_backward_op(backward_op);
    }

    result
}

struct AddScaledBackward {
    a: Tensor,
    b: Tensor,
    scale: f32,
    result_grad: Rc<RefCell<Option<Array1<f32>>>>,
}

impl BackwardOp for AddScaledBackward {
    fn backward(&self) {
        if let Some(grad) = self.result_grad.borrow().as_ref() {
            // ∂L/∂a = ∂L/∂result * 1 = grad
            if self.a.requires_grad() {
                self.a.accumulate_grad(grad.clone());
            }

            // ∂L/∂b = ∂L/∂result * scale = grad * scale
            if self.b.requires_grad() {
                let grad_b = grad * self.scale;
                self.b.accumulate_grad(grad_b);
            }

            // Recursively call backward on inputs
            if let Some(op) = self.a.backward_op() {
                op.backward();
            }
            if let Some(op) = self.b.backward_op() {
                op.backward();
            }
        }
    }
}

/// Sum all elements
pub fn sum(a: &Tensor) -> Tensor {
    let data = Array1::from(vec![a.data().sum()]);
    let requires_grad = a.requires_grad();

    let mut result = Tensor::new(data, requires_grad);

    if requires_grad {
        let a_clone = a.clone();
        let backward_op = Rc::new(SumBackward { a: a_clone, result_grad: result.grad_cell() });
        result.set_backward_op(backward_op);
    }

    result
}

struct SumBackward {
    a: Tensor,
    result_grad: Rc<RefCell<Option<Array1<f32>>>>,
}

impl BackwardOp for SumBackward {
    fn backward(&self) {
        if let Some(grad) = self.result_grad.borrow().as_ref() {
            if self.a.requires_grad() {
                // ∂L/∂a = ∂L/∂sum * 1 (broadcast)
                let grad_val = grad[0];
                let grad_a = Array1::from(vec![grad_val; self.a.len()]);
                self.a.accumulate_grad(grad_a);
            }

            if let Some(op) = self.a.backward_op() {
                op.backward();
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array1;

    #[test]
    fn test_add_forward() {
        let a = Tensor::new(Array1::from(vec![1.0, 2.0, 3.0]), false);
        let b = Tensor::new(Array1::from(vec![4.0, 5.0, 6.0]), false);
        let result = add(&a, &b);

        assert_eq!(result.data().as_slice().expect("operation should succeed"), &[5.0, 7.0, 9.0]);
        assert!(!result.requires_grad());
    }

    #[test]
    fn test_add_backward() {
        let a = Tensor::new(Array1::from(vec![1.0, 2.0, 3.0]), true);
        let b = Tensor::new(Array1::from(vec![4.0, 5.0, 6.0]), true);
        let result = add(&a, &b);

        assert!(result.requires_grad());

        // Simulate gradient from downstream
        result.accumulate_grad(Array1::from(vec![1.0, 1.0, 1.0]));

        // Trigger backward
        if let Some(op) = result.backward_op() {
            op.backward();
        }

        // Both inputs should receive gradient of 1.0 (since d(a+b)/da = 1 and d(a+b)/db = 1)
        let a_grad = a.grad().expect("gradient should be available");
        let b_grad = b.grad().expect("gradient should be available");
        assert_eq!(a_grad.as_slice().expect("operation should succeed"), &[1.0, 1.0, 1.0]);
        assert_eq!(b_grad.as_slice().expect("operation should succeed"), &[1.0, 1.0, 1.0]);
    }

    #[test]
    fn test_add_partial_grad() {
        // Only a requires grad
        let a = Tensor::new(Array1::from(vec![1.0, 2.0]), true);
        let b = Tensor::new(Array1::from(vec![3.0, 4.0]), false);
        let result = add(&a, &b);

        result.accumulate_grad(Array1::from(vec![2.0, 3.0]));
        if let Some(op) = result.backward_op() {
            op.backward();
        }

        let a_grad = a.grad().expect("gradient should be available");
        assert_eq!(a_grad.as_slice().expect("operation should succeed"), &[2.0, 3.0]);
        assert!(b.grad().is_none());
    }

    #[test]
    fn test_mul_forward() {
        let a = Tensor::new(Array1::from(vec![2.0, 3.0, 4.0]), false);
        let b = Tensor::new(Array1::from(vec![5.0, 6.0, 7.0]), false);
        let result = mul(&a, &b);

        assert_eq!(
            result.data().as_slice().expect("operation should succeed"),
            &[10.0, 18.0, 28.0]
        );
        assert!(!result.requires_grad());
    }

    #[test]
    fn test_mul_backward() {
        let a = Tensor::new(Array1::from(vec![2.0, 3.0]), true);
        let b = Tensor::new(Array1::from(vec![4.0, 5.0]), true);
        let result = mul(&a, &b);

        assert!(result.requires_grad());

        result.accumulate_grad(Array1::from(vec![1.0, 1.0]));
        if let Some(op) = result.backward_op() {
            op.backward();
        }

        // d(a*b)/da = b, d(a*b)/db = a
        let a_grad = a.grad().expect("gradient should be available");
        let b_grad = b.grad().expect("gradient should be available");
        assert_eq!(a_grad.as_slice().expect("operation should succeed"), &[4.0, 5.0]); // grad = b
        assert_eq!(b_grad.as_slice().expect("operation should succeed"), &[2.0, 3.0]);
        // grad = a
    }

    #[test]
    fn test_mul_partial_grad() {
        let a = Tensor::new(Array1::from(vec![2.0, 3.0]), false);
        let b = Tensor::new(Array1::from(vec![4.0, 5.0]), true);
        let result = mul(&a, &b);

        result.accumulate_grad(Array1::from(vec![1.0, 1.0]));
        if let Some(op) = result.backward_op() {
            op.backward();
        }

        assert!(a.grad().is_none());
        let b_grad = b.grad().expect("gradient should be available");
        assert_eq!(b_grad.as_slice().expect("operation should succeed"), &[2.0, 3.0]);
    }

    #[test]
    fn test_scale_forward() {
        let a = Tensor::new(Array1::from(vec![1.0, 2.0, 3.0]), false);
        let result = scale(&a, 2.5);

        assert_eq!(result.data().as_slice().expect("operation should succeed"), &[2.5, 5.0, 7.5]);
        assert!(!result.requires_grad());
    }

    #[test]
    fn test_scale_backward() {
        let a = Tensor::new(Array1::from(vec![1.0, 2.0, 3.0]), true);
        let result = scale(&a, 3.0);

        assert!(result.requires_grad());

        result.accumulate_grad(Array1::from(vec![1.0, 1.0, 1.0]));
        if let Some(op) = result.backward_op() {
            op.backward();
        }

        // d(scale*a)/da = scale
        let a_grad = a.grad().expect("gradient should be available");
        assert_eq!(a_grad.as_slice().expect("operation should succeed"), &[3.0, 3.0, 3.0]);
    }

    #[test]
    fn test_scale_no_grad() {
        let a = Tensor::new(Array1::from(vec![1.0, 2.0]), false);
        let result = scale(&a, 5.0);

        assert!(!result.requires_grad());
        assert!(result.backward_op().is_none());
    }

    #[test]
    fn test_add_scaled_forward() {
        let a = Tensor::new(Array1::from(vec![1.0, 2.0, 3.0]), false);
        let b = Tensor::new(Array1::from(vec![4.0, 5.0, 6.0]), false);
        let result = add_scaled(&a, &b, 0.5);

        // result = a + 0.5 * b = [1+2, 2+2.5, 3+3] = [3, 4.5, 6]
        let expected = vec![3.0, 4.5, 6.0];
        let actual = result.data().as_slice().expect("operation should succeed");
        for (e, a) in expected.iter().zip(actual.iter()) {
            assert!((e - a).abs() < 1e-6);
        }
    }

    #[test]
    fn test_add_scaled_backward() {
        let a = Tensor::new(Array1::from(vec![1.0, 2.0]), true);
        let b = Tensor::new(Array1::from(vec![3.0, 4.0]), true);
        let result = add_scaled(&a, &b, 2.0);

        result.accumulate_grad(Array1::from(vec![1.0, 1.0]));
        if let Some(op) = result.backward_op() {
            op.backward();
        }

        // d(a + scale*b)/da = 1, d(a + scale*b)/db = scale
        let a_grad = a.grad().expect("gradient should be available");
        let b_grad = b.grad().expect("gradient should be available");
        assert_eq!(a_grad.as_slice().expect("operation should succeed"), &[1.0, 1.0]);
        assert_eq!(b_grad.as_slice().expect("operation should succeed"), &[2.0, 2.0]);
        // scale = 2.0
    }

    #[test]
    fn test_add_scaled_partial_grad() {
        let a = Tensor::new(Array1::from(vec![1.0, 2.0]), true);
        let b = Tensor::new(Array1::from(vec![3.0, 4.0]), false);
        let result = add_scaled(&a, &b, 0.5);

        result.accumulate_grad(Array1::from(vec![2.0, 3.0]));
        if let Some(op) = result.backward_op() {
            op.backward();
        }

        let a_grad = a.grad().expect("gradient should be available");
        assert_eq!(a_grad.as_slice().expect("operation should succeed"), &[2.0, 3.0]);
        assert!(b.grad().is_none());
    }

    #[test]
    #[should_panic(expected = "Tensors must have same length")]
    fn test_add_scaled_length_mismatch() {
        let a = Tensor::new(Array1::from(vec![1.0, 2.0]), false);
        let b = Tensor::new(Array1::from(vec![3.0, 4.0, 5.0]), false);
        let _ = add_scaled(&a, &b, 1.0);
    }

    #[test]
    fn test_sum_forward() {
        let a = Tensor::new(Array1::from(vec![1.0, 2.0, 3.0, 4.0]), false);
        let result = sum(&a);

        assert_eq!(result.data().as_slice().expect("operation should succeed"), &[10.0]);
        assert!(!result.requires_grad());
    }

    #[test]
    fn test_sum_backward() {
        let a = Tensor::new(Array1::from(vec![1.0, 2.0, 3.0]), true);
        let result = sum(&a);

        assert!(result.requires_grad());

        result.accumulate_grad(Array1::from(vec![2.0]));
        if let Some(op) = result.backward_op() {
            op.backward();
        }

        // d(sum(a))/da = [1, 1, 1], scaled by incoming grad
        let a_grad = a.grad().expect("gradient should be available");
        assert_eq!(a_grad.as_slice().expect("operation should succeed"), &[2.0, 2.0, 2.0]);
    }

    #[test]
    fn test_sum_no_grad() {
        let a = Tensor::new(Array1::from(vec![1.0, 2.0, 3.0]), false);
        let result = sum(&a);

        assert!(!result.requires_grad());
        assert!(result.backward_op().is_none());
    }

    #[test]
    fn test_chained_ops_backward() {
        // Test: (a + b) * c, then sum
        let a = Tensor::new(Array1::from(vec![1.0, 2.0]), true);
        let b = Tensor::new(Array1::from(vec![3.0, 4.0]), true);
        let c = Tensor::new(Array1::from(vec![2.0, 3.0]), true);

        let ab = add(&a, &b); // [4, 6]
        let abc = mul(&ab, &c); // [8, 18]
        let result = sum(&abc); // 26

        assert_eq!(result.data()[0], 26.0);

        // Backward
        result.accumulate_grad(Array1::from(vec![1.0]));
        if let Some(op) = result.backward_op() {
            op.backward();
        }

        // d(sum)/d(abc) = [1, 1]
        // d(abc)/d(ab) = c = [2, 3]
        // d(abc)/dc = ab = [4, 6]
        // d(ab)/da = [1, 1], d(ab)/db = [1, 1]
        // So: d/da = [2, 3], d/db = [2, 3], d/dc = [4, 6]
        let a_grad = a.grad().expect("gradient should be available");
        let b_grad = b.grad().expect("gradient should be available");
        let c_grad = c.grad().expect("gradient should be available");

        assert_eq!(a_grad.as_slice().expect("operation should succeed"), &[2.0, 3.0]);
        assert_eq!(b_grad.as_slice().expect("operation should succeed"), &[2.0, 3.0]);
        assert_eq!(c_grad.as_slice().expect("operation should succeed"), &[4.0, 6.0]);
    }

    #[test]
    fn test_scale_chained_backward() {
        let a = Tensor::new(Array1::from(vec![1.0, 2.0]), true);
        let scaled = scale(&a, 3.0);
        let result = sum(&scaled);

        result.accumulate_grad(Array1::from(vec![1.0]));
        if let Some(op) = result.backward_op() {
            op.backward();
        }

        // d(sum(3*a))/da = 3 * [1, 1] = [3, 3]
        let a_grad = a.grad().expect("gradient should be available");
        assert_eq!(a_grad.as_slice().expect("operation should succeed"), &[3.0, 3.0]);
    }
}
