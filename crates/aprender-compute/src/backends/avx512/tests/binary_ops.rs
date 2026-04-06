use super::*;

#[test]
fn test_avx512_add() {
    avx512_test(|| assert_binary_op(1.0, 2.0, 3.0, Avx512Backend::add));
}

#[test]
fn test_avx512_sub() {
    avx512_test(|| assert_binary_op(5.0, 2.0, 3.0, Avx512Backend::sub));
}

#[test]
fn test_avx512_mul() {
    avx512_test(|| assert_binary_op(2.0, 3.0, 6.0, Avx512Backend::mul));
}

#[test]
fn test_avx512_div() {
    avx512_test(|| assert_binary_op(6.0, 2.0, 3.0, Avx512Backend::div));
}
