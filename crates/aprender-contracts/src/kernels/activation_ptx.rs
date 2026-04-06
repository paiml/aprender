/// PTX assembly for the `ReLU` kernel (elementwise, 1 thread per element).
pub fn relu_ptx() -> &'static str {
    r".version 8.5
.target sm_90
.address_size 64
.visible .entry relu_kernel(
    .param .u64 input,
    .param .u64 output,
    .param .u32 n
) {
    .reg .u32 %tid, %ntid, %ctaid, %idx, %n;
    .reg .u64 %in_ptr, %out_ptr, %off;
    .reg .f32 %x, %zero, %y;
    .reg .pred %p;

    mov.u32 %tid, %tid.x;
    mov.u32 %ntid, %ntid.x;
    mov.u32 %ctaid, %ctaid.x;
    mad.lo.u32 %idx, %ctaid, %ntid, %tid;
    ld.param.u32 %n, [n];
    setp.ge.u32 %p, %idx, %n;
    @%p bra DONE;

    ld.param.u64 %in_ptr, [input];
    ld.param.u64 %out_ptr, [output];
    mul.wide.u32 %off, %idx, 4;
    add.u64 %in_ptr, %in_ptr, %off;
    add.u64 %out_ptr, %out_ptr, %off;
    ld.global.f32 %x, [%in_ptr];
    mov.f32 %zero, 0f00000000;
    max.f32 %y, %x, %zero;
    st.global.f32 [%out_ptr], %y;

DONE:
    ret;
}
"
}

/// PTX assembly for the `GELU` kernel.
///
/// Uses the approximation: GELU(x) = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
/// where tanh(a) is computed as 1 - 2/(1 + exp(2a)) and
/// exp(a) = 2^(a / ln2) via `ex2.approx.f32`.
pub fn gelu_ptx() -> &'static str {
    r".version 8.5
.target sm_90
.address_size 64
.visible .entry gelu_kernel(
    .param .u64 input,
    .param .u64 output,
    .param .u32 n
) {
    .reg .u32 %tid, %ntid, %ctaid, %idx, %n;
    .reg .u64 %in_ptr, %out_ptr, %off;
    .reg .f32 %x, %y, %x3, %inner, %two_inner, %scaled;
    .reg .f32 %exp_val, %denom, %tanh_val, %one_plus_tanh, %half_x;
    .reg .f32 %k_sqrt2pi, %k_coeff, %k_half, %k_one, %k_two, %k_rcp_ln2;
    .reg .pred %p;

    mov.u32 %tid, %tid.x;
    mov.u32 %ntid, %ntid.x;
    mov.u32 %ctaid, %ctaid.x;
    mad.lo.u32 %idx, %ctaid, %ntid, %tid;
    ld.param.u32 %n, [n];
    setp.ge.u32 %p, %idx, %n;
    @%p bra DONE;

    ld.param.u64 %in_ptr, [input];
    ld.param.u64 %out_ptr, [output];
    mul.wide.u32 %off, %idx, 4;
    add.u64 %in_ptr, %in_ptr, %off;
    add.u64 %out_ptr, %out_ptr, %off;
    ld.global.f32 %x, [%in_ptr];

    // Constants
    mov.f32 %k_sqrt2pi, 0f3F4C422A;   // sqrt(2/pi) ~ 0.7978845608
    mov.f32 %k_coeff, 0f3D372713;     // 0.044715
    mov.f32 %k_half, 0f3F000000;      // 0.5
    mov.f32 %k_one, 0f3F800000;       // 1.0
    mov.f32 %k_two, 0f40000000;       // 2.0
    mov.f32 %k_rcp_ln2, 0f3FB8AA3B;   // 1/ln(2) ~ 1.442695

    // x^3
    mul.f32 %x3, %x, %x;
    mul.f32 %x3, %x3, %x;

    // inner = sqrt(2/pi) * (x + 0.044715 * x^3)
    fma.rn.f32 %inner, %k_coeff, %x3, %x;
    mul.f32 %inner, %k_sqrt2pi, %inner;

    // tanh(inner) = 1 - 2/(1 + exp(2*inner))
    // exp(2*inner) via ex2: exp(a) = 2^(a/ln2)
    mul.f32 %two_inner, %k_two, %inner;
    mul.f32 %scaled, %two_inner, %k_rcp_ln2;
    ex2.approx.f32 %exp_val, %scaled;
    add.f32 %denom, %k_one, %exp_val;
    rcp.approx.f32 %denom, %denom;
    // tanh = 1 - 2*rcp(1+exp(2*inner))
    mul.f32 %tanh_val, %k_two, %denom;
    sub.f32 %tanh_val, %k_one, %tanh_val;

    // 0.5 * x * (1 + tanh)
    add.f32 %one_plus_tanh, %k_one, %tanh_val;
    mul.f32 %half_x, %k_half, %x;
    mul.f32 %y, %half_x, %one_plus_tanh;

    st.global.f32 [%out_ptr], %y;

DONE:
    ret;
}
"
}

/// PTX assembly for the `SiLU` (Swish) kernel.
///
/// `SiLU`(x) = x / (1 + exp(-x)), where exp(-x) = 2^(-x / ln2) via `ex2.approx.f32`.
pub fn silu_ptx() -> &'static str {
    r".version 8.5
.target sm_90
.address_size 64
.visible .entry silu_kernel(
    .param .u64 input,
    .param .u64 output,
    .param .u32 n
) {
    .reg .u32 %tid, %ntid, %ctaid, %idx, %n;
    .reg .u64 %in_ptr, %out_ptr, %off;
    .reg .f32 %x, %y, %neg_x, %scaled, %exp_val, %denom, %rcp_denom;
    .reg .f32 %k_one, %k_rcp_ln2;
    .reg .pred %p;

    mov.u32 %tid, %tid.x;
    mov.u32 %ntid, %ntid.x;
    mov.u32 %ctaid, %ctaid.x;
    mad.lo.u32 %idx, %ctaid, %ntid, %tid;
    ld.param.u32 %n, [n];
    setp.ge.u32 %p, %idx, %n;
    @%p bra DONE;

    ld.param.u64 %in_ptr, [input];
    ld.param.u64 %out_ptr, [output];
    mul.wide.u32 %off, %idx, 4;
    add.u64 %in_ptr, %in_ptr, %off;
    add.u64 %out_ptr, %out_ptr, %off;
    ld.global.f32 %x, [%in_ptr];

    // Constants
    mov.f32 %k_one, 0f3F800000;       // 1.0
    mov.f32 %k_rcp_ln2, 0f3FB8AA3B;   // 1/ln(2) ~ 1.442695

    // exp(-x) = 2^(-x / ln2) = 2^(-x * (1/ln2))
    neg.f32 %neg_x, %x;
    mul.f32 %scaled, %neg_x, %k_rcp_ln2;
    ex2.approx.f32 %exp_val, %scaled;

    // silu = x / (1 + exp(-x))
    add.f32 %denom, %k_one, %exp_val;
    rcp.approx.f32 %rcp_denom, %denom;
    mul.f32 %y, %x, %rcp_denom;

    st.global.f32 [%out_ptr], %y;

DONE:
    ret;
}
"
}
