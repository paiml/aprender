//! Link shim: the prebuilt onnxruntime (pyke) references glibc 2.38 `__isoc23_*` symbols;
//! this host has glibc 2.35, where the plain functions behave the same for the calls it makes.
#![allow(unsafe_code)]
use std::os::raw::{c_char, c_int, c_long, c_longlong, c_ulong, c_ulonglong};

extern "C" {
    fn strtol(s: *const c_char, end: *mut *mut c_char, base: c_int) -> c_long;
    fn strtoul(s: *const c_char, end: *mut *mut c_char, base: c_int) -> c_ulong;
    fn strtoll(s: *const c_char, end: *mut *mut c_char, base: c_int) -> c_longlong;
    fn strtoull(s: *const c_char, end: *mut *mut c_char, base: c_int) -> c_ulonglong;
}

/// # Safety
/// Same contract as C `strtol`.
#[no_mangle]
pub unsafe extern "C" fn __isoc23_strtol(
    s: *const c_char,
    e: *mut *mut c_char,
    b: c_int,
) -> c_long {
    strtol(s, e, b)
}
/// # Safety
/// Same contract as C `strtoul`.
#[no_mangle]
pub unsafe extern "C" fn __isoc23_strtoul(
    s: *const c_char,
    e: *mut *mut c_char,
    b: c_int,
) -> c_ulong {
    strtoul(s, e, b)
}
/// # Safety
/// Same contract as C `strtoll`.
#[no_mangle]
pub unsafe extern "C" fn __isoc23_strtoll(
    s: *const c_char,
    e: *mut *mut c_char,
    b: c_int,
) -> c_longlong {
    strtoll(s, e, b)
}
/// # Safety
/// Same contract as C `strtoull`.
#[no_mangle]
pub unsafe extern "C" fn __isoc23_strtoull(
    s: *const c_char,
    e: *mut *mut c_char,
    b: c_int,
) -> c_ulonglong {
    strtoull(s, e, b)
}
