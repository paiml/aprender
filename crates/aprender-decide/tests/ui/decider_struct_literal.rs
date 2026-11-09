// decide-apr-v1 rung 8: no second minting path for a verified Decider.
//
// A struct literal must fail to compile because every field is private. This case
// holds ONLY the struct literal: rustc runs the privacy pass (E0451) after type
// checking, so a type error elsewhere in the same file (e.g. a missing
// `Decider::new`) would stop compilation first and hide the E0451 this case exists
// to prove. The missing constructor is its own case, `decider_no_constructor.rs`.

use aprender_decide::Decider;

/// Any value of any type, without an `unreachable expression` warning.
fn any<T>() -> T {
    loop {}
}

fn main() {
    let _forged = Decider {
        method: any(),
        identity: any(),
        manifest: any(),
    };
}
