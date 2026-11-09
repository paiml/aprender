// decide-apr-v1 rung 8: no second minting path for a verified Decider.
//
// There is no constructor besides the ladder doors `Decider::load_bytes`,
// `Decider::load_hashed` and `Decider::load_path`, so `Decider::new` must not resolve.

use aprender_decide::Decider;

fn main() {
    let _forged = Decider::new();
}
