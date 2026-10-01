// TRN-01 / TRN-05 / D-06 — stage two is not reachable from stage zero.
//
// `fit_head` is declared in `impl SetFitRun<EncoderTuned>`, so it does not exist on
// `SetFitRun<Prepared>` at all. This is the ORDER half of the two-stage claim: a head fitted
// on a `Prepared` run's encoder would be a frozen linear probe (SAFE-03's baseline) reported
// under SetFit's name, which is the precise confusion the phase exists to make impossible.
//
// Note what is NOT asserted here: that `fit_head` is absent from the whole crate. It is
// present, on the state that earned it. The claim is that the METHOD SET is a function of the
// STATE, and a state parameter is not something a caller can choose.
//
// Expected diagnostic: `no method named 'fit_head' found for struct
// 'SetFitRun<Prepared>' in the current scope` (E0599).

use entrenar::train::setfit::{Prepared, SetFitRun};

fn skip_stage_one(run: SetFitRun<Prepared>) {
    let _ = run.fit_head();
}

fn main() {}
