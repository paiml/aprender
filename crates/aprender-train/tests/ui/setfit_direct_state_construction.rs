// TRN-01 / D-06 — a lifecycle state cannot be MINTED, only reached.
//
// `SetFitRun`'s six fields are all private and there is no public constructor for any state
// but `Prepared` (`SetFitRun::<Prepared>::prepare`). The transitions consume `self`, so the
// only way to hold a `SetFitRun<EncoderTuned>` is to have run `tune_encoder()` and had the
// evidence gate pass. A struct literal would be a run that never tuned, carrying evidence
// nobody produced — and every later stage reads its provenance off exactly that evidence.
//
// # Why this file contains ONE attempt and not two — measured, not assumed
//
// The first draft also read a private field off a legitimately obtained run
// (`&run.encoder`, expecting E0616). Both errors do not appear: rustc's PRIVACY pass runs
// AFTER type checking, so the E0616 typeck error aborted the compile before E0451 was ever
// emitted, and the blessed snapshot then contained only the field read. The claim this case
// exists to make — that the state is not CONSTRUCTIBLE — would have been silently absent
// from its own evidence. Two claims that fail in different compiler passes cannot share a
// snapshot; the constructor is the load-bearing half, so it is the half that is here.
// (Reading the field is not a boundary in any case: `encoder()` is a public accessor.)
//
// The `allow` is deliberate and narrow: `unimplemented!()` has type `!`, so every field
// after the first is unreachable and the warning would otherwise ride in the snapshot as
// noise about macro expansion rather than about privacy.
//
// Expected diagnostic: `fields ... of struct 'SetFitRun' are private` (E0451).

use entrenar::train::setfit::{EncoderTuned, SetFitRun};

#[allow(unreachable_code)]
fn forge_a_tuned_run() -> SetFitRun<EncoderTuned> {
    SetFitRun {
        encoder: unimplemented!(),
        dataset: unimplemented!(),
        selection: unimplemented!(),
        config: unimplemented!(),
        evidence: unimplemented!(),
        _state: unimplemented!(),
    }
}

fn main() {}
