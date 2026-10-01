// SAFE-03 / D-11 — the frozen linear probe cannot become a SetFit run.
//
// `FrozenProbeRun` is the baseline: encode with a FROZEN encoder, fit a head. It is the
// control the phase's headline claim is measured against, so the one thing it must never be
// able to do is present itself as the method under test. There is no `From`/`Into` between it
// and any `SetFitRun<_>` state, and — because `SetFitRun`'s constructors are private — there
// cannot be one written outside this crate either.
//
// The conversion is attempted through `Into` rather than by naming a constructor, because
// `Into` is the door a caller would actually reach for and its absence is a TRAIT bound
// failure that names both types in one diagnostic.
//
// Expected diagnostic: `the trait bound 'SetFitRun<EncoderTuned>: From<FrozenProbeRun>' is not
// satisfied` (E0277), naming `FrozenProbeRun` and `SetFitRun`.

use entrenar::train::setfit::baseline::FrozenProbeRun;
use entrenar::train::setfit::{EncoderTuned, SetFitRun};

fn launder_the_baseline(probe: FrozenProbeRun) -> SetFitRun<EncoderTuned> {
    probe.into()
}

fn main() {}
