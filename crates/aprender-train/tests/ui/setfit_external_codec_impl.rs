// TRN-01 / D-07 as amended by the phase-3 review — an out-of-crate codec cannot enter the
// verification path.
//
// `verify_artifact<C: SetFitCodec>` is generic, which is what lets the crate's own tests
// substitute `EchoCodec` (a codec that returns the bundle it was given without a round trip)
// and observe the verification REFUSE it. Generic over a trait anybody can implement, that
// same seam would let a caller supply a codec whose `deserialize` returns the bundle it just
// serialized — and the artifact verification would then be comparing a model against itself
// while reporting a closed round trip.
//
// `SetFitCodec: sealed::Sealed` with `mod sealed` private is what closes it. The interesting
// property of this case is that the diagnostic must name the PRIVATE trait: `Sealed` is not
// nameable from outside, so there is no way to satisfy the supertrait, and rustc says so
// explicitly rather than merely reporting a missing bound the caller could add.
//
// Expected diagnostic: `the trait bound 'MyCodec: verify::sealed::Sealed' is not satisfied`
// (E0277), naming both `SetFitCodec` and the private `Sealed`.

use entrenar::train::setfit::bundle::SetFitBundle;
use entrenar::train::setfit::verify::{CodecError, SetFitCodec};

struct MyCodec;

impl SetFitCodec for MyCodec {
    fn format_id(&self) -> &'static str {
        "my-codec-v1"
    }

    fn serialize(&self, _bundle: &SetFitBundle) -> Result<Vec<u8>, CodecError> {
        Ok(Vec::new())
    }

    fn deserialize(&self, _bytes: &[u8]) -> Result<SetFitBundle, CodecError> {
        unimplemented!()
    }
}

fn main() {}
