// TRN-07 / D-14 — canonical test access requires a LOCK, and a token is not forgeable.
//
// `CanonicalTestToken`'s four fields are private and the only mint is
// `SelectionLock::mint_test_token(&run)`, which runs `verify_integrity()` first and reads the
// artifact hash off the run OBJECT. A struct literal here would hand `CanonicalTestAccess::
// grant` a token whose `lock_hash` names a lock that never existed, which is the whole point
// of gating the canonical test split: the number reported on it must be attributable to a
// selection decision that was committed BEFORE the split was read.
//
// The token is deliberately still a public TYPE with public accessors — a caller must be able
// to log which lock it holds. The claim is about the CONSTRUCTOR, not about hiding the type.
//
// Expected diagnostic: `cannot construct 'CanonicalTestToken' with struct literal syntax due
// to private fields` (E0451).

use entrenar::train::setfit::lock::CanonicalTestToken;

fn forge_a_token() -> CanonicalTestToken {
    CanonicalTestToken {
        lock_hash: String::new(),
        artifact_hash: String::new(),
        dataset_fingerprint: String::new(),
        validation_split_fingerprint: String::new(),
    }
}

fn main() {}
