# PMAT-4127 implementation receipt — unmaintained proc-macro crates out of published aprender

Issue: paiml/aprender#4127 (found by aprender-70 on apr-cookbook#454).

## Change
- `crates/apr-cli`: `tabled` 0.16 -> 0.22. `tabled_derive` 0.12 no longer depends on
  `proc-macro-error` (RUSTSEC-2024-0370). apr-cli uses only `builder::Builder` and
  `settings::{Style, Modify, Alignment, object::Columns}`; no source change was needed.
- `crates/aprender-train`, `crates/aprender-simulate`: `validator` 0.20 -> 0.21, and the lockfile
  moves `validator_derive` 0.20.0 -> 0.20.1, which uses `proc-macro-error3` in place of
  `proc-macro-error2` (RUSTSEC-2026-0173).
- `deny.toml`: both ignores (RUSTSEC-2024-0370, RUSTSEC-2026-0173) removed. CI's `cargo deny check`
  (ci.yml, "cargo-deny (licences, bans, sources, advisories)") now fails if either crate returns.

## Evidence (measured 2026-09-24)
| check | origin/main | this branch |
|---|---|---|
| `cargo deny check advisories`, ignores removed, workspace lock | FAILED `error[unmaintained]` RUSTSEC-2026-0173 (stale lock) | ok |
| consumer crate, fresh `generate-lockfile`, path deps on apr-cli + aprender-train + aprender-simulate: `cargo tree -i proc-macro-error` | `proc-macro-error v1.0.4` | no match (rc 101) |
| same consumer: `cargo tree -i proc-macro-error2` | no match | no match |
| same consumer: `cargo deny check advisories` (this deny.toml) | FAILED `error[unmaintained]` proc-macro-error | ok |
| full `cargo deny check` | ok | ok, same warning set |
| clippy `-p apr-cli -p aprender-train -p aprender-simulate --lib --no-deps -D warnings` (lambda) | — | rc 0 |
| `cargo test -p apr-cli --lib output::` (gx10 aarch64) | — | 73 passed |
| `cargo test -p aprender-train --lib -- config::validate storage::preflight` | — | 121 passed |
| `cargo test -p aprender-simulate --lib -- config error` | — | 158 passed |
| `cargo fmt --all -- --check` | — | rc 0 |

## Honest scope note
From a consumer's FRESH resolve, `proc-macro-error2` is already absent at origin/main:
validator 0.20 accepts validator_derive 0.20.1. apr-cookbook's hit came from its lockfile
pinning validator_derive 0.20.0. validator 0.21 also requires only `validator_derive = "0.20"`, so the bump
does not force 0.20.1 on a consumer with a stale lock; `cargo update -p validator_derive` does.
The tabled bump is the change a fresh consumer actually needs. apr-cookbook can delete both
ignores once it takes the release carrying this and refreshes its lock.

Not done here: the aarch64 clippy failure hit on gx10 is pre-existing and unrelated (#4134).
