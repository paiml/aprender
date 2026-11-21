# The one ask: crates.io Trusted Publishing (0.76 P0)

On crates.io, for EACH of the 77 crates in docs/release/oidc-trusted-publisher.md, open
https://crates.io/crates/<crate>/settings -> Trusted Publishing -> Add GitHub Actions, enter:
  Repository owner: paiml | Repository name: aprender | Workflow filename: release.yml | Environment: infra-read
Then write each configured crate name (one per line) to a file and run:
  bash scripts/release/oidc_ready.sh --attest <file>      # exit 0 only when all 77 are attested
Until it exits 0, release.yml keeps token_auth=true; nothing else changes.
The script cannot read crates.io state without an owner session, so the attestation is yours; it reads no credentials.
