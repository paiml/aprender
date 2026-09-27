# PVL-001 EV-2 (#4080): pv proof-status --binding resolves bindings; a ghost binding is a reject.
# Reads tests/fixtures/pvl/ and contracts/ from the workspace root (the resolver is CWD-sensitive).
cargo test -p aprender-contracts-cli --test pvl_ghost_binding
