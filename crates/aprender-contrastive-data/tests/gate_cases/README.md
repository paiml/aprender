# D-04 gate case table

The `contrastive-data-boundary` gate's source half is a set of text patterns, and a
pattern is only ever as good as the cases it was tested against. Five spellings of the
ban shipped originally; `use std::{fs, net::TcpStream};` matched none of them and
compiled, so the gate passed a module that wrote to disk and opened a socket.

These fixtures are the regression record. `make contrastive-data-boundary-cases` runs the
gate's real detectors over every file here and checks the verdict against the filename:

  must_match_*.rs      the detectors MUST flag this (real fs/net/path reach)
  must_not_match_*.rs  the detectors MUST stay silent (legitimate std usage)

They live in a subdirectory, so cargo does not auto-discover them as test targets.

When you change a pattern, re-run the table. Do not re-read the pattern and reason about
it -- every one of this gate's misses was found by a case, none by review.
