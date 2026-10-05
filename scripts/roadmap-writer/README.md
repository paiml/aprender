# roadmap-writer — the one writer of `docs/roadmaps/roadmap.yaml`

Operator ruling RQ-8 (T21): pull requests commit `docs/roadmaps/entries/<ID>.yaml` only. Once a night,
`scripts/roadmap_writer_nightly.sh` regenerates the aggregate on `origin/main` in a dedicated clone and
opens ONE pull request that changes `roadmap.yaml` and nothing else, then arms it into the merge queue
(`gh pr merge --squash --auto`). It never pushes to `main`, never force-pushes, never uses `--admin` or
`--delete-branch`: it lands only through the PR and the queue (rows N10, N11 go RED otherwise).
An open writer PR that is already fresh is re-armed (N14). A normal run ignores `ROADMAP_WRITER_MUTANT`; only the
self-test's `--selftest-child` mode, against a local fixture origin, reads it (N12, N13).

It runs as a host systemd **user** timer, not a GitHub Actions workflow, so it does not depend on the
repository setting that lets Actions create pull requests.

    make roadmap-writer-install     # dedicated clone + installed script + units, timer enabled
    systemctl --user list-timers roadmap-writer.timer
    journalctl --user -u roadmap-writer.service -n 40

The account whose `gh` and git credentials the unit runs under is the identity that pushes the
`roadmap-writer/<date>` branch and opens the PR; that choice is an operator ruling.

Self-test: `bash scripts/roadmap_writer_nightly.sh --selftest` and `--mutants` (bare local origin, fake `gh`).
