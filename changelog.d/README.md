# changelog.d/

One file per ticket: `changelog.d/<TICKET>.md` (e.g. `PMAT-1234.md`, `GH-5761.md`). A PR writes its
fragment and never edits `CHANGELOG.md`; the aggregate is generated at release, so N PRs never
contend on one file (FLOW-009 RH-1/RH-2, paiml/aprender#5761).

Fragment shape (schema only, checked per PR):

```
### Fixed

- fix(scope): what changed (Refs #N)
```

Exactly one `### <Section>` heading (Added, Changed, Fixed, Performance, Security, Removed) followed
by one or more `- ` bullets.

```
bash scripts/changelog_fragments.sh check        # schema of every fragment
bash scripts/changelog_fragments.sh aggregate    # release: fold into CHANGELOG.md [Unreleased], delete fragments
bash scripts/changelog_fragments.sh test  # case table
```

Roadmap fragments are `docs/roadmaps/entries/<id>.yaml` (apr-roadmap-fragments-v1);
`roadmap.yaml` is their generated aggregate.
