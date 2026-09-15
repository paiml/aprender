# silicon-coverage fixtures

Six committed cases for `scripts/check_silicon_coverage.sh --fixture <dir>`,
driven by `scripts/test_silicon_coverage_run_probe.sh`. Each directory is a
complete offline substitute for the two GitHub listings the guard reads:

| file | what it stands in for |
|---|---|
| `policy.txt` | `.github/silicon-coverage.txt` |
| `runners.tsv` | `orgs/paiml/actions/runners` + `repos/paiml/aprender/actions/runners`, filtered to `online` — `<name>\t<labels csv>` |
| `jobs.tsv.in` | the concluded-job ledger built from `runs?event=schedule` + `runs?event=workflow_dispatch` — `<completed_at>\t<conclusion>\t<runs-on csv>\t<job>\t<workflow>\t<run id>` |

`jobs.tsv.in` is a TEMPLATE, and that is deliberate. Freshness is measured
against `now`, so a committed literal timestamp would silently convert the
`covered` case into the `stale` case three days after it was written — a fixture
that changes its own verdict with the calendar proves whatever the calendar
says. The test substitutes `@FRESH@` (now) and `@STALE@` (10 days ago) into a
scratch copy.

The cases, one per form variant — the house rule is a fixture per VARIANT, not
per form:

| dir | runner online? | matching run? | axis status | verdict | exit |
|---|---|---|---|---|---|
| `covered` | yes | yes, fresh | required | `ok` | 0 |
| `uncovered` | yes | **no** | required | `UNCOVERED` | 1 |
| `wrongjob` | yes | right labels, **wrong job name** | required | `UNCOVERED` | 1 |
| `stale` | yes | yes, 10d old | required | `STALE` | 1 |
| `ready` | yes | no | pending | `ready` | 0 |
| `promote` | yes | yes, fresh | pending | `PROMOTE` | 1 |

`uncovered` is the one R-5 exists for: before the run probe, a runner online with
no run scored as coverage.

`wrongjob` is the second half of the same defect, and it was measured rather
than imagined — the first live run of the run probe scored `x86_64-cpu` covered
by `Silicon Nightly / summary`, the workflow's needs-aggregator, which runs on
the intel pool and executes nothing. A pool label set does not name an axis, so
the policy's 4th column names the job.

`ready` is the behaviour that CHANGED. It used to be a hard `PROMOTE` failure on
runner existence alone, which is promotion with no evidence that anything ever
executed — the exemption the ledger exists to prevent.

Two of these are also re-run against a MUTANT of the guard whose run probe has
been deleted; both must go green there, or the probe is not what produces the
verdict. See the header of the test.

A note on the glob. The real leg declares `name: ada-yoga (x86_64, sm_89)`, so
the API reports that string and not the job id — which is why the policy column
is `job:ada-yoga*` and why the fixtures carry the full name. A fixture built
from the job ID alone would have passed while the live guard scored the axis
UNCOVERED, which is the fixture proving something the fleet does not do.
