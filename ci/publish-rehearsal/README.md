# aprender-publish-rehearsal

Test artifact. The aprender release pipeline publishes this crate nightly, as version `0.0.<run-number>`,
to rehearse the real publish path (clean-room read, upload, index read-back) before release day.
It is not part of the aprender workspace and is not for use.

Driver: `scripts/release/publish_rehearsal.sh`. Case table: `scripts/release/falsify_publish_rehearsal.sh`.
The version in this tree stays `0.0.0`; the script sets the real version in a scratch copy.
