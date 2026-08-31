# TEST-ONLY signing key

`pr-review-test-TEST-ONLY.key` is an **unencrypted minisign secret key that signs test
fixtures and nothing else**. It is committed deliberately so
`tests/fixtures/pr-review/build-fixtures.sh` can regenerate and re-sign every fixture,
and so the committed signatures can be verified from a clean checkout with no setup.

It is not a credential. It authenticates nothing, protects nothing, and grants no access.

The **production** key of PR-REVIEW-SKILL-002 v2 §4.3 is a different key entirely, and it
now exists. **PRREV-013 owns it** — this paragraph used to hedge "which PRREV-005/006
owes" while SKILL.md said PRREV-006 and `ci.yml` said PRREV-005, so all three named a
ticket and none of them shipped it. `PR_REVIEW_SIGNING_KEY` appeared in zero workflows.

- **Public half: `.github/pr-review.pub`, committed.** `scripts/check_pr_review_receipt.sh`
  defaults `PR_REVIEW_PUBKEY` to that path and REJECTED every receipt while it was absent —
  an unverifiable signature is not a verified one. The first real receipt this repository
  produced (`evidence/pr-review/2795/f5fe147.../`) ACCEPTed only with `PR_REVIEW_PUBKEY`
  pointed at a throwaway; against the repository default it was
  `REJECT [B1] public key .github/pr-review.pub is absent`. It now ACCEPTs under the
  default, with no override.
- **Secret half: never in this repository.** It is held by whoever runs the reviewer, at
  the path `$PR_REVIEW_SIGNING_KEY` names (§4.3 signs with `minisign -S -s "$…"`, which
  takes a FILE PATH, not key material). A copy is escrowed in the repository secret
  `PR_REVIEW_SIGNING_KEY_B64` — base64 of that file — so a CI-side signer can materialise
  it into `$RUNNER_TEMP` and export the path.
- **Rotation, if you want a key whose custody never touched an agent:**
  `minisign -G -W -p .github/pr-review.pub -s <secret>`, re-set the secret, re-sign every
  committed receipt, commit the new public half. The signature's meaning does not change:
  §4.3 and `attestation_level: L1-self` already say it proves provenance and not
  diligence.

Per §4.3: a valid signature proves the receipt came from the signing environment. It does
not prove the review was honest, and the receipt says `"attestation_level": "L1-self"` for
exactly that reason. Conflating provenance with diligence would be the most sophisticated
form of theater this repository has yet produced.
