# GitHub Secrets Setup for Automated Releases

This guide explains how to configure GitHub secrets for automated crates.io publishing.

## Required Secrets

### CARGO_REGISTRY_TOKEN

This token is required for automated publishing to crates.io.

#### Steps to Configure:

1. **Get your crates.io API token:**
   - Log in to https://crates.io
   - Go to Account Settings: https://crates.io/me
   - Click "New Token" under "API Tokens"
   - Give it a name like "aprender-github-actions"
   - Copy the token (it won't be shown again!)

2. **Add token to GitHub Secrets:**
   - Go to your GitHub repository: https://github.com/paiml/aprender
   - Click "Settings" → "Secrets and variables" → "Actions"
   - Click "New repository secret"
   - Name: `CARGO_REGISTRY_TOKEN`
   - Value: Paste your crates.io token
   - Click "Add secret"

3. **Enable automatic publishing:**
   - Edit `.github/workflows/release.yml`
   - Find the `publish-crate` job
   - Uncomment the actual publish line:
     ```yaml
     - name: Publish to crates.io
       run: cargo publish --token ${{ secrets.CARGO_REGISTRY_TOKEN }}
     ```
   - Comment out or remove the dry-run line
   - Commit and push the changes

## Verification

To verify the secret is configured:

1. Go to Settings → Secrets and variables → Actions
2. You should see `CARGO_REGISTRY_TOKEN` listed
3. The value will be hidden (showing only `***`)

## Security Best Practices

- **Never commit tokens to git** - always use GitHub Secrets
- **Rotate tokens periodically** - create new tokens every 6-12 months
- **Use scoped tokens** - crates.io tokens can be scoped to specific crates
- **Monitor token usage** - check crates.io for API token activity

## Testing

Before enabling automatic publishing:

1. Test locally with dry-run:
   ```bash
   cargo publish --dry-run
   ```

2. Verify the release workflow works by creating a test tag:
   ```bash
   git tag test-v0.0.1
   git push origin test-v0.0.1
   ```

3. Check the GitHub Actions run completes successfully

4. Delete the test tag:
   ```bash
   git tag -d test-v0.0.1
   git push origin :refs/tags/test-v0.0.1
   ```

## Troubleshooting

### "Invalid token" error
- Regenerate token on crates.io
- Update GitHub secret with new token

### "Permission denied" error
- Ensure token has publish permissions
- Check crate ownership on crates.io

### Workflow doesn't trigger
- Ensure tag matches pattern `v*.*.*` (e.g., `v0.1.0`)
- Check workflow file syntax is valid YAML
