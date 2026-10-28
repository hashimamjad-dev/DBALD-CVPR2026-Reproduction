# Permanent project workflow

- Only original DBALD Table 1 reproduction is active. Do not modify paused V1/V2/V3 work.
- Remote development destination is `hashim-dev`; no additional remote branches or PRs without explicit permission. Never force-push. Merge to `main` requires explicit approval.
- Use separate local `milestone/...` branches; preserve them and all experimental evidence. Integrate tested work into local `hashim-dev` without discarding history.
- Verify live remote state before publication. If CLI authentication fails, connected GitHub publication is permitted, but disclose local/remote ancestry differences. Never expose credentials or claim synchronization without verification.
- Completion reports include local milestone SHA, remote hashim-dev SHA, tests, changed files, new measurements and remaining Table 1 gaps.
