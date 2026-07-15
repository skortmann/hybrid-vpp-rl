# Security policy

## Reporting a vulnerability

Please report security issues through
[GitHub security advisories](https://github.com/skortmann/hybrid-vpp-rl/security/advisories/new)
rather than public issues.

## Credentials and private data

* **Never commit credentials.** API tokens (Renewables.ninja, Weights &
  Biases) are read from environment variables or a gitignored `local.env` /
  `.env` file in the project root — see `.env.example` for the expected
  variable names.
* **Never commit market databases.** The private market database is
  configured via `MARKET_DATABASE_PATH`; `*.db` files and all data
  directories are gitignored. The framework generates a synthetic drop-in
  database for use without private data.
* Experiment outputs (`runs/`, `wandb/`) may embed host and path details
  and are gitignored — review anything you export from them.

If you discover a committed secret in the repository history, report it via
a security advisory; the affected credential must be rotated and history
rewritten.
