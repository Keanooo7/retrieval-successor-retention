# Owner rulings

One file per ruling Brendan made. **Only written in an interactive session in which
Brendan stated the ruling** — never by a headless/overnight agent. The orchestrator's
headless settings deny writes to `docs/owner/**` (a PreToolUse hook); the queue's
`ruling:` predicate is satisfied only by a file here.

Why a path rule and not an identity check: the Studio's `gh` and git are logged in as
Brendan, so an agent's commit is indistinguishable from his by author or account.
SSH-signed ruling commits are the optional hardening.

Front matter: `id`, `date`, `stated_in` (session / where), `supersedes` (optional).
