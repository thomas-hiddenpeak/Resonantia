<!--
Resonantia project skill. Adapted from addyosmani/agent-skills (MIT License).
-->

# Git Workflow and Versioning

Use this guide only when the task explicitly includes Git operations. Inspection with
`git status` or `git diff` does not authorize staging, committing, branching, pushing,
tagging, rebasing, or opening a pull request.

## Protect the working tree

1. Inspect `git status --short`, the current branch, and the relevant path-scoped
   diff before any mutation.
2. Treat every pre-existing modification and untracked file as user-owned. Do not
   discard, overwrite, stash, stage, or fold it into a commit unless the user names
   that exact scope.
3. Never use destructive cleanup (`reset --hard`, broad checkout/restore, `clean`, or
   recursive deletion) to make the tree convenient. Ask when ownership overlaps.
4. Keep generated build output, credentials, private model weights, audio assets, and
   local configuration out of Git unless the repository contract explicitly tracks
   them.

## Branches

- Create or switch branches only when requested or required by the requested publish
  workflow. Use the repository/app convention (`codex/<topic>`) unless the user
  supplies a branch name.
- Confirm the base and current branch before creating one. Do not rewrite shared
  history or rebase unrelated user work without explicit approval.

## Staging and commits

- Stage only reviewed pathspecs that belong to the authorized task. Re-run
  `git diff --staged --check` and inspect `git diff --staged` before committing.
- Commit only when requested. A successful implementation or test run is not implicit
  authorization to commit.
- Keep one logical change per commit when the existing worktree permits it. Explain
  why in the message and reference the relevant Spec/ADR when useful.
- Before a requested commit, run verification proportional to the change. Record
  failures, skips, unavailable assets, and checks not run; do not bypass hooks or
  claim a full pass from a focused test.
- For SDD changes, run `scripts/check_sdd.sh` before staging.

Suggested message types are `feat`, `fix`, `refactor`, `test`, `docs`, and `chore`;
follow the repository's existing history when it establishes a stronger convention.

## Remote and release operations

- Push, force-push, tag, merge, and open or update a PR only when explicitly requested.
- Never force-push or delete a remote branch without separate confirmation of the
  exact target.
- Inspect the outgoing commits and diff before pushing. Report the branch, remote,
  tests, and any known gaps.
- Treat release tags and version changes as separate release authorization; do not
  infer them from a completed feature.

## Handoff

Report the files staged, commit hash, branch, remote/PR URL, and verification results
only for operations that actually succeeded. If no Git mutation was requested, leave
the user's repository state unchanged.

See `.specify/memory/constitution.md` and `AGENTS.md` for governance and evidence
requirements.
