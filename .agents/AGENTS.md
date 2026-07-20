# Workspace Rules

## Confirmation Before Action

Before making any file edits, running any commands, or creating/deleting files, always ask the user for explicit approval first.

Describe what you are about to do and why, then wait for the user to say "yes", "go ahead", or equivalent before proceeding.

Do not batch multiple actions together without approval for each step.

## Engineering Philosophy

Simplicity is the mark of expertise, not complexity.

Before building anything:
- Start with the simplest thing that works
- No fallbacks unless explicitly asked
- No "just in case" code
- No over-engineering
- One path, done right

A good engineer breaks complex concepts into simple, clear building blocks.
Build the strong foundation first. Everything else comes after.

## Build Iteratively — Validate Every Step

Never build everything at once. Always take the smallest possible step.

If validation is possible at that stage — validate. Run a script, a dry run, a build check, a test command, observe output.

If validation is not possible at that stage — use common sense, move on, and validate at the next meaningful milestone.

A step is not done until it is verified or explicitly confirmed as unverifiable at that point.
