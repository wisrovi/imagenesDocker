---
name: custom-conventional-commits
description: Guides writing git commit messages that follow the User's Custom Conventional Commits specification (One file per commit + Bracket tags). Use this whenever the user asks to "write a commit message", "make a conventional commit", or review/lint an existing commit message. This skill generates commit message text only.
---

# Custom Conventional Commits (Strict Mode)

Reference for producing git commit messages that comply with the user's custom specification, which merges the [Conventional Commits](https://www.conventionalcommits.org/) philosophy with strict enterprise rules (One file = One commit).

**Scope: message generation only.** This skill produces commit message text — it does not stage files or run `git commit` unless explicitly ordered. Present the result(s) as fenced code block(s).

## Message structure

```text
[<TYPE>(optional scope)<optional !>] <description>

[optional body]

[optional footer(s)]
```

## Core rules (normative)

1. **One file = One commit (CRITICAL):** Every single modified file MUST have its own independent commit. Never group multiple files into a single commit message.
2. **Strictly English:** All commit messages, bodies, and footers MUST be written in perfect English.
3. The commit MUST be prefixed with a **type** in UPPERCASE enclosed in brackets (e.g., `[FEATURE]`, `[FIX]`, `[DOCS]`, `[REFACTOR]`, `[CHORE]`, `[TEST]`).
4. A **scope** MAY follow the type inside the brackets: a lowercase noun describing a section of the codebase in parentheses, e.g., `[FIX(parser)]`.
5. The **description** immediately follows the closing bracket and a space. It must be a short summary in the imperative mood, e.g., `[FIX(parser)] Prevent array parsing issue`.
6. An OPTIONAL longer **body** MAY follow, starting one blank line after the description. **Always write the body as a bulleted list of items (one change/point per line, `- ` prefix)** rather than a prose paragraph.
7. OPTIONAL **footers** MAY follow, one blank line after the body. Each footer is a token, then `: `, e.g., `Reviewed-by: Z` or `Refs: #123`.
8. **Breaking changes** MUST be flagged one of two ways (or both):
   - Append `!` right before the closing bracket in the prefix, e.g., `[FEATURE(api)!] Send an email when a product is shipped`. 
   - Add a footer: `BREAKING CHANGE: <description>` (uppercase, exact token).
9. Types other than `[FEATURE]` / `[FIX]` are allowed. Commonly used: `[BUILD]`, `[CHORE]`, `[CI]`, `[DOCS]`, `[STYLE]`, `[REFACTOR]`, `[PERF]`, `[TEST]`.

## Workflow for generating commit message(s)

### Step 1 — Determine what's actually staged

1. Run `git diff --name-only` to get the list of modified/staged files.
2. If multiple files are modified, remember Rule #1: **You must generate one distinct commit message per file**.

### Step 2 — Draft the messages (One per file)

For **each** file modified:
1. **Determine the type** (`[FEATURE]`, `[FIX]`, `[DOCS]`, etc.) based on what changed in that specific file.
2. **Determine the scope** if applicable (e.g., if modifying `src/api/auth.py`, the scope could be `(auth)` -> `[FEATURE(auth)]`).
3. **Check for breaking changes** in that file's logic. If yes, add `!` before the closing bracket.
4. **Write the description**: imperative mood ("add", not "added"), concise, no trailing period.
5. **Write the body**: Bullet points `- ` explaining "why" or "what" if the description isn't enough.

### Step 3 — Output

Present one fenced code block per file modified, clearly labeled with the filename above it.

## Examples

**Simple fix without scope:**
```text
[FIX] Prevent racing of requests

- Introduce a request id and a reference to the latest request
- Dismiss incoming responses other than from the latest request
```

**New feature with scope:**
```text
[FEATURE(lang)] Add Polish language translations
```

**Breaking change via `!` + scope:**
```text
[FEATURE(api)!] Drop support for Node 6 legacy endpoints

- Remove all deprecated v1 user endpoints
- Migrate routing logic to require v2 token format
```

**Docs-only, no body:**
```text
[DOCS] Correct spelling of CHANGELOG
```

## SemVer mapping (for reasoning about release impact)

| Commit contains | SemVer bump |
|---|---|
| `[FIX]` type | PATCH |
| `[FEATURE]` type | MINOR |
| `BREAKING CHANGE` (footer or `!`), any type | MAJOR |
| any other type, no breaking change | no implicit bump |
