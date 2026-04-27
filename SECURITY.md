# Security Policy

## Reporting a vulnerability

If you find a security issue in fewwords, **don't open a public issue.** Email:

`abhishekvyas02032001@gmail.com`

Include:

- A clear description of the vulnerability
- Steps to reproduce (or a working PoC)
- Your assessment of severity
- Whether you'd like credit when the fix ships (default: yes)

## What to expect

- Acknowledgment within 48 hours.
- A fix or mitigation timeline within 7 days.
- Public fix + CVE if applicable, with credit to you, after the fix ships.

## Scope

**In scope:**

- The fewwords runtime engine (`src/trajeval/`, `src/fewwords/`)
- The contract grammar parser (`.fewwords.yml` / `.trajeval.yml`)
- The CLI (`fewwords` and `trajeval` console scripts)
- Trace adapters that parse external formats

**Out of scope:**

- Third-party dependencies (report upstream)
- The hosted demo at fewword-ai.fly.dev (in scope only for issues affecting the repo's reputation)
- Social-engineering issues outside the codebase

## Hall of fame

Researchers who report valid security issues will be credited here, with their permission, after the fix ships.

*(empty for now, be the first)*
