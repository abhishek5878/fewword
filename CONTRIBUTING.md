# Contributing to fewwords

Two kinds of contribution move fewwords forward.

## 1. Send us a trace

If your production agent got hit by a failure that isn't in our
corpus, and you're willing to let us reconstruct it into a contract,
open an issue here or email abhishekvyas02032001@gmail.com. We read
every submission personally and reply within two business days
during early access.

## 2. Ship a pull request

Bug reports, adapter fixes, and new adapters for trace formats we
don't yet cover are all welcome. For larger changes (new contract
primitives, new engine paths), open an issue first — the engine
internals diverge slightly from this public snapshot and we'll
want to coordinate.

### Running tests

```bash
pip install -e ".[dev]"
pytest tests/
```

### Style

- `ruff check .` must pass.
- `mypy --strict src/trajeval/` must pass on every file you touch.
- One concern per PR.

## Scope

- **Public here**: SDK, adapters, assertions, the guard primitive,
  the CLI, basic examples.
- **Invite-only (private)**: the 14-incident corpus, the full contract
  pack, the benchmark methodology, the drift layer, the contract-
  suggestion engine. Email for a private-repo invite if you're
  operating in production and want the full stack.
