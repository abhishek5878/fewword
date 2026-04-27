## Summary

What does this PR change, and why? One paragraph.

## Type of change

- [ ] Bug fix (something didn't work as documented)
- [ ] New trace adapter (your format isn't auto-detected today)
- [ ] New vertical contract pack
- [ ] New contract primitive (a rule that doesn't fit existing primitives)
- [ ] CLI / DX improvement
- [ ] Documentation
- [ ] Test coverage
- [ ] Performance
- [ ] Other (describe)

## How to test

Specific commands a reviewer can run to verify this works. If the PR adds a new adapter or pack, paste a sanitized minimal trace + config that exercises it.

```bash
# example
pytest tests/unit/test_<your_thing>.py -v
fewwords run examples/<your_example>.json --config examples/<your_config>.yml
```

## Sanity checks

- [ ] Tests added for any new public function or check category
- [ ] If the PR touches contracts/*.yml, the YAML loads via `trajeval.action.load_config` (try `python -c "from trajeval.action import load_config; load_config(Path('contracts/<pack>.yml'))"`)
- [ ] If the PR adds a runtime dependency, it's justified in the PR body and added to `pyproject.toml` (or, if optional, behind an extra)
- [ ] No private trace data, customer information, API keys, or other secrets in the diff

## Linked issues / discussions

Closes #<issue>, related to #<issue>, or "n/a — direct PR."

## Anything else worth knowing?

If you're a first-time contributor: hi, thanks. If your PR is touching the LTL / Büchi runtime or `SymbolicState`, please mention so a maintainer can do a closer review — that surface ships in production for paying pilots and we're careful with semantic changes.
