# Code Style and Conventions

## Python Style
- `from __future__ import annotations` at top of files
- Type hints throughout: `list[str]`, `dict[str, int]`, `str | Path`, `Optional[...]`
- Module-level docstrings (triple-quoted, brief description)
- Function docstrings with Args/Returns sections
- Constants in UPPER_SNAKE_CASE at module level
- Private/internal helpers not prefixed (small codebase)

## Naming
- Snake_case for functions and variables
- Descriptive names; no abbreviations except well-known ones (e.g. `nt` for node_type in loops)

## Imports
- Standard lib first, then third-party
- Lazy imports inside functions when dependency is optional (e.g. `from datasets import Dataset` inside `to_hf_dataset`)

## File Format
- UTF-8 encoding
- JSONL for datasets

## Linting/Formatting
- Use `ruff check .` and `ruff format .`
- No mypy/pyright config visible; use LSP for diagnostics
