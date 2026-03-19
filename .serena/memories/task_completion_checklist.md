# Task Completion Checklist

When a coding task is complete:

1. Run `ruff check . --fix` — auto-fix linting issues
2. Run `ruff format .` — format code
3. Check LSP diagnostics for errors
4. If dataset changed: re-run `prepare_dataset_v2.py`, commit `data/processed/`
5. If pushing to HF Hub: run `push_dataset_to_hub.py`
6. Git commit with conventional message (feat:, fix:, refactor:, etc.)
7. Git push

Note: No test suite present in this project — verification is done by running scripts end-to-end.
