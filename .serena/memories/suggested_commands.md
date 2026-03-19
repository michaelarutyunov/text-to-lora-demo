# Suggested Commands

## Environment Setup
```bash
uv venv
uv pip install -r requirements.txt
```

## Dataset Preparation (local, WSL)
```bash
uv run python scripts/test2/prepare_dataset_v2.py
# or with explicit path:
uv run python scripts/test2/prepare_dataset_v2.py --data-dir /path/to/synthetic_interviews

# Push to HF Hub:
uv run scripts/test2/push_dataset_to_hub.py --repo michaelarutyunov/pain-point-detection
```

## HF Jobs Training
```bash
hf jobs uv run \
    --flavor a10g-large \
    --timeout 2h \
    --secrets HF_TOKEN \
    scripts/test2/train_qlora_hf_jobs.py

hf jobs ps                    # list jobs
hf jobs logs <job-id>         # stream logs
hf jobs inspect <job-id>      # details
```

## HF Jobs Evaluation
```bash
hf jobs uv run \
    --flavor a10g-small \
    --timeout 30m \
    --secrets HF_TOKEN \
    scripts/test2/evaluate_qlora_hf_jobs.py
```

## Git
```bash
git status
git add <files>
git commit -m "feat: ..."
git push
```

## System utilities (Linux/WSL)
- File search: `rg` (ripgrep), `fd`
- Directory listing: `ls`, `find`
- Text processing: `jq`, `awk`, `sed`
