# Job Trigger Binary Detection Dataset

Binary classification dataset for detecting JTBD job_trigger nodes in consumer interview utterances.

## Dataset Description

This dataset contains utterances from Jobs-to-Be-Done (JTBD) interviews, labeled for whether they contain a **job_trigger** node.

- **Task**: Binary classification (yes/no)
- **Classes**:
  - `no` (0): Utterance does not contain a job_trigger
  - `yes` (1): Utterance contains a job_trigger
- **Data Source**: 27 v2 JTBD interviews from interview-system-v2
- **Split**: Interview-level stratified split (19 train / 3 val / 5 test interviews)

## Labels

Labels are derived from graph node `source_quotes`: a turn is labeled "yes" if any job_trigger node has a source_quote that appears in that turn's response text.

## Splits

- **train**: 238 utterances from 19 interviews
- **val**: 27 utterances from 3 interviews
- **test**: 48 utterances from 5 interviews

## Data Fields

- `utterance` (string): Full interviewee response text
- `label` (int): Binary label (0=no, 1=yes)
- `source_file` (string): Original interview filename
- `turn_number` (int): Turn index within interview

## Usage

```python
from datasets import load_dataset

dataset = load_dataset("michaelarutyunov/jtbd-binary-job-trigger")
train = dataset["train"]
```

## License

Please cite the associated research paper if using this dataset.
