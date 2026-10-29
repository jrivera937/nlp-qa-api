# Project Report — Extractive QA as a Service

## 0. Team

| Field | Value |
|---|---|
| Team id | `TBD` |
| Members | `TBD` |
| Repository URL | `TBD` |
| `submission` commit hash | `TBD` |
| Public service URL used at the dry run | `TBD` |
| AI tools used (which, and for what) | OpenAI Codex: repository scaffolding and training-script drafting; team must verify and disclose all tools used. |

## 1. Model choice and justification

- **Base checkpoint:** `distilbert/distilbert-base-uncased`. It is a compact encoder checkpoint intended for general language understanding, not a SQuAD-fine-tuned QA checkpoint. The training script adds a QA head and fine-tunes it on SQuAD 2.0. DistilBERT is selected as a CPU-serving candidate; the required 50-question/150-second limit still needs measurement on the team's laptop.
- **Alternatives considered:** `roberta-base` and `microsoft/deberta-v3-base` are larger candidates with potentially stronger QA quality, but greater CPU latency and memory costs. No measured comparison has been run for this project.
- **Trade-off:** The smaller DistilBERT encoder favors model footprint and CPU throughput. Quality and the 150-second target must be confirmed using the trained artifact and supplied sample set.

## 2. Training

| Item | Value |
|---|---|
| Training data (full SQuAD 2.0 train or subset; how many examples) | Script default: seeded, proportion-preserving subset of 12,000 training examples from `squad_v2`; pass `--train-limit 0` for all examples. |
| Validation data (what you validated on, how many examples) | Script default: seeded subset of 3,000 official validation examples; pass `--eval-limit 0` for all. |
| Epochs | 2 (configurable) |
| Learning rate and schedule (warmup?) | 3e-5, linear Trainer schedule, 10% warmup |
| Batch size | 8 per device |
| `max_seq_length` / `doc_stride` | 384 / 128 |
| Hardware and training time | `TBD: record CPU/RAM and actual wall time from logs/metrics.json` |

- **Loss curve:** [logs/loss_curves.png](logs/loss_curves.png) (generate after training with `python plot_training.py`).
- **Validation results:** `TBD: populate from logs/metrics.json after running training; metrics are split into answerable and unanswerable subsets.`

Run the default CPU baseline with `pip install -r requirements.txt`, `python train.py`, then `python plot_training.py`. The full dataset is downloaded by the Hugging Face Datasets library on first run. The 12k/3k subset is a pragmatic initial run, not a claim of full-data training. The script writes model files under `weights/` and logs under `logs/`.

## 3. No-answer threshold

The QA head scores candidate answer spans and the CLS null span. Use the difference `best_non_null_score - null_score`; return `null` when it is at or below the selected threshold. The current training script exports raw predictions and EM/F1 for a baseline; threshold calibration and a deployable inference adapter must be completed against held-out validation data before submission. A threshold that is too high returns too many nulls; a threshold that is too low returns unsupported spans for unanswerable questions. **Final calibrated threshold: TBD.**

## 4. Serving architecture

- **Framework and server:** `TBD (not present in the starter repository)`.
- **Model loading:** `TBD`.
- **Request flow:** `TBD`.
- **Access control:** `TBD`.
- **Tunnel:** `TBD`.
- **How to run from a clean machine:** `TBD: document the verified API setup, dependencies, and environment variables.`

## 5. Results on the public sample set

| Item | Expected | Your answer | `start` | Correct? |
|---|---|---|---|---|
| sample-01 | `TBD: from assignment sample` | `TBD` | `TBD` | `TBD` |
| sample-02 | `TBD: from assignment sample` | `TBD` | `TBD` | `TBD` |
| sample-03 | `null` | `TBD` | `TBD` | `TBD` |
| sample-04 | `null` | `TBD` | `TBD` | `TBD` |
| sample-05 | `TBD: from assignment sample` | `TBD` | `TBD` | `TBD` |

No public sample set was present in the repository, so no sample answers or correctness results are claimed.

## 6. Model size and latency (not graded)

| Item | Value |
|---|---|
| Parameters | `TBD: inspect saved model` |
| Size on disk | `TBD` |
| Mean latency per `/predict` request on your laptop (CPU) | `TBD: measure 50 questions` |
| Laptop CPU and RAM | `TBD` |

## 7. Problems encountered and how you fixed them

1. **Problem:** `TBD: record a real implementation problem.` **Cause:** `TBD` **Fix:** `TBD` **Commit:** `TBD`
2. **Problem:** `TBD: record a second real implementation problem.` **Cause:** `TBD` **Fix:** `TBD` **Commit:** `TBD`

## 8. Decoder investigation

1. **Problem:** Extractive QA can only return text that appears as one contiguous context span. A GPT-style decoder could generate a concise answer or normalize a discontinuous fact, but could also produce unsupported text; grounding and answerability checks would be needed. It would cost more CPU time and memory, so the 150-second limit would need a fresh benchmark.
2. **Problem:** The encoder QA head's null threshold may confuse answerable and unanswerable questions. A decoder could be prompted to abstain explicitly, but abstention is not guaranteed and requires calibration against the same held-out set. Generation would add latency and model-size costs.

## 9. Known limitations

The project has not yet run training or inference. The supplied script is a reproducible training/evaluation scaffold; the serving API, calibrated no-answer threshold, public sample results, loss figure, latency measurements, team details, and actual training hardware/time remain to be completed from the team's run. The default subset may score below full-data fine-tuning, and the training script requires access to download SQuAD 2.0 and the base checkpoint.
