# Evaluation results

## Current status

The repository now contains the offline evaluation entry points in [evaluation/](../evaluation/), but the actual benchmark numbers depend on the local datasets and installed model weights.

## What should be measured

- EER, FAR, and FRR across matcher thresholds.
- Passive liveness true-positive and false-positive rates on spoof samples.
- Backend comparison for recognition models when DeepFace is available.

## Suggested report format

When you run the evaluation scripts locally, record the following:

1. Dataset name and size.
2. Model/backend used.
3. Threshold settings.
4. EER and operating threshold.
5. Passive liveness detection rate.
6. Any failure cases observed during replay or print attack tests.

## Notes

- The evaluation screen in [ui/eval_screen.py](../ui/eval_screen.py) intentionally stays lightweight and points users to the offline evaluation package.
- Final numerical results should be updated only after you run the benchmark against your local data.

