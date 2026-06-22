# PoissonGuard

A behavioral anomaly detector that models per-entity event rates as **Poisson
processes** and is explicitly **hardened against adversarial baseline poisoning**
("boil-the-frog" slow-drift attacks).

Most rate-based detectors silently re-learn their own baseline from recent
traffic. An attacker who ramps activity slowly enough can drag the baseline up
until their target rate looks "normal." PoissonGuard's centerpiece is a
**drift-integrity gate** that distinguishes benign regime changes from
adversarial slow poisoning and refuses to absorb the latter into the baseline.

## What it does
<img width="1469" height="812" alt="Screenshot 2026-06-23 at 4 12 53 AM" src="https://github.com/user-attachments/assets/a76d248c-f7bb-4190-804d-9d5a69043a83" />
<img width="1430" height="789" alt="Screenshot 2026-06-23 at 4 12 58 AM" src="https://github.com/user-attachments/assets/3efa6016-4406-49d3-bb12-1e919878d512" />
<img width="1454" height="812" alt="Screenshot 2026-06-23 at 4 17 00 AM" src="https://github.com/user-attachments/assets/d0e690f7-14ba-4df3-a383-65631d1f70d4" />
<img width="1452" height="797" alt="Screenshot 2026-06-23 at 4 13 07 AM" src="https://github.com/user-attachments/assets/4673dd39-914c-4bcb-871b-a6ab8f554f83" />



For each `(entity, event_type)` window of timestamped events, PoissonGuard:

1. **Bayes rate** — Gamma-Poisson posterior over the rate; predictive p-value
   for the observed count vs. the learned baseline.
2. **Shape tests** — Fano factor (dispersion) and an inter-arrival
   exponentiality (KS) test to catch bursty/regular processes that match the
   *count* but not the *shape* of a Poisson process.
3. **Drift gate** — Page-Hinkley + slow-drift CUSUM decide whether to accept,
   freeze, or flag a baseline update, defeating poisoning.
4. **Fusion** — Fisher's method combines the sub-test p-values; a hierarchical
   empirical-Bayes prior stabilizes sparse entities.
5. **Bucketing** — diurnal (business / off-hours) splitting removes the
   calendar-span dilution bias that plagues naive rate estimators.

## Layout

```
poissonguard/   core library (schemas, bayes_rate, shape_tests, drift_guard,
                fusion, bucketing, detector, legacy, generators, serving)
scripts/        train.py, detect.py, evaluate.py
tests/          pytest suite (one file per module)
```

## Quickstart

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pip install -e .

python scripts/train.py    --out artifacts/baselines.json
python scripts/detect.py   --baselines artifacts/baselines.json
python scripts/evaluate.py --baselines artifacts/baselines.json
uvicorn poissonguard.serving:app --reload
```
