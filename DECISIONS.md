# DECISIONS.md

Five key choices we made, with alternatives considered and reasons for each.

---

## 1. Why Machine Learning (area E)?

**Decision**: Machine Learning as the primary area, with Data Science (D) and MLOps (F) as necessary supporting work.

**Alternatives considered**:
- **Data Engineering (A)**: The data has real quality issues (encoding, ID format mismatches, decommissioned gateways still in telemetry) — it would have been a valid choice. But the baseline's 80% false positive rate screams "there's a better model here." Data engineering work is baked into our feature pipeline anyway.
- **Software Development (B)**: Wrapping this in a REST API is nice to have, but the core problem is *which 15 gateways to visit*. A beautiful API serving bad predictions is still bad.
- **DevOps (C)**: We do containerize everything (`docker compose`), but the value is in the model, not the container.

**Why ML**: The baseline (3-sigma on 3 metrics) catches only 3 of 60 truly bad gateways per week. That's a 95% miss rate. With 58 telemetry columns, field visit history, meter reads, and gateway metadata, there's far more signal than the baseline uses. A LightGBM model with 68 engineered features catches 5.5 bad gateways per week on average — nearly double the baseline. That translates to **€3,000 per evaluation window** in savings.

---

## 2. What "needs a visit" means

**Decision**: A gateway "needs a visit" if it has a high probability of an engineer finding and fixing a fault — specifically, if our model's risk score places it in the top 15 for that week.

**Alternatives rejected**:
- **Any statistical anomaly** (what the baseline does): Too broad. 60.7% of historical visits found nothing wrong. Statistical anomalies don't equal real faults.
- **Low meter read rate only**: A useful signal (our #5 feature by importance), but not sufficient alone. A gateway can have low reads for reasons other than hardware failure.
- **Expert rule system**: We have 58 telemetry columns. Manually setting rules for each is error-prone and doesn't generalize.

**What the model actually pays attention to** (top 5 features by gain):
1. `meters_at_risk` — how many meters are affected if this gateway fails
2. `read_rate_trend_4w` — declining meter read success over 4 weeks
3. `days_since_last_visit` — longer unvisited = higher risk
4. `n_past_visits` — gateways that keep needing visits tend to need more
5. `last_read_rate` — current meter read performance

---

## 3. LightGBM over alternatives

**Decision**: LightGBM binary classifier with cost-sensitive weighting (scale_pos_weight = 600/380 = 1.58).

**Alternatives considered**:
- **XGBoost**: Comparable performance, but slower training. In the live session, we need to retrain in under 2 minutes — LightGBM does it in 10 seconds.
- **Random Forest**: No native support for cost-sensitive learning. Would need sample weights as a workaround.
- **Neural networks**: With 6,400 training samples across 320 gateways, we're in small-data territory. Neural nets would overfit.
- **Unsupervised anomaly detection** (Isolation Forest, autoencoder): No way to optimize for the €380/€600 cost structure without labels.
- **Keeping the 3-sigma baseline**: Works, but wastes ~80% of visits on healthy gateways.

**Why LightGBM wins**: 
- Trains in 10 seconds (critical for live session)
- Native handling of missing values (meter reads are stale during the scored window)
- Built-in feature importance (the brief asks "what is it paying attention to?")
- Deterministic with fixed seed (MLOps reproducibility requirement)

---

## 4. Training labels from field visit outcomes, not engineer review

**Decision**: Use historical field visit outcomes ("Fehler behoben" = fault fixed) as proxy labels for training. Reserve the engineer review (Feb 15, 2026) strictly for validation.

**Alternatives rejected**:
- **Using engineer review for training**: Temporal leakage — the review is from Feb 15, 2026, which is *inside* the scored prediction window (Feb 2 – Mar 23). Using it for training would mean the model has seen the "answers" for some of the weeks it's predicting.
- **Unsupervised (no labels)**: Possible, but can't optimize for cost. We'd be guessing which anomalies matter.
- **Using raw anomaly counts as labels**: Circular — the model would learn to predict anomalies, which is what the baseline already does.

**Trade-off**: Field visit labels are imperfect. They're biased by which gateways the current team *chose* to visit. Gateways that were never visited might be broken but are labeled as healthy. We document this as a known limitation.

---

## 5. Excluding decommissioned gateways from predictions

**Decision**: Filter out any gateway whose `decommissioned_on` date falls before or during the prediction week.

**Alternatives considered**:
- **Include them anyway**: The baseline actually selects decommissioned gateway `02EBC6CD4398` for week 2026-02-02 — that gateway was decommissioned on Feb 4. Visiting it is a guaranteed €380 waste with zero chance of finding a fixable fault. We wrote a specific test for this bug.
- **Only exclude if decommissioned before the scored window**: Three gateways were decommissioned *during* the scored window (Feb 2026). Our approach handles this correctly — we check per-week, not globally.

**Why this matters**: With only 15 visits per week, wasting even one on a decommissioned gateway means missing a truly broken one. At €600 per missed fault per week, this is an expensive mistake.

---

## What this system cannot do

1. **Predict hardware failures with no telemetry precursors**: If a gateway dies suddenly (power outage, physical damage), there's no telemetry pattern to learn from. The model only works when failure has symptoms.

2. **Account for external factors**: Weather, construction near the antenna, firmware bugs in new releases — none of these are in the data. The model will see their *effects* on telemetry, but can't predict the cause.

3. **Validate on the full fleet**: The engineer review covers only 120 of 320 gateways (37.5%). We're evaluating on a sample. The model's performance on the unreviewed 200 gateways is unknown.

4. **Handle label bias**: Our training labels come from gateways the team *chose* to visit. Gateways that were never visited are assumed healthy — but some may have been broken and ignored. This biases the model toward the current team's selection pattern.

5. **Replace human judgment**: The risk score and reason are decision support, not decisions. The operations manager should override the model when they have information it doesn't (e.g., "we're replacing that gateway next Tuesday anyway").

**What two more weeks would fix**:
- Collect labels from the scored window to expand the training set
- Build a feedback loop: engineer reports from visits improve next week's predictions
- Add weather data as external features
- Calibrate the risk scores (currently they're relative rankings, not true probabilities)
