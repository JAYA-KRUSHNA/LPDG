# AI-USAGE.md

## Tools Used

- **Google Gemini (Antigravity IDE)**: Pair programming, architecture planning, code generation, debugging.
- **LightGBM**: Gradient boosting framework for the ML model.

## What I Used AI For

1. **Architecture planning**: Generated the initial project structure and audited it for completeness against the challenge brief. The AI identified 11 issues in the first draft (missing deliverables, cost model misinterpretation, temporal leakage risk).

2. **Boilerplate code generation**: Data loader, config utilities, logging setup, Dockerfile, docker-compose.yml, Makefile. The pattern is standard Python packaging — AI is good at this.

3. **Feature engineering ideas**: The AI suggested feature groups (connectivity trends, reboot patterns, signal quality ratios) based on the telemetry column names. I evaluated each against the actual data distributions.

4. **Test generation**: Unit and integration tests for all modules. The AI wrote the test structure; I verified assertions match the actual data (e.g., 332 gateways in master, 12 decommissioned).

5. **Debugging**: Fixed index alignment bugs in pandas `groupby` trend computations. The AI initially used `numpy` division which failed when groups had different sizes — switched to `pandas` division with explicit `.reindex()` for alignment.

## One Thing AI Got Wrong That I Caught

**The cost model interpretation**: The AI initially described the cost structure as:
- "€380 per unnecessary visit (false positive)"
- "€600 per missed problem (false negative)"

This is subtly wrong. The brief actually says:
- **€380 per visit** — ANY visit, whether it finds a fault or not
- **€600 per week** — for every broken gateway left unvisited, compounding weekly

The difference matters: since exactly 15 visits happen per week regardless, the €380 is a fixed cost (€5,700/week always paid). The optimization target is to maximize the number of truly broken gateways in your top-15 selection — not to minimize false positives in isolation. The AI's original framing would have led to a model optimized for precision (fewer false positives) rather than recall within the top-15 (catching more broken gateways).

I caught this during manual re-reading of the brief and corrected the plan before implementation.
