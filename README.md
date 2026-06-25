# Portfolio MDP

Research project for dynamic allocation across four asset classes using a
monthly Markov Decision Process.

## Development

Install the project and development tools:

```bash
uv sync
```

Run the checks:

```bash
uv run pytest
uv run ruff check .
uv run ruff format --check .
```

Build the processed monthly dataset:

```bash
uv run python scripts/build_dataset.py
```

Generate the passive benchmark report:

```bash
uv run python scripts/run_benchmarks.py
```

The data layer downloads real Yahoo Finance close prices for the configured
asset universe, then builds monthly USD return series for the Argentine equity
basket, CEDEAR basket, SPY, and GLD.

Implemented roadmap status:

- Milestone 0: project setup
- Milestone 1: real monthly data layer
- Milestone 2: passive benchmark framework
- Milestone 3: discrete action space
- Milestone 4: portfolio environment
- Milestone 5: feature/state builder

The environment follows the project Gym-like interface:

```python
state = env.reset()
next_state, reward, done, info = env.step(action_id)
```
