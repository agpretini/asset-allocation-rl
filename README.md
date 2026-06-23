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

The current implementation covers the discrete portfolio action space only.
