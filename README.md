# Prefix Cache Evaluation

A causal event framework and simulator for evaluating prefix cache reuse in large language model serving.

The toolkit represents user turns as prefill requests and assistant replies as decode events. It computes causal counting bounds and runs controlled comparisons of scheduling, eviction rules, request protection and conversation construction.

## Contents

| Directory | Purpose |
| :--- | :--- |
| `revision/sim` | Event model, counting theory, simulator, experiment runners and synthetic tests |
| `revision/engine_ref` | Serial comparison harness, vendored vLLM cache components and CPU stubs |
| `reproducibility` | Dependency versions, data source configuration and verified reconstruction tools |

This repository distributes code and operating instructions. Manuscripts, historical experiment outputs, figures and conversation datasets are maintained separately.

## Setup and tests

Use Python 3.13. Run these commands from the repository root:

```sh
python -m venv .venv
.venv/bin/python -m pip install -r reproducibility/requirements.txt
PYTHONPATH=revision PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest -p no:cacheprovider revision/sim/tests
```

The tests use synthetic token sequences and run on a CPU without corpus downloads. The component harness uses PyTorch for dtype information.

## Running experiments

Follow the [data and execution guide](reproducibility/README.md) to obtain upstream datasets, verify inputs and select an experiment runner. Each runner writes new outputs to the directory passed through `--out`. Store local data, caches and outputs under `work/` inside your checkout.

The simulator measures token reuse under explicit event and cache contracts. The reference harness compares per request hit counts with selected vLLM cache components under a serial driver. Its scope is described in [the component guide](revision/engine_ref/README.md).

## Third party code

Vendored vLLM files retain their upstream copyright headers and Apache 2.0 license. See [third party notices](THIRD_PARTY_NOTICES.md). An overall reuse license for original project code has not yet been selected.
