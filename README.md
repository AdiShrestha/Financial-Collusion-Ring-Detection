# Simulated AML cycle-membership benchmark

Event-preserving cycle-membership evaluation on controlled AMLSim worlds, with summary-feature baselines, local graph, cellular and simplicial models, and an audit of a temporal shortcut built into the simulator.

## What the study found

The task is to rank enumerated candidate cycles (a directed account cycle plus one logged transfer per arc) by whether they exactly match a generated laundering-cycle instance. Every world has 78 candidates and 6 members, and a member differs from its six decoys only in one transfer.

- **The simulator fixes the answer's timing.** AMLSim's cycle typology pins the first and last transfer of every cycle to the ends of its alert window. In this template every member spans exactly 29 steps, so the rule "span equals 29" reaches mean average precision (AP) 0.839 on the development worlds and 0.861 on the 30 held-out worlds.
- **Held-out evaluation (W5–W34).** Six fixed model variants, specified before the worlds were generated: gradient-boosted trees on 20 summary features, mean AP 0.916 (95% interval 0.872–0.957); a neural network on the same features, 0.701; four message-passing models, 0.229–0.275, close to the 0.143 obtained from cycle length alone.
- **Post hoc ablations.** Refitting the trees without the span feature lowers AP to 0.617, and without any time feature to 0.212.
- The comparison therefore measures how readily each learner picks up a scheduling regularity from 24 training members, not the value of relational structure. One template, one training corpus and synthetic data limit every conclusion.

Every unfavourable result is kept, including two held-out worlds where the summary neural network beats the trees.

## Repository contents

| Path | Contents |
|---|---|
| `source/scripts/` | Benchmark definition (`check_s1_benchmark.py`), world generation (`run_s1.py`), training, inference, statistics, plotting and post hoc analyses |
| `source/tests/` | Unit tests on synthetic fixtures and the released tables |
| `results/s1_benchmark/` | Per-model, per-world, per-seed metrics; development, pilot (W1–W4) and held-out (W5–W34) tables |
| `figures/s1_benchmark/` | Figures generated from those tables (PDF and SVG) |
| `docs/` | [Reproduction](docs/reproduction.md), [data card](docs/s1_benchmark_data_card.md), [results dictionary](docs/s1_results_dictionary.md), [licensing](docs/licensing.md) |
| `source/src/`, `docs/historical_*` | Earlier exploratory AMLworld and Elliptic++ work, not used by the benchmark study |

The simulated worlds, labels, fitted models and prediction files are distributed separately as a data and model archive (about 5 MB) rather than in this repository.

## Quick start

```sh
python3.12 -m venv .venv && . .venv/bin/activate
python -m pip install -r source/requirements-models.txt
python -B -m unittest discover -s source/tests
```

See [docs/reproduction.md](docs/reproduction.md) for statistics, figures, models and world generation.

## Licence and credits

Original software is licensed under [Apache-2.0](LICENSE). Documentation, result tables, figures and data are licensed under [CC BY 4.0](docs/licenses/CC-BY-4.0.txt); [docs/licensing.md](docs/licensing.md) maps each path to its licence. AMLSim is by IBM and contributors and keeps its own Apache-2.0 licence; see [NOTICE](NOTICE).
