# Reproduction

All commands run from the repository root. Tested with CPython 3.12.8 on macOS (arm64), CPU only.

## 1. Environment

```sh
python3.12 -m venv .venv
. .venv/bin/activate
python -m pip install -r source/requirements-supplement.txt   # tables, tests, figures
python -m pip install -r source/requirements-models.txt       # adds training, inference, scoring
```

`source/requirements-supplement.txt` is enough for everything in sections 2–4. `source/requirements-models.txt` is needed only to run models (section 5). Both files hold exact pins.

## 2. Tests

```sh
python -B -m unittest discover -s source/tests
```

All six test modules use small synthetic fixtures or the released aggregate tables; none needs worlds, models or labels. Expected: every test passes (57 tests, a few seconds). The same command runs in continuous integration (`.github/workflows/tests.yml`).

## 3. Statistics from the released tables

```sh
python -s -B source/scripts/reproduce_s1_statistics.py \
  --world-ap-csv results/s1_benchmark/replication/ap_by_world.csv \
  --output-json /tmp/statistics.json
```

Recomputes the equal-world means, the five exact signed-rank tests with Holm adjustment and the 20,000-resample paired world bootstrap (PCG64, seed 2026101103). Output matches `results/s1_benchmark/replication/summary.json`, `paired_ap_inference.csv` and `mean_ap_intervals.csv`.

## 4. Figures

The plotting script writes the PDF figures used in the paper, plus SVG copies (the SVGs are tracked here). It refuses to overwrite existing files, so write to an empty directory.

```sh
python -s -B source/scripts/plot_s1_results.py \
  --metrics-csv results/s1_benchmark/metrics_by_world_initialization.csv \
  --output-dir /tmp/figures
python -s -B source/scripts/plot_s1_results.py --panel replication \
  --metrics-csv results/s1_benchmark/replication/metrics_by_world_initialization.csv \
  --mean-intervals-csv results/s1_benchmark/replication/mean_ap_intervals.csv \
  --contrast-intervals-csv results/s1_benchmark/replication/paired_ap_inference.csv \
  --output-dir /tmp/figures
```

## 5. Models, worlds and predictions

The worlds, fitted models and prediction files are distributed separately as a data and model archive, not with this repository. With those files:

- `source/scripts/check_s1_benchmark.py` defines candidate tensors, the 20 summary features, the four neural graph and complex models and the AP and AUROC metrics.
- `source/scripts/train_s1_development.py`, `continue_s1_development.py` and `diagnose_s1_learning.py` reproduce training, continuation and the summary-MLP grid and microfits.
- `source/scripts/evaluate_s1_heldout.py` runs label-blind inference from a world index, a model lock and a study configuration.
- `source/scripts/run_s1.py` regenerates worlds. It needs a Java 11 JDK (`JAVA_HOME`), an AMLSim checkout at commit `7338a4b` and network access to fetch ten jars, which are checked against pinned SHA-256 values.

Checkpoints written by the training scripts are pickled objects; load them only from sources you trust. The supplement ships neural weights as plain tensor dictionaries (`torch.load(..., weights_only=True)`) and the GBDT as its training matrix, because the GBDT fit is deterministic and takes under a second.

## 6. Code versions and checks

The scripts here are tidied versions of the ones that produced the reported results: the same computations, formatted with Black, with clearer messages, argument-based paths and pinned jar hashes. The byte-exact executed files, with their SHA-256 values, are included in the data and model archive. `source/pyproject.toml` configures Ruff and Black; continuous integration runs both, the unit tests, the statistics and the figures.

## Legacy code

`source/src/` and `source/scripts/run_amlworld_benchmark.py`, `run_elliptic_benchmark.py` and `run_integration_test.py` belong to earlier exploratory work on AMLworld and Elliptic++ and are not used by the benchmark paper. They need the `legacy` extra in `source/pyproject.toml`. Data paths are taken from arguments, from `AMLWORLD_DATA_DIR` / `ELLIPTIC_DATA_DIR`, or from `data/raw/` in the repository; nothing else is searched. The historical `old` tag preserves the earlier repository; its metrics are not results of the current study.
