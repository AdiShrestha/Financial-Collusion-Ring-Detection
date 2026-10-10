# Reproduction

The source layout is preserved under `source/`. Use Python with the scientific dependencies declared in `source/pyproject.toml`; the retained dependency records distinguish declared package requirements from the broader observed development environment. Creating a clean, minimal environment lock for the final model campaign remains outstanding.

From the repository root, run the small data-independent implementation fixtures with:

```sh
python3 -B -m unittest source.tests.test_research_invariants
python3 -B -m unittest source.tests.test_s1_diagnostic_interface
python3 -B -m unittest source.tests.test_s1_heldout_adapter
```

These fixtures check extraction identities, split integrity, mathematical properties and the label-blind inference interface on explicitly constructed examples. They do not generate empirical benchmark results or validate an entire future model comparison.

## Frozen simulator inference

The inference adapter accepts an indexed set of already generated worlds, frozen model bytes, a study configuration and a new output directory:

```sh
python3 -s -B source/scripts/evaluate_s1_heldout.py \
  --world-index /path/to/world-index.json \
  --model-lock /path/to/model-lock.json \
  --study-config /path/to/study-config.json \
  --output-dir /path/to/prediction-results
```

It performs prediction only. It validates the registered model and input bindings, keeps labels out of the scoring input inventory, and writes a hash-bound prediction manifest before metric code joins labels. Existing prediction outputs are protected from overwrite. The public tests use synthetic fixtures; they do not run the frozen models.

The actual frozen model states, generated worlds and their index are not distributed in this repository. A public checkout can inspect the adapter and run its fixtures, but cannot reproduce the four-world empirical comparison without those separately supplied inputs.

## S1 learning diagnostic

The public diagnostic accepts ordinary paths for an indexed set of development worlds, a study configuration, and an output directory:

```sh
python3 -s -B source/scripts/diagnose_s1_learning.py \
  --world-index /path/to/world_index.json \
  --study-config source/configs/s1_learning_diagnostic.json \
  --output-dir /path/to/diagnostic-results --qualify-only

python3 -s -B source/scripts/diagnose_s1_learning.py \
  --world-index /path/to/world_index.json \
  --study-config source/configs/s1_learning_diagnostic.json \
  --output-dir /path/to/diagnostic-results
```

The first command checks the data and model paths without optimizer updates and binds the qualification to the current inputs and source. The second runs the study using that qualification. The study configuration records the fixed summary-control and microfit settings. The world index must resolve the six development worlds named there and their retained source artifacts inside the repository checkout. Optional `--contract-report`, `--failure-history`, and `--reference-json` arguments accept explicit paths; none has a project-specific default. Supply the same optional input paths to both commands. The output directory retains trial traces, checkpoints, joined predictions, diagnostic figures, and a final report. Use a new output directory for each run; existing trial evidence is protected from overwrite.

The repository does not distribute the six world artifacts or their index, so a clone alone cannot reproduce the empirical run. Public fixtures exercise only the interface and data-independent contracts; they do not run an optimizer or assert the empirical findings below.

The development summary publishes exact aggregated findings and metric definitions without distributing the source datasets. Full scientific reproduction also requires the publisher's version-pinned files, the complete family index, exact annotation joins and independently checked scoring artifacts. Those artifacts have been retained locally but are not presently a public reproduction package. An ordinary clone can inspect the implementation and run its fixtures; it cannot reconstruct the complete reported census from the summary alone.

IBM AMLworld's publisher source is [IBM AML-Data](https://github.com/IBM/AML-Data), which links the released data and licensing terms. Use the named v8 LI-Small source under the applicable terms. No source timestamp, amount, identity or negative laundering label may be invented to fill unavailable data. Account-sidecar metadata is unavailable because the literal account-table join failed; that finding is part of the data limitations.

The historical `old` tag preserves the earlier repository. Its checkpoints, manuscript and reported metrics are historical material, not outputs of the current exploratory study. The current repository does not inherit their performance claims.

## Reproduce released metric figures

The complete exported metric CSV is sufficient for the neutral plotting utility. Use the dependencies declared in `source/pyproject.toml`, including matplotlib. From the repository root:

```sh
python3 -s -B source/scripts/plot_s1_results.py \
  --metrics-csv results/s1_benchmark/metrics_by_world_initialization.csv \
  --output-dir /path/to/new-figures
python3 -s -B -m unittest discover -s source/tests -p test_s1_publication_plot.py
```

The utility refuses duplicate/incomplete model–world–setting coverage, nonfinite metrics and inconsistent counts, and protects existing figures from overwrite. Optional `--preview-dir` writes local PNG previews. It averages saved AP across settings within worlds and renders every predeclared contrast. It does not load a model, generate a world or compute scores. Null required AP makes the corresponding aggregate unavailable instead of dropping that world.

The manuscript, data card and result dictionary explain target/alias/phase/missingness conventions. Released CSVs contain full aggregate metrics and counts only. Figure reproduction and synthetic fixtures do not reproduce the empirical run: frozen states, raw worlds and candidate-level evidence are not distributed. The benchmark shares a fixed simulator mechanism, and the published draft does not establish population precision, qualified neural learning, financial generalization or external review/custody assurance.
