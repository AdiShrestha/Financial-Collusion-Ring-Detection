# Reproduction

The source layout is preserved under `source/`. Use Python with the scientific dependencies declared in `source/pyproject.toml`; the retained dependency records distinguish declared package requirements from the broader observed development environment. Creating a clean, minimal environment lock for the final model campaign remains outstanding.

From the repository root, run the small data-independent implementation fixtures with:

```sh
python3 -B -m unittest source.tests.test_research_invariants
```

These fixtures check extraction identities, split integrity and mathematical properties on explicitly constructed examples. They do not generate empirical benchmark results or validate an entire future model comparison.

The development summary publishes exact aggregated findings and metric definitions without distributing the source datasets. Full scientific reproduction also requires the publisher's version-pinned files, the complete family index, exact annotation joins and independently checked scoring artifacts. Those artifacts have been retained locally but are not presently a public reproduction package. An ordinary clone can inspect the implementation and run its fixtures; it cannot reconstruct the complete reported census from the summary alone.

IBM AMLworld's publisher source is [IBM AML-Data](https://github.com/IBM/AML-Data), which links the released data and licensing terms. Use the named v8 LI-Small source under the applicable terms. No source timestamp, amount, identity or negative laundering label may be invented to fill unavailable data. Account-sidecar metadata is unavailable because the literal account-table join failed; that finding is part of the data limitations.

The historical `old` tag preserves the earlier repository. Its checkpoints, manuscript and reported metrics are historical material, not outputs of the current exploratory study. The current repository does not inherit their performance claims.
