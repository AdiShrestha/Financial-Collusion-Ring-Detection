# Financial Collusion Ring Detection

Toporing studies whether graph, cellular, simplicial and persistence-based representations help recover transaction cycles. The current work focuses on exact membership in the supplied IBM AMLworld CYCLE annotations, using directed multigraphs that preserve individual transfers.

The study is exploratory. Candidate identities, transaction-preserving observations and several exact structural diagnostics have been verified. Independent evaluation, sufficient statistical precision and model qualification remain under development. No current trained-model superiority, real-world laundering efficacy or confirmatory result is claimed. Earlier repository results belong to the historical `old` tag and should not be interpreted as results of this study.

The repository contains the project implementation, small regression fixtures, scientific methods and a development summary. Large source datasets, credentials and operational working files are kept outside the published files.

Read [methods and limitations](docs/methods.md), [development findings](docs/development_findings.md), and [reproduction instructions](docs/reproduction.md). The machine-readable findings are in [results/development_summary.json](results/development_summary.json).

The implementation is under `source/src/`, with data loading, candidate extraction, graph and higher-order representations, model components and persistence utilities. These components are an implementation baseline: their availability does not establish that every helper satisfies the eventual transaction-level observation contract. In particular, aggregated views lose parallel-transfer information, and inference must respect the actual independent observation unit.

Python and project dependencies are described in `source/pyproject.toml`. From the repository root, the data-independent regression command is:

```sh
python3 -B -m unittest source.tests.test_research_invariants
```

The fixtures test implementation behavior; they are separate from empirical detection evidence. Access to the IBM source release is governed by its publisher's terms. The repository does not redistribute those raw files. Project code licensing and permissions are stated in [LICENSE](LICENSE).
