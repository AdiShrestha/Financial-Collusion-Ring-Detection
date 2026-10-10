# Methods and limitations

The primary target is equality to a supplied IBM AMLworld CYCLE transaction instance. A candidate selects exactly one original transaction on each arc of a directed simple account cycle. Accounts are identified by the original bank token paired with the original account token. Strings remain literal: numerical conversion or removal of leading zeros changes identity and is not permitted.

The current development frame includes directed simple account cycles of lengths 2 through 12 across the complete LI-Small v8 transaction period. For a cycle family with arc multiplicities m_1,...,m_k, its transaction-tuple capacity is the exact integer product of those multiplicities. A mixed-radix index reconstructs each tuple without expanding the product population. This compresses identities; it does not establish affordable scoring for arbitrary learned models.

Labels are exact supplied-set membership. A nonmember is not established licit activity, and supplied annotations need not exhaust laundering truth. All supplied CYCLE instances remain in extraction-recall accounting, including two length-13 instances outside the current frame. Conditional candidate metrics and extraction coverage have different denominators.

Structural diagnostics use cycle length, tuple capacity, sum and maximum arc multiplicity, and mean and maximum distinct nonself indegree plus outdegree. Each quantity is evaluated in both ranking directions, alongside a constant control. These scores are constant across transaction choices within a family. Accordingly, aggregating each family with its exact tuple capacity and exact member count reproduces full-frame threshold counts. Families are not equally weighted examples. Equal scores are admitted together without identifier-based tie breaking.

For each descending score block b, AP adds (p_b/P)*(TP_b/M_b), where p_b is the block's positive count, TP_b the cumulative positive count, and M_b the cumulative candidate count. AUROC gives half credit to positive/nonmember ties. Counts, products and rational arithmetic retain exact precision. Large AUROC values must be assessed beside AP and the exceptionally sparse supplied-member prevalence.

A separate label-aware information bound optimizes over ordered partitions of the positive-containing families, including tied blocks. It bounds scalar scorers that are constant within each canonical family on this development frame. It is an information diagnostic, not predictive performance or a general ceiling for models that distinguish selected transfers. Better input resolution does not guarantee useful prediction or a topological advantage.

The verified observation reference preserves every source transfer incident to a central-cycle account, its endpoint nodes, physical direction, self-loops, parallel edges and selected-candidate mask. The context spans the full source period. This is retrospective input engineering; a chronological deployment claim requires a separately justified observation-time rule. Source timestamps have an unestablished timezone. Paid and received amounts retain separate native currency units, with no cross-currency conservation claim.

Both Small releases are exposed development data. Larger releases have not been acquired or inspected for this study. Publisher statements about independent, non-nested releases support a conditional source-design proposal but do not establish independent identically distributed worlds or configuration-level replication. Shared-account and shared-observation groups describe dependence within a corpus. Training seeds do not supply additional independent graphs.

Independent evaluation custody, the nonconstant-scoring or probability-sampling design, precision, model qualification and fair comparator execution remain unresolved. Scientific comparisons will require equivalent observations and candidate masks, credible simple and directed-multigraph baselines, and uncertainty at the justified independent unit. The existing mean-score bootstrap helper is not qualified as AP inference. No confirmatory trained-model campaign is reported here.

Context-size summaries use nearest-rank quantiles: the sorted observation at one-based ceil(q*N), with no interpolation. Shared-context account and source-row partitions include transitive overlap. Context caching preserves individual source rows and selected-edge masks; it does not prove that a learned scorer is constant within a family or independent of those masks.

The within-arc resolution audit groups rows only by permitted timestamp, exact native paid/received values with their currencies, and payment format. Amount spelling and custody identifiers do not distinguish observations. Counts were checked by exhaustive enumeration of the finite choices in the supplied-positive families and by comparison with raw attributes. This label-conditioned audit cannot choose model features, predictive strata or runtime representatives.

For a fixed finite negative stratum of size Q, uniform sampling without replacement gives the exact hypergeometric mass choose(K,k)*choose(Q-K,n-k)/choose(Q,n). The candidate confidence set retains feasible K only when both inclusive tails exceed half the allocated cell error. Cell budgets sum to at most 1/20 across the registered threshold/stratum/scorer family. Summed negative-count bounds propagate monotonically to precision and grouped-tie AP; jointly valid AP bounds give a conservative paired difference without an independence approximation. All certainty positives are accounted for, and no-alert precision remains undefined.

Source-free exhaustive fixtures check this arithmetic and explicitly measure AP plug-in bias and interval width. With an incomplete sample, zero observed negative hits do not generally establish a zero population tail. Completed large-count boundary checks use exact integer/rational arithmetic; timeouts remain inconclusive. Fixture qualification supplies neither an adequate empirical sample size nor independent evaluation support. Arbitrary learned scoring and estimator/allocation feasibility remain unresolved.

For zero sample hits, the no-hit probability is exactly choose(Q-K,n)/choose(Q,n)=choose(Q-n,K)/choose(Q,K). Strict inclusive-tail inversion gives an upper bound at most u when the probability at K=u+1 is at most half the cell error budget, for u<Q; a bound u>=Q already covers the stratum. With p certainty positives at a threshold, precision bounds [p/(p+U),1] have half-width at most epsilon, for 0<epsilon<1/2, precisely when U<=floor(2*epsilon*p/(1-2*epsilon)). The checked frontiers retain exact successful-minimum and failing-predecessor inequalities. These statements concern a fixed zero-hit threshold; generic grouped-tie AP needs its full weighted propagation.

Exact short-tail recurrences and complements improve arithmetic feasibility while preserving the confidence definition. A coefficient product for a particular additive selected-transfer score can potentially aggregate its tuple weights, but matching that scalar score does not establish equivalent observations for a different learned model. Source independence, custody and actual comparison cost require separate evidence.

Static inspection found that existing model-input paths require integration work before they can implement the verified observation reference: selected-transfer masks are missing or replaced by central-endpoint membership, and some lifts aggregate parallel/directed transfers or omit self-loops. These differences must be disclosed or resolved under a common observation contract before comparative claims. This inspection did not execute models or establish a predictive result.

The current full-frame and bounded-population alternatives lack a jointly qualified independent-evaluation, prospective-access, model-input and affordable campaign design. Candidate restrictions change the population and require scientific justification and complete supplied-instance coverage accounting. A separately defined simulator study changes the target/source and requires its own realized-event truth and evaluation qualification. The bounded simulator comparison below is development evidence for that separate target; it does not resolve the AMLworld population design. Campaign count, storage-layout and hypothetical throughput scenarios do not establish universal resource minima or measured learned-model cost.

A planning snapshot dated 8 October 2026 described a controlled AMLSim simulation design before any worlds had been generated. The follow-up below reports what the later development-only work actually completed; the two targets remain separate.

## Controlled simulation development follow-up — 9 October 2026

The separate target is exact recovery of complete realized AMLSim CYCLE-alert transaction instances, including SAR and non-SAR instances. Six separately initialized development worlds used one fixed, disclosed generator mechanism and 32 accounts over 120 steps. Each produced 77 persisted noncash transfer rows, six complete and in-frame CYCLE instances, and the full 78-tuple candidate frame: six supplied-instance members and 72 nonmembers. These are controlled simulation outcomes, not financial records, real-world prevalence estimates, or evidence that a nonmember is licit activity. The six worlds are development realizations conditional on the specified generator; seed numbers alone do not establish independent institutions or an IID population.

A same-seed replay of the first world matched all 22 checked scientific artifacts: generated input CSVs, unmodified and instrumented native transaction rows, observation sidecars, candidate identities and labels, graph identity, and incidence operators. Only the absolute topology-file path and timing metadata were excluded from comparison. The replay is not an additional world or population unit.

Four fixed, untrained shortcut rankings were reported separately by world. Across the six worlds, cycle length had AP 0.1429 and tie-aware AUROC 0.7500 in every world. Minimum selected log amount ranged from AP 0.0657–0.0922 and AUROC 0.3495–0.5590; negative log-amount span ranged from AP 0.0605–0.0756 and AUROC 0.3079–0.4456; negative time span ranged from AP 0.1271–0.1627 and AUROC 0.6204–0.6921. These are construction diagnostics, not trained-model performance; no ranking was selected or tuned and candidates were not pooled as independent replications.

The matched untrained neural interfaces used 8,390–8,533 trainable parameters, a 1.68% maximum relative spread. Every arm passed finite-gradient, local-feature-use, and scalar basis/permutation checks on each world. The GBDT API smoke used a separate deterministic toy fixture; no model was fit to the simulated candidate labels. No predictive efficacy, convergence, or model qualification is established.

The reference computation found 53 essential H1 classes per world for the unfilled port-expanded graph, matching its internal graph-cycle reference. External persistent-homology backend parity remains unresolved because GUDHI was unavailable in the measured environment.

One 78-candidate forward/backward pass measured about 24–41 ms across the neural arms. The end-to-end world-generation/native execution subprocess took about 0.9–1.1 seconds per world; benchmark-check subprocesses took about 2.6–3.3 seconds. These measurements cover this small simulated frame and interface smoke only. They do not measure convergence, full training, repeated initialization, independent evaluation, storage at larger scale, or total campaign cost. The dense-linear forward FLOP subtotal omits incidence multiplication, aggregation, nonlinearities, loss, and backward work. Future campaign size and epochs remain symbolic.

Reproduce the data-independent mathematical checks with:

```sh
python3 -s -B -m unittest source.tests.test_s1_benchmark_contract
```

The benchmark CLI takes ordinary `--world-dir`, `--output-dir`, and `--model-config` arguments. Reproduction of the measured results additionally requires the corresponding locally generated world artifacts and the recorded configuration; those data artifacts are not part of the public source release.

No untouched evaluation worlds were created or inspected for the AMLworld candidate study. Independent evaluation access, a justified world-level precision plan, external backend parity, and a fully costed predictive campaign remain open. The benchmark qualification described above involved no model fitting; the following development diagnostic added bounded fitting on existing worlds.

## S1-F learning diagnostics

The bounded S1-F development diagnostic uses the four existing training worlds and two existing validation worlds without changing candidate labels or the generator. Its tabular control consumes the unchanged 20-column summary built from selected-event times and amounts plus full-world degree and directed-pair multiplicity statistics. Means and population standard deviations are fit in float64 on training rows only; exact constant columns use scale one. Model tensors are float32. The control is a 20–81–81–1 ReLU network with 8,425 trainable parameters.

Five initialization seeds are crossed with three learning rates for 15 control trials. Each trial uses unweighted equal-world binary cross-entropy, Adam, and a training-loss scheduler. Validation is evaluated every 25 epochs, with a 3,000-epoch cap. The selected checkpoint minimizes equal-world validation BCE, with ties resolved to the earliest epoch. A rate is selected by the mean best validation BCE across all five seeds; an incomplete grid has no selected rate. The finite stability rule is diagnostic and does not extend training.

The separate microfit uses the lexicographically first two exact members and first two nonmembers in each training world, for 16 label-conditioned training rows. The four previously defined graph/complex paths keep their existing architectures and inputs. Each arm uses five fresh initialization seeds and the same three learning rates. No validation rows are used in microfit. A trial stops only after BCE is at most 0.02 and all 16 rows are classified correctly at logit zero for 20 consecutive post-update evaluations, or at the 1,500-epoch cap.

These diagnostics answer limited learning and fitting-capability questions on exposed development data. The label-conditioned subset is not a cohort or evaluation split; microfit success does not imply population learning. The two validation worlds do not support population intervals or superiority claims. Finite stability, full-corpus model qualification, independent-world evaluation, external GUDHI parity, and any topology-mechanism or real-world AML claim remain unresolved.

## Complete simulator evidence package

The venue-neutral [manuscript](manuscript.md), [data card](s1_benchmark_data_card.md), [dictionary](s1_results_dictionary.md), and [complete numeric tables](../results/s1_benchmark/summary.json) report the bounded retained simulator finding. Whole-world AP means and every predeclared contrast are generated from the saved full panel. The summary neural exception and unresolved learning/support remain visible.

| Retained arm | W1 | W2 | W3 | W4 | Equal-world mean |
|---|:---:|:---:|:---:|:---:|:---:|
| Fixed-summary GBDT | 1.0000 | 0.5093 | 0.9151 | 1.0000 | 0.8561 |
| Fixed-summary neural control | 0.9095 | 0.7497 | 0.8304 | 0.5092 | 0.7497 |
| Directed local edge GNN | 0.2254 | 0.1434 | 0.5530 | 0.1955 | 0.2793 |
| Local simplicial MPSN-style | 0.2230 | 0.2496 | 0.2385 | 0.2675 | 0.2447 |
| Cellular Hasse control | 0.2259 | 0.2892 | 0.2194 | 0.2045 | 0.2348 |
| Local cellular CWN-style | 0.1623 | 0.2879 | 0.2282 | 0.1813 | 0.2149 |


| Neural arm minus GBDT | W1 | W2 | W3 | W4 | Equal-world mean |
|---|:---:|:---:|:---:|:---:|:---:|
| Fixed-summary neural control | -0.0905 | +0.2404 | -0.0847 | -0.4908 | -0.1064 |
| Directed local edge GNN | -0.7746 | -0.3659 | -0.3621 | -0.8045 | -0.5768 |
| Local simplicial MPSN-style | -0.7770 | -0.2596 | -0.6766 | -0.7325 | -0.6114 |
| Cellular Hasse control | -0.7741 | -0.2201 | -0.6957 | -0.7955 | -0.6213 |
| Local cellular CWN-style | -0.8377 | -0.2214 | -0.6868 | -0.8187 | -0.6411 |


Each world column averages every initialization; the final column weights worlds equally. These are finite conditional observations, with no population superiority or financial efficacy inference. Full per-setting secondary metrics/confusions and separate development tables accompany the paper. The measured GBDT policy is staged round 70 from 800-round fitted objects with 7 maximum leaves; learning rate is 0.1, minimum leaf sample count 2, and API early stopping disabled. Its strong result is retained. Current graph/complex learning remains unqualified. No microfit state was deployed.
