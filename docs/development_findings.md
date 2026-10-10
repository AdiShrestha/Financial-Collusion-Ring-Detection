# Development findings

The verified LI-Small development frame contains 6,924,049 source transactions, 5,810 directed cycle families and 96,361,568,254 implicit selected-transaction tuples. Exact source identities resolve 10 of 12 supplied CYCLE instances in the fixed length-2–12 frame. Both length-13 instances remain in the complete supplied-instance denominator.

All thirteen predeclared structural ranking diagnostics retain exact population weights and score ties. Several decreasing-multiplicity rankings have near-perfect AUROC with low AP on the extremely sparse supplied-member population. Independent raw-source and arithmetic checks found no defect in those diagnostics. Their legitimate structural signal remains a required input to appropriate future baselines.

The label-aware family-constant AP information bound is 9553/13770 (approximately 0.69375454). It is conditional on this frame and deliberately uses supplied labels. Selected-transaction attributes may distinguish choices inside a family, but neither identity preservation nor this bound establishes a useful predictive model.

The exact rational metric values and counts are retained in [the development summary](../results/development_summary.json). Model comparison, independent evaluation, statistical precision and complete scoring or probability-sampling feasibility remain outstanding. No trained-model superiority or real-world anti-money-laundering efficacy is claimed.

The transaction-preserving observation was verified for every family. It retains 261,838 distinct source rows across 372,017 family-row memberships in a shared cache. The largest context has 16,431 transfers and 1,428 nodes. Fixed engineering probes cover 7,981 unique ranks; they are not a probability sample.

Central-account overlap produces 4,356 components, compared with 3,020 under shared context accounts and 4,190 under shared context rows. Neighborhood observations can overlap across otherwise separate central-cycle groups. These partitions do not establish independent sources or valid evaluation splits.

A separate label-conditioned audit exhaustively checked 162 transaction choices across the 10 positive-containing families. For each of the 10 in-frame supplied members, no nonmember choice had identical permitted attributes within the same fixed central arcs. Raw-row attribute checks reproduced this finding. Arbitrary graph-isomorphism equivalence and learned separability were not established, and no predictive performance follows. The two out-of-frame length-13 instances remain in coverage accounting.

Separate finite-population engineering fixtures qualified exact confidence and metric arithmetic on small frames. The exhaustive panel contains 55 allocations and 849 subset outcomes. Joint coverage is one because every count interval spans its complete feasible range; useful precision is not established. 47 allocations miss the provisional AP half-width goal and 22 miss the bias goal. These are constructed fixture diagnostics, not predictive results.

In the rare-high-score fixture, true AP is 1/2 while the one-negative-sample plug-in expectation is 8/9 and bias is 7/18. The associated count estimate is unbiased. This demonstrates why count unbiasedness does not transfer automatically to AP ratios. The large-count computation panel verified 92 of 128 cases; 36 timed out under a 15-second stage cap. All outcomes and limitations appear in the development summary.

Exact zero-hit precision frontiers were independently checked for 72 hypothetical scenarios. The required sample fractions range from 521823/1000000 to 3999/4000 of the uncertain negative population. These are conditional one-threshold requirements under the declared positive counts, error budgets and width goals, not guaranteed realized precision or universal AP requirements. Hypothetical per-tuple costs must be distinguished from measured model throughput.

An optimized exact arithmetic implementation verified 122 of 128 stress cases under the unchanged 15-second stage limits, retaining 6 timeouts. It qualified 30 previously unfinished cases. The original unfavorable AP bias/width findings remain unchanged. Independent evaluation and scalable scoring for the intended learned comparison remain unresolved.

A review of the complete proposed experiment found no currently qualified route to a confirmatory learned comparison. Independently initialized evaluation support, prospective evaluation access boundaries, integration of the transaction-preserving inputs and measured full-campaign processing cost remain unresolved. This is a finding about the current design and available evidence; no trained-model result establishes predictive failure or efficacy.

## S1 learning diagnostics — 10 October 2026

One bounded development run used four existing training worlds (312 candidates) and two existing validation worlds (156 candidates). The fixed20 tabular neural control completed all 15 seed/rate trials at the 3,000-epoch cap. The selected learning rate was 0.001, with mean best equal-world validation BCE 0.1133 across five initializations. At that rate, mean AP was 0.7266 on one validation world and 0.6984 on the other; mean AUROC was 0.9727 and 0.9694, respectively. No trial established the declared finite stability rule. Training losses became very small, while validation BCE remained materially higher.

The retained fixed20 GBDT development reference remains strong: mean AP 0.9742, AUROC 0.9983, and BCE 0.0293 across ten seed-by-world records. This reference and the S1-F neural measurements are descriptive results on exposed development worlds, not an independent paired evaluation or a population comparison. The neural result does not establish that fixed summaries are inadequate; optimization and generalization remain unresolved.

All 60 unchanged graph/complex microfits met the fitting endpoint on the deliberately label-conditioned, balanced 16-row training subset. This supports fitting capability on those selected rows only. It does not establish full-corpus learning, generalization, representation adequacy, or a topology mechanism. The S1-F outputs remain development evidence; external GUDHI parity is still unresolved, and no independent test performance is reported.
