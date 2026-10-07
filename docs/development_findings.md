# Development findings

The verified LI-Small development frame contains 6,924,049 source transactions, 5,810 directed cycle families and 96,361,568,254 implicit selected-transaction tuples. Exact source identities resolve 10 of 12 supplied CYCLE instances in the fixed length-2–12 frame. Both length-13 instances remain in the complete supplied-instance denominator.

All thirteen predeclared structural ranking diagnostics retain exact population weights and score ties. Several decreasing-multiplicity rankings have near-perfect AUROC with low AP on the extremely sparse supplied-member population. Independent raw-source and arithmetic checks found no defect in those diagnostics. Their legitimate structural signal remains a required input to appropriate future baselines.

The label-aware family-constant AP information bound is 9553/13770 (approximately 0.69375454). It is conditional on this frame and deliberately uses supplied labels. Selected-transaction attributes may distinguish choices inside a family, but neither identity preservation nor this bound establishes a useful predictive model.

The exact rational metric values and counts are retained in [the development summary](../results/development_summary.json). Model comparison, independent evaluation, statistical precision and complete scoring or probability-sampling feasibility remain outstanding. No trained-model superiority or real-world anti-money-laundering efficacy is claimed.
