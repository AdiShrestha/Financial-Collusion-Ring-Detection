# Data card: controlled simulated cycle recovery

## Purpose and provenance

The target is exact complete persisted generated CYCLE tuple membership, including both SAR and non-SAR instances. This is a controlled synthetic proxy benchmark. It supplies no claim about actual laundering, legitimate activity, currency conservation, independent institutions or deployment. Earlier AMLworld and Elliptic studies remain separate exploratory context.

The fixed mechanism derives from IBM AMLSim at revision `7338a4bcb1af9bcfea2201ad7daccfe2a4d569ca`. It uses disclosed controlled input generation, Java runtime/build variants and observation/persistence instrumentation. Upstream authors and source license are cited in the manuscript. All realized source/observation joins were retained locally. The source license does not grant distribution of the project's raw/model artifacts.

## Design and units

The design has 32 accounts, 120 native simulator steps, 6 CYCLE templates and 2 noncycle alert templates, plus background events. Generation used the same tiny template with separate registered random draws. 4 training worlds and 2 validation worlds were used during development; aliases D1–D4 and D5–D6 keep those scopes distinct. The 4 prospective worlds W1–W4 follow registered order; I1–I5 follow initialization-setting order. The exact seed/namespace/state crosswalk remains private. A realization is the evaluation unit conditional on this fixed design. Rows, initializations and graph fragments do not multiply the number of worlds.

The candidate census includes every directed simple account cycle of lengths 2–12, and every combination choosing one physical transfer per directed arc. Rotations are canonicalized without reversing direction. Parallel equal-content transfers have distinct identities, reciprocal cycles are retained and self-loops remain in the world context. Cash-in/out events are excluded from candidate arcs. Overflow stops without clipping at the declared 128 noncash-event/10000 tuple guards; no replacement draw is allowed.

Each original pilot realization produced 77 physical events, 78 candidate tuples, 6 positives and 72 nonmembers. Actual candidate prevalence is 0.0769; no acceptance rule demanded these counts. Native amounts use emitted persistence truncation, and times are integer simulator steps scaled by the configured step count. There is no wall-clock interpretation or conversion to real currency.

## Annotations and permitted inputs

Positives require exact complete physical-event equality to a persisted generated CYCLE, reconciled through intended membership, finalized schedule, arc attempts and native event rows. Incomplete intentions cannot be treated as successful positives. Nonmembership means only absence from these exact annotations. SAR flags do not define the target.

Predictors see full-world event topology/attributes, selected-event and central-account masks, and the candidate's legitimate derived representations. Account/event identifiers serve joins and coordinate maps rather than predictor channels. Labels, annotation IDs, SAR, seed, world identity and candidate rank are excluded from features. Summary baselines receive lossy fixed summaries of permitted observations; graph/complex models retain their distinct full-world representation paths.

The GBDT reads staged round 70 of retained 800-round objects, with 7 maximum leaves, learning rate 0.1 and minimum leaf sample count 2. Summary neural normalization was fit only to the original training frame with population SD; exactly constant features use unit scale. Transformation is float64 and model tensors float32. No holdout transform fitting occurs.

## Access, custody and missing analyses

Generation reconciliation is annotation-aware. Prediction uses its blind inventory; every model–world block is bound before the metric label join. This is same-user local custody, without external label sealing or witnessed execution. The completed review used saved-metric arithmetic/native source joins in the same session after conclusions were exposed. This supports local consistency, not cold or externally independent assurance.

All evaluation worlds are now exposed. The original pilot remains descriptive with no retrospectively added p-values or intervals. The separate prospective replication reports conditional rank tests and marginal world-bootstrap intervals for 30 new realizations. Neural comparative learning/stability, feature/readout/tuning parity, causal interventions, root AMLworld precision, external PH parity and broad financial generalization remain unresolved. Nominal realization count is not qualification of the original population design. Candidates/settings are not independent units.

## Availability and intended use

The publication candidate contains complete aggregate model/world/setting metrics, class counts, fixed-threshold confusions, separate development tables and source-generated SVG figures. It excludes candidate identities, labels/logits, raw world rows, exact seed literals, checkpoints and operational files. Public plots/fixtures can be reproduced from those aggregate rows; the empirical comparison requires withheld inputs/weights. Original public project software is licensed under Apache License 2.0; the manuscript, documentation, aggregate tables and figures are licensed under CC BY 4.0. Third-party rights and credits remain under their original terms. These licenses cover released public material and do not release withheld worlds or weights. The work was conducted independently, without university funding or institutional support. This benchmark is intended for bounded methods inspection and negative-result reporting, not real financial decision making.

## Separate prospective replication

The W1–W4 study was prospectively registered as a four-world study; “Pre-Registered Primary Pilot” is a retrospective descriptor assigned after exposure. W5–W34 were registered before generation and evaluated with the same 30 previously selected states and unchanged transform/checkpoint/threshold policies. All 900 prediction blocks were completed and bound before separate metric label access. The two panels are not pooled. Every new realized census contained 78 candidates and 6 exact members; all actual census rows remain in the replication chance table. Shared accounts, templates and schedules limit the scope to realizations of one controlled design.

The replication exports every model/world/setting endpoint and all prespecified contrasts. Exact signed-rank tests use conditional sign symmetry and assumed independent realizations; they do not specifically test arithmetic mean differences. Paired world percentile intervals are marginal, not simultaneous or a precision/power guarantee. Actual-census constant-score AP and uniform-random tie-free AP are expectations, not floors. Review used distinct arithmetic in the same exposed local session, without external assurance. Full empirical reproduction remains unavailable without the withheld worlds and weights.
