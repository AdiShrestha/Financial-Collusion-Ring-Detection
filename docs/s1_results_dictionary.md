# Result table dictionary

All numeric values are exported from retained measurements at full JSON/CSV round-trip precision. Empty CSV cells represent null/missing fields rather than zero. Some heterogeneous development columns apply only to particular record types. Boolean cells retain source flags. Trial/checkpoint selection measurements and terminal measurements stay separate. Neither candidates nor settings increase the independent world count.

AP uses complete equal-logit threshold groups without identifier tie breaking; AUROC gives half credit for ties; BCE uses stable logit arithmetic. AP is unavailable for no-positive support, AUROC for a single class. Null AP blocks required aggregate/contrast output rather than removing an unfavorable world. The retained holdout has complete defined metrics. Counts are integer physical-candidate counts and fixed-zero-logit confusions. No empirical score computation is performed by the plotting utility.

World/initialization aliases preserve registered order. D aliases identify development worlds only. Mean rows average metrics across all settings, then weight every world equally. Contrasts are neural minus fixed-summary GBDT, with no population inference, confidence intervals or p-values. No raw labels, predictions or exact seed/state identities are distributed.

## metrics_by_world_initialization.csv

Rows: 120.

| Column | Meaning |
|---|---|
| `phase` | Study period; initial development, continuation, learning diagnostic or heldout. Never pool these phases. |
| `arm` | Stable public model-arm name, identical across tables; the original papers are credited as background. |
| `world` | Neutral registered holdout alias W or distinct development alias D; equal_world_mean is an aggregate row. |
| `initialization` | I alias in registered setting order, not an independent world. |
| `ap` | Grouped-threshold average precision from retained logits. |
| `ap_undefined_reason` | Literal reason if AP unavailable; empty cell for a defined AP. |
| `auroc` | AUROC with half credit for ties. |
| `auroc_undefined_reason` | Literal reason if AUROC unavailable; empty cell for defined AUROC. |
| `bce` | Mean stable binary cross-entropy on logits. |
| `candidate_count` | Full candidate census size in this world. |
| `positives` | Exact complete generated CYCLE member count. |
| `negatives` | Candidate nonmember count; not verified licit activity. |
| `prevalence` | positives/candidate_count; contextual baseline, not a random-ranking test. |
| `true_positive` | Positive rows with logit greater than or equal to zero. |
| `false_positive` | Nonmember rows with logit greater than or equal to zero. |
| `true_negative` | Nonmember rows with logit below zero. |
| `false_negative` | Positive rows with logit below zero. |
| `threshold` | Fixed raw-logit classification threshold. |
| `threshold_rule` | Retained rule; no validation/test calibration. |

## ap_by_world.csv

Rows: 24.

| Column | Meaning |
|---|---|
| `phase` | Study period; initial development, continuation, learning diagnostic or heldout. Never pool these phases. |
| `arm` | Stable public model-arm name, identical across tables; the original papers are credited as background. |
| `world` | Neutral registered holdout alias W or distinct development alias D; equal_world_mean is an aggregate row. |
| `mean_ap` | Arithmetic mean of AP over every initialization inside the given world. |
| `initialization_count` | All registered settings contributing to a world mean. |
| `undefined_reason` | Reason a mean or contrast is unavailable; none was missing in the retained holdout. |

## ap_contrasts.csv

Rows: 25.

| Column | Meaning |
|---|---|
| `phase` | Study period; initial development, continuation, learning diagnostic or heldout. Never pool these phases. |
| `arm` | Stable public model-arm name, identical across tables; the original papers are credited as background. |
| `reference` | Fixed summary GBDT in every predeclared contrast. |
| `world` | Neutral registered holdout alias W or distinct development alias D; equal_world_mean is an aggregate row. |
| `ap_difference` | Mean AP of named neural arm minus mean AP of reference, per world or equally over worlds. |
| `undefined_reason` | Reason a mean or contrast is unavailable; none was missing in the retained holdout. |

## development_initial_trials.csv

Rows: 100.

| Column | Meaning |
|---|---|
| `arm` | Stable public model-arm name, identical across tables; the original papers are credited as background. |
| `best_epoch` | Earliest epoch attaining lowest equal-world validation BCE in the source trial. |
| `best_equal_world_validation_bce` | Minimum equal-world validation BCE for this trial/policy setting. |
| `duration_seconds` | Original measured trial elapsed duration; not export duration. |
| `history_points` | Recorded initial trial evaluation/history point count. |
| `learning_rate` | Initial candidate optimizer/boosting learning rate. |
| `phase` | Study period; initial development, continuation, learning diagnostic or heldout. Never pool these phases. |
| `stage` | Source trial collection or selected_checkpoint versus terminal_endpoint; terminal values are not selected values. |
| `split` | Training, validation, or deliberately label-conditioned training subset; scope is retained. |
| `trial` | Neutral trial alias, unique within the table; private ledger retains original trial keys. |
| `initialization` | I alias in registered setting order, not an independent world. |
| `per_trial_status` | Null when original initial-trial summary does not record a status; overall completion is not substituted. |
| `status_unavailable_reason` | Reason the per-trial status is missing. |
| `best_iteration` | Selected evaluated GBDT staged iteration in initial trial. |
| `early_stopping` | Original GBDT API early-stopping flag; validation selection is recorded separately. |
| `fit_seconds` | Original measured GBDT fitting duration only. |
| `l2_regularization` | Original GBDT regularization coefficient. |
| `max_leaf_nodes` | Candidate/selected GBDT maximum leaf count. |
| `min_samples_leaf` | Declared minimum leaf sample count. |
| `best_epoch_from_0_to_800` | Original initial-extension checkpoint-selection epoch; not terminal epoch. |
| `best_evaluated_iteration_up_to_800` | Extension selection, distinct from fitted-object round count. |
| `max_iter` | GBDT fitted-object iteration cap; distinct from selected inference iteration. |

## development_selected_metrics.csv

Rows: 100.

| Column | Meaning |
|---|---|
| `arm` | Stable public model-arm name, identical across tables; the original papers are credited as background. |
| `auroc_tie_aware` | Original initial-development half-tie AUROC. |
| `average_precision_grouped_threshold` | Grouped-threshold AP; same metric name as the original development export. |
| `bce` | Mean stable binary cross-entropy on logits. |
| `candidate_count` | Full candidate census size in this world. |
| `positives` | Exact complete generated CYCLE member count. |
| `stage` | Source trial collection or selected_checkpoint versus terminal_endpoint; terminal values are not selected values. |
| `phase` | Study period; initial development, continuation, learning diagnostic or heldout. Never pool these phases. |
| `split` | Training, validation, or deliberately label-conditioned training subset; scope is retained. |
| `world` | Neutral registered holdout alias W or distinct development alias D; equal_world_mean is an aggregate row. |
| `initialization` | I alias in registered setting order, not an independent world. |

## development_continuation_metrics.csv

Rows: 240.

| Column | Meaning |
|---|---|
| `arm` | Stable public model-arm name, identical across tables; the original papers are credited as background. |
| `split` | Training, validation, or deliberately label-conditioned training subset; scope is retained. |
| `average_precision_grouped_threshold` | Grouped-threshold AP; same metric name as the original development export. |
| `auroc_half_tie` | Development AUROC with half credit for ties. |
| `binary_cross_entropy_logits` | Development logit BCE. |
| `positives` | Exact complete generated CYCLE member count. |
| `negatives` | Candidate nonmember count; not verified licit activity. |
| `n` | Number of rows in the source metric endpoint. |
| `phase` | Study period; initial development, continuation, learning diagnostic or heldout. Never pool these phases. |
| `stage` | Source trial collection or selected_checkpoint versus terminal_endpoint; terminal values are not selected values. |
| `world` | Neutral registered holdout alias W or distinct development alias D; equal_world_mean is an aggregate row. |
| `initialization` | I alias in registered setting order, not an independent world. |
| `selected_epoch` | Locked validation-selected checkpoint epoch. |
| `source_phase` | Exact source phase string, even if it names validation selection for a training-split metric. |
| `last_epoch` | Continuation terminal epoch; not selected checkpoint. |
| `selected_initial_rate` | Across-setting validation-selected continuation initial rate. |
| `stability_status` | Original terminal run stability disposition. |

## development_continuation_status.csv

Rows: 106.

| Column | Meaning |
|---|---|
| `confirmation_end_epoch` | End of the recorded finite confirmation; null if not established. |
| `first_qualifying_pair_end_epoch` | First candidate stability pair; may be null. |
| `interpretation` | Retained finite-stability interpretation. |
| `last_epoch` | Continuation terminal epoch; not selected checkpoint. |
| `last_learning_rate` | Rate at the last recorded continuation epoch. |
| `status` | Original retained trial or finite-stability status. |
| `phase` | Study period; initial development, continuation, learning diagnostic or heldout. Never pool these phases. |
| `record_type` | Trial, policy setting/mean, stability window, or selected summary validation record. |
| `arm` | Stable public model-arm name, identical across tables; the original papers are credited as background. |
| `initialization` | I alias in registered setting order, not an independent world. |
| `confirmed_window_count` | Number of actual confirmed rolling-window records. |
| `available` | Recorded window diagnostic availability. |
| `end_epoch` | End of a confirmed stability window. |
| `learning_rate` | Initial candidate optimizer/boosting learning rate. |
| `learning_rate_unchanged` | Original diagnostic required an unchanged rate within window. |
| `stable` | Recorded stability-window outcome. |
| `start_epoch` | Start of a confirmed stability window. |
| `training_BCE_relative_range` | Recorded within-window training BCE relative range. |
| `validation_ap_range_D5` | Recorded validation AP range for the named development-world alias within this confirmed window. |
| `validation_ap_range_D6` | Recorded validation AP range for the named development-world alias within this confirmed window. |
| `initial_rate` | Candidate continuation initial rate. |
| `best_equal_world_validation_bce` | Minimum equal-world validation BCE for this trial/policy setting. |
| `selected_initial_rate` | Across-setting validation-selected continuation initial rate. |
| `mean_best_equal_world_validation_bce` | Across-setting mean minimum validation BCE for a candidate policy. |

## learning_diagnostic_trials.csv

Rows: 88.

| Column | Meaning |
|---|---|
| `kind` | summary or microfit; a microfit is training capability evidence only. |
| `arm` | Stable public model-arm name, identical across tables; the original papers are credited as background. |
| `learning_rate` | Initial candidate optimizer/boosting learning rate. |
| `status` | Original retained trial or finite-stability status. |
| `epochs_completed` | Actual optimizer epochs in retained trial, not authorization for extension. |
| `endpoint_reason` | Retained literal stop reason; endpoint rule versus cap are distinct. |
| `best_validation_bce` | Summary trial selected validation BCE; null for microfit without validation. |
| `best_epoch` | Earliest epoch attaining lowest equal-world validation BCE in the source trial. |
| `consecutive_endpoint_evaluations` | Actual consecutive successful microfit endpoint evaluations. |
| `duration_seconds` | Original measured trial elapsed duration; not export duration. |
| `phase` | Study period; initial development, continuation, learning diagnostic or heldout. Never pool these phases. |
| `record_type` | Trial, policy setting/mean, stability window, or selected summary validation record. |
| `trial` | Neutral trial alias, unique within the table; private ledger retains original trial keys. |
| `initialization` | I alias in registered setting order, not an independent world. |
| `split` | Training, validation, or deliberately label-conditioned training subset; scope is retained. |
| `finite_stability_status` | Summary finite diagnostic; completion does not establish stability. |
| `confirmation_end_epoch` | End of the recorded finite confirmation; null if not established. |
| `selected_rate` | Across-setting selected summary rate. |
| `epoch` | Epoch at which terminal diagnostic was recorded. |
| `correct_at_logit_zero` | Number correct on the selected microfit rows at fixed zero logit. |
| `post_update_equal_world_bce` | Microfit terminal equal-world BCE. |
| `coordinate_checks_passed` | Count of retained endpoint coordinate checks passed. |
| `coordinate_checks_total` | Count of retained endpoint coordinate checks attempted. |
| `all_16_observed_logits_bitwise_equal_to_uninstrumented` | Retained instrumentation-fidelity endpoint status; no logits are released. |
| `endpoint_status` | Retained microfit terminal diagnostic status. |
| `ap` | Grouped-threshold average precision from retained logits. |
| `auroc` | AUROC with half credit for ties. |
| `bce` | Mean stable binary cross-entropy on logits. |
| `n` | Number of rows in the source metric endpoint. |
| `positives` | Exact complete generated CYCLE member count. |
| `world` | Neutral registered holdout alias W or distinct development alias D; equal_world_mean is an aggregate row. |
| `mean_best_equal_world_validation_bce` | Across-setting mean minimum validation BCE for a candidate policy. |
