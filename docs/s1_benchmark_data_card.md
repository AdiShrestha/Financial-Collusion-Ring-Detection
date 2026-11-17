# Data card: controlled AMLSim cycle-membership benchmark

## Purpose

The benchmark asks a model to rank enumerated candidate cycles by whether each one exactly matches a generated laundering-cycle instance in a simulated transaction world. It is a controlled synthetic benchmark for studying evaluation practice. It makes no claim about real laundering, legitimate activity or deployment.

## How worlds are produced

- **Simulator.** IBM AMLSim at commit `7338a4bcb1af9bcfea2201ad7daccfe2a4d569ca`, compiled for Java 8 and run on a Java 11 runtime, with a logging patch that records every attempted and persisted transfer. AMLSim's own Python graph generator is not used: `source/scripts/run_s1.py` writes the simulator's input tables directly. The ten third-party jars it needs are verified against pinned SHA-256 values.
- **Template.** 32 accounts and 120 steps. Six cycle alerts (lengths 3, 3, 4, 4, 5 and 6; two each with fixed-interval, ordered-random and unordered schedules), one fan-in and one fan-out alert, and periodic background transfers. Every cycle alert has the same alert window, steps 0 to 29.
- **What the seed changes.** Which account identifiers play which role, initial balances, and the simulator's draws of start steps, transfer times and amounts. Models and summary features never use account identifiers, so worlds differ in timing and amounts only.
- **Worlds.** 40 in total: six development worlds (seeds 101, 103, 107, 109 for training; 113, 127 for validation), four pilot worlds W1–W4 and thirty held-out worlds W5–W34. World identifiers contain the seed (`S1_test_<seed>`). Held-out seeds were drawn uniformly without replacement from 1 to 2³¹ − 1 after excluding all earlier seeds, and registered before generation.

## Candidates and labels

A candidate is a directed simple account cycle (length 2–12, modulo rotation) together with one logged non-cash transfer per arc. Parallel transfers stay distinct. A candidate is a **member** if it exactly matches one complete logged generated cycle instance, whether or not the simulator flags that instance as a SAR. A nonmember is outside the annotation target; it is not verified legitimate activity.

Every one of the 40 worlds has the same frame:

| Cycle length | 2 | 3 | 4 | 5 | 6 | Total |
|---|---|---|---|---|---|---|
| Candidates | 36 | 14 | 14 | 7 | 7 | 78 |
| Members | 0 | 2 | 2 | 1 | 1 | 6 |
| Nonmembers | 36 | 12 | 12 | 6 | 6 | 72 |

The 36 two-cycles come from one reciprocal pair of background accounts. Each generated cycle's first arc also carries six periodic background transfers, so each cycle yields seven tuples: the member and six decoys that differ only in the first-arc transfer.

## Known shortcut

AMLSim's cycle typology places the first transfer of every cycle at the start of its alert window and another at the end. In this template every member therefore spans exactly 29 steps, and its first-arc transfer is its earliest. The rule "selected-event span equals 29" has recall 1 by construction; its mean AP is 0.839 on the development worlds and 0.861 on W5–W34. Anyone using these worlds should report this rule and the cycle-length reference (AP 1/7, AUROC 0.75) beside any model.

## Inputs available to models

Event times (step / 120), log(1 + amount), transfer direction, the candidate's event and account masks, and the whole world's topology. Labels, alert flags, account and event identifiers, seeds and world identity are excluded. The 20 summary features are listed in the manuscript.

## Custody

Models and the analysis plan were fixed before W5–W34 were generated. Generation uses labels (it must reconcile transfers with generated instances); prediction read only observation and topology files, and all 900 prediction files were hashed before a separate process read the labels. Everything ran on one machine under one user account, without external registration or audit.

## Availability and licence

The anonymous review supplement contains all 40 worlds (candidate observations, identities, labels and topology files), the fitted models, all prediction files, the analysis plan and every result table. The public repository contains the code and the aggregate result tables. Original software is Apache-2.0; data, documentation, tables and figures are CC BY 4.0. AMLSim keeps its own Apache-2.0 licence.

## Intended use and limits

Use for studying benchmark construction, shortcut audits and evaluation protocols. Do not use scores from this benchmark as evidence of real-world AML detection ability, and never to make decisions about individuals. One template, one training corpus of 24 members and fixed models limit every conclusion.
