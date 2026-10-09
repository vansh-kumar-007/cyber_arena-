# Cyber Arena DQN Evaluation — Random Initialization vs Trained Candidate

**Experiment status:** completed; candidate **not promoted**  
**Workflow run:** [Reproducible policy evaluation #3](https://github.com/vansh-kumar-007/cyber_arena-/actions/runs/37976955123)  
**Report and checkpoint artifact:** [cyberarena-policy-evaluation (artifact 11639751748)](https://github.com/vansh-kumar-007/cyber_arena-/actions/runs/37976955123/artifacts/11639751748)  
**Experiment code commit:** `52bebcaab894748d4d9f41542fd42b3cf79b7109`  
**CI merge ref recorded by the report:** `d4ed52b343d4c1fd4e594762889d0894803a68b4`

## Design

The repository did not contain a tracked, deployable trained checkpoint pair to use as a production baseline. This experiment therefore compares a random-initialized baseline policy pair with a candidate trained from the *same seeded initialization*. It does **not** compare with the live production model.

| Setting | Value |
|---|---|
| Scenario | 1 attacker vs 1 defender (1v1) |
| State / action dimensions | 37 state features; 12 actions for each role |
| Algorithm | Double DQN with prioritized experience replay |
| Training episodes | 100 |
| Training seed | `20261010` |
| Evaluation episodes per policy | 100 paired episodes |
| Held-out evaluation seeds | `20361010` through `20361109` |
| Episode step cap | 50 |
| Evaluation exploration | Disabled (`epsilon = 0`) |
| Uncertainty | 10,000 paired-bootstrap replicates; Wilson intervals for win rate; exact two-sided McNemar test |
| Runtime | 33.291 seconds, CPU-only; Python 3.11.17, PyTorch 2.14.1+cpu, NumPy 2.4.6 |

Training replay restoration was disabled so the candidate started from the baseline's random initialization and learned only from this experiment's training episodes. Evaluation used an identical held-out seed list for baseline and candidate.

## Measured results

| Metric | Baseline | Trained candidate | Candidate minus baseline |
|---|---:|---:|---:|
| Attacker win rate | 0.0% (95% Wilson CI 0.0–3.7%) | 30.0% (95% Wilson CI 21.9–39.6%) | **+30.0 percentage points** |
| Mean attacker reward | 11.860 | 163.105 | +151.245 (paired-bootstrap 95% CI +133.720 to +169.355) |
| Mean defender reward | 0.000 | 570.240 | +570.240 (paired-bootstrap 95% CI +512.152 to +625.760) |
| Median episode length | 50 steps | 50 steps | 0 |
| Mean detections per episode | 2.88 | 0.03 | -2.85 |

**Paired attacker win outcome:** candidate-only wins = 30; baseline-only wins = 0; observed win-rate difference = +0.30. The paired-bootstrap 95% CI is **[+0.21, +0.39]**, and the exact two-sided McNemar test gives **p = 1.862645149 × 10⁻⁹** for this fixed seed suite.

The 100-episode training run recorded 8,060 experiences and 8,060 durable replay transitions in its run-local SQLite database (4,030 per role). The data and model artifacts are in the linked CI artifact.

## Interpretation and limitations

This provides **evidence of improvement in the attacker win-rate metric for this one 1v1 held-out scenario suite** after a bounded training run from a random initial policy. It does not prove broad/generalized improvement, and it is not a production checkpoint comparison. The single training seed, one team size, limited training budget, and test suite limit external validity. Attacker and defender reward numbers should be interpreted according to the environment's reward definition; the experiment does not establish a globally optimal or safer joint policy.

**Promotion decision: NOT PROMOTED.** No model was copied into production or made active. A follow-up should repeat training across multiple independent training seeds and scenario sizes, evaluate against frozen opponents as well as self-play, and compare candidate policies against a verified stable production checkpoint when one is available.

This experiment used an ephemeral CI workspace. Its SQLite artifact proves that records existed during the run; it does **not** demonstrate that Render's production filesystem survives restarts or redeployments.
