# Cyber Arena DQN Evaluation — Three-Seed Replication

**Status:** completed; experimental candidate **not promoted**  
**PR:** [#5 — paired-seed evaluation](https://github.com/vansh-kumar-007/cyber_arena-/pull/5)  
**Workflow run:** [Reproducible policy evaluation #8](https://github.com/vansh-kumar-007/cyber_arena-/actions/runs/37977440164)  
**Report and checkpoint artifact:** [cyberarena-policy-evaluation — artifact 11638569292](https://github.com/vansh-kumar-007/cyber_arena-/actions/runs/37977440164/artifacts/11638569292)  
**Experiment code commit:** `78621e384b27a88d5c09fb17a10c4edb1ed13d0a`

## Experimental setup

No tracked production-trained checkpoint pair was available in the repository. For each independent training seed, the baseline is the randomly initialized attacker/defender pair, and the candidate starts training from that exact same seeded initialization. This measures whether the DQN training run improves a controlled baseline; it is **not** a comparison against the currently deployed policy.

| Parameter | Value |
|---|---|
| Scenario | 1 attacker vs 1 defender (1v1) |
| Algorithm | Double DQN with prioritized experience replay |
| Training seeds | `20261010`, `20261011`, `20261012` |
| Training budget | 100 episodes per seed (300 total) |
| Holdout evaluation seeds | `20361010`–`20361109` (same 100 seeds for each run) |
| Evaluation | 100 baseline episodes + 100 candidate episodes per training seed (600 policy episodes total, 300 matched pairs) |
| Maximum episode length | 50 steps |
| Exploration during evaluation | Disabled (`epsilon = 0`) |
| Statistics | 10,000 paired-bootstrap replicates per run and a 10,000-replicate hierarchical bootstrap over training-seed clusters and episodes |
| Compute | CPU-only; approximately 35 seconds per training/evaluation replicate |

Training and evaluation seeds are separate. The candidate does not restore replay history from earlier runs; it learns from its own 100 training episodes. Within each training replicate, the baseline and candidate face the same ordered evaluation seeds.

## Results by training seed

| Training seed | Baseline attacker win rate | Candidate attacker win rate | Difference | Paired bootstrap 95% CI | Exact McNemar p-value |
|---:|---:|---:|---:|---:|---:|
| 20261010 | 0% | 30% | +30 pp | +21 to +39 pp | (1.86 	imes 10^{-9}) |
| 20261011 | 0% | 28% | +28 pp | +19 to +37 pp | (7.45 	imes 10^{-9}) |
| 20261012 | 0% | 24% | +24 pp | +16 to +33 pp | (1.19 	imes 10^{-7}) |

All three independently initialized training runs had a positive attacker win-rate difference on this suite.

## Aggregate results across 300 matched pairs

| Metric | Random-initialized baseline | Trained candidate | Difference (clustered paired-bootstrap 95% CI) |
|---|---:|---:|---:|
| Attacker win rate | 0.0% | 27.33% | **+27.33 percentage points** (+21.67 to +33.33 pp) |
| Mean attacker reward | 27.04 | 149.50 | +122.46 (+99.40 to +149.67) |
| Mean defender reward | 0.00 | 583.73 | +583.73 (+549.49 to +615.81) |

The aggregate intervals use a hierarchical paired bootstrap: training seeds are resampled as clusters, then evaluation episode pairs are resampled within each selected run. This avoids treating all 300 episodes as independent draws from independent trained policies.

Each temporary SQLite run recorded between 8,060 and 8,464 experiences/replay transitions. The report, model checkpoint pairs, hashes, and run-local SQLite databases are available in the linked 30-day CI artifact.

## Interpretation

**Improvement was demonstrated for the measured attacker win-rate objective within this fixed 1v1 scenario suite across all three training seeds.** The paired intervals are positive in all runs; the cluster-bootstrap aggregate interval is also positive. This is stronger evidence than a single training run.

This is still a bounded experiment. It does not establish performance for other team sizes, held-out environment definitions, different opponents, long training horizons, or a live production model. The rewards of both roles increased; because that result depends on the simulator's reward design, it should be examined with additional behavior and reward-hacking checks rather than interpreted as proof of a globally optimal joint policy.

**Promotion decision: NOT PROMOTED.** These candidates remain experiment artifacts and have not replaced production checkpoints. Before production promotion, repeat with more training seeds and scenario sizes, compare against frozen/independent opponents, review reward decomposition and failure categories, and run explicit acceptance/rollback tests against a verified stable checkpoint.

The run-local SQLite files prove persistence within the CI workspace only. They do not prove that Render's production data survives service restarts or redeployments.
