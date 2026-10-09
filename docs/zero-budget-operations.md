# Zero-budget operating mode

No paid plan, external database, persistent disk, or payment method is required for this workflow.

## Deployment guarantees

The frontend stays on Vercel and FastAPI stays on Render Free. Render Free does not provide
persistent local disks. Hosted SQLite, runtime registry files and checkpoints created by the
service must be treated as ephemeral. Cold starts, idle spin-downs, restarts, redeploys or
instance replacement can remove runtime writes. /health reports process/policy readiness, not
filesystem durability. The free deployment is suitable for demos and inference from tracked weights,
not growing episode history, hosted training or durable promotion state. Keep one API instance and
one Uvicorn worker because process memory/locks are not distributed.

## Data classification

| Data | Recommended location | Expected lifetime |
|---|---|---|
| Source, schemas, configs, tests, runbooks | Git | Versioned |
| Small non-sensitive experiment summaries and hashes | Git when useful | Versioned when committed |
| Training reports and generated checkpoints | User-chosen local output directory | Depends on local filesystem/backups |
| Episode logs, state vectors, replay, memory JSON exports | Private local storage, excluded from Git | Depends on backups |
| Render Free database/registry files | Render local filesystem | Ephemeral / not guaranteed |
| CI evaluation artifacts | Download and retain privately | Temporary CI handoff, not a backup |

Some trained pairs are tracked for reproducibility. Do not commit new large checkpoint binaries, secrets,
SQLite databases or private memory exports. Keep schemas/configs/metadata/hashes and compact summaries in Git.

## Local memory transfer

JSON memory packages contain episode content and replay state. Treat them as private training data.
The local-only CLI uses a versioned schema, byte/record limits, validation, atomic exports and a single
SQLite import transaction. It refuses ID collisions and broken replay foreign keys rather than overwriting.
Exports use owner-only POSIX permissions where supported. The source database must already exist.

    python scripts/memory_transfer.py --db data/agent_memory.sqlite3 export backups/cyberarena-memory.json
    python scripts/memory_transfer.py --db data/recovery.sqlite3 import backups/cyberarena-memory.json

Imports merge into an existing SQLite database; on an ID collision, use a fresh reviewed destination.
This tool cannot retrieve hosted Render files, and a local export cannot restore data that has already
disappeared. There is no public HTTP memory dump endpoint.

## Reproducible training

Training is distinct from production API inference. Save artifacts outside Git when possible:

    python train_dqn.py --episodes 500 --seed 20261010 --scenarios 1v1 2v2 3v2 --output-dir ../cyberarena-runs

CYBERARENA_TRAINING_OUTPUT_DIR sets the default output path; explicit --output-dir takes precedence.
Reports include the seed, scenario/config, Python/NumPy/PyTorch versions, per-episode attacker and defender
metrics, memory summary, optional source revision, and checkpoint sizes/SHA-256. The latest-state pair is
updated every 100 episodes; Ctrl+C/SIGTERM asks training to finish the current episode, save available
checkpoints and write an interrupted report. Two checkpoint files are not one atomic transaction: verify
the pair before resuming after any abrupt failure. Reports help reproduce experiments but do not prove
convergence, generalization or production quality. Seeds do not guarantee bitwise identity across hardware
and library versions.

## Evaluation, security and limits

Keep candidate evaluation report-only. Require held-out seeds, separate attacker/defender metrics,
paired confidence intervals, regression gates, runtime/schema validation, and checkpoint provenance.
The current candidate remains rejected: attacker improvement was statistically inconclusive and defender
success regressed materially. Never promote it because of a successful training job alone.

Memory listing/search and mutation require server-side CYBERARENA_ADMIN_TOKEN in X-Admin-Token.
Unset returns HTTP 503; missing/incorrect tokens return HTTP 403. Never bundle the secret into Vercel
or put it in local storage. Keep CORS allow-origins narrow. There is no distributed per-user rate limiter;
process-local limits do not coordinate instances and reset on deployment. Health/readiness cannot certify
durability, security or model quality.

Latest observed frontend audit: 86 advisories (0 critical, 71 high, 12 moderate, 3 low), including the
legacy Create React App toolchain. No blind force-upgrade: a CRA-to-Vite migration needs its own
compatibility-tested PR.

## Free datastore decision

No external database is added. SQLite remains the simplest fit for one local writer; private versioned
JSON exports support local backups/imports without new credentials or vendor coupling. GitHub Actions
artifacts are temporary—download reports that matter and retain them privately. A free remote datastore
still introduces access control, quota and retention policies and does not remove backup requirements.
