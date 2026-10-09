# Phase 3: durable storage and controlled model promotion

## Current production state

The API currently runs on a Render Free web service. Its SQLite database defaults to
`<repository-root>/data/agent_memory.sqlite3`; Render's health endpoint reported
`backend=sqlite`, `file_backed=true`, and `path_explicitly_configured=false`.
Those facts confirm the database is file-backed, not that the mount is durable.

Render Free web services do not support persistent disks. Do not treat a successful
`/health` response or `PRAGMA integrity_check` as proof that records will survive a
deploy/restart. The currently loaded legacy checkpoint pair is still:

- `models/final_marl_attacker_1v1.pt`
- `models/final_marl_defender_1v1_defender.pt`

The model registry is not enabled in production. No candidate has been promoted.

## Storage design decision

Keep SQLite for current workloads. The application currently runs one API instance and
one Uvicorn worker; the memory connection uses WAL, `synchronous=FULL`, a 10-second
busy timeout, and an in-process reentrant lock. A single-host SQLite database is a
reasonable low-operational-overhead choice for the current experience/replay workload.
It is not an appropriate shared write store for multiple service instances/processes
accessing separate local files, and the registry lock is process-local. Keep one serving
instance/process while using this registry. Reconsider a managed database only if the
product needs multiple writers/hosts, independent training and serving concurrently,
or stronger managed backup/restore operations.

## Required migration before claiming durability

1. **Choose/enable storage first.** Move this service to a Render plan that supports an
   attached persistent disk, then attach the disk at `/var/data`. This may change the
   service price. No plan or paid resource is changed by this PR.
2. **Back up the current SQLite database without changing it.** Use SQLite's online
   backup API (or the `sqlite3 .backup` command) rather than copying only the main
   database file while WAL is active. Record the backup timestamp and run
   `PRAGMA integrity_check` against the backup.
3. **Preserve the known checkpoint pair and registry together.** Copy the entire existing
   checkpoint pair to `/var/data/models/`. When the registry is eventually initialized,
   keep `registry/`, `versions/`, `active.json`, `registry.json`, and `history.jsonl`
   on the same persistent mount as the immutable checkpoint bytes. Never create an
   active pointer as part of ordinary boot.
4. **Configure both paths on Render** (merge-only environment-variable updates):
   - `CYBERARENA_MEMORY_DB=/var/data/agent_memory.sqlite3`
   - `CYBERARENA_MODEL_DIR=/var/data/models`
5. **Deploy only after the backup and copy are verified.** The API and explicit training
   pipeline now use `CYBERARENA_MODEL_DIR`; without it, both keep using the legacy
   project-root `models/` directory. Do not delete the old source files as cleanup until
   the live service has loaded the mounted pair and a rollback path is verified.
6. **Run a non-destructive persistence test.** Record a uniquely tagged test experience
   only after the database backup is verified, read it back through the API, record the
   resolved storage configuration and registry/checkpoint checksums, then perform a
   controlled restart in an approved window. Confirm the same ID survives, replay totals
   and registry history are intact, and run `PRAGMA integrity_check`. Remove only the
   known test experience through its ID after evidence is collected; do not reset the
   whole database. If the check fails, stop promotion and restore from the backup.
7. **Set Render's health check path to `/health`** in the service dashboard. The endpoint
   is now confirmed to return HTTP 200 with `ready=true` on the live service.

## Promotion gate

The registry validates recorded evidence at evaluation and again immediately before
promotion. A candidate comparison must use the active stable checkpoint pair and exact
checkpoint SHA-256 hashes, the same scenario/schema, a reproducible three-block holdout
suite with at least 300 matched episodes, strictly positive lower 95% confidence bounds
for both attacker cross-play win rate and defender cross-play success, reward-regression
lower bounds above -5, p95 inference latency at or below 100 ms, compatible schema/runtime
smoke checks, and zero invalid actions, non-finite Q outputs, or critical regressions.

The current tracked candidate failed the live-policy comparison: attacker win-rate change
was +2 percentage points (95% CI -0.67 to +5.0 pp), while defender success fell 10 points
(95% CI -14.0 to -6.33 pp). Keep the existing stable checkpoint pair active. A positive
training-vs-random-initialization experiment does not override this production cross-play
regression. Evaluation workflows are report-only and never promote a model.

## Concurrency and backup limits

- One Uvicorn worker and one Render instance only until registry/database coordination is
  designed for multi-process/multi-host operation.
- Back up SQLite with its online backup capability so WAL contents are included.
- Back up the checkpoint pair and registry metadata as one consistent set.
- Keep a dated, restorable backup outside the service's ephemeral filesystem. A disk
  attached to one service instance is durable across deploys, but it is not a substitute
  for an independent backup.
- Never test durability on the Free service by restarting it: there is no attached
  persistent disk to validate, and a test cannot demonstrate data surviving a filesystem
  replacement that the plan does not support.
