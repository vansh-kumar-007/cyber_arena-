"""Train shared-team Double DQN policies with bounded, persistent replay.

The deployed API remains inference-only. This script is the explicit route for
gradient updates, allowing training to be reproducible and auditable.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import random
import signal
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

from agents.dqn_attacker import DQNAttacker
from agents.dqn_defender import DQNDefender
from configs.network_config import ATTACK_TYPES, DEFENSE_TYPES
from env.network_env import NetworkEnvironment
from utils.experience_memory import ExperienceMemory
from utils.model_paths import resolve_model_dir
from utils.metrics import Metrics

PROJECT_ROOT = Path(__file__).resolve().parent
REPLAY_CAPACITY = 10_000


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


_STOP_REQUESTED = False


def _request_graceful_stop(signum, frame) -> None:
    """Finish the current episode, then write an interrupted-run report."""
    global _STOP_REQUESTED
    _STOP_REQUESTED = True
    print(f"Stop requested by signal {signum}; finishing this episode before saving.")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_training_report(
    report_path: Path, *, started_at: datetime, run_status: str, context: str,
    n_attackers: int, n_defenders: int, num_episodes_requested: int, seed: int | None,
    resume: bool, restore_replay: bool, save_models: bool, model_dir: Path,
    attacker_checkpoint: Path, defender_checkpoint: Path, metrics: Metrics,
    memory: ExperienceMemory,
) -> None:
    """Atomically record seed/configuration, separate team metrics and artifact hashes."""
    def safe_number(value):
        number = float(value)
        return number if math.isfinite(number) else None
    episodes = []
    for index, values in enumerate(zip(
        metrics.attacker_rewards, metrics.defender_rewards, metrics.attacker_success,
        metrics.defender_detections, metrics.episode_lengths,
    ), start=1):
        ar, dr, won, detections, steps = values
        episodes.append({"episode_in_this_run":index,"attacker_reward":safe_number(ar),
            "defender_reward":safe_number(dr),"attacker_won":bool(won),
            "defender_detections":int(detections),"steps":int(steps)})
    count=len(episodes)
    artifacts={}
    if save_models and count:
        artifact_paths = [
            ("attacker_resume", attacker_checkpoint),
            ("defender_resume", defender_checkpoint),
        ]
        # Only a completed run creates new candidate artifacts. A partial report
        # must not mislabel an old candidate file as the result of this run.
        if run_status == "completed":
            artifact_paths.extend([
                ("attacker_candidate", model_dir / f"final_marl_attacker_{context}.pt"),
                ("defender_candidate", model_dir / f"final_marl_defender_{context}_defender.pt"),
            ])
        for name,path in artifact_paths:
            if path.is_file():
                artifacts[name]={"filename":path.name,"bytes":path.stat().st_size,"sha256":_sha256_file(path)}
    report={
        "format":"cyberarena.dqn-training-report","format_version":1,"run_status":run_status,
        "started_at":started_at.isoformat(),"snapshot_at":datetime.now(timezone.utc).isoformat(),
        "git_commit":os.environ.get("GITHUB_SHA") or os.environ.get("CYBERARENA_GIT_COMMIT"),
        "scenario":{"name":context,"n_attackers":n_attackers,"n_defenders":n_defenders},
        "training_config":{
            "episodes_requested_this_run":num_episodes_requested,"episodes_completed_this_run":count,
            "seed":seed,"resume":resume,"restore_replay":restore_replay,"save_models":save_models,
            "gamma":0.95,"learning_rate":0.0005,"target_update_frequency":5,
            "replay_capacity_per_agent":REPLAY_CAPACITY,
            "reward_normalization":"team rewards / 30, clipped to [-3, 3]",
            "reproducibility_note":"Seeds recorded; bitwise determinism is not guaranteed across library versions or hardware.",
        },
        "runtime":{"python":platform.python_version(),"numpy":np.__version__,
                   "torch":str(torch.__version__),"cuda_available":bool(torch.cuda.is_available())},
        "output_directory":str(model_dir),
        "metrics_summary":{
            "attacker_win_rate":safe_number(sum(metrics.attacker_success)/count) if count else None,
            "mean_attacker_reward":safe_number(sum(metrics.attacker_rewards)/count) if count else None,
            "mean_defender_reward":safe_number(sum(metrics.defender_rewards)/count) if count else None,
            "mean_episode_steps":safe_number(sum(metrics.episode_lengths)/count) if count else None},
        "metrics_by_episode":episodes,"experience_memory_summary":memory.summary(),"checkpoint_artifacts":artifacts,
    }
    report_path.parent.mkdir(parents=True,exist_ok=True)
    temporary=report_path.with_suffix(report_path.suffix+".tmp")
    try:
        with temporary.open("w",encoding="utf-8",newline="\n") as stream:
            json.dump(report,stream,indent=2,ensure_ascii=False,allow_nan=False)
            stream.write("\n");stream.flush();os.fsync(stream.fileno())
        os.replace(temporary,report_path)
    finally:
        try: temporary.unlink()
        except FileNotFoundError: pass



def _restore_replay(memory: ExperienceMemory, agent, *, agent_name: str, context: str) -> int:
    """Reload compatible transitions into PER; priorities are reinitialized uniformly."""
    restored = 0
    for item in memory.load_transitions(agent=agent_name, context=context, limit=REPLAY_CAPACITY):
        state = item["state"]
        next_state = item["next_state"]
        if len(state) != agent.state_size or len(next_state) != agent.state_size:
            continue
        if item["action"] >= agent.action_size:
            continue
        agent.memory.push(
            tuple(state), item["action"], item["reward"], tuple(next_state),
            float(item["terminal"]),
        )
        restored += 1
    return restored


def _record_training_transition(
    memory: ExperienceMemory,
    *,
    agent_name: str,
    context: str,
    action: int,
    state: tuple[float, ...],
    reward: float,
    next_state: tuple[float, ...],
    done: bool,
    episode: int,
    step: int,
    action_name: str,
) -> None:
    """Store an audit record and the matching replay transition."""
    outcome = "success" if reward > 0 else "failure" if reward < 0 else "neutral"
    record = memory.record(
        agent=agent_name,
        task=f"{agent_name} training decision in {context}",
        state_summary={"state_vector": list(state)},
        action={"id": action, "name": action_name},
        outcome=outcome,
        reward=float(reward),
        lesson=(
            f"Observed normalized team reward {reward:.4f} after the {action_name} "
            f"action at episode {episode}, step {step}. This is an outcome observation, "
            "not proof that the action caused the reward."
        ),
        tags=[context, action_name, outcome],
        metadata={"episode": episode, "step": step, "context": context, "terminal": done},
    )
    memory.remember_transition(
        agent=agent_name,
        context=context,
        state=state,
        action=action,
        reward=float(reward),
        next_state=next_state,
        terminal=done,
        experience_id=record["id"],
        capacity=REPLAY_CAPACITY,
    )


def train_marl(
    n_attackers: int = 2,
    n_defenders: int = 2,
    num_episodes: int = 1000,
    save_models: bool = True,
    *,
    seed: int | None = None,
    memory: ExperienceMemory | None = None,
    resume: bool = False,
    restore_replay: bool = True,
    output_dir: str | Path | None = None,
):
    """Train a shared DQN per team.

    Args:
        seed: Seeds Python, NumPy, PyTorch, and environment randomness.
        memory: Optional externally-managed memory database.
        resume: Restore latest per-configuration checkpoint weights/optimizer.
        restore_replay: Reload stored transitions for this exact team-size scenario.
        output_dir: Directory for generated checkpoints and reports.
    """
    if not isinstance(num_episodes, int) or isinstance(num_episodes, bool) or num_episodes < 1:
        raise ValueError("num_episodes must be a positive integer")
    if seed is not None:
        if not isinstance(seed, int) or isinstance(seed, bool) or seed < 0:
            raise ValueError("seed must be a non-negative integer or None")
        _seed_everything(seed)

    context = f"{n_attackers}v{n_defenders}"
    configured_output = output_dir if output_dir is not None else os.environ.get("CYBERARENA_TRAINING_OUTPUT_DIR", "").strip()
    if output_dir is not None and not str(output_dir).strip():
        raise ValueError("output_dir must be a non-empty directory path")
    model_dir = Path(configured_output).expanduser().resolve() if configured_output else resolve_model_dir(PROJECT_ROOT)
    attacker_checkpoint = model_dir / f"marl_{context}_attacker.pt"
    defender_checkpoint = model_dir / f"marl_{context}_defender.pt"
    if resume:
        missing = [str(path) for path in (attacker_checkpoint, defender_checkpoint) if not path.is_file()]
        if missing:
            raise FileNotFoundError(
                f"Cannot resume {context}: both attacker and defender checkpoints are required; "
                f"missing: {', '.join(missing)}. Start a fresh run with resume=False."
            )

    started_at = datetime.now(timezone.utc)
    run_stamp = started_at.strftime("%Y%m%dT%H%M%S%fZ")
    report_path = model_dir / "reports" / f"training_{context}_seed-{seed if seed is not None else 'unseeded'}_{run_stamp}.json"
    owns_memory = memory is None
    memory = memory or ExperienceMemory()
    env = NetworkEnvironment(
        n_attackers=n_attackers, n_defenders=n_defenders, seed=seed, max_steps=50
    )
    state_size = len(env.reset())

    print("=" * 60)
    print("   CyberArena RL — Multi-Agent DQN Training (MARL)")
    print(f"   {n_attackers} Attackers vs {n_defenders} Defenders")
    print(f"   Scenario: {context} | State size: {state_size} | Seed: {seed}")
    print("   Architecture: Centralized Training, Decentralized Execution")
    print("=" * 60)

    # One shared brain per team; every individual action contributes a transition.
    attacker_brain = DQNAttacker(state_size=state_size)
    defender_brain = DQNDefender(state_size=state_size)
    for brain in (attacker_brain, defender_brain):
        brain.gamma = 0.95
        brain.learning_rate = 0.0005
        brain.target_update_freq = 5
        for group in brain.optimizer.param_groups:
            group["lr"] = brain.learning_rate

    if restore_replay:
        att_restored = _restore_replay(memory, attacker_brain, agent_name="attacker", context=context)
        def_restored = _restore_replay(memory, defender_brain, agent_name="defender", context=context)
        if att_restored or def_restored:
            print(f"Restored replay: attacker={att_restored}, defender={def_restored}")

    start_episode = 0
    if resume:
        try:
            # Restore the saved global RNG stream once, from the final checkpoint in
            # the pair; the two checkpoint writes happen sequentially at save time.
            attacker_brain.load(str(attacker_checkpoint), restore_rng=False)
            defender_brain.load(str(defender_checkpoint), restore_rng=True)
            # Keep the trainer's explicit configuration, even for older checkpoints.
            for brain in (attacker_brain, defender_brain):
                brain.gamma = 0.95
                brain.learning_rate = 0.0005
                brain.target_update_freq = 5
                for group in brain.optimizer.param_groups:
                    group["lr"] = brain.learning_rate
            if attacker_brain.episode_count != defender_brain.episode_count:
                raise ValueError(
                    "attacker/defender checkpoints have mismatched episode counters "
                    f"({attacker_brain.episode_count} != {defender_brain.episode_count})"
                )
            start_episode = attacker_brain.episode_count
            print(f"Resumed checkpoints for {context} at completed episode {start_episode}")
        except Exception as exc:
            if owns_memory:
                memory.close()
            raise RuntimeError(
                f"Resume requested for {context}, but the saved training state could not "
                "be restored safely; no fresh-weight fallback was performed."
            ) from exc

    metrics = Metrics()
    print(f"Training for {num_episodes} episodes...")

    for episode in range(start_episode + 1, start_episode + num_episodes + 1):
        if _STOP_REQUESTED:
            break
        local_episode = episode - start_episode
        episode_seed = ((seed + episode - 1) % (2**32 - 1)) if seed is not None else None
        state = env.reset(seed=episode_seed)
        attacker_brain.reset_episode_reward()
        defender_brain.reset_episode_reward()
        done = False
        step = 0
        att_losses = []
        def_losses = []

        while not done and step < env.max_steps:
            step += 1
            att_actions = [attacker_brain.choose_action(state) for _ in range(n_attackers)]
            def_actions = [defender_brain.choose_action(state) for _ in range(n_defenders)]
            next_state, att_reward, def_reward, done = env.step(att_actions, def_actions)

            att_reward_n = float(np.clip(att_reward / 30.0, -3.0, 3.0))
            def_reward_n = float(np.clip(def_reward / 30.0, -3.0, 3.0))

            for att_action in att_actions:
                attacker_brain.remember(state, att_action, att_reward_n, next_state, float(done))
                _record_training_transition(
                    memory, agent_name="attacker", context=context, action=att_action,
                    state=state, reward=att_reward_n, next_state=next_state, done=bool(done),
                    episode=episode, step=step, action_name=ATTACK_TYPES[att_action]["name"],
                )
            for def_action in def_actions:
                defender_brain.remember(state, def_action, def_reward_n, next_state, float(done))
                _record_training_transition(
                    memory, agent_name="defender", context=context, action=def_action,
                    state=state, reward=def_reward_n, next_state=next_state, done=bool(done),
                    episode=episode, step=step, action_name=DEFENSE_TYPES[def_action]["name"],
                )

            att_loss = attacker_brain.learn()
            def_loss = defender_brain.learn()
            if att_loss is not None:
                att_losses.append(att_loss)
            if def_loss is not None:
                def_losses.append(def_loss)
            state = next_state

        attacker_brain.decay_epsilon()
        defender_brain.decay_epsilon()
        if episode % attacker_brain.target_update_freq == 0:
            attacker_brain.update_target_network()
            defender_brain.update_target_network()

        info = env.get_info()
        metrics.record(
            ep_attacker_reward=attacker_brain.episode_reward,
            ep_defender_reward=defender_brain.episode_reward,
            attacker_won=info["attacker_won"],
            detections=info["detection_count"],
            steps=step,
        )

        if local_episode % 100 == 0:
            recent_wins = metrics.attacker_success[-100:]
            win_rate = sum(recent_wins) / len(recent_wins) * 100
            avg_att_loss = float(np.mean(att_losses)) if att_losses else 0.0
            avg_def_loss = float(np.mean(def_losses)) if def_losses else 0.0
            att_stats = attacker_brain.get_stats()
            def_stats = defender_brain.get_stats()
            print(f"Episode {episode}/{num_episodes} [{context}]")
            print(f"  Attacker reward={att_stats['episode_reward']:.2f} epsilon={att_stats['epsilon']} loss={avg_att_loss:.4f}")
            print(f"  Defender reward={def_stats['episode_reward']:.2f} epsilon={def_stats['epsilon']} loss={avg_def_loss:.4f}")
            print(f"  Attacker win rate (last 100): {win_rate:.1f}% | {att_stats['mode']}")
            if save_models:
                # Refresh latest completed state, not a single best-reward episode.
                model_dir.mkdir(parents=True, exist_ok=True)
                attacker_brain.save(str(attacker_checkpoint))
                defender_brain.save(str(defender_checkpoint))
            _write_training_report(report_path,started_at=started_at,run_status="partial",context=context,
                n_attackers=n_attackers,n_defenders=n_defenders,num_episodes_requested=num_episodes,
                seed=seed,resume=resume,restore_replay=restore_replay,save_models=save_models,model_dir=model_dir,
                attacker_checkpoint=attacker_checkpoint,defender_checkpoint=defender_checkpoint,metrics=metrics,memory=memory)

    print("=" * 60)
    print("   MARL Training Complete!")
    print("=" * 60)
    if metrics.attacker_rewards:
        metrics.summary(last_n=min(100, len(metrics.attacker_rewards)))

    run_status = ("completed" if len(metrics.attacker_rewards) == num_episodes
                  else "interrupted" if _STOP_REQUESTED else "stopped_early")
    if save_models and metrics.attacker_rewards:
        model_dir.mkdir(parents=True, exist_ok=True)
        # Resume checkpoints may represent partial training, but candidate files are
        # replaced only after the requested episode count has completed successfully.
        attacker_brain.save(str(attacker_checkpoint))
        defender_brain.save(str(defender_checkpoint))
        if run_status == "completed":
            attacker_brain.save(str(model_dir / f"final_marl_attacker_{context}.pt"))
            defender_brain.save(str(model_dir / f"final_marl_defender_{context}_defender.pt"))
        else:
            print("Training did not complete; saved resume state without replacing the last completed candidate pair.")
    elif save_models:
        print("No episode completed; no checkpoint was written and existing candidate files were left unchanged.")
    _write_training_report(report_path,started_at=started_at,run_status=run_status,context=context,
        n_attackers=n_attackers,n_defenders=n_defenders,num_episodes_requested=num_episodes,seed=seed,
        resume=resume,restore_replay=restore_replay,save_models=save_models,model_dir=model_dir,
        attacker_checkpoint=attacker_checkpoint,defender_checkpoint=defender_checkpoint,metrics=metrics,memory=memory)
    print(f"Training report saved: {report_path}")
    print(f"Experience memory summary (retention depends on SQLite host): {memory.summary()}")
    if owns_memory: memory.close()
    return metrics, attacker_brain, defender_brain


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description="Reproducible CPU-compatible CyberArena DQN training")
    parser.add_argument("--episodes",type=int,default=500)
    parser.add_argument("--seed",type=int,default=20261010)
    parser.add_argument("--output-dir",type=Path,default=None)
    parser.add_argument("--scenarios",nargs="+",choices=("1v1","2v2","3v2"),default=("1v1","2v2","3v2"))
    parser.add_argument("--resume",action="store_true")
    parser.add_argument("--no-save-models",action="store_true")
    args=parser.parse_args()
    signal.signal(signal.SIGINT,_request_graceful_stop)
    if hasattr(signal,"SIGTERM"): signal.signal(signal.SIGTERM,_request_graceful_stop)
    scenarios={"1v1":(1,1),"2v2":(2,2),"3v2":(3,2)}
    for scenario in args.scenarios:
        if _STOP_REQUESTED: break
        n_att,n_def=scenarios[scenario]
        train_marl(n_attackers=n_att,n_defenders=n_def,num_episodes=args.episodes,seed=args.seed,
                   output_dir=args.output_dir,resume=args.resume,save_models=not args.no_save_models)
