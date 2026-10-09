import pytest

from env.network_env import NetworkEnvironment
from env.state_encoder import state_size


def test_state_encoder_keeps_checkpoint_dimensions():
    env = NetworkEnvironment(seed=123)
    state = env.reset()
    assert len(state) == state_size(node_count=6) == 37


def test_seeded_environment_is_reproducible():
    first = NetworkEnvironment(seed=1234)
    second = NetworkEnvironment(seed=1234)
    assert first.reset() == second.reset()
    for _ in range(8):
        next_first = first.step([6], [2])
        next_second = second.step([6], [2])
        assert next_first == next_second


def test_scalar_actions_work_for_legacy_single_agent_callers():
    env = NetworkEnvironment(seed=2)
    # Older entry points call the 1v1 environment with scalar actions.
    state, attacker_reward, defender_reward, done = env.step(11, 11)
    assert len(state) == 37
    assert isinstance(attacker_reward, float)
    assert isinstance(defender_reward, float)
    assert done is False


def test_invalid_actions_and_agent_counts_are_rejected():
    with pytest.raises(ValueError):
        NetworkEnvironment(n_attackers=5)
    env = NetworkEnvironment(seed=3)
    with pytest.raises(ValueError):
        env.step([99], [0])
    with pytest.raises(ValueError):
        env.step([0, 1], [0])


def test_episode_horizon_matches_training_scale():
    env = NetworkEnvironment(seed=4, max_steps=2)
    assert env.step([11], [11])[3] is False
    assert env.step([11], [11])[3] is True
