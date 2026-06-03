# visualization/dashboard.py
# Streamlit dashboard — run with: streamlit run visualization/dashboard.py

import streamlit as st
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from env.network_env import NetworkEnvironment
from agents.attacker import AttackerAgent
from agents.defender import DefenderAgent
from utils.metrics import Metrics
from visualization.plots import (
    plot_rewards, plot_win_rate,
    plot_network, plot_episode_lengths
)
from configs.hyperparams import NUM_EPISODES, MAX_STEPS

# ─── PAGE SETUP ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="CyberArena RL",
    page_icon="🛡️",
    layout="wide"
)

st.title("🛡️ CyberArena RL — Adversarial Network Simulation")
st.markdown("*A multi-agent reinforcement learning simulation where an attacker "
            "and defender co-evolve strategies in a dynamic network environment.*")

st.divider()

# ─── SIDEBAR CONTROLS ─────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Training Settings")
    num_episodes = st.slider("Number of Episodes", 100, 2000, 1000, step=100)
    show_network = st.checkbox("Show Network Graph", value=True)
    st.divider()
    st.markdown("**How it works:**")
    st.markdown("- 🔴 **Attacker** tries to compromise critical nodes")
    st.markdown("- 🟢 **Defender** tries to detect and block attacks")
    st.markdown("- Both agents learn using **Q-Learning**")
    st.markdown("- Watch them get smarter over time!")

# ─── TRAIN BUTTON ─────────────────────────────────────────────────────────────
if st.button("🚀 Start Training", type="primary", use_container_width=True):

    env = NetworkEnvironment()
    attacker = AttackerAgent()
    defender = DefenderAgent()
    metrics = Metrics()

    # Progress display
    progress_bar = st.progress(0)
    status = st.empty()
    col1, col2, col3 = st.columns(3)
    att_metric = col1.empty()
    def_metric = col2.empty()
    win_metric = col3.empty()

    for episode in range(1, num_episodes + 1):
        state = env.reset()
        attacker.reset_episode_reward()
        defender.reset_episode_reward()
        done = False
        step = 0

        while not done and step < MAX_STEPS:
            step += 1
            attacker_action = attacker.choose_action(state)
            defender_action = defender.choose_action(state)
            next_state, att_r, def_r, done = env.step(attacker_action, defender_action)
            attacker.update(state, attacker_action, att_r, next_state)
            defender.update(state, defender_action, def_r, next_state)
            state = next_state

        attacker.decay_epsilon()
        defender.decay_epsilon()

        info = env.get_info()
        metrics.record(
            ep_attacker_reward=attacker.episode_reward,
            ep_defender_reward=defender.episode_reward,
            attacker_won=info["attacker_won"],
            detections=info["detection_count"],
            steps=step
        )

        # Update display every 50 episodes
        if episode % 50 == 0:
            progress_bar.progress(episode / num_episodes)
            status.markdown(f"**Training... Episode {episode}/{num_episodes}**")
            recent_wins = metrics.attacker_success[-50:]
            win_rate = sum(recent_wins) / len(recent_wins) * 100
            att_metric.metric("Attacker Reward", f"{attacker.episode_reward:.0f}")
            def_metric.metric("Defender Reward", f"{defender.episode_reward:.0f}")
            win_metric.metric("Attacker Win Rate", f"{win_rate:.1f}%")

    progress_bar.progress(1.0)
    status.markdown("✅ **Training Complete!**")

    st.divider()

    # ─── RESULTS ──────────────────────────────────────────────────────────────
    st.header("📊 Training Results")

    tab1, tab2, tab3, tab4 = st.tabs([
        "💰 Rewards", "🏆 Win Rate",
        "🌐 Network State", "⏱️ Episode Lengths"
    ])

    with tab1:
        fig = plot_rewards(metrics.attacker_rewards, metrics.defender_rewards)
        st.pyplot(fig)

    with tab2:
        fig = plot_win_rate(metrics.attacker_success)
        st.pyplot(fig)

    with tab3:
        if show_network:
            info = env.get_info()
            fig = plot_network(
                compromised=info["compromised"],
                blocked=set(info["blocked_nodes"]),
                attacker_position=info["attacker_position"]
            )
            st.pyplot(fig)

    with tab4:
        fig = plot_episode_lengths(metrics.episode_lengths)
        st.pyplot(fig)

    # ─── SUMMARY STATS ────────────────────────────────────────────────────────
    st.divider()
    st.header("📈 Final Summary")

    last100_wins = metrics.attacker_success[-100:]
    final_win_rate = sum(last100_wins) / len(last100_wins) * 100
    avg_att = sum(metrics.attacker_rewards[-100:]) / 100
    avg_def = sum(metrics.defender_rewards[-100:]) / 100

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Final Attacker Win Rate", f"{final_win_rate:.1f}%")
    c2.metric("Avg Attacker Reward (last 100)", f"{avg_att:.1f}")
    c3.metric("Avg Defender Reward (last 100)", f"{avg_def:.1f}")
    c4.metric("Total Episodes", num_episodes)