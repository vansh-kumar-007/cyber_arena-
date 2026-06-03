# visualization/plots.py
# This file creates all the charts and graphs for our dashboard.

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import networkx as nx

def plot_rewards(attacker_rewards, defender_rewards, save=False):
    """Line chart showing how rewards changed over training"""
    fig, ax = plt.subplots(figsize=(12, 5))

    episodes = range(1, len(attacker_rewards) + 1)

    ax.plot(episodes, attacker_rewards, color='#e74c3c',
            label='Attacker', linewidth=1.5, alpha=0.8)
    ax.plot(episodes, defender_rewards, color='#2ecc71',
            label='Defender', linewidth=1.5, alpha=0.8)

    # Smooth trend lines (moving average)
    window = max(1, len(attacker_rewards) // 20)
    if len(attacker_rewards) >= window:
        att_smooth = [
            sum(attacker_rewards[max(0, i-window):i+1]) /
            len(attacker_rewards[max(0, i-window):i+1])
            for i in range(len(attacker_rewards))
        ]
        def_smooth = [
            sum(defender_rewards[max(0, i-window):i+1]) /
            len(defender_rewards[max(0, i-window):i+1])
            for i in range(len(defender_rewards))
        ]
        ax.plot(episodes, att_smooth, color='#c0392b',
                linewidth=2.5, label='Attacker (trend)')
        ax.plot(episodes, def_smooth, color='#27ae60',
                linewidth=2.5, label='Defender (trend)')

    ax.set_title('Reward Over Training Episodes', fontsize=14, fontweight='bold')
    ax.set_xlabel('Episode')
    ax.set_ylabel('Total Reward')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color='gray', linewidth=0.8, linestyle='--')

    plt.tight_layout()
    if save:
        plt.savefig('rewards_chart.png', dpi=150)
        print("Saved rewards_chart.png")
    return fig


def plot_win_rate(attacker_success, save=False):
    """Rolling win rate chart"""
    fig, ax = plt.subplots(figsize=(12, 4))
    window = 50
    rolling_win = []

    for i in range(len(attacker_success)):
        start = max(0, i - window)
        chunk = attacker_success[start:i+1]
        rolling_win.append(sum(chunk) / len(chunk) * 100)

    ax.plot(range(1, len(rolling_win) + 1), rolling_win,
            color='#9b59b6', linewidth=2)
    ax.axhline(y=50, color='gray', linewidth=1,
               linestyle='--', label='50% line (balanced)')
    ax.fill_between(range(1, len(rolling_win) + 1),
                    rolling_win, 50,
                    where=[r > 50 for r in rolling_win],
                    alpha=0.2, color='red', label='Attacker winning')
    ax.fill_between(range(1, len(rolling_win) + 1),
                    rolling_win, 50,
                    where=[r <= 50 for r in rolling_win],
                    alpha=0.2, color='green', label='Defender winning')

    ax.set_title('Attacker Win Rate (Rolling 50 Episodes)', fontsize=14, fontweight='bold')
    ax.set_xlabel('Episode')
    ax.set_ylabel('Win Rate %')
    ax.set_ylim(0, 100)
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if save:
        plt.savefig('winrate_chart.png', dpi=150)
        print("Saved winrate_chart.png")
    return fig


def plot_network(compromised, blocked, attacker_position, save=False):
    """
    Network graph showing:
    - Green nodes = safe
    - Red nodes = compromised
    - Orange nodes = blocked
    - Yellow border = attacker is here
    """
    fig, ax = plt.subplots(figsize=(8, 6))

    G = nx.DiGraph()
    G.add_nodes_from(["N1", "N2", "N3", "N4"])
    G.add_edges_from([
        ("N1", "N2"), ("N1", "N3"),
        ("N2", "N1"), ("N2", "N4"),
        ("N3", "N1"), ("N4", "N2")
    ])

    pos = {
        "N1": (0, 0.5),
        "N2": (1, 0.5),
        "N3": (0, 0),
        "N4": (2, 0.5)
    }

    labels = {
        "N1": "N1\nWeb Server",
        "N2": "N2\nDB Server",
        "N3": "N3\nUser PC",
        "N4": "N4\nAdmin"
    }

    # Color each node based on its status
    node_colors = []
    node_borders = []
    for node in G.nodes():
        if compromised.get(node, False):
            node_colors.append('#e74c3c')      # Red = hacked
        elif node in blocked:
            node_colors.append('#e67e22')      # Orange = blocked
        else:
            node_colors.append('#2ecc71')      # Green = safe

        if node == attacker_position:
            node_borders.append('#f1c40f')     # Yellow border = attacker here
        else:
            node_borders.append('#2c3e50')

    nx.draw_networkx_nodes(G, pos, node_color=node_colors,
                           node_size=2000, ax=ax,
                           edgecolors=node_borders, linewidths=3)
    nx.draw_networkx_labels(G, pos, labels=labels,
                            font_size=8, font_color='white',
                            font_weight='bold', ax=ax)
    nx.draw_networkx_edges(G, pos, edge_color='#95a5a6',
                           arrows=True, arrowsize=20,
                           ax=ax, connectionstyle='arc3,rad=0.1')

    # Legend
    legend = [
        mpatches.Patch(color='#2ecc71', label='Safe'),
        mpatches.Patch(color='#e74c3c', label='Compromised'),
        mpatches.Patch(color='#e67e22', label='Blocked'),
        mpatches.Patch(color='#f1c40f', label='Attacker Position'),
    ]
    ax.legend(handles=legend, loc='lower right', fontsize=9)
    ax.set_title('Network State', fontsize=14, fontweight='bold')
    ax.axis('off')

    plt.tight_layout()
    if save:
        plt.savefig('network_graph.png', dpi=150)
        print("Saved network_graph.png")
    return fig


def plot_episode_lengths(episode_lengths, save=False):
    """Bar chart of how many steps each episode lasted"""
    fig, ax = plt.subplots(figsize=(12, 4))

    ax.bar(range(1, len(episode_lengths) + 1),
           episode_lengths, color='#3498db', alpha=0.6)

    window = max(1, len(episode_lengths) // 20)
    smooth = [
        sum(episode_lengths[max(0, i-window):i+1]) /
        len(episode_lengths[max(0, i-window):i+1])
        for i in range(len(episode_lengths))
    ]
    ax.plot(range(1, len(smooth) + 1), smooth,
            color='#e74c3c', linewidth=2, label='Trend')

    ax.set_title('Episode Length Over Training', fontsize=14, fontweight='bold')
    ax.set_xlabel('Episode')
    ax.set_ylabel('Steps')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    if save:
        plt.savefig('episode_lengths.png', dpi=150)
        print("Saved episode_lengths.png")
    return fig