
from pathlib import Path
from typing import NamedTuple, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from tqdm import tqdm

import gymnasium as gym
from gymnasium.envs.toy_text.frozen_lake import generate_random_map

from aif_pg.algos.rl_algos import Qlearning, EpsilonGreedy

sns.set_theme()

# %load_ext lab_black


class Params(NamedTuple):
    total_episodes: int  # Total episodes
    learning_rate: float  # Learning rate
    gamma: float  # Discounting rate
    epsilon: float  # Exploration probability
    map_size: int  # Number of tiles of one side of the squared environment
    seed: int  # Define a seed so that we get reproducible results
    is_slippery: bool  # If true the player will move in intended direction with probability of 1/3 else will move in either perpendicular direction with equal probability of 1/3 in both directions
    n_runs: int  # Number of runs
    action_size: int  # Number of possible actions
    state_size: int  # Number of possible states
    proba_frozen: float  # Probability that a tile is frozen
    savefig_folder: Path  # Root folder where plots are saved

def qtable_directions_map(qtable, map_size):
    """Get the best learned action & map it to arrows."""
    qtable_val_max = qtable.max(axis=1).reshape(map_size, map_size)
    qtable_best_action = np.argmax(qtable, axis=1).reshape(map_size, map_size)
    directions = {0: "←", 1: "↓", 2: "→", 3: "↑"}
    qtable_directions = np.empty(qtable_best_action.flatten().shape, dtype=str)
    eps = np.finfo(float).eps  # Minimum float number on the machine
    for idx, val in enumerate(qtable_best_action.flatten()):
        if qtable_val_max.flatten()[idx] > eps:
            # Assign an arrow only if a minimal Q-value has been learned as best action
            # otherwise since 0 is a direction, it also gets mapped on the tiles where
            # it didn't actually learn anything
            qtable_directions[idx] = directions[val]
    qtable_directions = qtable_directions.reshape(map_size, map_size)
    return qtable_val_max, qtable_directions


def plot_q_values_map(qtable, env, map_size, params:Params):
    """Plot the last frame of the simulation and the policy learned."""
    qtable_val_max, qtable_directions = qtable_directions_map(qtable, map_size)

    # Plot the last frame
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(15, 5))
    ax[0].imshow(env.render())
    ax[0].axis("off")
    ax[0].set_title("Last frame")

    # Plot the policy
    sns.heatmap(
        qtable_val_max,
        annot=qtable_directions,
        fmt="",
        ax=ax[1],
        cmap=sns.color_palette("Blues", as_cmap=True),
        linewidths=0.7,
        linecolor="black",
        xticklabels=[],
        yticklabels=[],
        annot_kws={"fontsize": "xx-large"},
    ).set(title="Learned Q-values\nArrows represent best action")
    for _, spine in ax[1].spines.items():
        spine.set_visible(True)
        spine.set_linewidth(0.7)
        spine.set_color("black")
    img_title = f"frozenlake_q_values_{map_size}x{map_size}.png"
    fig.savefig(params.savefig_folder / img_title, bbox_inches="tight")
    fig.show()

def plot_states_actions_distribution(states, actions, map_size, params:Params):
    """Plot the distributions of states and actions."""
    labels = {"LEFT": 0, "DOWN": 1, "RIGHT": 2, "UP": 3}

    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(15, 5))
    sns.histplot(data=states, ax=ax[0], kde=True)
    ax[0].set_title("States")
    sns.histplot(data=actions, ax=ax[1])
    ax[1].set_xticks(list(labels.values()), labels=labels.keys())
    ax[1].set_title("Actions")
    fig.tight_layout()
    img_title = f"frozenlake_states_actions_distrib_{map_size}x{map_size}.png"
    fig.savefig(params.savefig_folder / img_title, bbox_inches="tight")
    fig.show()


def postprocess(episodes, params, rewards, steps, map_size):
    """Convert the results of the simulation in dataframes."""
    res = pd.DataFrame(
        data={
            "Episode": np.tile(episodes, reps=params.n_runs),
            "Run": np.repeat(np.arange(params.n_runs), params.total_episodes),
            "Rewards": rewards.flatten('F'),
            # "Steps": steps.flatten(),
        }
    )
    res["cum_rewards"] = rewards.cumsum(axis=0).flatten(order="F")
    # res["avg_rewards"] = res["cum_rewards"] / (res["Episodes"] + 1)
    res["map_size"] = np.repeat(f"{map_size}x{map_size}", res.shape[0])

    # st = pd.DataFrame(data={"Episode": episodes, "Steps": steps.mean(axis=1), "Avg_rewards": rewards.mean(axis=1)})
    st = pd.DataFrame(data={"Episode": episodes, "Steps": steps.mean(axis=1)})
    st["map_size"] = np.repeat(f"{map_size}x{map_size}", st.shape[0])
    return res, st

def run_episode(env: gym.Env, learner: Qlearning, explorer: EpsilonGreedy, params:Params) -> tuple:
    state = env.reset(seed=params.seed)[0]  # Reset the environment
    step = 0
    done = False
    total_rewards = 0
    all_states = []
    all_actions = []

    while not done:
        action = explorer.choose_action(
            action_space=env.action_space, state=state, qtable=learner.qtable
        )

        # Log all states and actions
        all_states.append(state)
        all_actions.append(action)

        # Take the action (a) and observe the outcome state(s') and reward (r)
        new_state, reward, terminated, truncated, info = env.step(action)

        done = terminated or truncated

        learner.qtable[state, action] = learner.update(
            state, action, reward, new_state
        )

        total_rewards += reward
        step += 1

        # Our new state is state
        state = new_state

    return total_rewards, step, all_states, all_actions

def run_env_deterministic(env: gym.Env, learner: Qlearning, explorer: EpsilonGreedy, params:Params) -> tuple:
    rewards = np.zeros((params.total_episodes, params.n_runs))
    steps = np.zeros((params.total_episodes, params.n_runs))
    episodes = np.arange(params.total_episodes)
    qtables = np.zeros((params.n_runs, params.state_size, params.action_size))
    all_states = []
    all_actions = []

    for run in range(params.n_runs):  # Run several times to account for stochasticity
        learner.reset_qtable()  # Reset the Q-table between runs

        for episode in tqdm(episodes, desc=f"Run {run}/{params.n_runs} - Episodes", leave=False):
            total_rewards, step, all_states_eps, all_actions_eps = run_episode(env, learner, explorer, params)
            # Log all rewards and steps
            rewards[episode, run] = total_rewards
            steps[episode, run] = step
            all_states += all_states_eps
            all_actions += all_actions_eps

        qtables[run, :, :] = learner.qtable

    return rewards, steps, episodes, qtables, all_states, all_actions

def run_env_stochastic(envs: List[gym.Env], transition_eps: List[int], 
                       learner: Qlearning, explorer: EpsilonGreedy, params:Params) -> tuple:
    rewards = np.zeros((params.total_episodes, params.n_runs))
    steps = np.zeros((params.total_episodes, params.n_runs))
    episodes = np.arange(params.total_episodes)
    qtables = np.zeros((params.n_runs, params.state_size, params.action_size))
    all_states = []
    all_actions = []

    for run in range(params.n_runs):  # Run several times to account for stochasticity
        learner.reset_qtable()  # Reset the Q-table between runs
        transition_ctr = 0

        for episode in tqdm(episodes, desc=f"Run {run}/{params.n_runs} - Episodes", leave=False):
            if episode in transition_eps:
                transition_ctr += 1
            total_rewards, step, all_states_eps, all_actions_eps = run_episode(envs[transition_ctr%len(envs)], learner, explorer, params)
            # Log all rewards and steps
            rewards[episode, run] = total_rewards
            steps[episode, run] = step
            all_states += all_states_eps
            all_actions += all_actions_eps

        qtables[run, :, :] = learner.qtable

    return rewards, steps, episodes, qtables, all_states, all_actions

def plot_steps_and_rewards(rewards_df, steps_df, params:Params):
    """Plot the steps and rewards from dataframes."""
    fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(15, 10))
    ax = ax.flatten()
    sns.lineplot(
        data=rewards_df, x="Episode", y="cum_rewards", hue="map_size", ax=ax[0]
    )
    ax[0].set(ylabel="Cumulated rewards")

    sns.lineplot(data=steps_df, x="Episode", y="Steps", hue="map_size", ax=ax[1])
    ax[1].set(ylabel="Averaged steps number")

    sns.lineplot(
        data=rewards_df, x="Episode", y="Rewards", hue="map_size", ax=ax[2]
    )
    ax[2].set(ylabel="Avg rewards")

    for axi in ax[:-1]:
        axi.legend(title="map size")
    fig.delaxes(ax[3])
    fig.tight_layout()
    img_title = "frozenlake_steps_and_rewards.png"
    fig.savefig(params.savefig_folder / img_title, bbox_inches="tight")
    fig.show()

def ql_agent():

    params = Params(
        total_episodes=400,
        learning_rate=0.8,
        gamma=0.95,
        epsilon=0.1,
        map_size=5,
        seed=123,
        is_slippery=False,
        n_runs=20,
        action_size=None,
        state_size=None,
        proba_frozen=0.9,
        savefig_folder=Path("stuff/frozen_lake/ql_sandbox_stochastic"),
    )

    # Set the seed
    rng = np.random.default_rng(params.seed)

    # Create the figure folder if it doesn't exists
    params.savefig_folder.mkdir(parents=True, exist_ok=True)


    map_sizes = [4]
    res_all = pd.DataFrame()
    st_all = pd.DataFrame()

    for map_size in map_sizes:
        env = gym.make(
            "FrozenLake-v1",
            is_slippery=params.is_slippery,
            render_mode="rgb_array",
            # desc=generate_random_map(
            #     size=map_size, p=params.proba_frozen, seed=params.seed
            # ),
            desc=["SFFF", "FFFF", "FHFF", "FFFG"]
        )
        env.reset(seed=params.seed)
        env.action_space.seed(params.seed)  # Set the seed to get reproducible results when sampling the action space

        env2 = gym.make(
            "FrozenLake-v1",
            is_slippery=params.is_slippery,
            render_mode="rgb_array",
            # desc=generate_random_map(
            #     size=map_size, p=params.proba_frozen, seed=params.seed
            # ),
            desc=["SFFF", "FFFF", "FGFF", "FFFH"]
        )
        env2.reset(seed=params.seed)
        env2.action_space.seed(params.seed)  # Set the seed to get reproducible results when sampling the action space
        
        params = params._replace(action_size=env.action_space.n)
        params = params._replace(state_size=env.observation_space.n)
        
        learner = Qlearning(
            learning_rate=params.learning_rate,
            gamma=params.gamma,
            state_size=params.state_size,
            action_size=params.action_size,
        )
        explorer = EpsilonGreedy(
            epsilon=params.epsilon,
            seed=params.seed,
        )
        envs = [env, env2]
        transition_eps = [100, 200, 300]
        print(f"Map size: {map_size}x{map_size}")
        rewards, steps, episodes, qtables, all_states, all_actions = run_env_stochastic(envs, transition_eps, learner, explorer, params)

        # Save the results in dataframes
        res, st = postprocess(episodes, params, rewards, steps, map_size)
        res_all = pd.concat([res_all, res])
        st_all = pd.concat([st_all, st])
        qtable = qtables.mean(axis=0)  # Average the Q-table between runs

        plot_states_actions_distribution(
            states=all_states, actions=all_actions, map_size=map_size, params=params
        )  # Sanity check
        plot_q_values_map(qtable, env, map_size, params=params)

        env.close()


    plot_steps_and_rewards(res_all, st_all, params=params)
    plt.show()

if __name__ == "__main__":
    ql_agent()
