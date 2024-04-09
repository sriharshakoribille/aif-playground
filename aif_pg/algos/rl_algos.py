import numpy as np

class Qlearning:
    def __init__(self, learning_rate, gamma, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.reset_qtable()

    def update(self, state, action, reward, new_state):
        """Update Q(s,a):= Q(s,a) + lr [R(s,a) + gamma * max Q(s',a') - Q(s,a)]"""
        delta = (
            reward
            + self.gamma * np.max(self.qtable[new_state, :])
            - self.qtable[state, action]
        )
        q_update = self.qtable[state, action] + self.learning_rate * delta
        return q_update

    def reset_qtable(self):
        """Reset the Q-table."""
        self.qtable = np.zeros((self.state_size, self.action_size))

class EpsilonGreedy:
    def __init__(self, epsilon, seed):
        self.epsilon = epsilon
        self.rng = np.random.default_rng(seed)   # Set the seed

    def choose_action(self, action_space, state, qtable, rollout=False):
        """Choose an action `a` in the current world state (s)."""
        if rollout:
            # In a rollout we always choose the best action
            action = np.argmax(qtable[state, :])
            return action
        else:
            # First we randomize a number
            explor_exploit_tradeoff = self.rng.uniform(0, 1)

            # Exploration
            if explor_exploit_tradeoff < self.epsilon:
                action = action_space.sample()

            # Exploitation (taking the biggest Q-value for this state)
            else:
                # Break ties randomly
                # If all actions are the same for this state we choose a random one
                # (otherwise `np.argmax()` would always take the first one)
                if np.all(qtable[state, :]) == qtable[state, 0]:
                    action = action_space.sample()
                else:
                    action = np.argmax(qtable[state, :])
            return action