import numpy as np

# np.random.seed(0)


def softmax(x):
    n = np.exp(x)
    return n / np.sum(n)


class Bandit:
    """A simple multi-arm bandit environment."""

    def __init__(self, n_arms, max_reward=1):
        self.n_arms = n_arms
        self.probs = np.random.rand(n_arms)
        self.max_reward = max_reward

    def pull(self, arm: int) -> int:
        """
        Pull the arm and return the reward.

        Args:
            arm (int): The index of the arm to pull.

        Returns:
            int: The reward obtained by pulling the arm.
        """

        reward = sum(np.random.rand() < self.probs[arm] for _ in range(self.max_reward))
        return reward


class EpsilonGreedyAgent:
    """An epsilon-greedy algorithm for the multi-arm bandit problem."""

    def __init__(self, n_arms: int, epsilon=0.3):
        self.counts = np.zeros(n_arms)
        self.values = np.zeros(n_arms)
        self.epsilon = epsilon

    def select_action(self):
        if np.random.rand() < self.epsilon:
            return np.random.randint(len(self.values))
        return np.argmax(self.values)

    def update(self, action: int, reward: float):
        n = self.counts[action]
        self.counts[action] += 1

        self.values[action] = (self.values[action] * n + reward) / self.counts[action]


class SoftmaxActionSelectorAgent:
    """A softmax action selector (Boltzmann exploration) for the multi-arm bandit problem."""

    def __init__(self, n_arms: int, temperature=1.0):
        self.counts = np.zeros(n_arms)
        self.values = np.zeros(n_arms)
        self.temperature = temperature

    def select_action(self):
        prob = softmax(self.values / self.temperature)
        return np.random.choice(len(self.values), p=prob)

    def update(self, action: int, reward: float):
        n = self.counts[action]
        self.counts[action] += 1

        self.values[action] = (self.values[action] * n + reward) / self.counts[action]


if __name__ == "__main__":
    n_arms = 10
    max_reward = 10

    bandit = Bandit(n_arms, max_reward)

    eps_agent = EpsilonGreedyAgent(n_arms)

    avg_reward_history = [0]

    for i in range(500):
        action = eps_agent.select_action()
        reward = bandit.pull(action)
        eps_agent.update(action, reward)

        avg_reward_history.append((avg_reward_history[-1] * i + reward) / (i + 1))

    print("Actual Probability:")
    print(bandit.probs)
    print()

    print("Epsilon Policy:")
    print(eps_agent.values)
    print()

    print("Epsilon Reward trend:")
    print(list(map(lambda x: f"{x:.2f}", avg_reward_history[-10:-1])))
    print()
    print()

    avg_softmax_reward_history = [0]
    softmax_agent = SoftmaxActionSelectorAgent(n_arms, temperature=1.12)

    for i in range(500):
        action = softmax_agent.select_action()
        reward = bandit.pull(action)
        softmax_agent.update(action, reward)

        avg_softmax_reward_history.append(
            (avg_softmax_reward_history[-1] * i + reward) / (i + 1)
        )

    print("Softmax Policy:")
    print(softmax_agent.values)
    print()

    print("Softmax Reward trend:")
    print(list(map(lambda x: f"{x:.2f}", avg_softmax_reward_history[-10:-1])))
