import gym
import numpy as np
import matplotlib.pyplot as plt

def make_bins(n_bins):
    obs_low  = np.array([-2.4, -3.0, -0.2094, -3.5])
    obs_high = np.array([ 2.4,  3.0,  0.2094,  3.5])
    return [np.linspace(obs_low[i], obs_high[i], n_bins[i]-1)
            for i in range(len(n_bins))]

def discretize(obs, bins):
    state = []
    for i, val in enumerate(obs):
        state.append(np.digitize(val, bins[i]))
    return tuple(state)

def train_qlearning(env_name='CartPole-v1',
                    n_bins=(6,12,6,12),
                    alpha=0.1, gamma=0.99,
                    epsilon=1.0, eps_decay=0.995, min_epsilon=0.01,
                    n_episodes=10000):
    env = gym.make(env_name)
    bins = make_bins(n_bins)
    Q = np.zeros(n_bins + (env.action_space.n,))

    rewards = []
    for ep in range(n_episodes):
        obs = env.reset()
        state = discretize(obs, bins)
        total_reward = 0
        done = False

        while not done:
            if np.random.rand() < epsilon:
                action = env.action_space.sample()
            else:
                action = np.argmax(Q[state])

            obs2, reward, done, _ = env.step(action)
            state2 = discretize(obs2, bins)

            best_next = np.max(Q[state2])
            Q[state + (action,)] += alpha * (
                reward + gamma * best_next - Q[state + (action,)]
            )

            state = state2
            total_reward += reward

        epsilon = max(min_epsilon, epsilon * eps_decay)
        rewards.append(total_reward)

    return Q, rewards

def plot_rewards(rewards):
    mov_avg = np.convolve(rewards, np.ones(100)/100, mode='valid')
    plt.plot(mov_avg)
    plt.title('Recompensa média (janela=100)')
    plt.xlabel('Episódios')
    plt.ylabel('Recompensa média')
    plt.show()

if __name__ == '__main__':
    Q_table, rewards = train_qlearning()
    np.save('q_table.npy', Q_table)
    np.save('rewards.npy', rewards)
    plot_rewards(rewards)
