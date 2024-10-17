from matplotlib import pyplot as plt
import numpy as np
import random
def plot_histories(history_means, history_stds, history_names):
    for history_mean, history_std, history_name in zip(history_means, history_stds, history_names):
        plt.plot(history_mean, label=history_name)
        plt.fill_between(range(len(history_mean)), np.array(history_mean) - np.array(history_std), np.array(history_mean) + np.array(history_std), alpha=0.2)
    plt.title(f'Reward')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.legend()
    plt.show()

def dummy_reward_history(add_term,multiply_term):
    # emulate a noisy saturating reward history higher add_term means higher reward
    # higher multiply_term means faster saturation
    history_mean = []
    history_std = []
    for i in range(100):
        history_mean.append(add_term*(1-np.exp(-multiply_term*i)) + random.random())
        # std is noisily proportional to the mean
        std = history_mean[-1]*0.1+random.random()*0.01
        history_std.append(std)
    
    return history_mean, history_std

history_mean1, history_std1 = dummy_reward_history(10,0.1)
history_mean2, history_std2 = dummy_reward_history(15,0.15)
history_mean3, history_std3 = dummy_reward_history(20,0.2)
history_means = [history_mean1, history_mean2, history_mean3]
history_stds = [history_std1, history_std2, history_std3]
history_names = ['K=1', 'K=2', 'K=3']

# ax1.plot(rolling_mean, label='Rolling Mean')
# ax1.fill_between(range(len(history)), rolling_mean - std, rolling_mean + std, color='orange', alpha=0.2, label='±1 Std Dev')
# ax1.set_title(f'Reward ({window}-episode window)')
# ax1.set_xlabel('Episode')
# ax1.set_ylabel('Episode Length')
# ax1.legend()
# plt.plot(history_mean)
# plt.fill_between(range(len(history_mean)), np.array(history_mean) - np.array(history_std), np.array(history_mean) + np.array(history_std), color='orange', alpha=0.2)
# plt.title(f'Reward')
# plt.xlabel('Episode')
# plt.ylabel('Reward')
# plt.show()

plot_histories(history_means, history_stds, history_names)