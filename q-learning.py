import gym
import numpy as np
import matplotlib.pyplot as plt


# Q-Learning settings
# (0, 1)
LEARNING_RATE = 0.1
# (0, 1): Measure of how much we want to care about FUTURE reward rather than immediate reward
DISCOUNT = 0.95
EPISODES = 4000  # Configurable
SHOW_EVERY = 1000
STATS_EVERY = 100

env = gym.make("MountainCar-v0")
print('Available actions:', env.action_space.n)
print('State low values (position, velocity) :', env.observation_space.low)
print('State high values (position, velocity):', env.observation_space.high)

# 20 buckets for each range
DISCRETE_OS_SIZE = [20, 20]  # Configurable
discrete_os_win_size = (env.observation_space.high - env.observation_space.low)/DISCRETE_OS_SIZE
print('Calculated size of each bucket:', discrete_os_win_size)

# Create 20x20x3 table with initial value between [-2, 0)
q_table = np.random.uniform(low=-2, high=0, size=(DISCRETE_OS_SIZE + [env.action_space.n]))
print(q_table.shape)

# Exploration settings
epsilon = 1  # not a constant, will be decayed
EPISODE_START_EPSILON_DECAYING = 1
EPISODE_END_EPSILON_DECAYING = EPISODES // 2  # Configurable
epsilon_decay_value = epsilon / (EPISODE_END_EPSILON_DECAYING - EPISODE_START_EPSILON_DECAYING)

# For stats
ep_rewards = []
aggr_ep_rewards = {'ep': [], 'min': [], 'max': [], 'avg': []}


def get_discrete_state(state):
    discrete_state = (state - env.observation_space.low)/discrete_os_win_size
    return tuple(discrete_state.astype(np.int))


for episode in range(EPISODES):
    episode_reward = 0
    state = env.reset()
    discrete_state = get_discrete_state(state)

    if episode % SHOW_EVERY == 0:
        render = True
        # print(episode)
        # print('Initial state (position, velocity)    :', discrete_state)
    else:
        render = False

    done = False
    while not done:
        # Random # between (0, 1)
        if np.random.random() > epsilon:
            # Random # was larger: Get action from Q table (exploitation)
            # Given the current state: The index of the action with highest reward (0, 2)
            action = np.argmax(q_table[discrete_state])
        else:
            # Epsilon was larger: Get random action
            # Initially, epsilon will almost always be larger (promote exploration)
            action = np.random.randint(0, env.action_space.n)

        # Given the current state and action: Current Q value [-2, 0)
        current_q = q_table[discrete_state + (action,)]

        # Perform the action and receive (new state, reward, done)
        new_state, reward, done, _ = env.step(action)
        episode_reward += reward
        new_discrete_state = get_discrete_state(new_state)
        # Given the new state: The max possible Q value in next step max(0,1,2)
        max_future_q = np.max(q_table[new_discrete_state])

        if render:
            env.render()

        # Simulation ended - objective position is achieved
        # update Q value with reward directly
        if done and new_state[0] >= env.goal_position:
            # print(f"Achieved goal at episode {episode}")
            q_table[discrete_state + (action,)] = 0

        # Simulation did not end yet - update Q table
        else:
            # Calculate new Q value for current state and action
            new_q = (1 - LEARNING_RATE) * current_q + LEARNING_RATE * (reward + DISCOUNT * max_future_q)
            # Given the current state and action: Update Q table with new Q value
            q_table[discrete_state + (action,)] = new_q
            discrete_state = new_discrete_state

    # Decay: if episode number is within decaying range (EPISODES / 2)
    if episode <= EPISODE_END_EPSILON_DECAYING:
        epsilon -= epsilon_decay_value

    # Stats
    ep_rewards.append(episode_reward)
    # Record metrics every STATS_EVERY episode, for the last STATS_EVERY episodes
    if not episode % STATS_EVERY:
        aggr_ep_rewards['ep'].append(episode)
        aggr_ep_rewards['min'].append(min(ep_rewards[-STATS_EVERY:]))
        aggr_ep_rewards['max'].append(max(ep_rewards[-STATS_EVERY:]))
        average_reward = sum(ep_rewards[-STATS_EVERY:]) / STATS_EVERY
        aggr_ep_rewards['avg'].append(average_reward)
        print(f'Episode: {episode:>5d}, average reward: {average_reward:>4.1f}, current epsilon: {epsilon:>1.2f}')

    # np.save(f"qtables/{episode}-qtable.npy", q_table)

env.close()

# Visualize stats
plt.plot(aggr_ep_rewards['ep'], aggr_ep_rewards['min'], label="min rewards")
plt.plot(aggr_ep_rewards['ep'], aggr_ep_rewards['max'], label="max rewards")
plt.plot(aggr_ep_rewards['ep'], aggr_ep_rewards['avg'], label="average rewards")
plt.legend(loc=4)
plt.grid(True)
plt.show()
