from collections import deque
import time
import random
import os

from tqdm import tqdm  # progressbar decorator for iterators
import numpy as np  # for array stuff and random
from PIL import Image  # for creating visual of our env
import cv2  # for showing our env live

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.optimizers import Adam
from modified_tensorboard import ModifiedTensorBoard

# Environment settings
EPISODES = 20_000
MODEL_NAME = "256x2"

# Training settings
REPLAY_MEMORY_SIZE = 50_000  # How many previous steps to keep for model training
MIN_REPLAY_MEMORY_SIZE = 1000  # Minimum number of steps in a memory to start training
MINIBATCH_SIZE = 64  # How many steps (samples) to use for training
UPDATE_TARGET_EVERY = 5  # Terminal states (end of episodes)
DISCOUNT = 0.99
MIN_REWARD = -200  # For model save (model didn't get to the food, but also didn't hit an enemy)
MEMORY_FRACTION = 0.20  # Memory fraction, used mostly when training multiple agents

# Exploration settings
epsilon = 1  # not a constant, going to be decayed
EPSILON_DECAY = 0.99975
MIN_EPSILON = 0.001

#  Stats settings
AGGREGATE_STATS_EVERY = 50  # episodes
SHOW_PREVIEW = False


# Default behavior of tf is to map all of the GPU memory of all GPUs
# Set to only grow the memory usage as is needed by the process instead (to avoid GPU memory overload)
physical_devices = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)


class Blob:
    def __init__(self, size):
        self.size = size
        self.x = np.random.randint(0, size)
        self.y = np.random.randint(0, size)

    def __str__(self):
        return f"Blob ({self.x}, {self.y})"

    def __sub__(self, other):
        return (self.x-other.x, self.y-other.y)

    # Used to determine if two blobs are touching
    def __eq__(self, other):
        return self.x == other.x and self.y == other.y

    def action(self, choice):
        """
        Gives us 9 total movement options. (0,1,2,3,4,5,6,7,8)
        """
        # Diagonals
        if choice == 0:
            self.move(x=1, y=1)
        elif choice == 1:
            self.move(x=-1, y=-1)
        elif choice == 2:
            self.move(x=-1, y=1)
        elif choice == 3:
            self.move(x=1, y=-1)

        # Horizontal
        elif choice == 4:
            self.move(x=1, y=0)
        elif choice == 5:
            self.move(x=-1, y=0)

        # Vertical
        elif choice == 6:
            self.move(x=0, y=1)
        elif choice == 7:
            self.move(x=0, y=-1)

        # Don't move
        elif choice == 8:
            self.move(x=0, y=0)

    def move(self, x=False, y=False):
        # If no value for x, move randomly
        if not x:
            self.x += np.random.randint(-1, 2)
        else:
            self.x += x

        # If no value for y, move randomly
        if not y:
            self.y += np.random.randint(-1, 2)
        else:
            self.y += y

        # If we are out of bounds, fix!
        if self.x < 0:
            self.x = 0
        elif self.x > self.size-1:
            self.x = self.size-1
        if self.y < 0:
            self.y = 0
        elif self.y > self.size-1:
            self.y = self.size-1


class BlobEnv:
    SIZE = 10  # 10x10 "Q-Table"
    RETURN_IMAGES = True
    MOVE_PENALTY = 1  # Cost to move
    ENEMY_PENALTY = 300  # Cost if enemy encounter
    FOOD_REWARD = 25  # Reward if reached food
    OBSERVATION_SPACE_VALUES = (SIZE, SIZE, 3)  # for numpy visualization
    ACTION_SPACE_SIZE = 9  # 9 actions defined in Blob class
    PLAYER_N = 1  # player key in dict
    FOOD_N = 2  # food key in dict
    ENEMY_N = 3  # enemy key in dict
    # Blob colors
    d = {1: (255, 175, 0),  # blue
         2: (0, 255, 0),  # green
         3: (0, 0, 255)}  # red

    def __init__(self):
        self.player = None
        self.food = None
        self.enemy = None
        self.episode_step = None

    def reset(self):
        """
        returns observation used to calculate action
        """
        self.player = Blob(self.SIZE)
        self.food = Blob(self.SIZE)
        # Don't spawn the food on the player
        while self.food == self.player:
            self.food = Blob(self.SIZE)
        self.enemy = Blob(self.SIZE)
        # Don't spawn the enemy on the player
        while self.enemy == self.player or self.enemy == self.food:
            self.enemy = Blob(self.SIZE)

        self.episode_step = 0

        if self.RETURN_IMAGES:
            observation = np.array(self.get_image())
        else:
            observation = (self.player-self.food) + (self.player-self.enemy)
        return observation

    def step(self, action):
        """
        performs the action and returns (new_state, reward, done)
        """
        self.episode_step += 1
        self.player.action(action)

        #### MAYBE ###
        #enemy.move()
        #food.move()
        ##############

        # new observation
        if self.RETURN_IMAGES:
            new_state = np.array(self.get_image())
        else:
            new_state = (self.player-self.food) + (self.player-self.enemy)

        # reward
        # Player landed on enemy
        if self.player == self.enemy:
            reward = -self.ENEMY_PENALTY
        # Player reached the food
        elif self.player == self.food:
            reward = self.FOOD_REWARD
        else:
            reward = -self.MOVE_PENALTY

        # done condition
        done = False
        if reward == self.FOOD_REWARD or reward == -self.ENEMY_PENALTY or self.episode_step >= 200:
            done = True

        return new_state, reward, done

    def render(self):
        img = self.get_image()
        img = img.resize((300, 300))  # resizing so we can see our agent in all its glory.
        cv2.imshow("image", np.array(img))  # show it!
        cv2.waitKey(1)

    # FOR CNN #
    def get_image(self):
        env = np.zeros((self.SIZE, self.SIZE, 3), dtype=np.uint8)  # starts an rbg of our size
        env[self.food.x][self.food.y] = self.d[self.FOOD_N]  # sets the food location tile to green color
        env[self.enemy.x][self.enemy.y] = self.d[self.ENEMY_N]  # sets the enemy location to red
        env[self.player.x][self.player.y] = self.d[self.PLAYER_N]  # sets the player tile to blue
        img = Image.fromarray(env, 'RGB')  # reading to rgb. Apparently. Even tho color definitions are bgr. ???
        return img


class DQNAgent:
    def __init__(self):
        # Main model (used for fit)
        self.model = self.create_model()

        # Target network (used for predict)
        # update every every n episodes
        # used to determine future Q values
        self.target_model = self.create_model()
        self.target_model.set_weights(self.model.get_weights())

        # An array with last n steps for training
        # we will fit our model on a random selection of these previous n actions (to "smooth out" some of the crazy fluctuations)
        # List of tuples of: (curr_state, action, reward, new_state, done)
        self.replay_memory = deque(maxlen=REPLAY_MEMORY_SIZE)

        # Custom tensorboard object to reduce file IO (Keras wants to write a logfile per .fit())
        self.tensorboard = ModifiedTensorBoard(log_dir="logs/{}-{}".format(MODEL_NAME, int(time.time())))

        # Used to count when to update target network with main network's weights
        self.target_update_counter = 0

    # Adds step's data to a memory replay array
    def update_replay_memory(self, transition):
        # transition = (current_state, action, reward, new_state, done)
        self.replay_memory.append(transition)

    # Update the target model with the main model's weights
    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())
        self.target_update_counter = 0

    # Queries main network for Q values given current observation space (environment state)
    # (reshape because TensorFlow wants that exact explicit way to shape)
    # (-1 means a variable amount of this data could be fed through.)
    def get_qs(self, state):
        return self.model.predict(np.array(state).reshape(-1, *state.shape)/255)[0]

    def create_model(self):
        model = Sequential()

        # layer 1: Convolutional (2D layer)
        # OBSERVATION_SPACE_VALUES = (10, 10, 3) (a 10x10 RGB image)
        model.add(Conv2D(256, (3, 3), input_shape=env.OBSERVATION_SPACE_VALUES))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.2))

        # layer 2: Convolutional (2D layer)
        model.add(Conv2D(256, (3, 3)))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.2))

        # layer 3: Dense (1D "fully connected" layer)
        model.add(Flatten())  # this converts our 3D feature maps to 1D feature vectors
        model.add(Dense(64))

        # layer 4: Dense output (1D "fully connected" layer)
        model.add(Dense(env.ACTION_SPACE_SIZE, activation='linear'))  # ACTION_SPACE_SIZE = how many choices (9)
        model.compile(loss="mse", optimizer=Adam(lr=0.001), metrics=['accuracy'])
        return model

    # Trains main network every step during episode
    def train(self, done):
        # Start training only if certain number of samples is recorded
        if len(self.replay_memory) < MIN_REPLAY_MEMORY_SIZE:
            return

        # Get a minibatch of random samples from memory replay table
        minibatch = random.sample(self.replay_memory, MINIBATCH_SIZE)

        # transition = (current_state, action, reward, new_state, done)

        # Query main model model for Q values for current states
        current_states = np.array([transition[0] for transition in minibatch]) / 255
        current_qs_list = self.model.predict(current_states)

        # Query target model for Q values for future states
        new_states = np.array([transition[3] for transition in minibatch]) / 255
        future_qs_list = self.target_model.predict(new_states)

        features = []
        labels = []

        # Now we need to enumerate our batches
        for index, (current_state, action, reward, new_state, new_done) in enumerate(minibatch):
            if new_done:
                new_q = reward
            else:
                # Given the new state, the max possible Q value in next step: max(0..8)
                max_future_q = np.max(future_qs_list[index])
                # Calculate new Q value for current state and action
                new_q = reward + DISCOUNT * max_future_q

            # Given the current state and action: Update Q table with new Q value
            current_q = current_qs_list[index]
            current_q[action] = new_q

            # And append to our training data
            features.append(current_state)
            labels.append(current_q)

        # Fit on all samples as one batch, log only on terminal state
        self.model.fit(np.array(features) / 255,
                       np.array(labels),
                       batch_size=MINIBATCH_SIZE,
                       verbose=0,
                       shuffle=False,
                       callbacks=[self.tensorboard] if done else None)

        # Update target network counter at the end of each episode
        if done:
            self.target_update_counter += 1

        # If counter reaches UPDATE_TARGET_EVERY, update target network with weights of main network
        if self.target_update_counter > UPDATE_TARGET_EVERY:
            self.update_target_model()


env = BlobEnv()

# For stats
ep_rewards = [-200]

# For more repetitive results
random.seed(1)
np.random.seed(1)
tf.random.set_seed(1)

# Create models folder
if not os.path.isdir('models'):
    os.makedirs('models')

agent = DQNAgent()

# Iterate over episodes
for episode in tqdm(range(1, EPISODES + 1), unit='episodes'):
    # Update tensorboard step every episode
    agent.tensorboard.step = episode

    # Restarting episode - reset episode reward
    episode_reward = 0

    # Reset environment and get initial state
    current_state = env.reset()

    # Reset flag and start iterating until episode ends
    done = False
    while not done:
        # Determine action to take
        if np.random.random() > epsilon:
            # Get action from Q table
            action = np.argmax(agent.get_qs(current_state))
        else:
            # Get random action
            action = np.random.randint(0, env.ACTION_SPACE_SIZE)

        # Perform the action and receive (new state, reward, done)
        new_state, reward, done = env.step(action)

        # Transform new continuous state to new discrete state and count reward
        episode_reward += reward

        # Visualize every AGGREGATE_STATS_EVERY episode
        if SHOW_PREVIEW and not episode % AGGREGATE_STATS_EVERY:
            env.render()

        # Every step we update replay memory and train main network
        agent.update_replay_memory((current_state, action, reward, new_state, done))

        # Train to fit the agent at each step (and target_model UPDATE_TARGET_EVERY episode)
        agent.train(done)
        current_state = new_state

    # Append episode reward to a list
    ep_rewards.append(episode_reward)

    # Log stats (AGGREGATE_STATS_EVERY episodes)
    if not episode % AGGREGATE_STATS_EVERY or episode == 1:
        average_reward = sum(ep_rewards[-AGGREGATE_STATS_EVERY:])/len(ep_rewards[-AGGREGATE_STATS_EVERY:])
        min_reward = min(ep_rewards[-AGGREGATE_STATS_EVERY:])
        max_reward = max(ep_rewards[-AGGREGATE_STATS_EVERY:])
        agent.tensorboard.update_stats(reward_avg=average_reward, reward_min=min_reward, reward_max=max_reward, epsilon=epsilon)

        # Save model, but only when min reward is greater or equal a set value
        if average_reward >= MIN_REWARD:
            agent.model.save(f'models/{MODEL_NAME}__{max_reward:_>7.2f}max_{average_reward:_>7.2f}avg_{min_reward:_>7.2f}min__{int(time.time())}.model')

    # Decay epsilon
    if epsilon > MIN_EPSILON:
        epsilon *= EPSILON_DECAY
        epsilon = max(MIN_EPSILON, epsilon)


