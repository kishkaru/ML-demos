from collections import deque
from threading import Thread
import time
import random
import os
import math

from numpy import array
import carla

from tqdm import tqdm  # progressbar decorator for iterators
import numpy as np  # for array stuff and random
import cv2  # for showing our env live

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Flatten
from tensorflow.keras.layers import Conv2D, AveragePooling2D
from tensorflow.keras.optimizers import Adam
from modified_tensorboard import ModifiedTensorBoard

IMG_WIDTH = 640
IMG_HEIGHT = 480
SHOW_PREVIEW = False
SECONDS_PER_EPISODE = 10

REPLAY_MEMORY_SIZE = 5_000  # How many previous steps to keep for model training
MIN_REPLAY_MEMORY_SIZE = 1_000  # Minimum number of steps in a memory to start training
MINIBATCH_SIZE = 16  # How many steps (samples) to use for training
PREDICTION_BATCH_SIZE = 1  # How many steps (samples) to use for prediction
TRAINING_BATCH_SIZE = MINIBATCH_SIZE // 4
UPDATE_TARGET_EVERY = 5  # Terminal states (end of episodes)
ACTION_SPACE_SIZE = 3  # how many choices (3)
MODEL_NAME = "64x3-CNN"

MEMORY_FRACTION = 0.8  # Memory fraction, used mostly when training multiple agents
MIN_REWARD = -200  # For model save (model didn't get to the food, but also didn't hit an enemy)

EPISODES = 100

DISCOUNT = 0.99
epsilon = 1
EPSILON_DECAY = 0.95
MIN_EPSILON = 0.001

AGGREGATE_STATS_EVERY = 10  # episodes


# Default behavior of tf is to map all of the GPU memory of all GPUs
# Set to only grow the memory usage as is needed by the process instead (to avoid GPU memory overload)
physical_devices = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)


class CarEnv:
    SHOW_CAM = SHOW_PREVIEW
    front_camera = None

    im_width = IMG_WIDTH
    im_height = IMG_HEIGHT
    actor_list = []
    collision_hist = []

    STEER_AMT = 1.0

    def __init__(self):
        # Connect to Carla server, get world, and bp lib
        print('Connecting to Carla')
        self.client = carla.Client('localhost', 2000)
        self.client.set_timeout(2.0)
        self.world = self.client.get_world()
        self.bp_lib = self.world.get_blueprint_library()
        self.tm3_bp = self.bp_lib.filter('model3')[0]

        self.spawn_point = None
        self.vehicle = None
        self.rgba_cam_bp = None
        self.rgba_sensor = None
        self.colsensor_bp = None
        self.colsensor = None

    def process_img(self, image):
        i = array(image.raw_data)
        i2 = i.reshape((self.im_height, self.im_width, 4))
        i3 = i2[:, :, :3]
        # cv2.imwrite(f'output/{image.frame}.png', i3)
        if self.SHOW_CAM:
            cv2.imshow("Front Cam", i3)
            cv2.waitKey(1)
        self.front_camera = i3

    def collision_data(self, event):
        self.collision_hist.append(event)

    def reset(self):
        self.collision_hist = []
        self.actor_list = []

        # Spawn TM3 vehicle via bp
        self.spawn_point = random.choice(self.world.get_map().get_spawn_points())
        self.vehicle = self.world.spawn_actor(self.tm3_bp, self.spawn_point)
        self.actor_list.append(self.vehicle)

        # Get RGBA camera bp and config it
        self.rgba_cam_bp = self.bp_lib.find('sensor.camera.rgb')
        self.rgba_cam_bp.set_attribute('image_size_x', f'{IMG_WIDTH}')
        self.rgba_cam_bp.set_attribute('image_size_y', f'{IMG_HEIGHT}')
        self.rgba_cam_bp.set_attribute('fov', '110')

        # Adjust sensor relative to vehicle and spawn it
        spawn_point = carla.Transform(carla.Location(x=2.5, z=0.7))
        self.rgba_sensor = self.world.spawn_actor(self.rgba_cam_bp, spawn_point, attach_to=self.vehicle)
        self.actor_list.append(self.rgba_sensor)
        self.rgba_sensor.listen(lambda data: self.process_img(data))

        # initially passing some commands seems to help with time. Not sure why.
        self.vehicle.apply_control(carla.VehicleControl(throttle=0.0, brake=0.0))
        # sleep to get things started and to not detect a collision when the car spawns/falls from sky.
        time.sleep(4)

        self.colsensor_bp = self.bp_lib.find('sensor.other.collision')
        self.colsensor = self.world.spawn_actor(self.colsensor_bp, spawn_point, attach_to=self.vehicle)
        self.actor_list.append(self.colsensor)
        self.colsensor.listen(lambda event: self.collision_data(event))

        while self.front_camera is None:
            time.sleep(0.01)

        self.episode_start = time.time()
        # initially passing some commands seems to help with time. Not sure why.
        self.vehicle.apply_control(carla.VehicleControl(brake=0.0, throttle=0.0))
        return self.front_camera

    def step(self, action):
        """
        reinforcement learning paradigm
        return [observation, reward, done, any_extra_info]
        """
        if action == 0:
            self.vehicle.apply_control(carla.VehicleControl(throttle=1.0, steer=0))
        if action == 1:
            self.vehicle.apply_control(carla.VehicleControl(throttle=1.0, steer=-1 * self.STEER_AMT))
        if action == 2:
            self.vehicle.apply_control(carla.VehicleControl(throttle=1.0, steer=1 * self.STEER_AMT))

        v = self.vehicle.get_velocity()
        kmh = int(3.6 * math.sqrt(v.x**2 + v.y**2 + v.z**2))

        if len(self.collision_hist) != 0:
            done = True
            reward = -200
        elif kmh < 50:
            done = False
            reward = -1
        else:
            done = False
            reward = 1

        if self.episode_start + SECONDS_PER_EPISODE < time.time():
            done = True

        return self.front_camera, reward, done, None


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
        self.tensorboard = ModifiedTensorBoard(log_dir=f"logs/{MODEL_NAME}-{int(time.time())}")

        # Used to count when to update target network with main network's weights
        self.target_update_counter = 0
        self.graph = tf.compat.v1.get_default_graph()

        self.terminate = False  # Should we quit?
        self.last_logged_episode = 0
        self.training_initialized = False  # waiting for TF to get rolling

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

    def model_base_64x3_CNN(self, input_shape):
        model = Sequential()

        # layer 1: Convolutional (2D layer)
        model.add(Conv2D(64, (3, 3), input_shape=input_shape, padding='same'))
        model.add(Activation('relu'))
        model.add(AveragePooling2D(pool_size=(5, 5), strides=(3, 3), padding='same'))

        # layer 2: Convolutional (2D layer)
        model.add(Conv2D(64, (3, 3), padding='same'))
        model.add(Activation('relu'))
        model.add(AveragePooling2D(pool_size=(5, 5), strides=(3, 3), padding='same'))

        # layer 3: Convolutional (2D layer)
        model.add(Conv2D(64, (3, 3), padding='same'))
        model.add(Activation('relu'))
        model.add(AveragePooling2D(pool_size=(5, 5), strides=(3, 3), padding='same'))

        return model

    def create_model(self):
        # Use 64x3 as a base model
        base_model = self.model_base_64x3_CNN((IMG_HEIGHT, IMG_WIDTH, 3))

        # layer 4: Dense (1D "fully connected" layer)
        base_model.add(Flatten())  # this converts our 3D feature maps to 1D feature vectors
        base_model.add(Dense(64))

        # layer 5: Dense output (1D "fully connected" layer)
        base_model.add(Dense(ACTION_SPACE_SIZE, activation='linear'))

        base_model.compile(loss="mse", optimizer=Adam(lr=0.001), metrics=["accuracy"])
        return base_model

    # Trains main network every step during episode
    def train(self):
        # Start training only if certain number of samples is recorded
        if len(self.replay_memory) < MIN_REPLAY_MEMORY_SIZE:
            return

        # Get a minibatch of random samples from memory replay table
        minibatch = random.sample(self.replay_memory, MINIBATCH_SIZE)

        # transition = (current_state, action, reward, new_state, done)

        # Query main model model for Q values for current states
        current_states = np.array([transition[0] for transition in minibatch]) / 255
        current_qs_list = self.model.predict(current_states, PREDICTION_BATCH_SIZE)

        # Query target model for Q values for future states
        new_current_states = np.array([transition[3] for transition in minibatch]) / 255
        future_qs_list = self.target_model.predict(new_current_states, PREDICTION_BATCH_SIZE)

        features = []
        labels = []

        # Now we need to enumerate our batches
        for index, (current_state, action, reward, new_state, done) in enumerate(minibatch):
            if done:
                new_q = reward
            else:
                # Given the new state, the max possible Q value in next step: max(0..2)
                max_future_q = np.max(future_qs_list[index])
                # Calculate new Q value for current state and action
                new_q = reward + DISCOUNT * max_future_q

            # Given the current state and action: Update Q table with new Q value
            current_qs = current_qs_list[index]
            current_qs[action] = new_q

            # And append to our training data
            features.append(current_state)
            labels.append(current_qs)

        # Determine if we're at the end of an episode, so we can log
        log_this_step = False
        if self.tensorboard.step > self.last_logged_episode:
            log_this_step = True
            self.last_log_episode = self.tensorboard.step

        # Fit on all samples as one batch, log only on terminal state
        self.model.fit(np.array(features) / 255, np.array(labels), batch_size=TRAINING_BATCH_SIZE, verbose=0, shuffle=False,
                           callbacks=[self.tensorboard] if log_this_step else None)

        # Update target network counter at the end of each episode
        if log_this_step:
            self.target_update_counter += 1

        # If counter reaches UPDATE_TARGET_EVERY, update target network with weights of main network
        if self.target_update_counter > UPDATE_TARGET_EVERY:
            self.update_target_model()

    def train_in_loop(self):
        # Do a random warmup fitment to speed up actual training
        feature = np.random.uniform(size=(1, IMG_HEIGHT, IMG_WIDTH, 3)).astype(np.float32)
        label = np.random.uniform(size=(1, 3)).astype(np.float32)
        self.model.fit(feature, label, verbose=False, batch_size=1)

        self.training_initialized = True
        while True:
            if self.terminate:
                return
            self.train()
            time.sleep(0.01)


if __name__ == '__main__':
    # A random choice is going to be much faster than a predict operation,
    # so we can arbitrarily delay this by setting some sort of general FPS
    # Set this to be whatever your actual FPS is when epsilon is 0
    FPS = 60

    # For stats
    ep_rewards = [-200]

    # For more repetitive results
    random.seed(1)
    np.random.seed(1)
    tf.random.set_seed(1)

    # # Memory fraction, used mostly when training multiple agents
    # gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=MEMORY_FRACTION)
    # backend.set_session(tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)))

    # Create models folder
    if not os.path.isdir('models'):
        os.makedirs('models')

    # Create agent and environment
    agent = DQNAgent()
    env = CarEnv()

    # Start model training in a separate thread (using data from replay_memory)
    trainer_thread = Thread(target=agent.train_in_loop, daemon=True)
    trainer_thread.start()
    while not agent.training_initialized:
        time.sleep(0.01)

    # Initialize predictions - first prediction takes longer because of initialization that has to be done
    # It's better to do a first prediction before we start iterating over episode steps
    agent.get_qs(np.ones((env.im_height, env.im_width, 3)))

    # Iterate over episodes
    for episode in tqdm(range(1, EPISODES + 1), unit='episodes'):
        # env.collision_hist = []

        # Update tensorboard step every episode
        agent.tensorboard.step = episode

        # Restarting episode - reset episode reward and step number
        episode_reward = 0
        step = 1

        # Reset environment and get initial state
        current_state = env.reset()

        # Reset flag and start iterating until episode ends
        done = False
        episode_start = time.time()

        # Play for given number of seconds only
        while True:
            # Determine action to take
            if np.random.random() > epsilon:
                # Get action from Q table
                action = np.argmax(agent.get_qs(current_state))
            else:
                # Get random action
                action = np.random.randint(0, ACTION_SPACE_SIZE)
                # This takes no time, so we add a delay matching 60 FPS (prediction above takes longer)
                time.sleep(1 / FPS)

            # Perform the action and receive (new state, reward, done, any_extra_info)
            new_state, reward, done, _ = env.step(action)

            # Transform new continuous state to new discrete state and count reward
            episode_reward += reward

            # Every step we update replay memory
            agent.update_replay_memory((current_state, action, reward, new_state, done))

            current_state = new_state
            step += 1

            if done:
                break

        # Append episode reward to a list
        ep_rewards.append(episode_reward)

        # End of episode - destroy agents
        for actor in env.actor_list:
            actor.destroy()

        # Log stats (AGGREGATE_STATS_EVERY episodes)
        if not episode % AGGREGATE_STATS_EVERY or episode == 1:
            average_reward = sum(ep_rewards[-AGGREGATE_STATS_EVERY:])/len(ep_rewards[-AGGREGATE_STATS_EVERY:])
            min_reward = min(ep_rewards[-AGGREGATE_STATS_EVERY:])
            max_reward = max(ep_rewards[-AGGREGATE_STATS_EVERY:])
            agent.tensorboard.update_stats(reward_avg=average_reward, reward_min=min_reward, reward_max=max_reward, epsilon=epsilon)

            # Save model, but only when min reward is greater or equal a set value
            if min_reward >= MIN_REWARD:
                agent.model.save(f'models/{MODEL_NAME}__{max_reward:_>7.2f}max_{average_reward:_>7.2f}avg_{min_reward:_>7.2f}min__{int(time.time())}.model')

        # Decay epsilon
        if epsilon > MIN_EPSILON:
            epsilon *= EPSILON_DECAY
            epsilon = max(MIN_EPSILON, epsilon)

    # Set termination flag for training thread and wait for it to finish
    agent.terminate = True
    trainer_thread.join()
    agent.model.save(f'models/{MODEL_NAME}__{max_reward:_>7.2f}max_{average_reward:_>7.2f}avg_{min_reward:_>7.2f}min__{int(time.time())}.model')

