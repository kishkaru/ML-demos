import random
import time
import math
from numpy import array
import cv2

import carla


IMG_WIDTH = 1280
IMG_HEIGHT = 720
SHOW_PREVIEW = False
SECONDS_PER_EPISODE = 10


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

# finally:
#     print('Destroying actors...')
#     for actor in actor_list:
#         actor.destroy()
#     print('Done!')
