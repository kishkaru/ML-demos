import random
import time
from numpy import array
import cv2

import carla


IMG_WIDTH = 1280
IMG_HEIGHT = 720


def process_img(image):
    i = array(image.raw_data)
    i2 = i.reshape((IMG_HEIGHT, IMG_WIDTH, 4))
    i3 = i2[:, :, :3]
    cv2.imwrite(f'output/{image.frame}.png', i3)
    cv2.imshow("Camera Output", i3)
    cv2.waitKey(1)
    return i3/255.0


actor_list = []
try:
    # Connect to Carla server, get world, and bp lib
    print('Connecting to Carla')
    client = carla.Client('localhost', 2000)
    client.set_timeout(2.0)
    world = client.get_world()
    bp_lib = world.get_blueprint_library()

    # Get TM3 bp and spawn it
    print('Spawning vehicle')
    bp = bp_lib.filter('model3')[0]
    spawn_point = random.choice(world.get_map().get_spawn_points())
    vehicle = world.spawn_actor(bp, spawn_point)
    actor_list.append(vehicle)

    # Get RGB camera bp and config it
    bp = bp_lib.find('sensor.camera.rgb')
    bp.set_attribute('image_size_x', f'{IMG_WIDTH}')
    bp.set_attribute('image_size_y', f'{IMG_HEIGHT}')
    bp.set_attribute('fov', '110')

    # Adjust sensor relative to vehicle and spawn it
    print('Spawning sensor')
    spawn_point = carla.Transform(carla.Location(x=2.5, z=0.7))
    sensor = world.spawn_actor(bp, spawn_point, attach_to=vehicle)
    actor_list.append(sensor)
    sensor.listen(lambda data: process_img(data))

    # Apply forward control to vehicle
    print('Controlling vehicle')
    vehicle.apply_control(carla.VehicleControl(throttle=2.0, steer=0.0))
    time.sleep(5)

finally:
    print('Destroying actors...')
    for actor in actor_list:
        actor.destroy()
    print('Done!')
