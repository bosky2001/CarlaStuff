import glob
import os
import sys

try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

import carla
import random
import time
import numpy as np
import cv2
import queue
import math
#image dimensions
IM_WIDTH = 640
IM_HEIGHT = 480

def get_speed(vehicle):

	vel = vehicle.get_velocity()
	return 3.6*math.sqrt(vel.x**2 +vel.y**2+vel.z**2)

class PIDControl():
    
    def __init__(self, vehicle, args_lateral, args_longitudnal, offset=0, max_throttle = 0.75, max_brake = 0.3, max_steering = 0.8):
        self.max_brake = max_brake
        self.max_steering = max_steering
        self.max_throttle = max_throttle

        self.vehicle = vehicle
        self.world = vehicle.get_world()
        self.past_steering = self.vehicle.get_control().steer
        self.lon_controller = PIDLongitudnalController(self.vehicle, **args_longitudnal)
        self.lat_controller = PIDLateralController(self.vehicle, offset, **args_lateral)

    def run_step(self, target_speed, waypoint):
	
        acceleration = self.lon_controller.run_step(target_speed)
        current_steering = self.lat_controller.run_step(waypoint)
        control = carla.VehicleControl()
        if acceleration>=0.0:
            control.throttle = min(abs(acceleration), self.max_brake)
            control.brake = 0.0
        else :
            control.throttle = 0.0
            control.brake = min(abs(acceleration), self.max_brake)
        if current_steering > self.past_steering+0.1:
            current_steering = self.past_steering+0.1
        
        elif current_steering<self.past_steering-0.1:
            current_steering = self.past_steering-0.1
        if current_steering>=0:
            steering = min(self.max_steering , current_steering)
            
        else:
            steering = max(-self.max_steering, current_steering)

        control.steer = steering
        control.hand_brake = False
        control.manual_gear_shift = False
        self.past_steering = steering
        return control

    def change_longitudinal_PID(self, args_longitudinal):
        """Changes the parameters of the PIDLongitudinalController"""
        self.lon_controller.change_parameters(**args_longitudinal)

    def change_lateral_PID(self, args_lateral):
        """Changes the parameters of the PIDLongitudinalController"""
        self.lat_controller.change_parameters(**args_lateral)

class PIDLongitudnalController():
    
    def __init__(self, vehicle , K_P=1.0 , K_D= 0.0, K_I=0.0, dt=0.03):

        self.vehicle = vehicle
        self.K_D = K_D
        self.K_P = K_P
        self.K_I = K_I
        self.dt = dt
        self.errorBuffer = queue.deque(maxlen = 10)
        
        
    def run_step(self, target_speed, debug=False):
        current_speed = get_speed(self.vehicle)
        return self.pid_controller(target_speed, current_speed)

    def pid_controller(self, target_speed, current_speed):
        error = target_speed -current_speed
        self.errorBuffer.append(error)
        de = 0.0
        ie = 0.0
        if len(self.errorBuffer)>=2:
            de = (self.errorBuffer[-1]-self.errorBuffer[-2])/self.dt
            ie = sum(self.errorBuffer)/self.dt

        return np.clip(self.K_P*error + self.K_D*de + self.K_I *ie,0.0, 1.0)

    
    def change_parameters(self, K_P, K_I, K_D, dt):
        """Changes the PID parameters"""
        self._k_p = K_P
        self._k_i = K_I
        self._k_d = K_D
        self._dt = dt

   


class PIDLateralController():

    def __init__(self, vehicle , offset=0, K_P=1.0 , K_D= 0.0, K_I=0.0, dt=0.03): 

        self.vehicle = vehicle
        self.K_D = K_D
        self.K_P = K_P
        self.K_I = K_I
        self.dt = dt
        self.offset = offset
        self.errorBuffer = queue.deque(maxlen = 10)

    def run_step(self,waypoint):
        return self.pid_controller(waypoint, self.vehicle.get_transform())

    def pid_controller(self,waypoint,vehicle_transform):

        v_begin = vehicle_transform.location
        v_end = v_begin+carla.Location(x=math.cos(math.radians(vehicle_transform.rotation.yaw)),y=math.sin(math.radians(vehicle_transform.rotation.yaw)))
        v_vec = np.array([v_end.x - v_begin.x, v_end.y - v_begin.y,0.0])
        w_vec = np.array([waypoint.transform.location.x - v_begin.x , waypoint.transform.location.y - v_begin.y,0.0])
        dot = math.acos(np.clip(np.dot(w_vec,v_vec)/np.linalg.norm(w_vec)*np.linalg.norm(v_vec), -1.0,1.0))
        cross = np.cross(v_vec,w_vec)

        if cross[2]<0:
            dot*=-1
            self.errorBuffer.append(dot)

        if len(self.errorBuffer)>=2:
            de = (self.errorBuffer[-1]-self.errorBuffer[-2]/self.dt)
            ie = sum(self.errorBuffer)*self.dt
            
        else:
            de = 0.0
            ie = 0.0

        return np.clip((self.K_P*dot)+(self.K_I*ie)+(self.K_D*de),-1.0,1.0)
    
    def change_parameters(self, K_P, K_I, K_D, dt):
        """Changes the PID parameters"""
        self._k_p = K_P
        self._k_i = K_I
        self._k_d = K_D
        self._dt = dt

'''def process_img(image):
    i = np.array(image.raw_data)
    i2 = i.reshape((IM_HEIGHT, IM_WIDTH, 4))
    i3 = i2[:, :, :3]
    cv2.imshow("", i3)
    cv2.waitKey(1)
    return i3/255.0'''

actor_list=[]

try:
    #creation of the world and actors
    client = carla.Client('localhost', 2000)
    client.set_timeout(20.0)

    world = client.get_world()

    blueprint_library = world.get_blueprint_library()
    map = world.get_map()


    bp = blueprint_library.filter('model3')[0]

    spawn_point = random.choice(world.get_map().get_spawn_points())

    vehicle = world.spawn_actor(bp, spawn_point)           # creating the car to be controlled
    

    actor_list.append(vehicle)
    #(vehicle, args_lateral={'K_P'=1.0, 'K_I'=0.0, 'K_D'=0.0} ,args_longitudnal={'K_P'=1.0, 'K_I'=0.0, 'K_D'=0.0})
    vehicle_control = PIDControl(vehicle, args_lateral={'K_P':1.0, 'K_I':0.0, 'K_D':0.0}, args_longitudnal= {'K_P':1.0, 'K_I':0.0, 'K_D':0.0})

    '''# Find the blueprint of the sensor.
    cam_bp = world.get_blueprint_library().find('sensor.camera.rgb')
    # Modify the attributes of the blueprint to set image resolution and field of view.
    cam_bp.set_attribute('image_size_x', f'{IM_WIDTH}')
    cam_bp.set_attribute('image_size_y', f'{IM_HEIGHT}')
    cam_bp.set_attribute('fov', '110')
    

    #adjusting camera to be on top of vehicle
    spawn_point = carla.Transform(carla.Location(x=2.5, z=0.7))

    cam_sen = world.spawn_actor(cam_bp, spawn_point, attach_to=vehicle)

    actor_list.append(cam_sen)
    cam_sen.listen(lambda data: process_img(data))'''

    while True:
        waypoints = world.get_map().get_waypoint(vehicle.get_location())
        waypoints = np.random.choice(waypoints.next(0.3))
        control_signal = vehicle_control.run_step(5,waypoints)
        vehicle.apply_control(control_signal)


finally: 
    for actor in actor_list:           # clearing up the actor after we are done using the server
        actor.destroy()
        print("Done cleaning up")


