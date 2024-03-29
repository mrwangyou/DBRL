import random
import os
import re
import sys
import time
import warnings
import math

import gym
import numpy as np
from gym import Env
from gym.spaces import Box, Discrete
from pathlib import Path


if re.findall('dogfightEnv\.py$', str(Path(__file__).resolve())) and \
   re.findall('dogfightEnv$', str(Path(__file__).resolve().parents[0])) and \
   re.findall('envs$', str(Path(__file__).resolve().parents[1])) and \
   re.findall('gym$', str(Path(__file__).resolve().parents[2])):
    print("Using Gym Version")
    time.sleep(1)

elif re.findall('dogfightEnv\.py$', str(Path(__file__).resolve())) and \
     re.findall('dogfightEnv$', str(Path(__file__).resolve().parents[0])) and \
     re.findall('environments$', str(Path(__file__).resolve().parents[1])) and \
     re.findall('src$', str(Path(__file__).resolve().parents[2])) and \
     re.findall('DBRL$', str(Path(__file__).resolve().parents[3])):
    print("Using DBRL Version")
    time.sleep(1)

sys.path.append(str(Path(__file__).resolve().parents[2]) + '/envs/dogfightEnv')
# You can also replace `import socket_lib` in `dogfight_sandbox_hg2\network_client_example\dogfight_client.py` with `from . import socket_lib` to achieve the same function as the following line of code
sys.path.append(str(Path(__file__).resolve().parents[2]) + '/envs/dogfightEnv/dogfight_sandbox_hg2/network_client_example/')

from initialization import *
from models import *

try:
    from .dogfight_sandbox_hg2.network_client_example import \
        dogfight_client as df
    print("Gym")
    time.sleep(1)
except:
    from .dogfight_sandbox_hg2.network_client_example import \
        dogfight_client as df
    print("DBRL")
    time.sleep(1)


class DogfightEnv(Env):

    def __init__(
        self,
        host='10.184.0.0',
        port='50888',
        plane_slot=1,
        enemy_slot=3,
        missile_slot=1,
        rendering=False,
        record_status=0,
        initial_state='carrier',
        throttle_enable=False,
        flare_enable=False,
        ego_pose_enable=True,
        oppo_pose_enable=False,
        missile_pose_enable=True,
        missile_relative_azimuth_enable=False,
        msg=None,
    ) -> None:

        self.host = host
        self.port = port
        self.nof = 0
        self.rendering = rendering
        self.plane_slot = plane_slot
        self.enemy_slot = enemy_slot
        self.missile_slot = missile_slot
        self.record_status = record_status
        self.initial_state = initial_state
        self.throttle_enable = throttle_enable
        self.flare_enable = flare_enable
        self.flare_active = False
        self.ego_pose_enable = ego_pose_enable
        self.oppo_pose_enable = oppo_pose_enable
        self.missile_pose_enable = missile_pose_enable
        self.missile_relative_azimuth_enable = missile_relative_azimuth_enable
        self.msg = msg

        try:
            planes = df.get_planes_list()
        except:
            print('Connecting...')
            df.connect(host, int(port))
            time.sleep(2)
            planes = df.get_planes_list()

        df.disable_log()

        for i in planes:
            df.reset_machine(i)
        
        self.planeID = planes[self.plane_slot]
        self.enemyID = planes[self.enemy_slot]

        missiles = df.get_machine_missiles_list(self.enemyID)
        self.missileID = missiles[self.missile_slot]

        df.set_plane_thrust(self.enemyID, 1)
        df.set_plane_thrust(self.planeID, 1)

        df.set_client_update_mode(True)

        df.set_renderless_mode(True)

        if self.initial_state == 'carrier':
            start_on_carrier(
                df=df, 
                planeID=self.planeID,
                enemyID=self.enemyID,
                missile_slot=self.missile_slot,
                missileID=self.missileID,
            )
        elif self.initial_state == 'air':
            start_in_sky(
                df=df, 
                planeID=self.planeID,
                enemyID=self.enemyID,
                missile_slot=self.missile_slot,
                missileID=self.missileID,
            )
        else:
            raise Exception('Invalid initial state {}!'.format(self.initial_state))

        self.action_space, self.action_type = action_space(
            pitch_enable=True,
            roll_enable=True,
            yaw_enable=True,
            flaps_enable=True,
            throttle_enable=throttle_enable,
            flare_enable=flare_enable
        )

        self.observation_space = observation_space(
            ego_plane_position=ego_pose_enable,
            ego_plane_attitude=ego_pose_enable,
            oppo_plane_position=oppo_pose_enable,
            oppo_plane_attitude=oppo_pose_enable,
            missile_position=missile_pose_enable,
            missile_attitude=missile_pose_enable,
            missile_relative_azimuth=missile_relative_azimuth_enable,
        )

        
        if self.rendering:
            df.set_renderless_mode(False)

    def getProperty(
        self,
        prop
    ):
        if prop == 'position':
            plane_state = df.get_plane_state(self.planeID)
            return [
                plane_state['position'][0],
                plane_state['position'][2],
                plane_state['position'][1],
            ]
        elif prop == 'positionEci':
            warnings.warn('Dogfight simulation environments have no global data!')
            plane_state = df.get_plane_state(self.planeID)
            return [
                plane_state['position'][0],
                plane_state['position'][2],
                plane_state['position'][1],
            ]
        elif prop == 'positionEcef':
            warnings.warn('Dogfight simulation environments have no global data!')
            plane_state = df.get_plane_state(self.planeID)
            return [
                plane_state['position'][0],
                plane_state['position'][2],
                plane_state['position'][1],
            ]
        elif prop == 'attitudeRad':
            plane_state = df.get_plane_state(self.planeID)
            return [
                plane_state['heading'] / 180 * np.pi,
                plane_state['pitch_attitude'] / 180 * np.pi,
                plane_state['roll_attitude'] / 180 * np.pi,
            ]
        elif prop == 'attitudeDeg':
            plane_state = df.get_plane_state(self.planeID)
            return [
                plane_state['heading'],
                plane_state['pitch_attitude'],
                plane_state['roll_attitude'],
            ]
        elif prop == 'pose':
            plane_state = df.get_plane_state(self.planeID)
            return [
                plane_state['position'][0],
                plane_state['position'][2],
                plane_state['position'][1],
                plane_state['heading'],
                plane_state['pitch_attitude'],
                plane_state['roll_attitude'],
            ]
        elif prop == 'poseNorm':
            plane_state = df.get_plane_state(self.planeID)
            return [
                plane_state['position'][0] / 100,
                plane_state['position'][2] / 100,
                plane_state['position'][1] / 50,
                plane_state['heading'],
                plane_state['pitch_attitude'] * 4,
                plane_state['roll_attitude'] * 4,
            ]
        elif prop == 'velocity':
            warnings.warn('三个值为速度在欧拉角上的分量, 与JSBSim中的速度不同')
            plane_state = df.get_plane_state(self.planeID)
            return [
                plane_state['horizontal_speed'],
                plane_state['linear_speed'],
                -plane_state['vertical_speed'],
            ]
        elif prop == 'poseMissile':
            missile_state = df.get_missile_state(self.missileID)
            return [
                missile_state['position'][0],
                missile_state['position'][2],
                missile_state['position'][1],
                missile_state['Euler_angles'][0],
                missile_state['Euler_angles'][1],
                missile_state['Euler_angles'][2],
            ]
        elif prop == 'poseMissleNorm':
            missile_state = df.get_missile_state(self.missileID)
            return [
                missile_state['position'][0] / 100,
                missile_state['position'][2] / 100,
                missile_state['position'][1] / 50,
                missile_state['Euler_angles'][0] * 100,
                missile_state['Euler_angles'][1] * 100,
                missile_state['Euler_angles'][2] * 100,
            ]
        elif prop == 'azimuthRel':
            plane_state = df.get_plane_state(self.planeID)
            missile_state = df.get_missile_state(self.missileID)
            azimuth = [
                missile_state['position'][0] - plane_state['position'][0],
                missile_state['position'][2] - plane_state['position'][2],
                missile_state['position'][1] - plane_state['position'][1],
            ]
            azimuthRel = [0, 0, 0]
            for idx, value in enumerate(azimuth):
                azimuthRel[idx] = value / np.linalg.norm(azimuth)
            return azimuthRel

        else:
            raise Exception("Property {} doesn't exist!".format(prop))

    def getDistance(self):
        plane_state = df.get_plane_state(self.planeID)
        missile_state = df.get_missile_state(self.missileID)

        return ((plane_state['position'][0] - missile_state['position'][0]) ** 2 +\
        (plane_state['position'][1] - missile_state['position'][1]) ** 2 +\
        (plane_state['position'][2] - missile_state['position'][2]) ** 2) ** .5

    def getHP(self):
        return df.get_health(self.planeID)['health_level']

    def terminate(self):
        if not df.get_missile_state(self.missileID)['active']:
            if self.getHP() >= .9:
                return 1
            else:
                return -1
        else:
            return 0

    def setProperty(
        self,
        prop,
        value,
    ):
        if prop == 'plane':
            self.plane_slot = value
        elif prop == 'enemy':
            self.enemy_slot = value
        elif prop == 'missile':
            self.missile_slot = value
        else:
            raise Exception("Property {} doesn't exist!".format(prop))

    def sendAction(
        self,
        action,
        actionType=None,
    ):
        if actionType == None:
            for idx, value in enumerate(self.action_type):
                if value[0] == '_':
                    eval(value[1:])(self.planeID, float(action[idx]))

                elif value == 'Flare' and not self.flare_active:
                    if float(action[idx]) >= 0.99:
                        self.flare_active = True
                        self.setFlare()

        elif actionType == 'Flaps' or actionType == 'flaps':
            df.set_plane_flaps(self.planeID, action)
        elif actionType == 'Pitch' or actionType == 'pitch':
            df.set_plane_pitch(self.planeID, action)
        elif actionType == 'Roll' or actionType == 'roll':
            df.set_plane_roll(self.planeID, action)
        elif actionType == 'Yaw' or actionType == 'yaw':
            df.set_plane_yaw(self.planeID, action)

    def setFlare(
        self,
    ):
        planes = df.get_planes_list()
        plane_id = planes[self.plane_slot]

        missiles = df.get_machine_missiles_list(plane_id)

        self.flare_slot = 0
        self.flare_id = missiles[self.flare_slot]

        df.fire_missile(plane_id, self.flare_slot)

        df.set_machine_custom_physics_mode(self.flare_id, True)

        df.set_missile_life_delay(self.flare_id, 5)

        df.update_scene()


        flare_state = df.get_missile_state(self.flare_id)
        self.x, self.y, self.z = flare_state["position"][0], flare_state["position"][1], flare_state["position"][2]
        self.y_init = self.y
        self.z_init = self.z
        self.v_init = df.get_plane_state(planes[self.plane_slot])['linear_speed']
        self.w_init = df.get_plane_state(planes[self.plane_slot])['vertical_speed']

        self.flare_matrix = [
            1, 0, 0,
            0, 1, 0,
            0, 0, 1,
            self.x, self.y, self.z,
        ]

        # Linear displacement vector in m.s-1
        self.flare_speed_vector = [1, 1, 1]

        # Custom missile movements
        self.flare_active_time = 0

    def flare_step(
        self,
    ):
        frame_time_step = 1/60
        
        flare_state = df.get_missile_state(self.flare_id)
        if not flare_state["wreck"]:
            self.flare_matrix[9] = self.x
            self.flare_matrix[10] = self.y
            self.flare_matrix[11] = self.z

            df.update_machine_kinetics(self.flare_id, self.flare_matrix, self.flare_speed_vector)
            df.update_scene()
            self.x = self.x
            self.y = self.y_init + self.w_init * self.flare_active_time - 0.5 * 10 * self.flare_active_time * self.flare_active_time
            self.z = self.z_init + self.v_init * self.flare_active_time
            
            # Compute speed vector, used by missile engine smoke
            self.flare_speed_vector = [(self.x-self.flare_matrix[9]) / frame_time_step, (self.y - self.flare_matrix[10]) / frame_time_step, (self.z - self.flare_matrix[11]) / frame_time_step]

            self.flare_active_time += frame_time_step
        
        if df.get_missile_state(self.flare_id)['active']:

            if random.random() < .5:
                df.set_missile_target(self.missileID, self.planeID)
            else:
                df.set_missile_target(self.missileID, self.flare_id)

        else:
            df.set_missile_target(self.missileID, self.planeID)

    def XYtoGPS(self, x, y, ref_lat=0, ref_lon=0):
        CONSTANTS_RADIUS_OF_EARTH = 6378137.  # meters (m)
        x_rad = float(x) / CONSTANTS_RADIUS_OF_EARTH
        y_rad = float(y) / CONSTANTS_RADIUS_OF_EARTH
        c = math.sqrt(x_rad * x_rad + y_rad * y_rad)

        ref_lat_rad = math.radians(ref_lat)
        ref_lon_rad = math.radians(ref_lon)

        ref_sin_lat = math.sin(ref_lat_rad)
        ref_cos_lat = math.cos(ref_lat_rad)

        if abs(c) > 0:
            sin_c = math.sin(c)
            cos_c = math.cos(c)

            lat_rad = math.asin(cos_c * ref_sin_lat + (x_rad * sin_c * ref_cos_lat) / c)
            lon_rad = (ref_lon_rad + math.atan2(y_rad * sin_c, c * ref_cos_lat * cos_c - x_rad * ref_sin_lat * sin_c))

            lat = math.degrees(lat_rad)
            lon = math.degrees(lon_rad)

        else:
            lat = math.degrees(ref_lat)
            lon = math.degrees(ref_lon)

        return lat, lon

    def getObservation(self):
        ob = np.array([])
        if self.ego_pose_enable:
            ob = np.append(ob, self.getProperty('poseNorm'))
        if self.oppo_pose_enable:
            pass
        if self.missile_pose_enable:
            ob = np.append(ob, self.getProperty('poseMissleNorm'))
        if self.missile_relative_azimuth_enable:
            ob = np.append(ob, self.getProperty('azimuthRel'))
        
        return ob

    def step(self, action):

        t_begin = time.time()

        self.sendAction(action)
        
        if self.flare_enable and self.flare_active:
            self.flare_step()

        df.update_scene()
        self.nof += 1

        if self.record_status:
            if self.nof == 1:
                path = r'./log/'
                if not os.path.exists(path):
                    os.mkdir(path)
                file = open('./log/{}_status_record.txt'.format(self.msg), 'w')
                file.write('FileType=text/acmi/tacview\nFileVersion=2.1\n0,ReferenceTime=2022-10-01T00:00:00Z\n0,Title = test simple aircraft\n1000000,T=160.123456|24.8976763|0, Type=Ground+Static+Building, Name=Competition, EngagementRange=30000\n')
            else:
                file = open('./log/{}_status_record.txt'.format(self.msg), 'a')
            file.write('#{}\n'.format(self.nof * df.get_timestep()['timestep']))
            if self.nof == 1:
                initMsg = ',Name=F16,Type=Air+FixedWing,Coalition=Enemies,Color=Blue,Mach=0.800,ShortName=F16,RadarMode=1,RadarRange=40000,RadarHorizontalBeamwidth=60,RadarVerticalBeamwidth=60'
            else:
                initMsg = ''
            file.write("1,T={0[0]}|{0[1]}|{1}{2}\n".format(
                self.XYtoGPS(self.getProperty('position')[0], self.getProperty('position')[1]), 
                self.getProperty('position')[2],
                initMsg
                )
            )
            file.close()

        terminate = self.terminate()

        if terminate == 1:
            reward = 50
        elif terminate == -1:
            reward = -50
        else:
            reward = .1
            if self.getHP() <= .1:
                reward = -1

        terminate = True if terminate else False
        
        # plane_state = df.get_plane_state(self.planeID)
        # missile_state = df.get_missile_state(self.missileID)

        ob = self.getObservation()

        if self.rendering:
            time.sleep(
                max(0, df.get_timestep()['timestep'] - (time.time() - t_begin))
            )

        return ob, reward, terminate, {}

    def render(self, id=0):
        
        df.set_renderless_mode(False)


    def reset(
        self,
    ): 

        self.__init__(
            host=self.host,
            port=self.port,
            plane_slot=self.plane_slot,
            enemy_slot=self.enemy_slot,
            missile_slot=self.missile_slot,
            rendering=self.rendering,
            record_status=self.record_status,
            initial_state=self.initial_state,
            throttle_enable=self.throttle_enable,
            flare_enable=self.flare_enable,
            ego_pose_enable = self.ego_pose_enable,
            oppo_pose_enable = self.oppo_pose_enable,
            missile_pose_enable = self.missile_pose_enable,
            missile_relative_azimuth_enable = self.missile_relative_azimuth_enable,
            msg=self.msg
        )

        ob = self.getObservation()

        return ob

