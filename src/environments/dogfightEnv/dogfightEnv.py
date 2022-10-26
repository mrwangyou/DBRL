import random
import re
import sys
import time
import warnings

import gym
import numpy as np
from gym import Env
from gym.spaces import Box, Discrete

sys.path.append('./src/')
sys.path.append('./src/environments/dogfightEnv/')
sys.path.append('./src/environments/dogfightEnv/dogfight_sandbox_hg2/network_client_example/')
sys.path.append('gym.envs.dogfightEnv.dogfight_sandbox_hg2.network_client_example/')


try:
    from gym.envs.dogfightEnv.dogfight_sandbox_hg2.network_client_example import \
        dogfight_client as df
    print("Gym")
    time.sleep(1)
except:
    from dogfightEnv.dogfight_sandbox_hg2.network_client_example import \
        dogfight_client as df
    print("DBRL")
    time.sleep(1)


class DogfightEnv(Env):

    def __init__(
        self,
        host='10.184.0.0',
        port='50888',
        render=False
    ) -> None:

        self.host = host
        self.port = port
        self.nof = 0

        try:
            df.get_planes_list()
        except:
            print('Run for the first time')
            df.connect(host, int(port))
            time.sleep(2)
        
        planes = df.get_planes_list()

        df.disable_log()
        
        self.planeID = planes[1]

        for i in planes:
            df.reset_machine(i)
        
        df.set_plane_thrust(planes[3], 1)
        df.set_plane_thrust(planes[1], 1)

        df.set_client_update_mode(True)

        df.set_renderless_mode(True)

        t = 0
        while t < 1:
            plane_state = df.get_plane_state(planes[3])
            df.update_scene()
            t = plane_state["thrust_level"]
        
        df.activate_post_combustion(planes[3])
        df.activate_post_combustion(planes[1])

        df.set_plane_pitch(planes[3], -0.5)
        df.set_plane_pitch(planes[1], -0.5)

        p = 0
        while p < 15:
            plane_state = df.get_plane_state(planes[3])
            df.update_scene()
            p = plane_state["pitch_attitude"]

        df.stabilize_plane(planes[3])
        df.stabilize_plane(planes[1])

        df.retract_gear(planes[3])
        df.retract_gear(planes[1])

        s = 0
        while s < 1000:
            plane_state = df.get_plane_state(planes[3])
            df.update_scene()
            s = plane_state["altitude"]
        
        df.set_plane_yaw(self.planeID, 1)

        missiles = df.get_machine_missiles_list(planes[3])

        missile_slot = 1
        self.missileID = missiles[missile_slot]

        df.fire_missile(planes[3], missile_slot)

        df.set_missile_target(self.missileID, 'ally_2')
        df.set_missile_life_delay(self.missileID, 30)


        self.action_space = Box(
            low=np.array([
                0,  # Flaps 襟翼
                -1,  # Pitch 俯仰角
                -1,  # Roll 翻滚角
                -1,  # Yaw 偏航角
            ]),
            high=np.array([
                1,
                1,
                1,
                1,
            ]),
        )

        self.observation_space = Box(
            low=np.array([  # simple normalized
                -300,  # x / 100
                -300,  # y / 100
                -1,    # z / 50
                0,     # heading
                -360,  # pitch_attitude * 4
                -360,  # roll_attitude * 4
                -300,  # x / 100
                -300,  # y / 100
                -1,    # z / 50
                -315,  # heading * 100
                -315,  # pitch_attitude * 100
                -315,  # roll_attitude * 100
            ]),
            high=np.array([
                300,
                300,
                200,
                360,
                360,
                360,
                300,
                300,
                200,
                315,
                315,
                315,
            ])
        )

        if render == True:
            df.set_renderless_mode(False)

    def getProperty(
        self,
        prop
    ):
        if prop == 'position':
            return [
                df.get_plane_state(self.planeID)['position'][0],
                df.get_plane_state(self.planeID)['position'][2],
                df.get_plane_state(self.planeID)['position'][1],
            ]
        elif prop == 'positionEci':
            warnings.warn('Dogfight simulation environments have no global data!')
            return [
                df.get_plane_state(self.planeID)['position'][0],
                df.get_plane_state(self.planeID)['position'][2],
                df.get_plane_state(self.planeID)['position'][1],
            ]
        elif prop == 'positionEcef':
            warnings.warn('Dogfight simulation environments have no global data!')
            return [
                df.get_plane_state(self.planeID)['position'][0],
                df.get_plane_state(self.planeID)['position'][2],
                df.get_plane_state(self.planeID)['position'][1],
            ]
        elif prop == 'attitudeRad':
            return [
                df.get_plane_state(self.planeID)['heading'] / 180 * np.pi,
                df.get_plane_state(self.planeID)['pitch_attitude'] / 180 * np.pi,
                df.get_plane_state(self.planeID)['roll_attitude'] / 180 * np.pi,
            ]
        elif prop == 'attitudeDeg':
            return [
                df.get_plane_state(self.planeID)['heading'],
                df.get_plane_state(self.planeID)['pitch_attitude'],
                df.get_plane_state(self.planeID)['roll_attitude'],
            ]
        elif prop == 'pose':
            return [
                df.get_plane_state(self.planeID)['position'][0],
                df.get_plane_state(self.planeID)['position'][2],
                df.get_plane_state(self.planeID)['position'][1],
                df.get_plane_state(self.planeID)['heading'],
                df.get_plane_state(self.planeID)['pitch_attitude'],
                df.get_plane_state(self.planeID)['roll_attitude'],
            ]
        elif prop == 'velocity':
            warnings.warn('三个值为速度在欧拉角上的分量, 与JSBSim中的速度不同')
            return [
                df.get_plane_state(self.planeID)['horizontal_speed'],
                df.get_plane_state(self.planeID)['linear_speed'],
                -df.get_plane_state(self.planeID)['vertical_speed'],
            ]
        elif prop == 'poseMissile':
            return [
                df.get_missile_state(self.missileID)['position'][0],
                df.get_missile_state(self.missileID)['position'][1],
                df.get_missile_state(self.missileID)['position'][2],
                df.get_missile_state(self.missileID)['Euler_angles'][0],
                df.get_missile_state(self.missileID)['Euler_angles'][1],
                df.get_missile_state(self.missileID)['Euler_angles'][2],
            ]
        else:
            raise Exception("Property {} doesn't exist!".format(prop))

    def getDistance(self):
        return ((df.get_plane_state(self.planeID)['position'][0] - df.get_missile_state(self.missileID)['position'][0]) ** 2 +\
        (df.get_plane_state(self.planeID)['position'][1] - df.get_missile_state(self.missileID)['position'][1]) ** 2 +\
        (df.get_plane_state(self.planeID)['position'][2] - df.get_missile_state(self.missileID)['position'][2]) ** 2) ** .5

    def getHP(self):
        return df.get_health(self.planeID)['health_level']

    def terminate(self):
        if not df.get_missile_state('Meteorennemy_2.1')['active']:
            if self.getHP() >= .9:
                return 1
            else:
                return -1
        else:
            return 0

    def sendAction(
        self,
        action,
        actionType=None,
    ):
        if actionType == None:
            df.set_plane_flaps(self.planeID, float(action[0]))
            df.set_plane_pitch(self.planeID, float(action[1]))
            df.set_plane_roll(self.planeID, float(action[2]))
            df.set_plane_yaw(self.planeID, float(action[3]))
        elif actionType == 'Flaps' or actionType == 'flaps':
            df.set_plane_flaps(self.planeID, action)
        elif actionType == 'Pitch' or actionType == 'pitch':
            df.set_plane_pitch(self.planeID, action)
        elif actionType == 'Roll' or actionType == 'roll':
            df.set_plane_roll(self.planeID, action)
        elif actionType == 'Yaw' or actionType == 'yaw':
            df.set_plane_yaw(self.planeID, action)

    def step(self, action):
        
        self.sendAction(action)
        
        df.update_scene()
        self.nof += 1

        if self.terminate() == 1:
            reward = 50
        elif self.terminate() == -1:
            reward = -50
        else:
            reward = .1
            if self.getHP() <= .1:
                reward = -1

        terminate = True if self.terminate() else False
        
        ob = [  # normalized
            df.get_plane_state(self.planeID)['position'][0] / 100,
            df.get_plane_state(self.planeID)['position'][2] / 100,
            df.get_plane_state(self.planeID)['position'][1] / 50,
            df.get_plane_state(self.planeID)['heading'],
            df.get_plane_state(self.planeID)['pitch_attitude'] * 4,
            df.get_plane_state(self.planeID)['roll_attitude'] * 4,

            df.get_missile_state(self.missileID)['position'][0] / 100,
            df.get_missile_state(self.missileID)['position'][2] / 100,
            df.get_missile_state(self.missileID)['position'][1] / 50,
            df.get_missile_state(self.missileID)['Euler_angles'][0] * 100,
            df.get_missile_state(self.missileID)['Euler_angles'][1] * 100,
            df.get_missile_state(self.missileID)['Euler_angles'][2] * 100,
        ]

        return ob, reward, terminate, {}

    def render(self, id=0):
        
        df.set_renderless_mode(False)


    def reset(
        self,
    ):
        self.__init__(
            self.host,
            self.port,
        )
        
        ob = [  # normalized
            df.get_plane_state(self.planeID)['position'][0] / 100,
            df.get_plane_state(self.planeID)['position'][2] / 100,
            df.get_plane_state(self.planeID)['position'][1] / 50,
            df.get_plane_state(self.planeID)['heading'],
            df.get_plane_state(self.planeID)['pitch_attitude'] * 4,
            df.get_plane_state(self.planeID)['roll_attitude'] * 4,

            df.get_missile_state(self.missileID)['position'][0] / 100,
            df.get_missile_state(self.missileID)['position'][2] / 100,
            df.get_missile_state(self.missileID)['position'][1] / 50,
            df.get_missile_state(self.missileID)['Euler_angles'][0] * 100,
            df.get_missile_state(self.missileID)['Euler_angles'][1] * 100,
            df.get_missile_state(self.missileID)['Euler_angles'][2] * 100,
        ]

        return ob

