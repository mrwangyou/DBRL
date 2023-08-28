import random
import re
import sys
import time
import warnings

import gym
import numpy as np
from gym import Env
from gym.spaces import Box, Discrete
from pathlib import Path

if str(Path(__file__).resolve().parents[2])[-3:] == 'gym':  # 后续改成re判断
    print("Using Gym Version")
    time.sleep(1)
elif str(Path(__file__).resolve().parents[3])[-4:] == 'DBRL':
    print("Using DBRL Version")
    time.sleep(1)

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
        plane_slot=1,
        enemy_slot=3,
        missile_slot=1,
        rendering=False,
        record_status=0,
        flare_enable=False,
    ) -> None:

        self.host = host
        self.port = port
        self.nof = 0
        self.rendering = rendering
        self.plane_slot = plane_slot
        self.enemy_slot = enemy_slot
        self.missile_slot = missile_slot
        self.record_status = record_status
        self.flare_enable = flare_enable
        self.flare_active = False

        if self.record_status > 0:
            try:
                self.status
            except AttributeError:
                self.status = []

            self.epoch_status = []

        try:
            planes = df.get_planes_list()
        except AttributeError:
            print('Run for the first time')
            df.connect(host, int(port))
            time.sleep(2)
            planes = df.get_planes_list()

        df.disable_log()
        
        self.planeID = planes[self.plane_slot]
        self.enemyID = planes[self.enemy_slot]

        for i in planes:
            df.reset_machine(i)
        

        df.set_plane_thrust(self.enemyID, 1)
        df.set_plane_thrust(self.planeID, 1)

        df.set_client_update_mode(True)

        df.set_renderless_mode(True)

        t = 0
        while t < 1:
            plane_state = df.get_plane_state(self.enemyID)
            df.update_scene()
            t = plane_state["thrust_level"]
        
        df.activate_post_combustion(self.enemyID)
        df.activate_post_combustion(self.planeID)

        df.set_plane_pitch(self.enemyID, -0.5)
        df.set_plane_pitch(self.planeID, -0.5)

        p = 0
        while p < 15:
            plane_state = df.get_plane_state(self.enemyID)
            df.update_scene()
            p = plane_state["pitch_attitude"]

        df.stabilize_plane(self.enemyID)
        df.stabilize_plane(self.planeID)

        df.retract_gear(self.enemyID)
        df.retract_gear(self.planeID)

        s = 0
        while s < 1000:
            plane_state = df.get_plane_state(self.enemyID)
            df.update_scene()
            s = plane_state["altitude"]
        
        df.set_plane_yaw(self.planeID, 1)

        missiles = df.get_machine_missiles_list(self.enemyID)
        self.missileID = missiles[self.missile_slot]

        df.fire_missile(self.enemyID, self.missile_slot)

        df.set_missile_target(self.missileID, self.planeID)
        df.set_missile_life_delay(self.missileID, 30)

        action_space_low = np.array([
            0,  # Flaps 襟翼
            -1,  # Pitch 俯仰角
            -1,  # Roll 翻滚角
            -1,  # Yaw 偏航角
        ])

        action_space_high = np.array([
            1,
            1,
            1,
            1,
        ])

        if self.flare_enable:  # Flare 干扰弹
            action_space_low = np.append(action_space_low, 0)
            action_space_high = np.append(action_space_high, 1)

        self.action_space = Box(
            low=action_space_low,
            high=action_space_high,
        )

        self.observation_space = Box(
            low=np.array([  # simple normalized
                # ego
                -300,  # x / 100
                -300,  # y / 100
                -1,    # z / 50
                0,     # heading
                -360,  # pitch_attitude * 4
                -360,  # roll_attitude * 4
                # oppo
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
            ]),
            dtype=np.float64
        )

        
        if self.rendering:
            df.set_renderless_mode(False)

    def getProperty(
        self,
        prop
    ):
        plane_state = df.get_plane_state(self.planeID)
        missile_state = df.get_missile_state(self.missileID)
        if prop == 'position':
            return [
                plane_state['position'][0],
                plane_state['position'][2],
                plane_state['position'][1],
            ]
        elif prop == 'positionEci':
            warnings.warn('Dogfight simulation environments have no global data!')
            return [
                plane_state['position'][0],
                plane_state['position'][2],
                plane_state['position'][1],
            ]
        elif prop == 'positionEcef':
            warnings.warn('Dogfight simulation environments have no global data!')
            return [
                plane_state['position'][0],
                plane_state['position'][2],
                plane_state['position'][1],
            ]
        elif prop == 'attitudeRad':
            return [
                plane_state['heading'] / 180 * np.pi,
                plane_state['pitch_attitude'] / 180 * np.pi,
                plane_state['roll_attitude'] / 180 * np.pi,
            ]
        elif prop == 'attitudeDeg':
            return [
                plane_state['heading'],
                plane_state['pitch_attitude'],
                plane_state['roll_attitude'],
            ]
        elif prop == 'pose':
            return [
                plane_state['position'][0],
                plane_state['position'][2],
                plane_state['position'][1],
                plane_state['heading'],
                plane_state['pitch_attitude'],
                plane_state['roll_attitude'],
            ]
        elif prop == 'velocity':
            warnings.warn('三个值为速度在欧拉角上的分量, 与JSBSim中的速度不同')
            return [
                plane_state['horizontal_speed'],
                plane_state['linear_speed'],
                -plane_state['vertical_speed'],
            ]
        elif prop == 'poseMissile':
            return [
                missile_state['position'][0],
                missile_state['position'][1],
                missile_state['position'][2],
                missile_state['Euler_angles'][0],
                missile_state['Euler_angles'][1],
                missile_state['Euler_angles'][2],
            ]
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
            df.set_plane_flaps(self.planeID, float(action[0]))
            df.set_plane_pitch(self.planeID, float(action[1]))
            df.set_plane_roll(self.planeID, float(action[2]))
            df.set_plane_yaw(self.planeID, float(action[3]))
            if self.flare_enable and not self.flare_active:
                if float(action[4]) >= 0.99:
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

        df.set_missile_life_delay(self.flare_id, 10)

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
        
        if random.random() < .5:
            df.set_missile_target(self.missileID, self.planeID)
        else:
            df.set_missile_target(self.missileID, self.flare_id)


    def step(self, action):

        t_begin = time.time()

        self.sendAction(action)
        
        if self.flare_enable and self.flare_active:
            self.flare_step()

        df.update_scene()
        self.nof += 1
        
        if self.record_status > 0 and self.nof % self.record_status == 0:

            self.epoch_status.append(
                df.get_plane_state(self.planeID)['position'] +
                [df.get_plane_state(self.planeID)['heading']] + 
                [df.get_plane_state(self.planeID)['pitch_attitude']] + 
                [df.get_plane_state(self.planeID)['roll_attitude']] + 
                [df.get_plane_state(self.planeID)['horizontal_speed']] +
                [df.get_plane_state(self.planeID)['vertical_speed']] +
                [df.get_plane_state(self.planeID)['linear_speed']] +

                df.get_missile_state(self.missileID)['position'] +
                [df.get_missile_state(self.missileID)['heading']] + 
                [df.get_missile_state(self.missileID)['pitch_attitude']] + 
                [df.get_missile_state(self.missileID)['roll_attitude']] + 
                [df.get_missile_state(self.missileID)['horizontal_speed']] +
                [df.get_missile_state(self.missileID)['vertical_speed']] +
                [df.get_missile_state(self.missileID)['linear_speed']]
            )

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
        
        plane_state = df.get_plane_state(self.planeID)
        missile_state = df.get_missile_state(self.missileID)

        ob = np.array([  # normalized
            plane_state['position'][0] / 100,
            plane_state['position'][2] / 100,
            plane_state['position'][1] / 50,
            plane_state['heading'],
            plane_state['pitch_attitude'] * 4,
            plane_state['roll_attitude'] * 4,

            missile_state['position'][0] / 100,
            missile_state['position'][2] / 100,
            missile_state['position'][1] / 50,
            missile_state['Euler_angles'][0] * 100,
            missile_state['Euler_angles'][1] * 100,
            missile_state['Euler_angles'][2] * 100,
        ])

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

        if self.record_status > 0:
            self.status.append(self.epoch_status)

            np.save('./log/evade_status_record', self.status)

        self.__init__(
            host=self.host,
            port=self.port,
            plane_slot=self.plane_slot,
            enemy_slot=self.enemy_slot,
            missile_slot=self.missile_slot,
            rendering=self.rendering,
            record_status=self.record_status,
            flare_enable=self.flare_enable,
        )

        plane_state = df.get_plane_state(self.planeID)
        missile_state = df.get_missile_state(self.missileID)

        ob = np.array([  # normalized
            plane_state['position'][0] / 100,
            plane_state['position'][2] / 100,
            plane_state['position'][1] / 50,
            plane_state['heading'],
            plane_state['pitch_attitude'] * 4,
            plane_state['roll_attitude'] * 4,

            missile_state['position'][0] / 100,
            missile_state['position'][2] / 100,
            missile_state['position'][1] / 50,
            missile_state['Euler_angles'][0] * 100,
            missile_state['Euler_angles'][1] * 100,
            missile_state['Euler_angles'][2] * 100,
        ])

        return ob

