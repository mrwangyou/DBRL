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
sys.path.append('./src/environments/dfEnv/')
sys.path.append('./src/environments/dfEnv/dogfight_sandbox_hg2/network_client_example/')
sys.path.append('gym.envs.dfEnv.dogfight_sandbox_hg2.network_client_example/')


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


class DfEnv(Env):

    def __init__(
        self,
        host='10.184.0.0',
        port='50888',
        plane_slot=1,
        enemy_slot=3,
        rendering=False,
    ) -> None:
        
        self.host = host
        self.port = port
        self.nof = 0
        self.rendering = rendering
        self.plane_slot = plane_slot
        self.enemy_slot = enemy_slot


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

        df.set_target_id(self.enemyID, self.planeID)
        df.activate_IA(self.enemyID)

        self.action_space = Box(  # Same as Evade mode
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
                0,     # heading
                -360,  # pitch_attitude * 4
                -360,  # roll_attitude * 4
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
                360,
                360,
                360,
            ]),
            dtype=np.float64
        )

        if self.rendering:
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
        elif prop == 'positionEnemy':
            return [
                df.get_plane_state(self.enemyID)['position'][0],
                df.get_plane_state(self.enemyID)['position'][2],
                df.get_plane_state(self.enemyID)['position'][1],
            ]
        elif prop == 'attitudeRad':
            return [
                df.get_plane_state(self.planeID)['heading'] / 180 * np.pi,
                df.get_plane_state(self.planeID)['pitch_attitude'] / 180 * np.pi,
                df.get_plane_state(self.planeID)['roll_attitude'] / 180 * np.pi,
            ]
        elif prop == 'attitudeRadEnemy':
            return [
                df.get_plane_state(self.enemyID)['heading'] / 180 * np.pi,
                df.get_plane_state(self.enemyID)['pitch_attitude'] / 180 * np.pi,
                df.get_plane_state(self.enemyID)['roll_attitude'] / 180 * np.pi,
            ]
        elif prop == 'attitudeDeg':
            return [
                df.get_plane_state(self.planeID)['heading'],
                df.get_plane_state(self.planeID)['pitch_attitude'],
                df.get_plane_state(self.planeID)['roll_attitude'],
            ]
        elif prop == 'attitudeDegEnemy':
            return [
                df.get_plane_state(self.enemyID)['heading'],
                df.get_plane_state(self.enemyID)['pitch_attitude'],
                df.get_plane_state(self.enemyID)['roll_attitude'],
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
        elif prop == 'poseEnemy':
            return [
                df.get_plane_state(self.enemyID)['position'][0],
                df.get_plane_state(self.enemyID)['position'][2],
                df.get_plane_state(self.enemyID)['position'][1],
                df.get_plane_state(self.enemyID)['heading'],
                df.get_plane_state(self.enemyID)['pitch_attitude'],
                df.get_plane_state(self.enemyID)['roll_attitude'],
            ]
        elif prop == 'velocity':
            warnings.warn('三个值为速度在欧拉角上的分量, 与JSBSim中的速度不同')
            return [
                df.get_plane_state(self.planeID)['horizontal_speed'],
                df.get_plane_state(self.planeID)['linear_speed'],
                -df.get_plane_state(self.planeID)['vertical_speed'],
            ]
        elif prop == 'velocityEnemy':
            warnings.warn('三个值为速度在欧拉角上的分量, 与JSBSim中的速度不同')
            return [
                df.get_plane_state(self.enemyID)['horizontal_speed'],
                df.get_plane_state(self.enemyID)['linear_speed'],
                -df.get_plane_state(self.enemyID)['vertical_speed'],
            ]
        else:
            raise Exception("Property {} doesn't exist!".format(prop))
    
    def getDistanceVector(self, ego):
        
        positionEci1 = self.getProperty("position")
        positionEci2 = self.getProperty("positionEnemy")

        if ego == 1:
            return np.array(positionEci2) - np.array(positionEci1)
        elif ego == 2:
            return np.array(positionEci1) - np.array(positionEci2)
        else:
            raise Exception("Plane {} doesn\'t exist".format(ego))

    def getDistance(self):
        plane_state = df.get_plane_state(self.planeID)
        enemy_state = df.get_plane_state(self.enemyID)

        tmp1 = ((plane_state['position'][0] - enemy_state['position'][0]) ** 2 +\
        (plane_state['position'][1] - enemy_state['position'][1]) ** 2 +\
        (plane_state['position'][2] - enemy_state['position'][2]) ** 2) ** .5

        tmp2 = np.linalg.norm(self.getDistanceVector(1))
        tmp3 = np.linalg.norm(self.getDistanceVector(2))
        try:
            assert abs(tmp1 - tmp2) < 1, [tmp1, tmp2, abs(tmp1 - tmp2), abs(tmp1 - tmp2) < 1]
            assert abs(tmp3 - tmp2) < 1, [tmp3, tmp2, abs(tmp3 - tmp2), abs(tmp3 - tmp2) < 1]
        except:
            warnings.warn('飞机之间距离计算误差较大！')

        return tmp2
    
    def getDamage(self, ego):
        attitude1 = self.getProperty("attitudeRad")
        attitude2 = self.getProperty("attitudeRadEnemy")

        theta_1 = np.pi / 2 - attitude1[1]
        psi_1 = np.pi * 2 - ((attitude1[0] - np.pi / 2) % (np.pi * 2))
        heading_1 = np.array([
            np.sin(theta_1) * np.cos(psi_1),
            np.sin(theta_1) * np.sin(psi_1),
            np.cos(theta_1)
        ])

        theta_2 = np.pi / 2 - attitude2[1]
        psi_2 = np.pi * 2 - (attitude2[0] - np.pi / 2) % (np.pi * 2)
        heading_2 = np.array([
            np.sin(theta_2) * np.cos(psi_2),
            np.sin(theta_2) * np.sin(psi_2),
            np.cos(theta_2)
        ])

        if 500 <= self.getDistance() <= 3000:

            angle1 = np.arccos(
                np.dot(self.getDistanceVector(ego=1), heading_1) / 
                (self.getDistance() * np.linalg.norm(heading_1))
            )

            if -1 <= angle1 / np.pi * 180 <= 1:
                if ego == 2:
                    return (3000 - self.getDistance()) / 2500 / 120

            angle2 = np.arccos(
                np.dot(self.getDistanceVector(ego=2), heading_2) / 
                (self.getDistance() * np.linalg.norm(heading_2))
            )

            if -1 <= angle2 / np.pi * 180 <= 1:
                if ego == 1:
                    return (3000 - self.getDistance()) / 2500 / 120
            
            # print("angle1 {}\tangle2 {}".format(angle1 / np.pi * 180, angle2 / np.pi * 180))

        return 0

    def damage(self, ego):
        if ego == 1:
            df.set_health(self.planeID, df.get_health(self.planeID)['health_level'] - self.getDamage(1))
        if ego == 2:
            df.set_health(self.enemyID, df.get_health(self.enemyID)['health_level'] - self.getDamage(2))

    def getHP(self):
        return df.get_health(self.planeID)['health_level']

    def getHPEnemy(self):
        return df.get_health(self.enemyID)['health_level']

    def terminate(self):
        if self.getHP() <= 0.0 and self.getHPEnemy() > 0.0:  # Oppo
            return -1
        elif self.getHP() > 0.0 and self.getHPEnemy() <= 0.0:  # Ego
            return 1
        elif self.getHP() <= 0.0 and self.getHPEnemy() <= 0.0:  # Tie
            return 2
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

        t_begin = time.time()

        self.sendAction(action)
        
        df.update_scene()
        self.nof += 1

        # self.damage(1)
        self.damage(2)

        terminate = self.terminate()

        if terminate == 1:
            reward = 50
        elif terminate == -1:
            reward = 0
        elif terminate == 2:
            reward = 0
        else:
            reward = 0

        terminate = True if terminate else False
        
        ob = np.array([  # normalized
            df.get_plane_state(self.planeID)['position'][0] / 100,
            df.get_plane_state(self.planeID)['position'][2] / 100,
            df.get_plane_state(self.planeID)['position'][1] / 50,
            df.get_plane_state(self.planeID)['heading'],
            df.get_plane_state(self.planeID)['pitch_attitude'] * 4,
            df.get_plane_state(self.planeID)['roll_attitude'] * 4,

            df.get_plane_state(self.enemyID)['position'][0] / 100,
            df.get_plane_state(self.enemyID)['position'][2] / 100,
            df.get_plane_state(self.enemyID)['position'][1] / 50,
            df.get_plane_state(self.enemyID)['heading'],
            df.get_plane_state(self.enemyID)['pitch_attitude'] * 4,
            df.get_plane_state(self.enemyID)['roll_attitude'] * 4,

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
        self.__init__(
            host=self.host,
            port=self.port,
            plane_slot=self.plane_slot,
            enemy_slot=self.enemy_slot,
            rendering=self.rendering,
        )
        
        ob = np.array([  # normalized
            df.get_plane_state(self.planeID)['position'][0] / 100,
            df.get_plane_state(self.planeID)['position'][2] / 100,
            df.get_plane_state(self.planeID)['position'][1] / 50,
            df.get_plane_state(self.planeID)['heading'],
            df.get_plane_state(self.planeID)['pitch_attitude'] * 4,
            df.get_plane_state(self.planeID)['roll_attitude'] * 4,

            df.get_plane_state(self.enemyID)['position'][0] / 100,
            df.get_plane_state(self.enemyID)['position'][2] / 100,
            df.get_plane_state(self.enemyID)['position'][1] / 50,
            df.get_plane_state(self.enemyID)['heading'],
            df.get_plane_state(self.enemyID)['pitch_attitude'] * 4,
            df.get_plane_state(self.enemyID)['roll_attitude'] * 4,
        ])

        return ob

