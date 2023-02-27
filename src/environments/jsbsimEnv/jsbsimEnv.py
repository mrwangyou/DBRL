import random
import re
import sys
import time

import gym
import numpy as np
from gym import Env
from gym.spaces import Box, Discrete

sys.path.append('./src/environments/jsbsimEnv')
try:
    from gym.envs.jsbsimEnv.jsbsimFdm import JsbsimFdm as Fdm
    print("Gym")
    time.sleep(2)
except:
    from jsbsimFdm import JsbsimFdm as Fdm
    print("DBRL")
    time.sleep(2)
    

class JsbsimEnv(Env):

    def __init__(
        self,
        fdm1=Fdm(
            fdm_id=1,
            fdm_fgfs=True
        ),
        fdm2=Fdm(
            fdm_id=2,
            fdm_ic_lat=.003,
            fdm_ic_psi=0,
            fdm_fgfs=True
        ),
        policy2='Level',
    ) -> None:
        super(JsbsimEnv, self).__init__()

        self.fdm1 = fdm1
        self.fdm2 = fdm2

        self.param1 = self.fdm1.param
        self.param2 = self.fdm2.param

        self.policy2 = policy2

        self.action_space = Box(
            low=np.array([
                -1,  # Aileron 副翼
                -1,  # Elevator 升降舵
                -1,  # Rudder 方向舵
                0,   # Throttle 油门
                0,   # Flap 襟翼
                0,   # Speed brake 减速板
                0,   # Spoiler 扰流片
            ]),
            high=np.array([
                1,
                1,
                1,
                1,
                1,
                1,
                1,
            ]),
        )

        self.observation_space = Box(
            low=np.array([
                -500,  # Latitude 纬度
                -500,  # Longitude 经度
                -500,  # Height above sea level 海拔
                0,     # Yaw 偏航角
                -180,  # Pitch 俯仰角
                -180,  # Roll 翻滚角
            ] * 2),
            high=np.array([
                500,
                500,
                500,
                360,
                180,
                180,
            ] * 2),
            dtype=np.float64
        )

    def getDistanceVector(self, ego):
        positionEci1 = self.fdm1.getProperty("positionEci")  # A list of size [3]
        positionEci2 = self.fdm2.getProperty("positionEci")  # A list of size [3]
        if ego == 1:
            return np.array(positionEci2) - np.array(positionEci1)
        elif ego == 2:
            return np.array(positionEci1) - np.array(positionEci2)
        else:
            raise Exception("Plane {} doesn\'t exist".format(ego))

    def getDistance(self):
        assert abs(np.linalg.norm(self.getDistanceVector(1)) - np.linalg.norm(self.getDistanceVector(2))) < 1e-5
        return np.linalg.norm(self.getDistanceVector(1))
    
    def getDamage(self, ego):
        attitude1 = self.fdm1.getProperty("attitudeRad")  # A list of size [3]
        attitude2 = self.fdm2.getProperty("attitudeRad")  # A list of size [3]

        theta_1 = np.pi / 2 - attitude1[1]
        psi_1 = np.pi / 2 - attitude1[0]
        heading_1 = np.array([
            np.cos(theta_1),
            np.sin(theta_1) * np.cos(psi_1),
            np.sin(theta_1) * np.sin(psi_1),
        ])

        theta_2 = np.pi / 2 - attitude2[1]
        psi_2 = np.pi / 2 - attitude2[0]
        heading_2 = np.array([
            np.cos(theta_2),
            np.sin(theta_2) * np.cos(psi_2),
            np.sin(theta_2) * np.sin(psi_2),
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
            
        return 0
    
    def damage(self):
        self.fdm1.damage(self.getDamage(1))
        self.fdm2.damage(self.getDamage(2))

    def reward(self):

        # sparse reward
        if self.terminate() == 2:
            r = -10
        elif self.terminate() == 1:
            r = 10
        elif self.terminate() == -1:
            r = -0.001
        else:
            r = -0.001 + (self.getDamage(2) - self.getDamage(1)) * 100

        return r

    def terminate(self):

        if self.fdm1.getProperty('position')[2] <= 10000:  # Height protection
            return 2

        if self.fdm2.getProperty('position')[2] <= 10000:
            return 1
    
        if self.fdm1.nof >= 2000:  # Tied
            return -1

        if self.fdm1.fdm_hp <= 0 and self.fdm2.fdm_hp > 0:
            return 2
        elif self.fdm2.fdm_hp <= 0 and self.fdm1.fdm_hp > 0:
            return 1
        elif self.fdm1.fdm_hp <= 0 and self.fdm2.fdm_hp <= 0:
            return -1
        else:
            return 0

    def step(self, action):
        
        self.fdm1.sendAction(action)

        if self.policy2 == 'Level':
            self.fdm2.sendAction(  # level flight
                action=-np.tanh((0 - self.fdm2.getProperty('attitudeDeg')[1]) * 1.5),
                actionType='fcs/elevator-cmd-norm'
            )
        elif self.policy2 == 'Random':
            self.fdm2.sendAction(
                self.action_space.sample()
            )

        self.damage()
        
        self.fdm1.step()
        self.fdm2.step()

        terminate = True if self.terminate() else False
        
        ob = np.array([  # ~100
            (self.fdm1.getProperty('position')[0] - self.fdm1.param['fdm_ic_lat']) * 1e4,
            (self.fdm1.getProperty('position')[1] - self.fdm1.param['fdm_ic_long']) * 1e4,
            (self.fdm1.getProperty('position')[2] - self.fdm1.param['fdm_ic_h']) / 10,
            (self.fdm1.getProperty('attitudeDeg')[0] - self.fdm1.param['fdm_ic_psi']),
            (self.fdm1.getProperty('attitudeDeg')[1] - self.fdm1.param['fdm_ic_theta']),
            (self.fdm1.getProperty('attitudeDeg')[2] - self.fdm1.param['fdm_ic_phi']),
            (self.fdm2.getProperty('position')[0] - self.fdm2.param['fdm_ic_lat']) * 1e4,
            (self.fdm2.getProperty('position')[1] - self.fdm2.param['fdm_ic_long']) * 1e4,
            (self.fdm2.getProperty('position')[2] - self.fdm2.param['fdm_ic_h']) / 10,
            (self.fdm2.getProperty('attitudeDeg')[0] - self.fdm2.param['fdm_ic_psi']),
            (self.fdm2.getProperty('attitudeDeg')[1] - self.fdm2.param['fdm_ic_theta']),
            (self.fdm2.getProperty('attitudeDeg')[2] - self.fdm2.param['fdm_ic_phi']),
        ])

        return ob, self.reward(), terminate, {}

    def render(self, id=0):

        if id == 1 or id == 0:
            self.fdm1.set_output_directive('./data_output/flightgear1.xml')

        if id == 2 or id == 0:
            self.fdm2.set_output_directive('./data_output/flightgear2.xml')

    def reset(self):
        
        self.fdm1 = Fdm(
            fdm_id=self.param1['fdm_id'],
            fdm_aircraft=self.param1['fdm_aircraft'],
            fdm_ic_v=self.param1['fdm_ic_v'],
            fdm_ic_lat=self.param1['fdm_ic_lat'],
            fdm_ic_long=self.param1['fdm_ic_long'],
            fdm_ic_h=self.param1['fdm_ic_h'],
            fdm_ic_psi=self.param1['fdm_ic_psi'],
            fdm_ic_theta=self.param1['fdm_ic_theta'],
            fdm_ic_phi=self.param1['fdm_ic_phi'],
            fdm_hp=self.param1['fdm_hp'],
            fdm_fgfs=self.param1['fdm_fgfs'],
            flight_mode=self.param1['flight_mode'],
        )

        self.fdm2 = Fdm(
            fdm_id=self.param2['fdm_id'],
            fdm_aircraft=self.param2['fdm_aircraft'],
            fdm_ic_v=self.param2['fdm_ic_v'],
            fdm_ic_lat=self.param2['fdm_ic_lat'],
            fdm_ic_long=self.param2['fdm_ic_long'],
            fdm_ic_h=self.param2['fdm_ic_h'],
            fdm_ic_psi=self.param2['fdm_ic_psi'],
            fdm_ic_theta=self.param2['fdm_ic_theta'],
            fdm_ic_phi=self.param2['fdm_ic_phi'],
            fdm_hp=self.param2['fdm_hp'],
            fdm_fgfs=self.param2['fdm_fgfs'],
            flight_mode=self.param2['flight_mode'],
        )

        ob = np.array([  # ~100
            (self.fdm1.getProperty('position')[0] - self.fdm1.param['fdm_ic_lat']) * 1e4,
            (self.fdm1.getProperty('position')[1] - self.fdm1.param['fdm_ic_long']) * 1e4,
            (self.fdm1.getProperty('position')[2] - self.fdm1.param['fdm_ic_h']) / 10,
            (self.fdm1.getProperty('attitudeDeg')[0] - self.fdm1.param['fdm_ic_psi']),
            (self.fdm1.getProperty('attitudeDeg')[1] - self.fdm1.param['fdm_ic_theta']),
            (self.fdm1.getProperty('attitudeDeg')[2] - self.fdm1.param['fdm_ic_phi']),
            (self.fdm2.getProperty('position')[0] - self.fdm2.param['fdm_ic_lat']) * 1e4,
            (self.fdm2.getProperty('position')[1] - self.fdm2.param['fdm_ic_long']) * 1e4,
            (self.fdm2.getProperty('position')[2] - self.fdm2.param['fdm_ic_h']) / 10,
            (self.fdm2.getProperty('attitudeDeg')[0] - self.fdm2.param['fdm_ic_psi']),
            (self.fdm2.getProperty('attitudeDeg')[1] - self.fdm2.param['fdm_ic_theta']),
            (self.fdm2.getProperty('attitudeDeg')[2] - self.fdm2.param['fdm_ic_phi']),
        ])
        
        return ob

