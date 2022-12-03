import os
import time

import gym
import numpy as np
from stable_baselines3 import SAC
from stable_baselines3.common.noise import (NormalActionNoise,
                                            OrnsteinUhlenbeckActionNoise)



env = gym.make(
    "DBRLDogfight-v0",
    host='192.168.239.1',
    port='50888',
    rendering=True,
)

n_actions = env.action_space.shape[-1]
action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))

model = SAC(
    "MlpPolicy",
    env, 
    verbose=1,
    action_noise=action_noise,
)

path = r'./log/'
if not os.path.exists(path):
    os.mkdir(path)


model = SAC.load("./log/sac_df2")

win = 0
episode = 0


env.setProperty('plane', 0)  # TFX
env.setProperty('enemy', 3)  # 
env.setProperty('missile', 0)  # Mica
obs = env.reset()
times = 0
sett = 0
while True:
    if times == 1 and sett == 0:
        env.setProperty('plane', 0)  # TFX
        env.setProperty('enemy', 3)  # 
        env.setProperty('missile', 1)  # Meteor
        obs = env.reset()
        sett = 1
    if times == 2 and sett == 0:
        env.setProperty('plane', 1)  # F-16
        env.setProperty('enemy', 3)  # 
        env.setProperty('missile', 0)  # Mica
        obs = env.reset()
        sett = 1
    if times == 3 and sett == 0:
        env.setProperty('plane', 1)  # F-16
        env.setProperty('enemy', 3)  # 
        env.setProperty('missile', 1)  # Meteor
        obs = env.reset()
        sett = 1
    if times == 4 and sett == 0:
        env.setProperty('plane', 2)  # Eurofighter Typhoon
        env.setProperty('enemy', 1)  # 
        env.setProperty('missile', 0)  # AIM
        obs = env.reset()
        sett = 1
    if times == 5 and sett == 0:
        env.setProperty('plane', 3)  # Rafale
        env.setProperty('enemy', 1)  # 
        env.setProperty('missile', 2)  # Karaok
        obs = env.reset()
        sett = 1
    if times == 6:
        break


    action, _states = model.predict(obs)
    obs, rewards, dones, info = env.step(action)
    env.render()
    if dones == True:
        if rewards == 50:
            win += 1
        episode += 1
        print("Done! episode: {}\tacc: {}".format(episode, win / episode))
        time.sleep(2)
        env.reset()
        times = times + 1
        sett = 0
