import time

import gym
import numpy as np
from stable_baselines3 import DDPG
from stable_baselines3.common.noise import (NormalActionNoise,
                                            OrnsteinUhlenbeckActionNoise)

env = gym.make("DBRLDogfight-v0", host='192.168.239.1')

n_actions = env.action_space.shape[-1]
action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))

print("Loading model")
model = DDPG(
    "MlpPolicy",
    env, 
    verbose=1,
    action_noise=action_noise
)

try:
    model.set_parameters("ddpg_pendulum")
except:
    pass
print("Training")
model.learn(total_timesteps=10000000, log_interval=1)
print("Done")
model.save("ddpg_pendulum")
# env = model.get_env()

# del model # remove to demonstrate saving and loading

# model = DDPG.load("ddpg_pendulum")

# print("line 24")


# obs = env.reset()
# while True:
#     action, _states = model.predict(obs)
#     obs, rewards, dones, info = env.step(action)
#     env.render()
#     if dones == True:
#         print("Done!")
#         time.sleep(2)
#         env.reset()
