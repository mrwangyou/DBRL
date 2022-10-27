import argparse
import os
import time

import gym
import numpy as np
from stable_baselines3 import SAC
from stable_baselines3.common.noise import (NormalActionNoise,
                                            OrnsteinUhlenbeckActionNoise)


def parse_args():
    parser = argparse.ArgumentParser(description='TBD')
    parser.add_argument('--host', default='10.184.0.0', metavar='str', help='specifies Harfang host id')
    parser.add_argument('--port', default='50888', metavar='str', help='specifies Harfang port id')
    # parser.add_argument('--modelPath', default='/data/wnn_data/bestModel/', metavar='str', help='specifies the pre-trained model')
    parser.add_argument('--playSpeed', default=0, metavar='double', help='specifies to run in real world time')
    parser.add_argument('--train', action='store_true', help='specifies the running mode of DBRL')
    parser.add_argument('--test', action='store_true', help='specifies the running mode of DBRL')
    parser.add_argument('--timesteps', default=10000000, metavar='double', help='specifies the training timesteps. Only works when --train is specified')
    args = parser.parse_args()
    return args

args = parse_args()

env = gym.make(
    "DBRLDogfight-v0", 
    host=args.host,
    port=args.port,
    render=True if args.playSpeed else False
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

if args.train:
    # try:
    #     model.set_parameters("./log/sac_df2")
    # except:
    #     pass
    model.learn(total_timesteps=10000000, log_interval=1)
    model.save("./log/sac_df2")

if args.test:
    model = SAC.load("./log/sac_df2")

    f = open('./log/sac_df_record.txt', 'r')
    for line in f:
        pass
    win = int(line.split()[0])
    episode = int(line.split()[2])

    f.close()

    obs = env.reset()
    while True:
        action, _states = model.predict(obs)
        obs, rewards, dones, info = env.step(action)
        env.render()
        if dones == True:
            f = open('./log/sac_df_record.txt', 'a')
            if rewards == 50:
                win += 1
            episode += 1
            f.write("{} / {}\n".format(win, episode))
            f.close()
            print("Done! episode: {}\tacc: {}".format(episode, win / episode))
            time.sleep(2)
            env.reset()
