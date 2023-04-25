import argparse
import os
import time

import gym
import numpy as np
from stable_baselines3 import DDPG, TD3, SAC
from stable_baselines3.common.noise import (NormalActionNoise,
                                            OrnsteinUhlenbeckActionNoise)


def parse_args():
    parser = argparse.ArgumentParser(description='TBD')
    parser.add_argument('--host', default='10.184.0.0', help='specifies Harfang host id')
    parser.add_argument('--port', default='50888', help='specifies Harfang port id')
    parser.add_argument('--planeSlot', default=1, help='specifies the ego plane')
    parser.add_argument('--enemySlot', default=3, help='specifies the enemy plane')
    parser.add_argument('--missileSlot', default=1, help='specifies the missile')
    parser.add_argument('--playSpeed', default=0, help='specifies to run in real world time')
    parser.add_argument('--train', action='store_true', help='specifies the running mode of DBRL')
    parser.add_argument('--test', action='store_true', help='specifies the running mode of DBRL')
    parser.add_argument('--timesteps', type=int, default=10000000, help='specifies the training timesteps. Only works when --train is specified')
    parser.add_argument('--model', default='SAC', help='specifies the DRL model used in algorithm training. Only works when --train or --test is specified')
    # parser.add_argument('--modelPath', default=None, help='specifies the pre-trained model. Only works when --train is specified')
    parser.add_argument('--record', action='store_true', help='specifies whether to record the evaluating result of DBRL. Only works when --test is specified')
    args = parser.parse_args()
    return args

args = parse_args()

env = gym.make(
    "DBRLDogfight-v0",
    host=args.host,
    port=args.port,
    plane_slot=args.planeSlot,
    enemy_slot=args.enemySlot,
    missile_slot=args.missileSlot,
    rendering=True if args.playSpeed else False
)

n_actions = env.action_space.shape[-1]
action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))

method = {
    'DDPG': 'ddpg',
    'TD3': 'td3',
    'SAC': 'sac',
}

assert args.model in method, 'Model {} not defined'.format(args.model)

msg = "evade_{}_{}".format(args.model, args.timesteps)

model = eval(args.model)(
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
    #     model = eval(args.model).load("./log/{}_evade".format(method[args.model]), env)
    #     model.load_replay_buffer("./log/{}_evade_rb".format(method[args.model]))
    # except:
    #     pass
    model.learn(total_timesteps=args.timesteps, log_interval=1)
    model.save("./log/{}".format(msg))
    model.save_replay_buffer("./log/{}_rb".format(msg))

if args.test:
    model = eval(args.model).load("./log/{}".format(msg))

    win = 0
    episode = 0

    if args.record:
        try:
            f = open('./log/{}_record.txt'.format(msg), 'r')
            for line in f:
                pass
            win = int(line.split()[0])
            episode = int(line.split()[2])
            f.close()
        except:
            pass

    obs = env.reset()
    while True:
        action, _states = model.predict(obs)
        obs, rewards, dones, info = env.step(action)
        env.render()
        if dones == True:
            if rewards == 50:
                win += 1
            episode += 1
            if args.record:
                f = open('./log/{}_record.txt'.format(msg), 'a')
                f.write("{} / {}\n".format(win, episode))
                f.close()
            print("Episode: {}\tacc: {}".format(episode, win / episode))
            time.sleep(2)
            env.reset()
