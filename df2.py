import argparse
import os
import time

import gym
import numpy as np
from stable_baselines3 import DDPG, TD3, SAC
from stable_baselines3.common.noise import (NormalActionNoise,
                                            OrnsteinUhlenbeckActionNoise)


def parse_args():
    parser = argparse.ArgumentParser(description='DBRL is an air combat simulation benchmark based on JSBSim and Dogfight 2. For more information, please visit https://github.com/mrwangyou/DBRL')
    parser.add_argument('--task', choices=['evade', 'dogfight'], required=True, help='specifies the simulation task')
    parser.add_argument('--host', default='10.184.0.0', help='specifies Harfang host id')
    parser.add_argument('--port', default='50888', help='specifies Harfang port id')
    parser.add_argument('--plane-slot', type=int, default=1, help='specifies the ego plane')
    parser.add_argument('--enemy-slot', type=int, default=3, help='specifies the enemy plane')
    parser.add_argument('--missile-slot', type=int, default=1, help='specifies the missile to escape from')
    parser.add_argument('--realtime', action='store_true', help='specifies to run in real world time while training. Only works when --trian is specified')
    parser.add_argument('--train', action='store_true', help='specifies the running mode of DBRL')
    parser.add_argument('--test', action='store_true', help='specifies the running mode of DBRL')
    parser.add_argument('--timesteps', type=int, default=10000000, help='specifies the training timesteps')
    parser.add_argument('--model', choices=['DDPG', 'TD3', 'SAC'], default='SAC', help='specifies the DRL model used in algorithm training')
    parser.add_argument('--initial-state', choices=['air', 'carrier'], default='carrier', help='')
    # parser.add_argument('--modelPath', default=None, help='specifies the pre-trained model. Only works when --train is specified')
    parser.add_argument('--record-result', action='store_true', help='specifies to record the evaluating result of DBRL. Only works when --test is specified')
    parser.add_argument('--record-status', type=int, default=0, help='specifies the recording period for aircraft status recording during test flights. Only works when --test is specified')
    parser.add_argument('--throttle-enable', action='store_true', help='specifies whether to enable the throttle control')
    parser.add_argument('--flare-enable', action='store_true', help='specifies whether to enable the decoy flare')
    parser.add_argument('--ego-pose-enable', action='store_true', help='')
    parser.add_argument('--oppo-pose-enable', action='store_true', help='')
    parser.add_argument('--missile-pose-enable', action='store_true', help='')
    parser.add_argument('--missile-relative-azimuth-enable', action='store_true', help='')
    args = parser.parse_args()
    return args

args = parse_args()

msg = "{}_{}_{}".format(args.task, args.model, args.timesteps)
print(msg)

env = gym.make(
    "DBRLDogfight-v0",
    host=args.host,
    port=args.port,
    task=args.task,
    plane_slot=args.plane_slot,
    enemy_slot=args.enemy_slot,
    missile_slot=args.missile_slot,
    # rendering=args.realtime,
    rendering=False,
    record_status=args.record_status,
    initial_state=args.initial_state,
    throttle_enable=args.throttle_enable,
    flare_enable=args.flare_enable,
    ego_pose_enable=True,
    oppo_pose_enable=False,
    missile_pose_enable=True,
    missile_relative_azimuth_enable=True,
    msg=msg
)

n_actions = env.action_space.shape[-1]
action_noise = NormalActionNoise(
    mean=np.zeros(n_actions), 
    sigma=0.1 * np.ones(n_actions)
)

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

    if args.record_result and os.path.exists('./log/{}_record_result.txt'.format(msg)):
        f = open('./log/{}_record_result.txt'.format(msg), 'r')
        for line in f:
            pass
        win = int(line.split()[0])
        episodes = int(line.split()[2])
        f.close()
    else:
        win = 0
        episodes = 0

    obs = env.reset()
    while True:
        action, _states = model.predict(obs)
        obs, rewards, dones, info = env.step(action)
        env.render()
        if dones == True:
            if rewards == 50:
                win += 1
            episodes += 1
            if args.record_result:
                f = open('./log/{}_record_result.txt'.format(msg), 'a')
                f.write("{} / {}\n".format(win, episodes))
                f.close()
            print("Episodes: {}\tacc: {}".format(episodes, win / episodes))
            time.sleep(2)
            env.reset()
