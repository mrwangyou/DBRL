import numpy as np
from gym.spaces import Box

def observation_space(
    ego_plane_position=True,
    ego_plane_attitude=True,
    oppo_plane_position=True,
    oppo_plane_attitude=True,
    missile_position=True,
    missile_attitude=True,
    missile_relative_attitude=False,
):
    observation_infimum = np.array([])
    observation_supermum = np.array([])
    
    if ego_plane_position:  # 我方飞机位置
        observation_infimum = np.append(observation_infimum, -1)
        observation_supermum = np.append(observation_supermum, 1)
    if ego_plane_attitude:  # 我方飞机姿态
        observation_infimum = np.append(observation_infimum, -1)
        observation_supermum = np.append(observation_supermum, 1)
    if oppo_plane_position:  # 敌方飞机位置
        observation_infimum = np.append(observation_infimum, -1)
        observation_supermum = np.append(observation_supermum, 1)
    if oppo_plane_attitude:  # 敌方飞机姿态
        observation_infimum = np.append(observation_infimum, 0)
        observation_supermum = np.append(observation_supermum, 1)
    if missile_position:  # 导弹位置
        observation_infimum = np.append(observation_infimum, 0)
        observation_supermum = np.append(observation_supermum, 1)
    if missile_attitude:  # 导弹姿态
        observation_infimum = np.append(observation_infimum, 0)
        observation_supermum = np.append(observation_supermum, 1)
    if missile_relative_attitude:  # 导弹相对位置
        observation_infimum = np.append(observation_infimum, 0)
        observation_supermum = np.append(observation_supermum, 1)

    observation = Box(
        low=observation_infimum,
        high=observation_supermum,
        dtype=float
    )

    return observation
