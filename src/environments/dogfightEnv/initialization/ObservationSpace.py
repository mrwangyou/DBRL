import numpy as np
from gym.spaces import Box

def observation_space(
    ego_plane_position=True,
    ego_plane_attitude=True,
    oppo_plane_position=False,
    oppo_plane_attitude=False,
    missile_position=True,
    missile_attitude=True,
    missile_relative_azimuth=False,
):
    observation_infimum = np.array([])
    observation_supermum = np.array([])
    observation_type = []
    
    if ego_plane_position:  # 我方飞机位置
        observation_infimum = np.append(observation_infimum, [-300, -300, -1])
        observation_supermum = np.append(observation_supermum, [300, 300, 200])
    if ego_plane_attitude:  # 我方飞机姿态
        observation_infimum = np.append(observation_infimum, [0, -360, -360])
        observation_supermum = np.append(observation_supermum, [360, 360, 360])
    if oppo_plane_position:  # 敌方飞机位置
        observation_infimum = np.append(observation_infimum, [-300, -300, -1])
        observation_supermum = np.append(observation_supermum, [300, 300, 200])
    if oppo_plane_attitude:  # 敌方飞机姿态
        observation_infimum = np.append(observation_infimum, [0, -360, -360])
        observation_supermum = np.append(observation_supermum, [360, 360, 360])
    if missile_position:  # 导弹位置
        observation_infimum = np.append(observation_infimum, [-300, -300, -1])
        observation_supermum = np.append(observation_supermum, [300, 300, 200])
    if missile_attitude:  # 导弹姿态
        observation_infimum = np.append(observation_infimum, [-315, -315, -315])
        observation_supermum = np.append(observation_supermum, [315, 315, 315])
    if missile_relative_azimuth:  # 导弹相对方位角
        observation_infimum = np.append(observation_infimum, [-1, -1, -1])
        observation_supermum = np.append(observation_supermum, [1, 1, 1])

    observation = Box(
        low=observation_infimum,
        high=observation_supermum,
        dtype=np.float64
    )

    return observation
