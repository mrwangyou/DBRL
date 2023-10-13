import numpy as np
from gym.spaces import Box

def action_space(
    pitch_enable=True,
    roll_enable=True,
    yaw_enable=True,
    flaps_enable=True,
    throttle_enable=False,
    flare_enable=False,
):
    action_infimum = np.array([])
    action_supermum = np.array([])
    action_type = []
    
    if pitch_enable:  # 俯仰角
        action_infimum = np.append(action_infimum, -1)
        action_supermum = np.append(action_supermum, 1)
        action_type = action_type + ['_df.set_plane_pitch']
    if roll_enable:  # 滚转角
        action_infimum = np.append(action_infimum, -1)
        action_supermum = np.append(action_supermum, 1)
        action_type = action_type + ['_df.set_plabe_roll']
    if yaw_enable:  # 偏航角
        action_infimum = np.append(action_infimum, -1)
        action_supermum = np.append(action_supermum, 1)
        action_type = action_type + ['_df.set_plane_yaw']
    if flaps_enable:  # 襟翼
        action_infimum = np.append(action_infimum, 0)
        action_supermum = np.append(action_supermum, 1)
        action_type = action_type + ['_df.set_plane_flaps']
    if throttle_enable:  # 油门
        action_infimum = np.append(action_infimum, 0)
        action_supermum = np.append(action_supermum, 1)
        action_type = action_type + ['_df.set_plane_thrust']
    if flare_enable:  # 干扰弹
        action_infimum = np.append(action_infimum, 0)
        action_supermum = np.append(action_supermum, 1)
        action_type = action_type + ['Flare']

    action = Box(
        low=action_infimum,
        high=action_supermum,
        dtype=float
    )

    return action, action_type
