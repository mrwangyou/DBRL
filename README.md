<div align="center">
  <h1>DBRL: A Gym <b>D</b>ogfighting Simulation <b>B</b>enchmark for <b>R</b>einforcement <b>L</b>earning Research</h1>

  <!-- ## <div align="center">Quick Start</div> -->
  <a align="center" href="https://github.com/mrwangyou/DBRL" target="_blank"><img width="850" src="images/logo.png"></a>

  [Reference Manual](doc/QuickStart_en.md) | [简体中文使用手册](doc/QuickStart_zh.md)

</div>

## <div align="center">Introduction</div>

DBRL is an air combat simulation benchmark based on <a href="http://jsbsim.sourceforge.net/">JSBSim</a> and <a href="https://github.com/harfang3d/dogfight-sandbox-hg2">Dogfight 2</a>. 

We adopt JSBSim as flight dynamic model to build our dogfight environment, and Dogfight 2 to simulate the situation when the plane is confronted with a missile.

If you are new to air combat simulation or reinforcement learning algorithm, we advise you to learn [DBRL tutorial](tutorial.ipynb) first.

## <div align="center">Gym Parameters</div>

```python
# JSBSim
observation_space = Box(
    low=np.array([
        -360,  # Latitude
        -360,  # Longitude
        0,     # Height above sea level
        -360,  # Yaw
        -360,  # Pitch
        -360,  # Roll
    ] * 2),
    high=np.array([
        360,
        360,
        60000,
        360,
        360,
        360,
    ] * 2)
)

action_space = Box(
    low=np.array([
        -1,  # Aileron
        -1,  # Elevator
        -1,  # Rudder
        0,   # Throttle
        0,   # Flap
        0,   # Speed brake
        0,   # Spoiler
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


# Dogfight 2
if ego_plane_position:
    observation_infimum = np.append(observation_infimum, [-300, -300, -1])
    observation_supermum = np.append(observation_supermum, [300, 300, 200])
if ego_plane_attitude:
    observation_infimum = np.append(observation_infimum, [0, -360, -360])
    observation_supermum = np.append(observation_supermum, [360, 360, 360])
if oppo_plane_position:
    observation_infimum = np.append(observation_infimum, [-300, -300, -1])
    observation_supermum = np.append(observation_supermum, [300, 300, 200])
if oppo_plane_attitude:
    observation_infimum = np.append(observation_infimum, [0, -360, -360])
    observation_supermum = np.append(observation_supermum, [360, 360, 360])
if missile_position:
    observation_infimum = np.append(observation_infimum, [-300, -300, -1])
    observation_supermum = np.append(observation_supermum, [300, 300, 200])
if missile_attitude:
    observation_infimum = np.append(observation_infimum, [-315, -315, -315])
    observation_supermum = np.append(observation_supermum, [315, 315, 315])
if missile_relative_azimuth:
    observation_infimum = np.append(observation_infimum, [-1, -1, -1])
    observation_supermum = np.append(observation_supermum, [1, 1, 1])
observation_space = Box(
    low=observation_infimum,
    high=observation_supermum,
)

if pitch_enable:
    action_infimum = np.append(action_infimum, -1)
    action_supermum = np.append(action_supermum, 1)
if roll_enable:
    action_infimum = np.append(action_infimum, -1)
    action_supermum = np.append(action_supermum, 1)
if yaw_enable:
    action_infimum = np.append(action_infimum, -1)
    action_supermum = np.append(action_supermum, 1)
if flaps_enable:
    action_infimum = np.append(action_infimum, 0)
    action_supermum = np.append(action_supermum, 1)
if throttle_enable:
    action_infimum = np.append(action_infimum, 0)
    action_supermum = np.append(action_supermum, 1)
if flare_enable:
    action_infimum = np.append(action_infimum, 0)
    action_supermum = np.append(action_supermum, 1)
action_space = Box(
    low=action_infimum,
    high=action_supermum,
)

```


## <div align="center">Welcome PR</div>

If you find any mistakes while using, or have any suggestions and advices, please point it out in Github Issues. We're also looking forward to your contributions, such as your air combat reinforcement learning models, to this dogfight benchmark. <!-- If you are interesting in this project, feel free to contact `mrwangyou@stu.xjtu.edu.cn`. -->

