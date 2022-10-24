<div align="center">
  <h1>DBRL: A Gym <b>D</b>ogfighting Simulation <b>B</b>enchmark for <b>R</b>einforcement <b>L</b>earning Research</h1>

</div>


## <div align="center">Quick Start</div>

<details open>
<summary>Install</summary>

DBRL is an air combat simulation benchmark based on <a href="http://jsbsim.sourceforge.net/">JSBSim</a> and <a href="https://github.com/harfang3d/dogfight-sandbox-hg2">Dogfight 2</a>. 

If you want to run with JSBSim dogfight environment, please run the code below to install JSBSim.

```bash
pip install jsbsim
```

Dogfight 2 environment could only run in Windows OS. If you want to run reinforcement learning methods with Dogfight 2 missile-avoiding environment, please run the code below in Windows PowerShell.

```bash
wget -Uri https://github.com/harfang3d/dogfight-sandbox-hg2/releases/download/1.0.2/dogfight-sandbox-hg2-win64.7z -OutFile "df2.7z"

7z x -odf2/ df2.7z

mv ./df2/dogfight-sandbox-hg2/ {DBRL}/src/environments/dogfightEnv/dogfight_sandbox_hg2/

del df2/, df2.7z
```
Replace `{DBRL}` with the path of this project. 

> If you want to build the Gym environment with `Gym.make`, you need to replace `import socket_lib` with `from gym.envs.dogfightEnv.dogfight_sandbox_hg2.network_client_example import socket_lib` in `{DBRL}/src/environments/dogfightEnv/dogfight_sandbox_hg/network_client_example/dogfight_client.py`. We are trying to simplify this step.

If you want to visualize the aircraft model in FlightGear while running the FDM with the JSBSim executable, please download FlightGear following the instructions on the <a href="https://www.flightgear.org/">FlightGear website</a>. If you need to visualize the two aircrafts in an engagement simultaneously, please copy two `{JSBSim}/data_output/flightgear.xml`, name them `flightgear{1/2}.xml`, and replace `5550` in line 18 of `flightgear2.xml` to `5551`. `{JSBSim}` represents the path of Python JSBSim. You could run the code below to get the default JSBSim path.

```python
import jsbsim

print(jsbsim.get_default_root_dir())
```

Run FlightGear with the attributes below.

```bash
--fdm=null --native-fdm=socket,in,60,,5550,udp
```

If you want to visualize two aircraft at the same time, use the attributes below in two FlightGear separately.

```bash
--fdm=null --native-fdm=socket,in,60,,5550,udp --multiplay=out,10,127.0.0.1,5000 --multiplay=in,10,127.0.0.1,5001 --callsign=Test1
```

```bash
--fdm=null --native-fdm=socket,in,60,,5550,udp --multiplay=out,10,127.0.0.1,5001 --multiplay=in,10,127.0.0.1,5000 --callsign=Test2
```

If you want to run the reinforcement learning methods in `src/models/*.py`, please put this project in `JSBSim/`.

```
JSBSim/
└── DBRL/
    ├── doc/
    ├── log/
    ├── src/
    ├── test/
    └── ...
```

</details>


<details open>
<summary>Adopting</summary>

DBRL builds the reinforcement learning environment in <a href="https://github.com/openai/gym">OpenAI Gym</a> form. You could use `gym.make` to build the environment with the following register. 

```python
register(
    id="DBRL{Jsbsim/Dogfight}-v0",
    entry_point="gym.envs.{jsbsim/dogfight}Env:{Jsbsim/Dogfight}Env",
)
```

Use the code below to get the path of Gym.

```bash
pip show gym
```

After that, you could use DBRL environments with Gym in the way like following.

```python
import gym

env = gym.make('DBRL{Jsbsim/Dogfight}-v0')
```

<!-- You could also use an instance of the environment class with out register.

```python
from DBRL.src.environments import jsbsimEnv as Env

env = Env.Env()
``` -->

</details>

You can download pretrained models in [link](https://1drv.ms/u/s!AjYxuXuKCgSEvjodBexf7MkaToL9?e=nREa7B).

<details open>
<summary>Properties</summary>

The action space of DBRL-JSBSim is:

```python
gym.spaces.Box(
    low=np.array([-1, -1, -1, 0]),
    high=np.array([1, 1, 1, 1])
)
```
which represents the control of aileron, elevator, rudder and throttle.

The observation space is:

```python
gym.spaces.Box(
    low=np.array([-360, -360, 0, -360, -360, -360] * 2),
    high=np.array([360, 360, 60000, 360, 360, 360] * 2)
)
```
which represents the latitude(degree), longitude(degree), height above sea level(feet), yaw(degree), pitch(degree) and roll(degree) of the plane. 

Two dimensions of the action space and the observation space represents the two aircraft in an engagement.

JSBSimEnv class's constructor function takes two JSBSimFdm class as input.

```python
class JsbsimEnv(Env):

    def __init__(
        self,
        fdm1=Fdm(fdm_id=1),
        fdm2=Fdm(fdm_id=2),
    ) -> None:
```

You could import JSBSimFdm class from `DBRL/src/environments/jsbsimEnv/jsbsimFdm.py`. It takes the arguments below as input.

```python
class JsbsimFdm():

    def __init__(
        self,
        fdm_id       = 1,
        fdm_aircraft = 'f16',
        fdm_ic_v     = 500,
        fdm_ic_lat   = 0,
        fdm_ic_long  = 0,
        fdm_ic_h     = 20005.5,
        fdm_ic_psi   = 0,
        fdm_ic_theta = 0,
        fdm_ic_phi   = 0,
        fdm_hp       = 1,
        fdm_fgfs     = False,
        flight_mode  = 1
    ) -> None:
```
`fdm_id` represents the number of aircraft, which needs to be set to `1` or `2`. `fdm_aircraft` represents the aircraft used in the engagement. `fdm_ic_v` represents the initial calibrated velocity. `fdm_ic_lat` represents the initial latitude. `fdm_ic_long` represents the initial longitude. `fdm_ic_h` represents the initial height. `fdm_ic_psi`, `fdm_ic_theta` and `fdm_ic_phi` represents the initial yaw, pitch and roll angle of the aircraft. `fdm_hp` represents the initial health point. `fdm_fgfs` denotes whether the aircraft will be visualized in FlightGear.

The reward function of DBRL are set to be the damage ego plane takes to the opposite minus the opposite takes to the ego. Various reward function could be customized by `getProperty` function of class JSBSimFdm.

```python
def getProperty(
        self,
        prop,
    ) -> list:
```

---

DBRL-Dogfight simulate the situation when the plane is confronted with a missile. The agent needs to take actions to avoid the attack of the missile.

The action space of DBRL-Dogfight is:

```python
gym.spaces.Box(
    low=np.array([0, -1, -1, -1]),
    high=np.array([1, 1, 1, 1])
)
```
which represents the control of flaps, pitch, roll and yaw.

The observation space is:

```python
gym.spaces.Box(
    low=np.array([-300, -300, -1, 0, -360, -360, -300, -300, -1, -315, -315, -315]),
    high=np.array([300, 300, 200, 360, 360, 360, 300, 300, 200, 315, 315, 315])
)
```
which represents the X-coordinate(÷100), Y-coordinate(÷100), Z-coordinate(÷50), heading, pitch(×4) and roll(×4) of the plane, and the X-coordinate(÷100), Y-coordinate(÷100), Z-coordinate(÷50), heading(×100), pitch(×100) and roll(×100) of the missile.

DogfightEnv needs to be connected to Dogfight 2 while simulating. Its constructor function takes host and port as input.

```python
class DogfightEnv(Env):

    def __init__(
        self,
        host='10.184.0.0',
        port='50888'
    ) -> None:
```

You need to start the Dogfight 2 at first and choose the Network mode mission. The host and port IP are in the upper left corner of the screen.


</details>


## <div align="center">Future works</div>

1. Offer various action spaces, observation spaces and reward functions to support different RL methods.

2. Reproduce air combat baseline methods from international conference or journal.

3. Output flight data in various forms to support visualization in different software like Tacview.


## <div align="center">Welcome PR</div>

If you find any mistakes while using, or have any suggestions and advices, please point it out in Github Issues. We're looking forward to your contributions, such as your air combat reinforcement learning models, to this dogfight benchmark. If you are interesting in this project, feel free to contact `mrwangyou@stu.xjtu.edu.cn`.
