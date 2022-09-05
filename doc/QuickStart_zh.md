<div align="center">
  <h1>DBRL: A Gym <b>D</b>ogfighting Simulation <b>B</b>enchmark for <b>R</b>einforcement <b>L</b>earning Research</h1>

</div>


## <div align="center">快速入门</div>

<details open>
<summary>准备工作</summary>

DBRL基于空气动力学软件<a href="http://jsbsim.sourceforge.net/">JSBSim</a>与<a href="https://github.com/harfang3d/dogfight-sandbox-hg2">Dogfight 2</a>搭建空战仿真平台。<u>如需使用JSBSim环境</u>，可以运行如下代码：

```bash
pip install jsbsim
```

Dogfight 2环境仅支持在Windows下运行，<u>如需使用Dogfight 2环境</u>，可以在Windows PowerShell中运行如下代码进行下载与配置：

```bash
wget -Uri https://github.com/harfang3d/dogfight-sandbox-hg2/releases/download/1.0.2/dogfight-sandbox-hg2-win64.7z -OutFile "df2.7z"

7z x -odf2/ df2.7z

mv ./df2/dogfight-sandbox-hg2/ {DBRL}/src/environments/dogfightEnv/dogfight_sandbox_hg2/
```
其中`{DBRL}`需要替换为本项目所在的路径，配置7z压缩软件的命令行功能可见[链接](https://www.cnblogs.com/conorblog/p/14543286.html)。


JSBSim的可视化功能基于FlightGear实现，<u>如需在使用JSBSim的过程中将飞行过程可视化显示</u>，请在<a href="https://www.flightgear.org/">FlightGear官网</a>中安装FlightGear软件。如需对缠斗中的两架飞机分别进行可视化，请在`{JSBSim}/data_output/`下复制两份`flightgear.xml`文件，分别命名为`flightgear{1/2}.xml`，并将`flightgear2.xml`第18行中的`5550`修改为`5551`。其中`{JSBSim}`需要替换为Python JSBSim包所在的路径，如需查看`JSBSim`源文件地址，可以运行如下代码：

```python
import jsbsim

print(jsbsim.get_default_root_dir())
```

在需要可视化的接战仿真中，请在FlightGear启动时使用如下参数

```bash
--fdm=null --native-fdm=socket,in,60,,5550,udp
```

在同时对两架飞机可视化的仿真中，请分别使用
```bash
--fdm=null --native-fdm=socket,in,60,,5550,udp --multiplay=out,10.127.0.0.1,5000 --multiplay=in,10.127.0.0.1,5001 --callsign=Test1
```
与
```bash
--fdm=null --native-fdm=socket,in,60,,5550,udp --multiplay=out,10.127.0.0.1,5001 --multiplay=in,10.127.0.0.1,5000 --callsign=Test2
```
参数

</details>


<details open>
<summary>使用方法</summary>

DBRL提供了<a href="https://github.com/openai/gym">OpenAI Gym</a>格式的强化学习环境，可以通过Gym库进行调用。如需使用`gym.make`调用环境，请在`Gym/envs`下复制`src/environments/*Env/`文件夹，并在`Gym/envs/__init__.py`中添加如下代码：

```python
register(
    id="DBRL-v0",
    entry_point="gym.envs.{jsbsim/dogfight}Env:{JSBSim/Dogfight}Env",
    max_episode_steps=10000,
    reward_threshold=100.0,
)
```

如需查看`gym`源文件地址，可以运行如下代码：

```bash
pip show gym
```

调用环境时可以采用如下代码：

```python
import gym

env = gym.make('DBRL-v0')
```

如果不从`Gym`库直接调用环境，也可以直接使用环境类的实例，可以采用如下代码：

```python
from DBRL.src.environments import jsbsimEnv as Env

env = Env.Env()
```

</details>


<details open>
<summary>环境特征</summary>

DBRL提供了一个基于强化学习框架Gym的智能空战仿真环境，它的动作空间为：

```python
gym.spaces.Box(
    low=np.array([[-1, -1, -1, 0]] * 2),
    high=np.array([[1, 1, 1, 1]] * 2)
)
```
其中四个维度分别表示对副翼（Aileron）、升降舵（Elevator）、方向舵（Rudder）、油门（Throttle）的操控。

状态空间为：

```python
gym.spaces.Box(
    low=np.array([[-360, -360, 0, -360, -360, -360]] * 2),
    high=np.array([[360, 360, 60000, 360, 360, 360]] * 2)
)
```
其中六个维度分别表示飞机的纬度（Latitude）、经度（Longitude）、海拔（Height above sea level）、偏航角（Yaw）、俯仰角（Pitch）、翻滚角（Roll），单位为度（Degree）与英尺（Feet）。

动作空间与状态空间的外层两个维度分别表示一场接战中的两架飞机。

作为一场接战环境的初始化，`class JSBSimEnv()`接受两个`class JSBSimFdm()`作为输入变量：

```python
class JsbsimEnv(Env):

    def __init__(
        self,
        fdm1=Fdm(fdm_id=1),
        fdm2=Fdm(fdm_id=2),
    ) -> None:
```

JSBSimFdm类储存在`DBRL/src/environments/jsbsimEnv/jsbsimFdm.py`中，它的构造函数接受如下参数作为输入：

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
其中：`fdm_id`表示接战中飞机的编号，需要分别设置为`1`与`2`；`fdm_aircraft`表示接战中飞机的型号；`fdm_ic_*`表示飞机的初始化特征；`fdm_ic_v`表示飞机的初始校正空速，单位为节（knot）；`fdm_ic_lat`表示飞机的初始纬度；`fdm_ic_long`表示飞机的初始经度；`fdm_ic_h`表示飞机的初始海拔高度，单位为英尺；`fdm_ic_psi`、`fdm_ic_theta`、`fdm_ic_phi`分别表示飞机的偏航角，俯仰角与翻滚角，值得注意的是，模型将按照此处列出的三个欧拉角的先后顺序进行飞机姿态的计算；`fdm_hp`表示飞机的初始血量；`fdm_fgfs`表示飞机是否需要在FlightGear上进行可视化；`flight_mode`表示飞行模式（预留变量，暂无效果）。

</details>


## <div align="center">未来工作</div>

1. 为不同强化学习模型提供不同的动作空间，包括现有的对飞机操纵面的控制，以及可能的以航向角，目标位姿等作为动作空间；

2. 提供不同的状态空间，除了现有的描述飞机六个自由度的值之外，额外增加速度、加速度、剩余油量等飞机状态供不同需要的强化学习模型选择；

3. 实现不同强化学习基本模型，复现近年来国内外顶会顶刊上的智能空战算法；


## <div align="center">欢迎PR</div>

如果您在使用过程中发现了任何错误，或者有任何需求与改进的建议，欢迎在Github Issues中提出，或者直接联系我的邮箱`mrwangyou [at] stu [dot] xjtu [dot] edu [dot] cn`，谢谢！
