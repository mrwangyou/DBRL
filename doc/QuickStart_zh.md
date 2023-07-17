<div align="center">
  <h1>DBRL: A Gym <b>D</b>ogfighting Simulation <b>B</b>enchmark for <b>R</b>einforcement <b>L</b>earning Research</h1>

</div>


## <div align="center">快速入门</div>

<details open>
<summary>准备工作</summary>

DBRL基于空气动力学软件<a href="http://jsbsim.sourceforge.net/">JSBSim</a>与<a href="https://github.com/harfang3d/dogfight-sandbox-hg2">Dogfight 2</a>搭建空战仿真平台。

<u>如需使用基于JSBSim搭建的视距内缠斗仿真环境</u>，可以运行如下代码安装Python-JSBSim：

```bash
pip install jsbsim
```

安装的JSBSim可能不包括[JSBSim Github页面](https://github.com/JSBSim-Team/jsbsim)中的全部文件，建议您将如上链接中的文件拷贝进`JSBSim`源文件地址中。

这里只提供Dogfight 2环境在Windows系统下运行的示例，如果需要在Linux系统中运行Harfang3D Dog-Fight，可以下载Linux版Harfang。<u>如需使用基于Dogfight 2搭建的导弹躲避仿真环境</u>，可以在Windows PowerShell中运行如下代码进行软件下载与配置：

```bash
wget -Uri https://github.com/harfang3d/dogfight-sandbox-hg2/releases/download/1.0.2/dogfight-sandbox-hg2-win64.7z -OutFile "df2.7z"

7z x -odf2/ df2.7z

mv ./df2/dogfight-sandbox-hg2/ {DBRL}/src/environments/dogfightEnv/dogfight_sandbox_hg2/

del df2/, df2.7z
```
其中，`{DBRL}`需要替换为本项目所在的路径，在命令行中使用7z指令需要将您的7zip安装位置添加至path系统变量，您同样可以手动进行文件下载，解压缩，重命名与移动操作。


> 如果需要使用`Gym.make`指令搭建环境，需要将生成的`{DBRL}/src/environments/dogfightEnv/dogfight_sandbox_hg/network_client_example/dogfight_client.py`中的`import socket_lib`修改为`from gym.envs.dogfightEnv.dogfight_sandbox_hg2.network_client_example import socket_lib`。我们正在尝试寻找简化这步操作的方法。

JSBSim的可视化功能基于FlightGear实现，<u>如需在使用JSBSim的过程中将空战的飞行过程可视化显示</u>，请在<a href="https://www.flightgear.org/">FlightGear官网</a>中安装FlightGear软件。如需对缠斗中的两架飞机分别进行可视化，请在`{JSBSim}/data_output/`下复制两份`flightgear.xml`文件，分别命名为`flightgear{1/2}.xml`，并将`flightgear2.xml`第18行中的`5550`修改为`5551`。上述操作中的`{JSBSim}`为Python JSBSim库所在的路径；如需查看`JSBSim`源文件地址，可以运行如下代码：

```python
import jsbsim

print(jsbsim.get_default_root_dir())
```

在运行需要可视化的接战仿真中，请在FlightGear启动时使用如下参数：

```bash
--fdm=null --native-fdm=socket,in,60,,5550,udp
```
> 如果您的系统语言非英语，且出现了FlightGear在操纵键盘后卡住的情况，您可以尝试使用其他输入法来解决这个问题。详情可见[论坛](https://forum.flightgear.org/viewtopic.php?f=22&t=40704)。

在运行同时对两架飞机可视化的仿真中，请分别使用
```bash
--fdm=null --native-fdm=socket,in,60,,5550,udp --multiplay=out,10,127.0.0.1,5000 --multiplay=in,10,127.0.0.1,5001 --callsign=Test1
```
与
```bash
--fdm=null --native-fdm=socket,in,60,,5550,udp --multiplay=out,10,127.0.0.1,5001 --multiplay=in,10,127.0.0.1,5000 --callsign=Test2
```
参数。

如需使用本项目中的强化学习模型`src/models/*.py`，请将本项目文件夹置于JSBSim目录下，即：

```
JSBSim/
├── DBRL/
│   ├── doc/
│   ├── log/
│   ├── src/
│   ├── test/
│   └── ...
├── aircraft/
├── scripts/
└── ...
```

</details>


<details open>
<summary>使用方法</summary>

DBRL提供了<a href="https://github.com/openai/gym">OpenAI Gym</a>格式的强化学习环境，可以通过Gym库进行调用。如需使用`gym.make`调用环境，请在`Gym/envs`下复制`src/environments/*Env/`文件夹，并在`Gym/envs/__init__.py`中添加如下代码：

```python
register(
    id="DBRL{Jsbsim/Dogfight}-v0",
    entry_point="gym.envs.{jsbsim/dogfight}Env:{Jsbsim/Dogfight}Env",
)
```

如需查看`gym`源文件地址，可以运行如下代码：

```bash
pip show gym
```

调用环境时可以采用如下代码：

```python
import gym

env = gym.make('DBRL{Jsbsim/Dogfight}-v0')
```

<!-- 如果不从`Gym`库直接调用环境，也可以直接使用环境类的实例，可以采用如下代码：

```python
from DBRL.src.environments import jsbsimEnv as Env

env = Env.Env()
``` -->

<!-- > 您可以在[OneDrive链接]()中获取已经完成训练的模型。 -->

</details>


<details open>
<summary>环境特征</summary>

DBRL-JSBSim提供了一个基于强化学习框架Gym的智能空战仿真环境，它的动作空间为：

```python
gym.spaces.Box(
    low=np.array([-1, -1, -1, 0, 0, 0, 0]),
    high=np.array([1, 1, 1, 1, 1, 1, 1])
)
```
其中六个维度分别表示对副翼（Aileron）、升降舵（Elevator）、方向舵（Rudder）、油门（Throttle）、襟翼（Flap）、减速板（Speed brake）、扰流片（Spoiler）的操控。

状态空间为：

```python
gym.spaces.Box(
    low=np.array([-360, -360, 0, -360, -360, -360] * 2),
    high=np.array([360, 360, 60000, 360, 360, 360] * 2)
)
```
其中六个维度分别表示飞机的纬度（Latitude）、经度（Longitude）、海拔（Height above sea level）、偏航角（Yaw）、俯仰角（Pitch）、滚转角（Roll），单位为度（Degree）与英尺（Feet）。

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
其中：`fdm_id`表示接战中飞机的编号，需要分别设置为`1`与`2`；`fdm_aircraft`表示接战中飞机的型号；`fdm_ic_*`表示飞机的初始化特征；`fdm_ic_v`表示飞机的初始校正空速，单位为节（knot）；`fdm_ic_lat`表示飞机的初始纬度；`fdm_ic_long`表示飞机的初始经度；`fdm_ic_h`表示飞机的初始海拔高度，单位为英尺；`fdm_ic_psi`、`fdm_ic_theta`、`fdm_ic_phi`分别表示飞机的偏航角，俯仰角与滚转角；`fdm_hp`表示飞机的初始血量；`fdm_fgfs`表示飞机是否需要在FlightGear上进行可视化；`flight_mode`表示飞行模式（暂无效果）。

> JSBSim将按照此处列出的三个欧拉角的先后顺序进行飞机姿态的计算，即，先设定飞机的偏航角，再在此基础上，依照参数设定飞机的俯仰角与滚转角。

DBRL提供的回报函数为当前帧对敌机造成的血量损失大小与敌机对自身造成伤害的差值，但也同样可以用JSBSimFdm类的`getProperty()`函数获得一些其他特征作为不同强化学习模型的回报函数。

```python
def getProperty(
        self,
        prop,
    ) -> list:
```

---

DBRL-Dogfight提供了一个基于强化学习框架Gym的战斗机机动躲避智能决策仿真环境，作为环境的初始状态，智能体需要操控的飞机被另一架飞机发射的“流星”空空导弹锁定，智能体需要采取机动操纵躲避导弹的威胁。

Dogfight 2环境的动作空间为：

```python
gym.spaces.Box(
    low=np.array([0, -1, -1, -1]),
    high=np.array([1, 1, 1, 1])
)
```
其中四个维度分别表示对襟翼（Flaps）、俯仰角（Pitch）、滚转角（Roll）、偏航角（Yaw）的操控。

状态空间为：

```python
gym.spaces.Box(
    low=np.array([-300, -300, -1, 0, -360, -360, -300, -300, -1, -315, -315, -315]),
    high=np.array([300, 300, 200, 360, 360, 360, 300, 300, 200, 315, 315, 315])
)
```
其中前六个维度分别表示飞机的X坐标（÷100）、Y坐标（÷100）、Z坐标（÷50）、偏航角、俯仰角（×4）、滚转角（×4），后六个维度分别表示导弹的X坐标（÷100）、Y坐标（÷100）、Z坐标（÷50）、偏航角（×100）、俯仰角（×100）、滚转角（×100）。

DogfightEnv的初始化需要与Dogfight 2软件进行连接，接受如下参数作为输入变量：

```python
class DogfightEnv(Env):

    def __init__(
        self,
        host='10.184.0.0',
        port='50888',
        plane_slot=1,
        enemy_slot=3,
        missile_slot=0,
        rendering=False,
    ) -> None:
```

在使用DogfightEnv环境前，需要先启动Dogfight 2软件，选择Network Mode，并将软件左上角显示的Host与Port作为参数传进环境中。


</details>


## <div align="center">未来工作</div>

1. 为不同强化学习模型提供不同的动作空间，包括现有的对飞机操纵面的控制，以及可能的以航向角，目标位姿等作为动作空间；

2. 提供不同的状态空间，除了现有的描述飞机六个自由度的值之外，额外增加速度、加速度、剩余油量等飞机状态供不同需要的强化学习模型选择；

3. 实现不同强化学习基本模型，复现近年来国内外顶会顶刊上的智能空战算法；

4. 提供多种格式的空战数据输出功能，例如支持JSBSim环境输出至Tacview等；


## <div align="center">欢迎交流</div>

如果您在使用过程中发现了任何错误，或者有任何需求与改进的建议，欢迎在Github Issues中提出，<!-- 或者直接联系我的邮箱`mrwangyou@stu.xjtu.edu.cn`，-->谢谢！
