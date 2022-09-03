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

<u>如需使用Dogfight 2环境</u>，请在<a href="https://github.com/harfang3d/dogfight-sandbox-hg2">Dogfight 2 Github仓库</a>中下载最新版本的Dogfight 2，并将源码放在`DBRL/src/envorinments/dogfightEnv/`下，并重命名为`dogfight_sandbox_hg2/`。

JSBSim的可视化功能基于FlightGear实现，<u>如需在使用JSBSim的过程中将飞行过程可视化显示</u>，请在<a href="https://www.flightgear.org/">FlightGear官网</a>中安装FlightGear软件。如需对缠斗中的两架飞机分别进行可视化，请在`JSBSim/data_output/`下复制两份`flightgear.xml`文件，分别命名为`flightgear{1/2}.xml`，并将`flightgear2.xml`第18行中的`5550`修改为`5551`。如需查看`JSBSim`源文件地址，可以运行如下代码：

```python
import jsbsim

print(jsbsim.get_default_root_dir())
```

</details>


<details open>
<summary>使用方法</summary>

DBRL提供了<a href="https://github.com/openai/gym">OpenAI Gym</a>格式的强化学习环境，可以通过Gym库进行调用。如需使用`gym.make`调用环境，请在`Gym/envs`下复制`src/environments/*Env/`文件夹，并在`Gym/envs/__init__.py`中添加如下代码：

```python
register(
    id="DBRL-v0",
    entry_point="gym.envs.{jsbsim/dogfight}Env:{JSBSim/Dogfight}Env",
    max_episode_steps=200,
    reward_threshold=100.0,
)
```

调用环境时可以采用如下代码：

```python
import gym

env = gym.make('DBRL-v0')
```

如需查看`gym`源文件地址，可以运行如下代码：

```bash
pip show gym
```

如果不从`Gym`库直接调用环境，也可以直接使用环境类的实例，可以采用如下代码：

```python
from DBRL.src.environments import jsbsimEnv as Env

env = Env.Env()
```


</details>
