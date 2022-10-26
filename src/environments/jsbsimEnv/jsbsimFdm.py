import time

import jsbsim


class JsbsimFdm():

    def __init__(
        self,
        fdm_id       = 1,
        fdm_aircraft = 'f16',    
        fdm_ic_v     = 500,      # Calibrated Velocity (knots) https://skybrary.aero/articles/calibrated-airspeed-cas
        fdm_ic_lat   = 0,        # Latitude (degree) 纬度
        fdm_ic_long  = 0,        # Longitude (degree) 经度
        fdm_ic_h     = 20005.5,  # Height above sea level (feet) 海拔
        fdm_ic_psi   = 0,        # Yaw 偏航角，绕Z轴，JSBSim将按照这里列出的顺序先后进行欧拉角计算
        fdm_ic_theta = 0,        # Pitch 俯仰角，绕Y轴
        fdm_ic_phi   = 0,        # Roll 翻滚角，绕X轴

        fdm_hp       = 1,
        fdm_fgfs     = False,    # Visualization
        flight_mode  = 1,        # 0 for flight test
    ) -> None:
        
        # FDM Initialization 空气动力学模型初始化
        self.fdm = jsbsim.FGFDMExec(None)

        # Aircraft Loading 加载飞机模型
        self.fdm.load_model(fdm_aircraft)

        # FlightGear Visualization 可视化
        if fdm_fgfs is True:
            self.fdm.set_output_directive('./data_output/flightgear{}.xml'.format(fdm_id))
        
        # Velocity Initialization 速度初始化
        self.fdm['ic/vc-kts'] = fdm_ic_v

        # Position Initialization 位置初始化
        self.fdm["ic/lat-gc-deg"] = fdm_ic_lat
        self.fdm["ic/long-gc-deg"] = fdm_ic_long
        self.fdm["ic/h-sl-ft"] = fdm_ic_h

        # Attitude Initialization 姿态初始化
        self.fdm["ic/psi-true-deg"] = fdm_ic_psi
        self.fdm["ic/theta-deg"] = fdm_ic_theta
        self.fdm["ic/phi-deg"] = fdm_ic_phi

        ##########################
        ## Model Initialization ##
        ## 模型初始化            ##
        self.fdm.run_ic()       ##
        ##########################

        # Turning on the Engine 启动引擎
        self.fdm["propulsion/starter_cmd"] = 1

        # Refueling 无限燃料
        if flight_mode == 0:
            self.fdm["propulsion/refuel"] = 1

        # First but not Initial 第一帧设置
        self.fdm.run()
        self.fdm["propulsion/active_engine"] = True
        self.fdm["propulsion/set-running"] = -1

        # Number of Frames 帧数
        self.nof = 1

        # Setting HP 生命值
        self.fdm_hp = fdm_hp

        # Getting parameters 记录参数
        self.param = {
            'fdm_id'       : fdm_id,
            'fdm_aircraft' : fdm_aircraft,
            'fdm_ic_v'     : fdm_ic_v,
            'fdm_ic_lat'   : fdm_ic_lat,
            'fdm_ic_long'  : fdm_ic_long,
            'fdm_ic_h'     : fdm_ic_h,
            'fdm_ic_psi'   : fdm_ic_psi,
            'fdm_ic_theta' : fdm_ic_theta,
            'fdm_ic_phi'   : fdm_ic_phi,
            'fdm_hp'       : fdm_hp,
            'fdm_fgfs'     : fdm_fgfs,
            'flight_mode'  : flight_mode,
        }

    def getProperty(
        self,
        prop,
    ) -> list:
        if prop == 'position':
            prop = [
                "position/lat-gc-deg",  # Latitude 纬度
                "position/long-gc-deg",  # Longitude 经度
                "position/h-sl-ft",  # Altitude above sea level(feet) 海拔（英尺）
            ]
        elif prop == 'positionEci':  # Earth-centered inertial(feet) 地心惯性坐标系（英尺），与Ecef相比相对静止
            prop = [
                "position/eci-x-ft",  # 指向春分点
                "position/eci-y-ft",  # 左手系决定
                "position/eci-z-ft",  # 指向北极点
            ]
        elif prop == 'positionEcef':  # Earth-centered, Earth-fixed coordinate system(feet) 地心地固坐标系（英尺） https://zhuanlan.zhihu.com/p/360744867
            prop = [
                "position/ecef-x-ft",  # 指向经纬度为0的点（本初子午线与赤道的交点）
                "position/ecef-y-ft",  # 左手系决定
                "position/ecef-z-ft",  # 指向北极点
            ]
        elif prop == 'attitudeRad':  # Attitude(Rad) 姿态（弧度）
            prop = [
                "attitude/psi-rad",  # Yaw 偏航角
                "attitude/theta-rad",  # Pitch 俯仰角
                "attitude/phi-rad",  # Roll 翻滚角
            ]
        elif prop == 'attitudeDeg':  # Attitude(Deg) 姿态（角度）
            prop = [
                "attitude/psi-deg",  # Yaw 偏航角
                "attitude/theta-deg",  # Pitch 俯仰角
                "attitude/phi-deg",  # Roll 翻滚角
            ]
        elif prop == 'pose':  # Pose(deg) 位姿（角度）
            prop = [
                "position/lat-gc-deg",
                "position/long-gc-deg",
                "position/h-sl-ft",
                "attitude/psi-deg",
                "attitude/theta-deg",
                "attitude/phi-deg",
            ]
        elif prop == 'velocity':  # Velocity(fps) 速度（英尺每秒）
            prop = [
                "velocities/v-north-fps",
                "velocities/v-east-fps",
                "velocities/v-down-fps",
            ]
        elif prop == 'acceleration':  # Acceleration 加速度
            raise NotImplementedError('Hasn\'t finished yet!')
            prop = [

            ]
        elif prop == 'all':  # nominal 'all'
            prop = [
                "position/lat-gc-deg",
                "position/long-gc-deg",
                "position/h-sl-ft",
                "attitude/psi-deg",
                "attitude/theta-deg",
                "attitude/phi-deg",
                "velocities/v-north-fps",
                "velocities/v-east-fps",
                "velocities/v-down-fps",
            ]
        else:
            return self.fdm[prop]

        return [
            self.fdm[item] for item in prop
        ]

    def sendAction(
        self,
        action,  # List of size [num_of_action]
        actionType=None,
    ):
        if actionType is None:
            action_space = [
                "fcs/aileron-cmd-norm",  # 副翼
                "fcs/elevator-cmd-norm",  # 升降舵
                "fcs/rudder-cmd-norm",  # 方向舵
                "fcs/throttle-cmd-norm",  # 油门
            ]
            for i in range(len(action_space)):
                self.fdm[action_space[i]] = action[i]
        else:
            self.fdm[actionType] = action

    def damage(self, value):
        self.fdm_hp = self.fdm_hp - value

    def step(self, playSpeed=0):
        self.nof = self.nof + 1
        self.fdm.run()

        if playSpeed != 0:
            time.sleep(self.fdm.get_delta_t() / playSpeed)

    def terminate(self):
        if self.fdm_hp <= 0:
            return 1
        else:
            return 0
