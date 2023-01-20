import numpy as np
from common.realtime import sec_since_boot, DT_MDL
from common.numpy_fast import interp
from system.swaglog import cloudlog
from selfdrive.controls.lib.lateral_mpc_lib.lat_mpc import LateralMpc
from selfdrive.controls.lib.lateral_mpc_lib.lat_mpc import N as LAT_MPC_N
from selfdrive.controls.lib.drive_helpers import CONTROL_N, MIN_SPEED
from selfdrive.controls.lib.desire_helper import DesireHelper
import cereal.messaging as messaging
from cereal import log

TRAJECTORY_SIZE = 33
CAMERA_OFFSET = 0.04


PATH_COST = 1.0
LATERAL_MOTION_COST = 0.11
LATERAL_ACCEL_COST = 1.0
LATERAL_JERK_COST = 0.05
# Extreme steering rate is unpleasant, even
# when it does not cause bad jerk.
# TODO this cost should be lowered when low
# speed lateral control is stable on all cars
STEERING_RATE_COST = 8.0


class LateralPlanner:
  def __init__(self, CP):
    self.DH = DesireHelper()

    # Vehicle model parameters used to calculate lateral movement of car
    self.factor1 = CP.wheelbase - CP.centerToFront
    self.factor2 = (CP.centerToFront * CP.mass) / (CP.wheelbase * CP.tireStiffnessRear)
    self.last_cloudlog_t = 0
    self.solution_invalid_cnt = 0

    self.path_xyz = np.zeros((TRAJECTORY_SIZE, 3))
    self.plan_yaw = np.zeros((TRAJECTORY_SIZE,))
    self.plan_yaw_rate = np.zeros((TRAJECTORY_SIZE,))
    self.t_idxs = np.arange(TRAJECTORY_SIZE)
    self.y_pts = np.zeros(TRAJECTORY_SIZE)

    self.lat_mpc = LateralMpc()
    self.reset_mpc(np.zeros(4))

  def reset_mpc(self, x0=np.zeros(4)):
    self.x0 = x0
    self.lat_mpc.reset(x0=self.x0)

  def update(self, sm):
    #MPC模拟的横向控制里假定车速不变，所以这里车速不能为0得有最低值 1.0 m/s
    # clip speed , lateral planning is not possible at 0 speed
    self.v_ego = max(MIN_SPEED, sm['carState'].vEgo)
    #读取车辆的当前的曲率，controlsState里所有信息由上一轮控制器使用车辆模型计等算后发布
    measured_curvature = sm['controlsState'].curvature

    #modelV2是模型推理部分发布的消息，包含模型给出的轨迹规划
    # Parse model predictions
    md = sm['modelV2']
    if len(md.position.x) == TRAJECTORY_SIZE and len(md.orientation.x) == TRAJECTORY_SIZE:
      #读取轨迹规划里的33个点的XYZ坐标
      #使用np.column_stack形成[[x0,y0,z0], [x1,y1,z1], ... , [x32, y32, z32]] 这样的数组
      self.path_xyz = np.column_stack([md.position.x, md.position.y, md.position.z])
      #读取轨迹里的33个点的时间，这里的时间是非均匀分布的
      # t_ids = [t0, t1, t2, ... , t32]
      self.t_idxs = np.array(md.position.t)
      # 读取轨迹里的33个点的横摆角psi, 
      # plan_yaw = [psi0, psi1, psi2, ... , psi32]
      self.plan_yaw = np.array(md.orientation.z)
      # 读取轨迹里的33个点的横摆角速度psi_dot
      # plan_yaw_rate = [psi_dot0, psi_dot1, psi_dot2, ... , psi_dot32]
      self.plan_yaw_rate = np.array(md.orientationRate.z)

    # Lane change logic
    desire_state = md.meta.desireState
    if len(desire_state):
      self.l_lane_change_prob = desire_state[log.LateralPlan.Desire.laneChangeLeft]
      self.r_lane_change_prob = desire_state[log.LateralPlan.Desire.laneChangeRight]
    lane_change_prob = self.l_lane_change_prob + self.r_lane_change_prob
    self.DH.update(sm['carState'], sm['carControl'].latActive, lane_change_prob)

    d_path_xyz = self.path_xyz
    # lat_mpc 是MPC控制器，给它设定代价的系数
    # PATH_COST 是与目标 y 的error的代价系数
    # LATERAL_MOTION_COST 是与目标 psi*v 的error的代价系数
    # LATERAL_ACCEL_COST 是与目标 psi_dot*v 的error的代价系数
    # 模型预测的plan里，只有y, psi, psi_dot信息在这里可以用作yref一部分
    #
    # psi_dot_dot 是该MPC里状态方程的控制量U
    # LATERAL_JERK_COST 是 psi_dot_dot*v 与 0 的error的代价系数，可以看成是U的cost
    # STEERING_RATE_COST 是 psi_dot_dot/v 与 0 的error的代价系数，可以看成是U的cost
    self.lat_mpc.set_weights(PATH_COST, LATERAL_MOTION_COST,
                             LATERAL_ACCEL_COST, LATERAL_JERK_COST,
                             STEERING_RATE_COST)
    # d_path_xyz[:, 1]是模型预测轨迹里的33个点的y
    # np.linalg.norm(d_path_xyz, axis=1) 计算出了33个点分别到当前位置（0,0,0）的距离
    # self.t_idxs是33个点的时间（10秒钟非均匀的分布在33个点上，越近越密，越远越疏）
    # LAT_MPC_N 是MPC里horizon的步数，16，不过考虑到还有终止状态，后续需要准备17个ref
    # 使用np.interp，用距离来计算插值近似值，y_pts是17个数的数组，代表按照t_idxs时间分布规律下的，17个时刻的y
    y_pts = np.interp(self.v_ego * self.t_idxs[:LAT_MPC_N + 1], np.linalg.norm(d_path_xyz, axis=1), d_path_xyz[:, 1])
    # 同理，17个时刻的psi
    heading_pts = np.interp(self.v_ego * self.t_idxs[:LAT_MPC_N + 1], np.linalg.norm(self.path_xyz, axis=1), self.plan_yaw)
    # 同理，17个时刻的psi_dot
    yaw_rate_pts = np.interp(self.v_ego * self.t_idxs[:LAT_MPC_N + 1], np.linalg.norm(self.path_xyz, axis=1), self.plan_yaw_rate)
    self.y_pts = y_pts

    assert len(y_pts) == LAT_MPC_N + 1
    assert len(heading_pts) == LAT_MPC_N + 1
    assert len(yaw_rate_pts) == LAT_MPC_N + 1
    lateral_factor = max(0, self.factor1 - (self.factor2 * self.v_ego**2))
    # 准备MPC里模型动态方程的参数p
    p = np.array([self.v_ego, lateral_factor])
    # self.x0含义是t0时刻[x, y, pi, pi_dot]，会被作为constraint
    # 以为t0时刻就是当前车辆现在的状态，所以self.x0 = [0, 0, 0, pi_dot]， pi_dot是车辆横摆角速度
    # 使用MPC求解
    self.lat_mpc.run(self.x0,
                     p,
                     y_pts,
                     heading_pts,
                     yaw_rate_pts)
    # init state for next iteration
    # mpc.u_sol is the desired second derivative of psi given x0 curv state.
    # with x0[3] = measured_yaw_rate, this would be the actual desired yaw rate.
    # instead, interpolate x_sol so that x0[3] is the desired yaw rate for lat_control.
    # self.lat_mpc.x_sol[:, 3]是MPC求解后，按照最优轨迹的17个时刻的psi_dot
    # DT_MDL是0.05s，表示模型每次推理的时间间隔，即模型会按照20HZ来推理，lateral_plan也会按照同样的频率来更新
    # 这里对self.x0[3] 的设置，是给下一次lateral_plan调用做准备，用本次MPC求解的结果，来估计0.05s后车辆的横摆角速度psi_dot
    self.x0[3] = interp(DT_MDL, self.t_idxs[:LAT_MPC_N + 1], self.lat_mpc.x_sol[:, 3])

    #  Check for infeasible MPC solution
    mpc_nans = np.isnan(self.lat_mpc.x_sol[:, 3]).any()
    t = sec_since_boot()
    if mpc_nans or self.lat_mpc.solution_status != 0:
      # MPC求解可能失败，若失败则需要对齐reset，同时下一次lateral_plan调用时横摆角速度psi_dot不能使用MPC的结算结果了，
      # 只能用当前的测量曲率和速度来近似计算
      self.reset_mpc()
      self.x0[3] = measured_curvature * self.v_ego
      if t > self.last_cloudlog_t + 5.0:
        self.last_cloudlog_t = t
        cloudlog.warning("Lateral mpc - nan: True")

    if self.lat_mpc.cost > 20000. or mpc_nans:
      self.solution_invalid_cnt += 1
    else:
      self.solution_invalid_cnt = 0

  def publish(self, sm, pm):
    plan_solution_valid = self.solution_invalid_cnt < 2
    # 构造要发布的lateralPlan消息
    plan_send = messaging.new_message('lateralPlan')
    plan_send.valid = sm.all_checks(service_list=['carState', 'controlsState', 'modelV2'])

    lateralPlan = plan_send.lateralPlan
    lateralPlan.modelMonoTime = sm.logMonoTime['modelV2']
    # 未经MPC处理的，插值后的未来17个时刻的 y，该信息后续不参与控制，只参与计算车道偏离预警
    lateralPlan.dPathPoints = self.y_pts.tolist()
    # CONTROL_N为17，lat_mpc.x_sol[0:CONTROL_N, 2]表示MPC求解的最优控制的未来17个时刻的psi
    lateralPlan.psis = self.lat_mpc.x_sol[0:CONTROL_N, 2].tolist()
    # lat_mpc.x_sol[0:CONTROL_N, 3]表示MPC求解的最优控制的未来17个时刻的很横摆角速度psi_dot
    # 使用 psi_dot / v = kappa 得到未来17个时刻的曲率
    lateralPlan.curvatures = (self.lat_mpc.x_sol[0:CONTROL_N, 3]/self.v_ego).tolist()
    # lat_mpc.u_sol[0:CONTROL_N - 1]]表示MPC求解的最优控制的未来16个时刻的psi_dot_dot
    # 使用 psi_dot_dot / v = kappa_dot 得到未来17个时刻的曲率变化率
    lateralPlan.curvatureRates = [float(x/self.v_ego) for x in self.lat_mpc.u_sol[0:CONTROL_N - 1]] + [0.0]

    lateralPlan.mpcSolutionValid = bool(plan_solution_valid)
    lateralPlan.solverExecutionTime = self.lat_mpc.solve_time

    lateralPlan.desire = self.DH.desire
    lateralPlan.useLaneLines = False
    lateralPlan.laneChangeState = self.DH.lane_change_state
    lateralPlan.laneChangeDirection = self.DH.lane_change_direction

    # 发布lateralPlan消息，其它sub该消息的地方就可以接收到
    pm.send('lateralPlan', plan_send)
