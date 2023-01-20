#!/usr/bin/env python3
import math
import numpy as np
from common.numpy_fast import clip, interp

import cereal.messaging as messaging
from common.conversions import Conversions as CV
from common.filter_simple import FirstOrderFilter
from common.realtime import DT_MDL
from selfdrive.modeld.constants import T_IDXS
from selfdrive.controls.lib.longcontrol import LongCtrlState
from selfdrive.controls.lib.longitudinal_mpc_lib.long_mpc import LongitudinalMpc, MIN_ACCEL, MAX_ACCEL
from selfdrive.controls.lib.longitudinal_mpc_lib.long_mpc import T_IDXS as T_IDXS_MPC
from selfdrive.controls.lib.drive_helpers import V_CRUISE_MAX, CONTROL_N
from system.swaglog import cloudlog

LON_MPC_STEP = 0.2  # first step is 0.2s
AWARENESS_DECEL = -0.2  # car smoothly decel at .2m/s^2 when user is distracted
A_CRUISE_MIN = -1.2
A_CRUISE_MAX_VALS = [1.6, 1.2, 0.8, 0.6]
A_CRUISE_MAX_BP = [0., 10.0, 25., 40.]

# Lookup table for turns
_A_TOTAL_MAX_V = [1.7, 3.2]
_A_TOTAL_MAX_BP = [20., 40.]


def get_max_accel(v_ego):
  return interp(v_ego, A_CRUISE_MAX_BP, A_CRUISE_MAX_VALS)


def limit_accel_in_turns(v_ego, angle_steers, a_target, CP):
  """
  This function returns a limited long acceleration allowed, depending on the existing lateral acceleration
  this should avoid accelerating when losing the target in turns
  """

  # FIXME: This function to calculate lateral accel is incorrect and should use the VehicleModel
  # The lookup table for turns should also be updated if we do this
  a_total_max = interp(v_ego, _A_TOTAL_MAX_BP, _A_TOTAL_MAX_V)
  a_y = v_ego ** 2 * angle_steers * CV.DEG_TO_RAD / (CP.steerRatio * CP.wheelbase)
  a_x_allowed = math.sqrt(max(a_total_max ** 2 - a_y ** 2, 0.))

  return [a_target[0], min(a_target[1], a_x_allowed)]


class LongitudinalPlanner:
  def __init__(self, CP, init_v=0.0, init_a=0.0):
    self.CP = CP
    self.mpc = LongitudinalMpc()
    self.fcw = False

    self.a_desired = init_a
    self.v_desired_filter = FirstOrderFilter(init_v, 2.0, DT_MDL)
    self.v_model_error = 0.0

    self.v_desired_trajectory = np.zeros(CONTROL_N)
    self.a_desired_trajectory = np.zeros(CONTROL_N)
    self.j_desired_trajectory = np.zeros(CONTROL_N)
    self.solverExecutionTime = 0.0

  @staticmethod
  def parse_model(model_msg, model_error, v_ego):
    # 当前supercombo模型输出的规划结果是33个路点，所以这里只考虑if为true时的情况
    if (len(model_msg.position.x) == 33 and
       len(model_msg.velocity.x) == 33 and
       len(model_msg.acceleration.x) == 33):
      # T_IDXS_MPC是把10s时间按指数分布到MPC求解用的13个时刻， 把模型给出的规划结果中的纵向位移x，
      # 从33个时刻插值到13个时刻再减去速度偏差model_error和T_IDXS_MPC计算出的距离偏差，得到新的位移x
      x = np.interp(T_IDXS_MPC, T_IDXS, model_msg.position.x) - model_error * T_IDXS_MPC
      # 把模型给出的规划结果中的速度，从33个时刻插值到13个时刻再减去速度偏差model_error，得到新的速度v
      v = np.interp(T_IDXS_MPC, T_IDXS, model_msg.velocity.x) - model_error
      # 把模型给出的规划结果中的加速度，从33个时刻插值到13个时刻
      a = np.interp(T_IDXS_MPC, T_IDXS, model_msg.acceleration.x)
      j = np.zeros(len(T_IDXS_MPC))
    else:
      x = np.zeros(len(T_IDXS_MPC))
      v = np.zeros(len(T_IDXS_MPC))
      a = np.zeros(len(T_IDXS_MPC))
      j = np.zeros(len(T_IDXS_MPC))
    # 根据当前车数v_ego, 使用插值方式取得最大横向加速度。车速小于等于5时，最大横向加速度为1.5
    max_lat_accel = interp(v_ego, [5, 10, 20], [1.5, 2.0, 3.0])
    # 根据轨迹规划的33个未来路点的横摆角速度orientationRate.z，使用插值到13个路点的结果除以速度，得到13个路点的曲率k
    # omega = phi_dot
    # k = omega / v
    curvatures = np.interp(T_IDXS_MPC, T_IDXS, model_msg.orientationRate.z) / np.clip(v, 0.3, 100.0)
    # 根据最大横向加速度和曲率k，计算出13个路点的最大速度max_v
    # v^2 = a / k
    max_v = np.sqrt(max_lat_accel / (np.abs(curvatures) + 1e-3)) - 2.0
    # 限制13个路点的速度v的最大值为max_v，这样横向加速度就不会超过最大横向加速度
    v = np.minimum(max_v, v)
    # 这里虽然限制和修改了轨迹规划结果中每个路点的速度上限，但x和a的值并没有修改，由后续的MPC来统一计算代价
    return x, v, a, j

  # 每轮supercombo模型输出规划轨迹后，都会调用该函数，用于优化纵向控制所需的参数
  def update(self, sm):
    # acc模式表示只是用雷达数据，blended模式表示使用雷达和视觉感知的融合数据
    self.mpc.mode = 'blended' if sm['controlsState'].experimentalMode else 'acc'

    # 该carState.vEgo，是由其他模块在获取了相关车辆传感器的的CAN数据后，
    # 又进行了卡尔曼滤波，从而估计出的当前自车速度
    v_ego = sm['carState'].vEgo
    # 获取以m/s为单位的，驾驶员所设定的巡航速度 v_cruise
    v_cruise_kph = sm['controlsState'].vCruise
    v_cruise_kph = min(v_cruise_kph, V_CRUISE_MAX)
    v_cruise = v_cruise_kph * CV.KPH_TO_MS

    # longControlState由上一轮的控制模块给出，其数据类型是枚举体，
    # 有 off, pid, starting, stopping 四种取值，表示相关控制状态机当前的的状态
    long_control_off = sm['controlsState'].longControlState == LongCtrlState.off
    # forceDecel由上一轮的控制模块给出，若司机注意力不集中，或正在关闭openpilot中，则为true
    force_slow_decel = sm['controlsState'].forceDecel

    # openpilotLongitudinalControl 表示端到端纵向控制功能是否已打开，且车辆类型也支持
    # 后面的分析只考虑车辆支持支持，且已打开，所以reset_state在这里被设置为=long_control_off，即longControlState是否处于off状态
    # Reset current state when not engaged, or user is controlling the speed
    reset_state = long_control_off if self.CP.openpilotLongitudinalControl else not sm['controlsState'].enabled

    # 如果不是处于停止中/off状态中，则需要让后续的MPC优化中考虑纵向加速度的cost
    # No change cost when user is controlling the speed, or when standstill
    prev_accel_constraint = not (reset_state or sm['carState'].standstill)

    if self.mpc.mode == 'acc':
      accel_limits = [A_CRUISE_MIN, get_max_accel(v_ego)]
      accel_limits_turns = limit_accel_in_turns(v_ego, sm['carState'].steeringAngleDeg, accel_limits, self.CP)
    else:
      # MIN_ACCEL=-3.5 , MAX_ACCEL=2.0
      accel_limits = [MIN_ACCEL, MAX_ACCEL]
      accel_limits_turns = [MIN_ACCEL, MAX_ACCEL]

    # a_desired是上一轮规划后的车辆在0.05s后的加速度
    # 若当前处于停止中/off状态中，则本次a_desired不采用上一轮的规划结果，而是采用车辆当前的加速度sm['carState'].aEgo
    if reset_state:
      self.v_desired_filter.x = v_ego
      # Clip aEgo to cruise limits to prevent large accelerations when becoming active
      self.a_desired = clip(sm['carState'].aEgo, accel_limits[0], accel_limits[1])
    
    # 对自车速度再次滤波
    # Prevent divergence, smooth in current v_ego
    self.v_desired_filter.x = max(0.0, self.v_desired_filter.update(v_ego))
    # 计算了模型给出的速度估计，与车辆传感器给出的速度，两者之差 v_model_error
    # 因为模型估计的自车速度的偏差，也会影响模型给出的轨迹规划，所以后续会用该偏差对规划进行修正
    # Compute model v_ego error
    if len(sm['modelV2'].temporalPose.trans):
      self.v_model_error = sm['modelV2'].temporalPose.trans[0] - v_ego

    if force_slow_decel:
      # if required so, force a smooth deceleration
      accel_limits_turns[1] = min(accel_limits_turns[1], AWARENESS_DECEL)
      accel_limits_turns[0] = min(accel_limits_turns[0], accel_limits_turns[1])
    # clip limits, cannot init MPC outside of bounds
    accel_limits_turns[0] = min(accel_limits_turns[0], self.a_desired + 0.05)
    accel_limits_turns[1] = max(accel_limits_turns[1], self.a_desired - 0.05)
    
    # 设置MPC求解需要的cost权重、constraint、参数等
    self.mpc.set_weights(prev_accel_constraint)
    self.mpc.set_accel_limits(accel_limits_turns[0], accel_limits_turns[1])
    self.mpc.set_cur_state(self.v_desired_filter.x, self.a_desired)
    # sm['modelV2']是模型推理的所有输出，这里结合速度偏差 v_model_error， 和当前真实车速v_ego，调用parse_model函数
    # 对模型的轨迹规划结果进行修正，得到修正后的未来轨迹的13个路点的 x, velocity, acceleration, jerk
    # movelV2给出的规划是33个路点，但MPC的滚动窗口horizon的步数是13，所以需要插值转换
    x, v, a, j = self.parse_model(sm['modelV2'], self.v_model_error, v_ego)
    # 调用纵向规划的MPC进行最优化问题求解
    self.mpc.update(sm['carState'], sm['radarState'], v_cruise, x, v, a, j)
    
    # 将MPC求解后的v, a, j  从13个时刻(10s)插值到17个时刻（约2.5s）
    self.v_desired_trajectory = np.interp(T_IDXS[:CONTROL_N], T_IDXS_MPC, self.mpc.v_solution)
    self.a_desired_trajectory = np.interp(T_IDXS[:CONTROL_N], T_IDXS_MPC, self.mpc.a_solution)
    self.j_desired_trajectory = np.interp(T_IDXS[:CONTROL_N], T_IDXS_MPC[:-1], self.mpc.j_solution)
    
    # 如果MPC中的crash_cnt累积值大于2则触发碰撞预警fcw。雷达不够可靠所以需要累积多次的结果再触发。
    # TODO counter is only needed because radar is glitchy, remove once radar is gone
    self.fcw = self.mpc.crash_cnt > 2 and not sm['carState'].standstill
    if self.fcw:
      cloudlog.info("FCW triggered")

    # Interpolate 0.05 seconds and save as starting point for next iteration
    a_prev = self.a_desired
    # DT_MDL表示模型推理的周期时间，为0.02s。这个周期时间也是规划模块被调用的间隔
    # 从优化后的加速度规划结果中，插值取得0.02s后的加速度
    self.a_desired = float(interp(DT_MDL, T_IDXS[:CONTROL_N], self.a_desired_trajectory))
    # 根据当前加速度，下一轮（0.02s后）的加速度，计算出平均加速度，再基于此计算下一轮的 v
    self.v_desired_filter.x = self.v_desired_filter.x + DT_MDL * (self.a_desired + a_prev) / 2.0
  
  # 将update函数得到的优化结果，填充到longitudinalPlan消息中，然后发布出去
  def publish(self, sm, pm):
    plan_send = messaging.new_message('longitudinalPlan')

    plan_send.valid = sm.all_checks(service_list=['carState', 'controlsState'])

    longitudinalPlan = plan_send.longitudinalPlan
    longitudinalPlan.modelMonoTime = sm.logMonoTime['modelV2']
    longitudinalPlan.processingDelay = (plan_send.logMonoTime / 1e9) - sm.logMonoTime['modelV2']
    # 填充longitudinalPlan消息中的，优化后的未来17个时刻的，速度、加速度、加速度变化率
    # 后续的控制模块会使用longitudinalPlan消息中的这些数据，再产生控制量，控制车辆的运动
    longitudinalPlan.speeds = self.v_desired_trajectory.tolist()
    longitudinalPlan.accels = self.a_desired_trajectory.tolist()
    longitudinalPlan.jerks = self.j_desired_trajectory.tolist()

    longitudinalPlan.hasLead = sm['radarState'].leadOne.status
    longitudinalPlan.longitudinalPlanSource = self.mpc.source
    longitudinalPlan.fcw = self.fcw

    longitudinalPlan.solverExecutionTime = self.mpc.solve_time

    pm.send('longitudinalPlan', plan_send)
