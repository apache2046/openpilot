#!/usr/bin/env python3
import os
import numpy as np

from common.realtime import sec_since_boot
from common.numpy_fast import clip
from system.swaglog import cloudlog
from selfdrive.modeld.constants import index_function
from selfdrive.controls.lib.radar_helpers import _LEAD_ACCEL_TAU

if __name__ == '__main__':  # generating code
  from pyextra.acados_template import AcadosModel, AcadosOcp, AcadosOcpSolver
else:
  from selfdrive.controls.lib.longitudinal_mpc_lib.c_generated_code.acados_ocp_solver_pyx import AcadosOcpSolverCython  # pylint: disable=no-name-in-module, import-error

from casadi import SX, vertcat

MODEL_NAME = 'long'
LONG_MPC_DIR = os.path.dirname(os.path.abspath(__file__))
EXPORT_DIR = os.path.join(LONG_MPC_DIR, "c_generated_code")
JSON_FILE = os.path.join(LONG_MPC_DIR, "acados_ocp_long.json")

SOURCES = ['lead0', 'lead1', 'cruise', 'e2e']

X_DIM = 3
U_DIM = 1
PARAM_DIM = 6
COST_E_DIM = 5
COST_DIM = COST_E_DIM + 1
CONSTR_DIM = 4

X_EGO_OBSTACLE_COST = 3.
X_EGO_COST = 0.
V_EGO_COST = 0.
A_EGO_COST = 0.
J_EGO_COST = 5.0
A_CHANGE_COST = 200.
DANGER_ZONE_COST = 100.
CRASH_DISTANCE = .25
LEAD_DANGER_FACTOR = 0.75
LIMIT_COST = 1e6
ACADOS_SOLVER_TYPE = 'SQP_RTI'


# Fewer timestamps don't hurt performance and lead to
# much better convergence of the MPC with low iterations
N = 12
MAX_T = 10.0
T_IDXS_LST = [index_function(idx, max_val=MAX_T, max_idx=N) for idx in range(N+1)]

T_IDXS = np.array(T_IDXS_LST)
FCW_IDXS = T_IDXS < 5.0
T_DIFFS = np.diff(T_IDXS, prepend=[0.])
MIN_ACCEL = -3.5
MAX_ACCEL = 2.0
T_FOLLOW = 1.45
COMFORT_BRAKE = 2.5
STOP_DISTANCE = 6.0

def get_stopped_equivalence_factor(v_lead):
  return (v_lead**2) / (2 * COMFORT_BRAKE)

# 计算距离障碍物的安全距离，COMFORT_BRAKE是刹车的加速度（2.5m/s），STOP_DISTANCE是刹停后希望还能距离障碍物的距离
# v^2 / 2a + t * v + stop_dist， 制动距离 + 跟车距离 + 刹停后距离障碍物的距离
# 表示的是，障碍物突然静止，此时自车继续以v_ego速度行驶t_follow时间后，开始以COMFORT_BRAKE加速度刹车，刹停后距离障碍物的距离
def get_safe_obstacle_distance(v_ego, t_follow=T_FOLLOW):
  return (v_ego**2) / (2 * COMFORT_BRAKE) + t_follow * v_ego + STOP_DISTANCE

def desired_follow_distance(v_ego, v_lead):
  return get_safe_obstacle_distance(v_ego) - get_stopped_equivalence_factor(v_lead)

# 定义系统模型
def gen_long_model():
  model = AcadosModel()
  model.name = MODEL_NAME
  # 系统状态量有3个，包括自车位置，速度，加速度
  # set up states & controls
  x_ego = SX.sym('x_ego')
  v_ego = SX.sym('v_ego')
  a_ego = SX.sym('a_ego')
  model.x = vertcat(x_ego, v_ego, a_ego)

  # 控制量有1个，加加速度jerk
  # controls
  j_ego = SX.sym('j_ego')
  model.u = vertcat(j_ego)

  # xdot
  x_ego_dot = SX.sym('x_ego_dot')
  v_ego_dot = SX.sym('v_ego_dot')
  a_ego_dot = SX.sym('a_ego_dot')
  model.xdot = vertcat(x_ego_dot, v_ego_dot, a_ego_dot)

  # MPC的参数有6个，分别是最小加速度，最大加速度，障碍物位置，前一时刻的加速度，前车跟车时间，前车危险系数
  # 这些参数不随滚动窗口horizon变化，由调用者在使用MPC求解前提前传入
  # live parameters
  a_min = SX.sym('a_min')
  a_max = SX.sym('a_max')
  x_obstacle = SX.sym('x_obstacle')
  prev_a = SX.sym('prev_a')
  lead_t_follow = SX.sym('lead_t_follow')
  lead_danger_factor = SX.sym('lead_danger_factor')
  model.p = vertcat(a_min, a_max, x_obstacle, prev_a, lead_t_follow, lead_danger_factor)

  # 系统的状态方程，f_expl定义了系统的3状态量x_ego, v_ego, a_ego的关于时间的微分该如何计算
  # dynamics model
  f_expl = vertcat(v_ego, a_ego, j_ego)
  model.f_expl_expr = f_expl
  # 按照acados的要求， f_impl_expr在integrator_type为IRK时是必须要定义的，否则可默认为None
  model.f_impl_expr = model.xdot - f_expl
  return model

# 定义OCP(optimal control problem)
def gen_long_ocp():
  ocp = AcadosOcp()
  ocp.model = gen_long_model()

  Tf = T_IDXS[-1]

  # set dimensions
  ocp.dims.N = N

  # 采用非线性损失函数
  # set cost module
  ocp.cost.cost_type = 'NONLINEAR_LS'
  ocp.cost.cost_type_e = 'NONLINEAR_LS'

  QR = np.zeros((COST_DIM, COST_DIM))
  Q = np.zeros((COST_E_DIM, COST_E_DIM))

  # 设置代价权重的初值
  ocp.cost.W = QR
  ocp.cost.W_e = Q

  # 读取系统的3个状态量到新的变量，方便后续计算代价
  x_ego, v_ego, a_ego = ocp.model.x[0], ocp.model.x[1], ocp.model.x[2]
  # 读取系统的1个控制量到新的变量，方便后续计算代价
  j_ego = ocp.model.u[0]

  # 读取6个参数到新的变量，方便后续计算代价
  a_min, a_max = ocp.model.p[0], ocp.model.p[1]
  x_obstacle = ocp.model.p[2]
  prev_a = ocp.model.p[3]
  lead_t_follow = ocp.model.p[4]
  lead_danger_factor = ocp.model.p[5]

  # 设置代价计算的目标yref和yref_e初值
  ocp.cost.yref = np.zeros((COST_DIM, ))
  ocp.cost.yref_e = np.zeros((COST_E_DIM, ))

  # 计算满足舒适刹车的最短跟车距离
  desired_dist_comfort = get_safe_obstacle_distance(v_ego, lead_t_follow)

  # The main cost in normal operation is how close you are to the "desired" distance
  # from an obstacle at every timestep. This obstacle can be a lead car
  # or other object. In e2e mode we can use x_position targets as a cost
  # instead.
  # blender模式下，cost的系数是[0., 0.1, 0.2, 5.0, 40.0, 1.0]
  # 和上一轮MPC求解结果prev_a差值的权重最大，是为了避免加速度的震荡
  costs = [((x_obstacle - x_ego) - (desired_dist_comfort)) / (v_ego + 10.),
           x_ego,
           v_ego,
           a_ego,
           a_ego - prev_a,
           j_ego]
  ocp.model.cost_y_expr = vertcat(*costs)
  ocp.model.cost_y_expr_e = vertcat(*costs[:-1])

  # Constraints on speed, acceleration and desired distance to
  # the obstacle, which is treated as a slack constraint so it
  # behaves like an asymmetrical cost.
  # 软约束的代价系数是[1e6, 1e6, 1e6, 50.0]，安全距离的约束系数最小
  constraints = vertcat(v_ego,
                        (a_ego - a_min),
                        (a_max - a_ego),
                        ((x_obstacle - x_ego) - lead_danger_factor * (desired_dist_comfort)) / (v_ego + 10.))
  ocp.model.con_h_expr = constraints

  x0 = np.zeros(X_DIM)
  ocp.constraints.x0 = x0
  ocp.parameter_values = np.array([-1.2, 1.2, 0.0, 0.0, T_FOLLOW, LEAD_DANGER_FACTOR])

  # We put all constraint cost weights to 0 and only set them at runtime
  cost_weights = np.zeros(CONSTR_DIM)
  ocp.cost.zl = cost_weights
  ocp.cost.Zl = cost_weights
  ocp.cost.Zu = cost_weights
  ocp.cost.zu = cost_weights

  ocp.constraints.lh = np.zeros(CONSTR_DIM)
  ocp.constraints.uh = 1e4*np.ones(CONSTR_DIM)
  ocp.constraints.idxsh = np.arange(CONSTR_DIM)

  # The HPIPM solver can give decent solutions even when it is stopped early
  # Which is critical for our purpose where compute time is strictly bounded
  # We use HPIPM in the SPEED_ABS mode, which ensures fastest runtime. This
  # does not cause issues since the problem is well bounded.
  ocp.solver_options.qp_solver = 'PARTIAL_CONDENSING_HPIPM'
  ocp.solver_options.hessian_approx = 'GAUSS_NEWTON'
  ocp.solver_options.integrator_type = 'ERK'
  ocp.solver_options.nlp_solver_type = ACADOS_SOLVER_TYPE
  ocp.solver_options.qp_solver_cond_N = 1

  # More iterations take too much time and less lead to inaccurate convergence in
  # some situations. Ideally we would run just 1 iteration to ensure fixed runtime.
  ocp.solver_options.qp_solver_iter_max = 10
  ocp.solver_options.qp_tol = 1e-3

  # set prediction horizon
  ocp.solver_options.tf = Tf
  ocp.solver_options.shooting_nodes = T_IDXS

  ocp.code_export_directory = EXPORT_DIR
  return ocp


class LongitudinalMpc:
  def __init__(self, mode='acc'):
    self.mode = mode
    self.solver = AcadosOcpSolverCython(MODEL_NAME, ACADOS_SOLVER_TYPE, N)
    self.reset()
    self.source = SOURCES[2]

  def reset(self):
    # self.solver = AcadosOcpSolverCython(MODEL_NAME, ACADOS_SOLVER_TYPE, N)
    self.solver.reset()
    # self.solver.options_set('print_level', 2)
    self.v_solution = np.zeros(N+1)
    self.a_solution = np.zeros(N+1)
    self.prev_a = np.array(self.a_solution)
    self.j_solution = np.zeros(N)
    self.yref = np.zeros((N+1, COST_DIM))
    for i in range(N):
      self.solver.cost_set(i, "yref", self.yref[i])
    self.solver.cost_set(N, "yref", self.yref[N][:COST_E_DIM])
    self.x_sol = np.zeros((N+1, X_DIM))
    self.u_sol = np.zeros((N,1))
    self.params = np.zeros((N+1, PARAM_DIM))
    for i in range(N+1):
      self.solver.set(i, 'x', np.zeros(X_DIM))
    self.last_cloudlog_t = 0
    self.status = False
    self.crash_cnt = 0.0
    self.solution_status = 0
    # timers
    self.solve_time = 0.0
    self.time_qp_solution = 0.0
    self.time_linearization = 0.0
    self.time_integrator = 0.0
    self.x0 = np.zeros(X_DIM)
    self.set_weights()

  def set_cost_weights(self, cost_weights, constraint_cost_weights):
    W = np.asfortranarray(np.diag(cost_weights))
    for i in range(N):
      # TODO don't hardcode A_CHANGE_COST idx
      # reduce the cost on (a-a_prev) later in the horizon.
      W[4,4] = cost_weights[4] * np.interp(T_IDXS[i], [0.0, 1.0, 2.0], [1.0, 1.0, 0.0])
      self.solver.cost_set(i, 'W', W)
    # Setting the slice without the copy make the array not contiguous,
    # causing issues with the C interface.
    self.solver.cost_set(N, 'W', np.copy(W[:COST_E_DIM, :COST_E_DIM]))

    # Set L2 slack cost on lower bound constraints
    Zl = np.array(constraint_cost_weights)
    for i in range(N):
      self.solver.cost_set(i, 'Zl', Zl)

  def set_weights(self, prev_accel_constraint=True):
    if self.mode == 'acc':
      a_change_cost = A_CHANGE_COST if prev_accel_constraint else 0
      cost_weights = [X_EGO_OBSTACLE_COST, X_EGO_COST, V_EGO_COST, A_EGO_COST, a_change_cost, J_EGO_COST]
      constraint_cost_weights = [LIMIT_COST, LIMIT_COST, LIMIT_COST, DANGER_ZONE_COST]
    elif self.mode == 'blended':
      a_change_cost = 40.0 if prev_accel_constraint else 0
      # 设置代价计算中，每一步与6个yref的二次型的权重Q
      cost_weights = [0., 0.1, 0.2, 5.0, a_change_cost, 1.0]
      # 设置soft constraint的系数，突破softconstraint的值会乘以该系数作为cost
      # LIMIT_COST是1e6，代价很大。constraint_cost_weights[3]被作用于安全距离，系数50，表示惩罚代价相对较小
      constraint_cost_weights = [LIMIT_COST, LIMIT_COST, LIMIT_COST, 50.0]
    else:
      raise NotImplementedError(f'Planner mode {self.mode} not recognized in planner cost set')
    self.set_cost_weights(cost_weights, constraint_cost_weights)

  def set_cur_state(self, v, a):
    v_prev = self.x0[1]
    self.x0[1] = v
    self.x0[2] = a
    if abs(v_prev - v) > 2.:  # probably only helps if v < v_prev
      for i in range(0, N+1):
        self.solver.set(i, 'x', self.x0)

  # 该静态方法使用当前时刻下的，前车的位置、速度、加速度、加速度衰减系数这4个标量信息，
  # 计算前车未来的13个时刻的位置、速度、加速度，返回的是一个13x2的矩阵，第一列是位置，第二列是速度
  @staticmethod
  def extrapolate_lead(x_lead, v_lead, a_lead, a_lead_tau):
    # a_lead_tau是一个加速度衰减系数，使用以下公式计算前车未来的13个时刻加速度
    a_lead_traj = a_lead * np.exp(-a_lead_tau * (T_IDXS**2)/2.)
    # 计算前车未来的13个时刻速度
    v_lead_traj = np.clip(v_lead + np.cumsum(T_DIFFS * a_lead_traj), 0.0, 1e8)
    # 计算前车未来的13个时刻的纵向距离
    x_lead_traj = x_lead + np.cumsum(T_DIFFS * v_lead_traj)
    # 将前车未来的13个时刻的纵向位移和速度组合成一个13行2列的矩阵，每行分别表示一个时刻的纵向距离和速度
    lead_xv = np.column_stack((x_lead_traj, v_lead_traj))
    return lead_xv

  # 
  def process_lead(self, lead):
    v_ego = self.x0[1]
    # 若前车信息有效，则使用前车的位置、速度、加速度、加速度衰减系数这4个标量信息
    if lead is not None and lead.status:
      x_lead = lead.dRel
      v_lead = lead.vLead
      a_lead = lead.aLeadK
      a_lead_tau = lead.aLeadTau
    # 若前车信息无效，则使用一些默认值，模拟一个50m远，车速比自车快10m/s的前车
    # 这样做可以让MPC用同一套逻辑来进行最优化
    else:
      # Fake a fast lead car, so mpc can keep running in the same mode
      x_lead = 50.0
      v_lead = v_ego + 10.0
      a_lead = 0.0
      a_lead_tau = _LEAD_ACCEL_TAU

    # min_x_lead = 平均速度 * 两车速度差 / 2倍最大减速度，含义是，
    # 若自车以2倍最大减速度刹车，以与前车速度达到相同作为减速目标，所需要的最小距离。低于这个距离将会碰撞
    # 这里进行clip操作目的是为了MPC求解不会出错，并不会影响FCW碰撞预警
    # MPC will not converge if immediate crash is expected
    # Clip lead distance to what is still possible to brake for
    min_x_lead = ((v_ego + v_lead)/2) * (v_ego - v_lead) / (-MIN_ACCEL * 2)
    x_lead = clip(x_lead, min_x_lead, 1e8)
    v_lead = clip(v_lead, 0.0, 1e8)
    a_lead = clip(a_lead, -10., 5.)
    # 使用加速度衰减因子，预计出前车未来的13个时刻的纵向距离（相对于当前自车位置）、速度
    lead_xv = self.extrapolate_lead(x_lead, v_lead, a_lead, a_lead_tau)
    return lead_xv

  def set_accel_limits(self, min_a, max_a):
    # TODO this sets a max accel limit, but the minimum limit is only for cruise decel
    # needs refactor
    self.cruise_min_a = min_a
    self.max_a = max_a

  def update(self, carstate, radarstate, v_cruise, x, v, a, j):
    v_ego = self.x0[1]
    self.status = radarstate.leadOne.status or radarstate.leadTwo.status

    lead_xv_0 = self.process_lead(radarstate.leadOne)
    lead_xv_1 = self.process_lead(radarstate.leadTwo)

    # lead_xv_0/1 是前车未来的13个时刻的纵向距离（相对于当前自车位置）、速度，
    # 这里假设前车在第n个时刻开始以COMFORT_BRAKE减速刹停（n in 0..13），则可以得到13个刹停后距离当前时刻自车的距离
    # 也就是把前方移动的障碍物，转换成了距离更远一些的静止的障碍物，方便后续的MPC求解
    # To estimate a safe distance from a moving lead, we calculate how much stopping
    # distance that lead needs as a minimum. We can add that to the current distance
    # and then treat that as a stopped car/obstacle at this new distance.
    lead_0_obstacle = lead_xv_0[:,0] + get_stopped_equivalence_factor(lead_xv_0[:,1])
    lead_1_obstacle = lead_xv_1[:,0] + get_stopped_equivalence_factor(lead_xv_1[:,1])

    # 设置代价计算中需要用到的，最小加速度、最大加速度，这2个参数
    self.params[:,0] = MIN_ACCEL
    self.params[:,1] = self.max_a

    # 以下只考虑blended模式
    # Update in ACC mode or ACC/e2e blend
    if self.mode == 'acc':
      self.params[:,5] = LEAD_DANGER_FACTOR

      # Fake an obstacle for cruise, this ensures smooth acceleration to set speed
      # when the leads are no factor.
      v_lower = v_ego + (T_IDXS * self.cruise_min_a * 1.05)
      v_upper = v_ego + (T_IDXS * self.max_a * 1.05)
      v_cruise_clipped = np.clip(v_cruise * np.ones(N+1),
                                 v_lower,
                                 v_upper)
      cruise_obstacle = np.cumsum(T_DIFFS * v_cruise_clipped) + get_safe_obstacle_distance(v_cruise_clipped)
      x_obstacles = np.column_stack([lead_0_obstacle, lead_1_obstacle, cruise_obstacle])
      self.source = SOURCES[np.argmin(x_obstacles[0])]

      # These are not used in ACC mode
      x[:], v[:], a[:], j[:] = 0.0, 0.0, 0.0, 0.0

    elif self.mode == 'blended':
      # 设置lead_danger_factor系数，比ACC模式的0.75更大，这样就会更加保守
      self.params[:,5] = 1.0

      x_obstacles = np.column_stack([lead_0_obstacle,
                                     lead_1_obstacle])
      # 使用司机设定的巡航速度v_cruise，来计算若按照这个速度行驶，未来13个时刻的自车距离
      cruise_target = T_IDXS * np.clip(v_cruise, v_ego - 2.0, 1e3) + x[0]
      # 使用调用者传进来的规划结果，计算未来12个时刻路点的自车距离间隔，使用每两个路点的平均速度，乘以两个路点之间的时间差
      xforward = ((v[1:] + v[:-1]) / 2) * (T_IDXS[1:] - T_IDXS[:-1])
      # 累加后，得到未来13个时刻的自车距离
      x = np.cumsum(np.insert(xforward, 0, x[0]))

      # 把规划结果中未来13个时刻的自车距离，和未来13个时刻的巡航距离，进行比较，取最小值
      x_and_cruise = np.column_stack([x, cruise_target])
      x = np.min(x_and_cruise, axis=1)
      # 设置source，表示速度限制来自e2e规划，还是巡航速度，该信息后续只用于UI显示
      self.source = 'e2e' if x_and_cruise[1,0] < x_and_cruise[1,1] else 'cruise'

    else:
      raise NotImplementedError(f'Planner mode {self.mode} not recognized in planner update')

    # 设置N步的目标值, yref[:, 0]与yref[:, 4]使用初始化的默认值0，MPC求解会用yref来计算代价
    self.yref[:,1] = x
    self.yref[:,2] = v
    self.yref[:,3] = a
    self.yref[:,5] = j
    # 设置MPC求解的N+1步的目标值
    for i in range(N):
      self.solver.set(i, "yref", self.yref[i])
    self.solver.set(N, "yref", self.yref[N][:COST_E_DIM])

    # 设置代价计算中需要用到的，障碍物距离、上一次求解时的自车加速度、跟车时间（标量） 这3个参数
    # x_obstacles是未来13个时刻的障碍物距离，prev_a是上一次MPC求解结果的未来13个时刻的自车加速度
    self.params[:,2] = np.min(x_obstacles, axis=1)
    self.params[:,3] = np.copy(self.prev_a)
    self.params[:,4] = T_FOLLOW  # 1.45s
    
    # 调用MPC求解
    self.run()

    # 使用求解后的未来N步的纵向位移 x_sol，等信息来计算是否可能产生碰撞，crash_cnt>2则会触发FCW警告
    # FCW_IDXS是未来2秒内的时间步索引
    if (np.any(lead_xv_0[FCW_IDXS,0] - self.x_sol[FCW_IDXS,0] < CRASH_DISTANCE) and
            radarstate.leadOne.modelProb > 0.9):
      self.crash_cnt += 1
    else:
      self.crash_cnt = 0

    # 设置source，表示和哪个前车信息可能产生不舒适的刹车，该信息后续只用于UI显示
    # Check if it got within lead comfort range
    # TODO This should be done cleaner
    if self.mode == 'blended':
      if any((lead_0_obstacle - get_safe_obstacle_distance(self.x_sol[:,1], T_FOLLOW))- self.x_sol[:,0] < 0.0):
        self.source = 'lead0'
      if any((lead_1_obstacle - get_safe_obstacle_distance(self.x_sol[:,1], T_FOLLOW))- self.x_sol[:,0] < 0.0) and \
         (lead_1_obstacle[0] - lead_0_obstacle[0]):
        self.source = 'lead1'

  def run(self):
    # t0 = sec_since_boot()
    # reset = 0
    for i in range(N+1):
      self.solver.set(i, 'p', self.params[i])
    self.solver.constraints_set(0, "lbx", self.x0)
    self.solver.constraints_set(0, "ubx", self.x0)

    self.solution_status = self.solver.solve()
    self.solve_time = float(self.solver.get_stats('time_tot')[0])
    self.time_qp_solution = float(self.solver.get_stats('time_qp')[0])
    self.time_linearization = float(self.solver.get_stats('time_lin')[0])
    self.time_integrator = float(self.solver.get_stats('time_sim')[0])

    # qp_iter = self.solver.get_stats('statistics')[-1][-1] # SQP_RTI specific
    # print(f"long_mpc timings: tot {self.solve_time:.2e}, qp {self.time_qp_solution:.2e}, lin {self.time_linearization:.2e}, integrator {self.time_integrator:.2e}, qp_iter {qp_iter}")
    # res = self.solver.get_residuals()
    # print(f"long_mpc residuals: {res[0]:.2e}, {res[1]:.2e}, {res[2]:.2e}, {res[3]:.2e}")
    # self.solver.print_statistics()

    for i in range(N+1):
      self.x_sol[i] = self.solver.get(i, 'x')
    for i in range(N):
      self.u_sol[i] = self.solver.get(i, 'u')

    self.v_solution = self.x_sol[:,1]
    self.a_solution = self.x_sol[:,2]
    self.j_solution = self.u_sol[:,0]

    # 准备下一轮MPC求解要用的prev_a，下一轮将于0.05秒后进行，所以用这种方式对齐
    self.prev_a = np.interp(T_IDXS + 0.05, T_IDXS, self.a_solution)

    t = sec_since_boot()
    if self.solution_status != 0:
      if t > self.last_cloudlog_t + 5.0:
        self.last_cloudlog_t = t
        cloudlog.warning(f"Long mpc reset, solution_status: {self.solution_status}")
      self.reset()
      # reset = 1
    # print(f"long_mpc timings: total internal {self.solve_time:.2e}, external: {(sec_since_boot() - t0):.2e} qp {self.time_qp_solution:.2e}, lin {self.time_linearization:.2e} qp_iter {qp_iter}, reset {reset}")


if __name__ == "__main__":
  ocp = gen_long_ocp()
  AcadosOcpSolver.generate(ocp, json_file=JSON_FILE)
  # AcadosOcpSolver.build(ocp.code_export_directory, with_cython=True)
