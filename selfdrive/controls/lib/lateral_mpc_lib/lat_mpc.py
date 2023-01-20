#!/usr/bin/env python3
import os
import numpy as np

from casadi import SX, vertcat, sin, cos

from common.realtime import sec_since_boot
from selfdrive.modeld.constants import T_IDXS

if __name__ == '__main__':  # generating code
  from pyextra.acados_template import AcadosModel, AcadosOcp, AcadosOcpSolver
else:
  from selfdrive.controls.lib.lateral_mpc_lib.c_generated_code.acados_ocp_solver_pyx import AcadosOcpSolverCython  # pylint: disable=no-name-in-module, import-error

LAT_MPC_DIR = os.path.dirname(os.path.abspath(__file__))
EXPORT_DIR = os.path.join(LAT_MPC_DIR, "c_generated_code")
JSON_FILE = os.path.join(LAT_MPC_DIR, "acados_ocp_lat.json")
X_DIM = 4
P_DIM = 2
N = 16
COST_E_DIM = 3
COST_DIM = COST_E_DIM + 2
SPEED_OFFSET = 10.0
MODEL_NAME = 'lat'
ACADOS_SOLVER_TYPE = 'SQP_RTI'

#定义模型
def gen_lat_model():
  model = AcadosModel()
  model.name = MODEL_NAME

  # set up states
  x_ego = SX.sym('x_ego')
  y_ego = SX.sym('y_ego')
  psi_ego = SX.sym('psi_ego')
  psi_rate_ego = SX.sym('psi_rate_ego')
  # 模型的状态有4个分量，[x, y, psi, psi_dot]
  model.x = vertcat(x_ego, y_ego, psi_ego, psi_rate_ego)

  # parameters
  v_ego = SX.sym('v_ego')
  rotation_radius = SX.sym('rotation_radius')
  # 求解时需要再给定的参数[v, rotation_radius]
  model.p = vertcat(v_ego, rotation_radius)

  # controls
  psi_accel_ego = SX.sym('psi_accel_ego')
  # 模型的控制量只有1个参数，psi_dot_dot
  model.u = vertcat(psi_accel_ego)

  # xdot
  # 定义状态4个分量的符号和xdot
  x_ego_dot = SX.sym('x_ego_dot')
  y_ego_dot = SX.sym('y_ego_dot')
  psi_ego_dot = SX.sym('psi_ego_dot')
  psi_rate_ego_dot = SX.sym('psi_rate_ego_dot')

  model.xdot = vertcat(x_ego_dot, y_ego_dot, psi_ego_dot, psi_rate_ego_dot)

  # dynamics model
  # 模型的动态方程，4行分别代表xdot的4个分量的计算方式，[x_dot, y_dot, psi_dot, psi_dot_dot]
  f_expl = vertcat(v_ego * cos(psi_ego) - rotation_radius * sin(psi_ego) * psi_rate_ego,
                   v_ego * sin(psi_ego) + rotation_radius * cos(psi_ego) * psi_rate_ego,
                   psi_rate_ego,
                   psi_accel_ego)
  model.f_expl_expr = f_expl
  # 按照acados的要求， f_impl_expr在integrator_type为IRK时是必须要定义的，否则可默认为None
  model.f_impl_expr = model.xdot - f_expl
  return model


def gen_lat_ocp():
  # 定义ocp (optimal control problem)
  ocp = AcadosOcp()
  ocp.model = gen_lat_model()

  Tf = np.array(T_IDXS)[N]

  # set dimensions
  # horizon分成N步，16步
  ocp.dims.N = N

  # set cost module
  # 代价函数采用非线性模式
  ocp.cost.cost_type = 'NONLINEAR_LS'
  # 终止状态的代价函数采用非线性模式
  ocp.cost.cost_type_e = 'NONLINEAR_LS'

  Q = np.diag(np.zeros(COST_E_DIM))
  QR = np.diag(np.zeros(COST_DIM))

  # 初始化代价函数的系数
  ocp.cost.W = QR
  ocp.cost.W_e = Q

  y_ego, psi_ego, psi_rate_ego = ocp.model.x[1], ocp.model.x[2], ocp.model.x[3]
  psi_rate_ego_dot = ocp.model.u[0]
  v_ego = ocp.model.p[0]

  ocp.parameter_values = np.zeros((P_DIM, ))

  ocp.cost.yref = np.zeros((COST_DIM, ))
  ocp.cost.yref_e = np.zeros((COST_E_DIM, ))
  # Add offset to smooth out low speed control
  # TODO unclear if this right solution long term
  v_ego_offset = v_ego + SPEED_OFFSET
  # TODO there are two costs on psi_rate_ego_dot, one
  # is correlated to jerk the other to steering wheel movement
  # the steering wheel movement cost is added to prevent excessive
  # wheel movements
  # 中间状态的代价函数，产生5个分量，[y, psi*v, psi_dot*v, psi_dot_dot*v, psi_dot_dot / v]
  # 和later_planer里self.lat_mpc.set_weights(PATH_COST, LATERAL_MOTION_COST... 是对应的
  ocp.model.cost_y_expr = vertcat(y_ego,
                                  v_ego_offset * psi_ego,
                                  v_ego_offset * psi_rate_ego,
                                  v_ego_offset * psi_rate_ego_dot,
                                  psi_rate_ego_dot / (v_ego + 0.1))
  # 终止状态的代价函数，产生3个分量，去掉了2个和u相关的代价                                  
  ocp.model.cost_y_expr_e = vertcat(y_ego,
                                   v_ego_offset * psi_ego,
                                   v_ego_offset * psi_rate_ego)

  # set constraints
  ocp.constraints.constr_type = 'BGH'
  # constraint这里限制x里的第2、3项，即psi, psi_dot
  ocp.constraints.idxbx = np.array([2,3])
  # 设置psi, psi_dot的上限
  ocp.constraints.ubx = np.array([np.radians(90), np.radians(50)])
  # 设置psi, psi_dot的下限
  ocp.constraints.lbx = np.array([-np.radians(90), -np.radians(50)])
  x0 = np.zeros((X_DIM,))
  ocp.constraints.x0 = x0

  ocp.solver_options.qp_solver = 'PARTIAL_CONDENSING_HPIPM'
  ocp.solver_options.hessian_approx = 'GAUSS_NEWTON'
  ocp.solver_options.integrator_type = 'ERK'
  ocp.solver_options.nlp_solver_type = ACADOS_SOLVER_TYPE
  ocp.solver_options.qp_solver_iter_max = 1
  ocp.solver_options.qp_solver_cond_N = 1

  # set prediction horizon
  # 设置MPC的horizon时间为2.5s
  ocp.solver_options.tf = Tf
  # 设置MPC的horizon窗口里，每个step的时刻
  ocp.solver_options.shooting_nodes = np.array(T_IDXS)[:N+1]

  ocp.code_export_directory = EXPORT_DIR
  return ocp


class LateralMpc():
  def __init__(self, x0=np.zeros(X_DIM)):
    self.solver = AcadosOcpSolverCython(MODEL_NAME, ACADOS_SOLVER_TYPE, N)
    self.reset(x0)

  def reset(self, x0=np.zeros(X_DIM)):
    self.x_sol = np.zeros((N+1, X_DIM))
    self.u_sol = np.zeros((N, 1))
    self.yref = np.zeros((N+1, COST_DIM))
    for i in range(N):
      self.solver.cost_set(i, "yref", self.yref[i])
    self.solver.cost_set(N, "yref", self.yref[N][:COST_E_DIM])

    # Somehow needed for stable init
    for i in range(N+1):
      self.solver.set(i, 'x', np.zeros(X_DIM))
      self.solver.set(i, 'p', np.zeros(P_DIM))
    self.solver.constraints_set(0, "lbx", x0)
    self.solver.constraints_set(0, "ubx", x0)
    self.solver.solve()
    self.solution_status = 0
    self.solve_time = 0.0
    self.cost = 0

  def set_weights(self, path_weight, heading_weight,
                  lat_accel_weight, lat_jerk_weight,
                  steering_rate_weight):
    # 把传入的不同代价系数组成对角矩阵
    W = np.asfortranarray(np.diag([path_weight, heading_weight,
                                   lat_accel_weight, lat_jerk_weight,
                                   steering_rate_weight]))
    # 设置中间状态的代价函数的系数                                       
    for i in range(N):
      self.solver.cost_set(i, 'W', W)
    # 设置终止状态的代价函数的系数
    self.solver.cost_set(N, 'W', W[:COST_E_DIM,:COST_E_DIM])

  def run(self, x0, p, y_pts, heading_pts, yaw_rate_pts):
    x0_cp = np.copy(x0)
    p_cp = np.copy(p)
    # 给0时刻设定x的约束，通过这种方式给模型的初始状态赋值
    self.solver.constraints_set(0, "lbx", x0_cp)
    self.solver.constraints_set(0, "ubx", x0_cp)
    # 代价函数计算后的第0个分量的目标值，包含N步
    self.yref[:,0] = y_pts
    v_ego = p_cp[0]
    # rotation_radius = p_cp[1]
    # 代价函数计算后的第1个分量的目标值，包含N步
    self.yref[:,1] = heading_pts * (v_ego + SPEED_OFFSET)
    # 代价函数计算后的第2个分量的目标值，包含N步
    self.yref[:,2] = yaw_rate_pts * (v_ego + SPEED_OFFSET)

    for i in range(N):
      # 把中间状态目标值yref设置到优化器的cost的yref属性里
      self.solver.cost_set(i, "yref", self.yref[i])
      # 设置模型状态方程和代价函数里要用到的参数p
      self.solver.set(i, "p", p_cp)
    # 设置终止时刻的p和yref
    self.solver.set(N, "p", p_cp)
    self.solver.cost_set(N, "yref", self.yref[N][:COST_E_DIM])

    t = sec_since_boot()
    # 调用求解器开始计算该最优化问题的解
    self.solution_status = self.solver.solve()
    self.solve_time = sec_since_boot() - t

    # 把最优化结果的状态量x的N+1步的轨迹取出
    for i in range(N+1):
      self.x_sol[i] = self.solver.get(i, 'x')
    # 把最优化结果的控制量u的N步的轨迹取出
    for i in range(N):
      self.u_sol[i] = self.solver.get(i, 'u')
    # 把最优化结果的所计算出的代价取出
    self.cost = self.solver.get_cost()


if __name__ == "__main__":
  ocp = gen_lat_ocp()
  AcadosOcpSolver.generate(ocp, json_file=JSON_FILE)
  # AcadosOcpSolver.build(ocp.code_export_directory, with_cython=True)
