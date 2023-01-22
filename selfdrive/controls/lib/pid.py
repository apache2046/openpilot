import numpy as np
from numbers import Number

from common.numpy_fast import clip, interp


class PIDController():
  # 初始化参数的 k_p, k_i, k_d 三个参数可以不是Number类型，而是2xN的数组，用来提供在不同车速下的PID参数
  # rate 是pid的调用频率，作用是给积分部分提供dt
  def __init__(self, k_p, k_i, k_f=0., k_d=0., pos_limit=1e308, neg_limit=-1e308, rate=100):
    self._k_p = k_p
    self._k_i = k_i
    self._k_d = k_d
    self.k_f = k_f   # feedforward gain
    if isinstance(self._k_p, Number):
      self._k_p = [[0], [self._k_p]]
    if isinstance(self._k_i, Number):
      self._k_i = [[0], [self._k_i]]
    if isinstance(self._k_d, Number):
      self._k_d = [[0], [self._k_d]]

    self.pos_limit = pos_limit
    self.neg_limit = neg_limit

    self.i_unwind_rate = 0.3 / rate
    self.i_rate = 1.0 / rate
    self.speed = 0.0

    self.reset()

  @property
  def k_p(self):
    return interp(self.speed, self._k_p[0], self._k_p[1])

  @property
  def k_i(self):
    return interp(self.speed, self._k_i[0], self._k_i[1])

  @property
  def k_d(self):
    return interp(self.speed, self._k_d[0], self._k_d[1])

  @property
  def error_integral(self):
    return self.i/self.k_i

  def reset(self):
    self.p = 0.0
    self.i = 0.0
    self.d = 0.0
    self.f = 0.0
    self.control = 0

  def update(self, error, error_rate=0.0, speed=0.0, override=False, feedforward=0., freeze_integrator=False):
    # 读取当前车速到self.speed，后续k_p,k_i,k_d属性的计算中会用到当前车速来做插值
    self.speed = speed

    # 读取k_p, k_f, k_d 属性（已根据速度做插值），来计算p, f, d
    self.p = float(error) * self.k_p
    self.f = feedforward * self.k_f
    self.d = error_rate * self.k_d

    if override:
      # override 为true时表示用户接管了方向盘（横向控制中），此时不能继续让误差进行积分，而需要将已有积分值缓慢的衰减到0
      # 为何只是去掉了积分部分而保留了pd部分，可能是依然给一个控制扭矩输出的反馈，使用户能感觉到openpilot此时的控制意图，
      # 但又不会使的误差积累的太大
      self.i -= self.i_unwind_rate * float(np.sign(self.i))
    else:
      # 计算新的积分值临时变量 i，并用它计算控制量
      i = self.i + error * self.k_i * self.i_rate
      control = self.p + i + self.d + self.f

      # 检查控制量是否越界，若越界或调用者禁止积分，或误差的方向与积分值方向不一样，则不更新self.i
      # Update when changing i will move the control away from the limits
      # or when i will move towards the sign of the error
      if ((error >= 0 and (control <= self.pos_limit or i < 0.0)) or
          (error <= 0 and (control >= self.neg_limit or i > 0.0))) and \
         not freeze_integrator:
        self.i = i

    # 重新计算控制量，使用了self.i而不是i
    control = self.p + self.i + self.d + self.f

    self.control = clip(control, self.neg_limit, self.pos_limit)
    return self.control
