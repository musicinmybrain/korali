#!/user/bin/env python3

##  Copyright (c) 2018 CSE-Lab, ETH Zurich, Switzerland. All rights reserved.
##  Distributed under the terms of the MIT license.
##
##  Created by Daniel Waelchli (wadaniel@ethz.ch).

import math
from math import sin, cos
import numpy as np, sys
from scipy.integrate import ode


class SinglePendulum:
  def __init__(self):
    self.dt = 0.02
    self.step=0
    # x, th1, xdot, th1dot,
    self.u = np.asarray([0, 0, 0, 0 ])
    self.F=0
    self.t=0
    self.ODE = ode(self.system).set_integrator('dopri5')

  def reset(self):
    self.u = np.random.uniform(-0.05, 0.05, 4)
    self.u[1] += math.pi # start from bottom
    self.step = 0
    self.F = 0
    self.t = 0

  def isFailed(self):
    return (abs(self.u[0])>7.0)

  def isOver(self): # is episode over
    return self.isFailed()

  def isTruncated(self):  # check that cause for termination is time limits
    return (abs(self.u[0])<=7.0)

  """Based on 'Swing-up Control of a Single Inverted Pendulum
  on a Cart With Input and Output Constraints' [Meta Tum, et. al.]"""

  @staticmethod
  def system(t, y, fact): #dynamics function
    #comi: mass of link i
    #lci: half length of links
    #g: gravitational constant
    com1, lc1, J1, c, g = 0.41, 0.22, 0.116, 0.005, 9.81

    # simplify
    x = y[0]
    th1 = y[1]
    xdot = y[2]
    th1dot = y[3]

    qdot = np.array([xdot, th1dot])
    th1dotdot = (com1*g*lc1*sin(th1)-c*th1dot)/(J1+com1*lc1*lc1) - com1*lc1*cos(th1)/(J1+com1*lc1*lc1)*fact
    
    # xdot, th1dot, xdotdot, th1dotdot
    return np.array([xdot, th1dot, fact, th1dotdot]) 
    
  def wrapToNPiPi(self, rad):
    return (rad + np.pi) % (2 * np.pi) - np.pi

  def advance(self, action):
    
    self.F = action[0]
    #self.u += self.dt*self.system(self.t, self.u, self.F)
    self.ODE.set_initial_value(self.u, self.t).set_f_params(self.F)
    self.u = self.ODE.integrate(self.t + self.dt)
    self.u[1] = self.wrapToNPiPi(self.u[1])
    self.u[2] = self.wrapToNPiPi(self.u[2])
 
    self.t = self.t + self.dt
    self.step = self.step + 1
    #print("{}: {}".format(self.step,self.u), flush=True)
    if self.isOver(): return 1
    else: return 0

  def getState(self):
    state = np.zeros(5)
    state[0] = np.copy(self.u[0])
    state[1] = np.copy(self.u[1])
    state[2] = np.copy(self.u[2])
    state[3] = np.copy(self.u[3])
    state[4] = cos(state[1])

    # maybe transform state 
    # ..
    # ..
    #print(self.F)
    #print(state, flush=True)
    
    return state

  def getReward(self):
    th1 = np.copy(self.u[1])
    return 1 + cos(th1)

if __name__ == '__main__':
    print("init..")
    sw = SinglePendulum()
    sp.reset()
    state = sp.getState()
    print("state:")
    print(state)
    sp.advance([1.0])
    state = sp.getState()
    print("state after one step:")
    print(state)
    print("exit.. BYE!")
