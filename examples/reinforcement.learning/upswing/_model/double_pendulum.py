#!/user/bin/env python3

##  Copyright (c) 2018 CSE-Lab, ETH Zurich, Switzerland. All rights reserved.
##  Distributed under the terms of the MIT license.
##
##  Created by Daniel Waelchli (wadaniel@ethz.ch).

import math
from math import sin, cos
import numpy as np, sys
from scipy.integrate import ode


class DoublePendulum:
  def __init__(self):
    self.dt = 0.01
    self.step=0
    # x, th1, th2, xdot, th1dot, th2dot
    self.u = np.asarray([0, 0, 0, 0, 0, 0])
    self.F=0
    self.t=0
    self.ODE = ode(self.system).set_integrator('dopri5')

  def reset(self):
    self.u = np.random.uniform(-0.05, 0.05, 6)
    self.u[1] += math.pi # start from bottom
    self.u[2] += math.pi
    self.step = 0
    self.F = 0
    self.t = 0

  def isFailed(self):
    return (abs(self.u[0])>7.0)

  def isOver(self): # is episode over
    return self.isFailed()

  def isTruncated(self):  # check that cause for termination is time limits
    return (abs(self.u[0])<=7.0)

  """Based on 'Control Design and Analysis for 
  Underactuated Tobotic Systems [Xin, Liu] (Chapter14)"""

  @staticmethod
  def system(t, y, fact): #dynamics function
    #mc: mass cart
    #comi: center of mass of link i (at li/2)
    #li: length of links
    #g: gravitational constant
    mc, com1, com2, l1, l2, g = 1.0, 0.5, 0.5, 1.0, 1.0, 9.81

    # internal params
    lc1, lc2 = 0.5*l1, 0.5*l2
    J1 = com1*lc1*lc1/3
    J2 = com2*lc2*lc2/3
    a0 = com1 + com2 + mc
    a1 = J1+com1*lc1*lc1+com2*lc2*lc2
    a2 = J2+com2*lc2*lc2
    a3 = com2*l1*lc2
    b1 = (com1*lc1+com2*l1)*g
    b2 = com2*lc2*g

    # simplify
    x = y[0]
    th1 = y[1]
    th2 = y[2]
    xdot = y[3]
    th1dot = y[4]
    th2dot = y[5]

    qdot = np.array([xdot, th1dot, th2dot])
    
    B = np.array([1, 0, 0])

    M = np.array([
        [a0, b1/g*cos(th1), b2/g*cos(th2)], 
        [b1/g*cos(th1), a1, a3*cos(th1-th2)], 
        [b2/g*cos(th2), a3*cos(th1-th2), a2]])

    C = np.array([
        [0.0, -b1/g*th1dot*sin(th1), -b2/g*th2dot*sin(th2)],
        [0.0, 0.0, a3*th2dot*sin(th1-th2)],
        [0.0, -a3*th1dot*sin(th1-th2), 0]])

    G = np.array([0.0, -b1*sin(th1), -b2*sin(th2)])
    
    RHS = np.linalg.solve(M, B*fact-G-np.matmul(C,qdot))
    
    # xdot, th1dot, th2dot, xdotdot, th1dotdot, th2dotdot
    return np.concatenate((qdot,RHS))
    
  def wrapToNPiPi(self, rad):
    return (rad + np.pi) % (2 * np.pi) - np.pi

  def advance(self, action):
    
    self.F = action[0]
    self.u += self.dt*self.system(self.t, self.u, self.F)
    #self.ODE.set_initial_value(self.u, self.t).set_f_params(self.F)
    #self.u = self.ODE.integrate(self.t + self.dt)
    self.u[1] = self.wrapToNPiPi(self.u[1])
    self.u[2] = self.wrapToNPiPi(self.u[2])
 
    self.t = self.t + self.dt
    self.step = self.step + 1
    
    if self.isOver(): 
        return 1
    else: 
        return 0

  def getState(self):
    state = np.zeros(7)
    state[0] = np.copy(self.u[0])
    state[1] = np.copy(self.u[1])
    state[2] = np.copy(self.u[2]) 
    state[3] = np.copy(self.u[3])
    state[4] = np.copy(self.u[4])
    state[5] = np.copy(self.u[3])
    state[6] = cos(state[1])+cos(state[2])

    # maybe transform state 
    # ..
    # ..
    
    assert np.any(np.isfinite(state) == False) == False, "State is nan: {}".format(state)
    
    return state

  def getReward(self):

    th1 = np.copy(self.u[1])
    th2 = np.copy(self.u[2])
#    return 2 + cos(th1) + cos(th2) - 200.*float(self.isFailed())
    return 2 + cos(th1) + cos(th2) 

if __name__ == '__main__':
    print("init..")
    dp = DoublePendulum()
    dp.reset()
    state = dp.getState()
    print("state:")
    print(state)
    dp.advance([1.0, 0.0, 0.0])
    state = dp.getState()
    print("state after one step:")
    print(state)
    print("exit.. BYE!")
