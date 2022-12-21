# State-reduced wrapper for Ant environments
# Created by Sergio Martin, adapted from HumanoidWrapper by Guido Novati

import gym, numpy as np

# This wrapper only shrinks the state space, eliminating variables that are
# always 0. By definition it does not affect the RL in any way except by making
# network forward/backward prop faster.
# Again, because the state vars omitted by this wrapper are always 0, the RL
# task itself is neither harder nor easier. Just cheaper to run.
# These are the indices of state variables actually used by OpenAI gym Humanoid:
INDS=[  0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16, 17,
       18, 19, 20, 21, 22, 23, 24, 25, 26 ] 

class AntWrapper():
  def __init__(self, env):
    self.env = env
    assert(self.env.observation_space.shape[0] == 111)
    assert(len(self.env.observation_space.shape) == 1)
    self.env.observation_space.shape = [ 27 ]

  def reset(self):
    observation = self.env.reset()
    return observation[INDS]

  @property
  def action_space(self):
    return self.env.action_space

  @property
  def observation_space(self):
    return self.env.observation_space
    
  @property
  def _max_episode_steps(self):
    return self.env._max_episode_steps

  def step(self, action):
    observation, reward, done, info = self.env.step(action)
    return observation[INDS], reward, done, info

  def render(self, mode):
    return self.env.render(mode=mode)

  def close(self):
    return self.env.close()

