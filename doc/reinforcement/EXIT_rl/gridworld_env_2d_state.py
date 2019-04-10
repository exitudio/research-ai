import sys
if "../" not in sys.path:
  sys.path.append("../") 
from lib.envs.gridworld import GridworldEnv
import numpy as np
import math

class GridworldEnv2DState(GridworldEnv):
    def __init__(self, shape=[4, 4]):
        super().__init__(shape)
        
    def convert_to_2_dimension_state(self, state):
        return np.array([math.floor(state/4), state%4], dtype=int)
    
    def reset(self):
        return self.convert_to_2_dimension_state(super(GridworldEnv, self).reset())
    
    def step(self, action):
        state, reward, done, info = super(GridworldEnv, self).step(action)
        state = self.convert_to_2_dimension_state(state)
        return state, reward, done, info