import sys
import os
sys.path.append(os.getcwd())

if __name__ == '__main__':
    import gym
    import envs
    import numpy as np
    
    env = gym.make('EpiODEContinuous-v0')
    s = env.reset()
    d = False
    while not d:
        a = env.action_space.sample()
        s, r, d, _ = env.step(a)
        print(s, r)