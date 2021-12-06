import sys
import os
sys.path.append(os.getcwd())

if __name__ == '__main__':
    from ode_epi_model import OdeEpiModel
    from envs.epi_env import EpiEnv
    import numpy as np

    model = OdeEpiModel(2, np.array([1000, 0, 0., 1000, 0, 0]), .2, .2)
    
    env = EpiEnv(model)
    s = env.reset()
    print(s)
    breakpoint()
    for _ in range(3):
        a = env.action_space.sample()
        s, r, d, _ = env.step(a)
        print(s, r)