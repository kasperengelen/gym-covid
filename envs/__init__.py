from gym.envs.registration import register
from envs.epi_env import EpiEnv
from envs.ode_sir_model import OdeEpiModel
import numpy as np



def epi_ode():
    # run for `steps` weeks
    # model with 2 age groups
    model = OdeEpiModel(10000000, 2, np.array([7500000-1, 1, 0., 2500000, 1, 0]), 0.05, 1/3)
    env = EpiEnv(model)
    return env


# split_env
register(
    id='EpiODEContinuous-v0',
    entry_point='envs:epi_ode',
    )