from gym.envs.registration import register
from gym.wrappers import TimeLimit
from envs.epi_env import EpiEnv
import numpy as np

# temp while waiting for true model
class OdeEpiModel(object):
    
    def __init__(self, K, init_model_state, beta, gamma):
        #age groups
        self.K = K
        #initial model state
        self.init_model_state = init_model_state
        #current model state
        self.current_model_state = init_model_state
        #parameters
        self.beta = beta
        self.gamma = gamma

        #number of compartments in each age groups
        self.n_comp = 3

    def S(self, k):
        return (self.n_comp*k) + 0

    def I(self, k):
        return (self.n_comp*k) + 1

    def R(self, k):
        return (self.n_comp*k) + 2

    #C is the contact matrix to be used,
    #would be used in beta
    def simulate_day(self, C):
        return self.current_model_state


def epi_ode(steps=16):
    # run for `steps` weeks
    # model with 2 age groups
    model = OdeEpiModel(2, np.array([1000, 0, 0., 1000, 0, 0]), .2, .2)
    env = EpiEnv(model)
    env = TimeLimit(env, steps)
    return env

# split_env
register(
    id='EpiODEContinuous-v0',
    entry_point='envs:epi_ode',
    kwargs={'steps': 16}
    )