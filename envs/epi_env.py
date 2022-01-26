import numpy as np
import gym
from gym.spaces import Box


def gradual_compliance_weights(t, beta_0, beta_1):
    x = beta_0 + beta_1*t
    w1 = np.minimum(1, np.exp(x)/(1+np.exp(x)))
    w0 = 1-w1
    return w0, w1


class EpiEnv(gym.Env):

    def __init__(self, model, C=None, beta_0=None, beta_1=None, datapoints=None):
        super(EpiEnv, self).__init__()
        # under the hood we run this epi model
        self.model = model
        # population size of each age-group is sum of the compartments
        N = self.model.init_state.sum(axis=0)
        self.K = len(N)
        # contact matrix
        self.C = np.ones((1, K, K)) if C is None else C
        # factors of contact matrix for symptomatic people, reshape to match C shape
        self.C_sym_factor = np.array([1., 0.09, 0.13, 0.09, 0.06, 0.25])[:, None, None]

        self.beta_0 = beta_0
        self.beta_1 = beta_1

        # the observation space are compartments x age_groups
        self.observation_space = Box(low=np.zeros((model.n_comp, self.K)), high=np.tile(N, (model.n_comp, 1)), dtype=np.float32)
        # action space is proportional reduction of work, school, leisure
        self.action_space = Box(low=np.zeros(3), high=np.ones(3), dtype=np.float32)
        # reward_space is attack-rate for infections, hospitalizations and reduction in social contact
        self.reward_space = Box(low=np.zeros(3), high=np.array([N.sum(), N.sum(), 1]), dtype=np.float32)

        self.datapoints = datapoints

    def reset(self):
        self.model.current_state = self.model.init_state.copy()
        self.current_C = self.C
        return self.model.current_state

    def step(self, action):
        # action is a 3d continuous vector
        p_w, p_s, p_l = action
        
        # match all C components, reshape to match C shape
        p = np.array([1, p_w, p_w, p_s, p_l, p_l])[:, None, None]
        C_target = self.C*p

        s = self.model.current_state

        # simulate for a whole week, sum the daily rewards
        r_ari = r_arh = r_sr = 0.
        state_n = np.empty((7,) + self.observation_space.shape)
        for day in range(7):
            # gradual compliance, C_target is only reached after a number of days
            w0, w1 = gradual_compliance_weights(day, self.beta_0, self.beta_1)
            
            C = self.current_C*w0 + C_target*w1
            C_asym = C.sum(axis=0)
            C_sym = (C*self.C_sym_factor).sum(axis=0)

            s_n = self.model.simulate_day(C_asym, C_sym)
            state_n[day] = s_n
            # attack rate infected
            S_s = s[self.model.S]
            S_s_n = s_n[self.model.S]
            r_ari += -(np.sum(S_s) - np.sum(S_s_n))
            # TODO attack rate hospitalization
            r_arh += 0.
            # reduction in social contact
            R_s_n = s_n[self.model.R]
            # all combinations of age groups
            i, j = np.meshgrid(range(self.K), range(self.K))
            r_sr += (C_asym*S_s_n[i]*S_s_n[j] + C_asym*R_s_n[i]*R_s_n[j]).sum()
            # update state
            s = s_n

        # update current contact matrix
        self.current_C = C_target

        # next-state, reward, terminal?, info
        return state_n, np.array([r_ari, r_arh, r_sr]), False, {}