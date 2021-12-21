import numpy as np
import gym
from gym.spaces import Box


def gradual_compliance_weights(t, beta_0, beta_1):
    x = beta_0 + beta_1*t
    w1 = np.minimum(1, np.exp(x)/(1+np.exp(x)))
    w0 = 1-w1
    return w0, w1


class EpiEnv(gym.Env):

    def __init__(self, model, funcs, C=None):
        super(EpiEnv, self).__init__()
        # under the hood we run this epi model
        self.model = model
        self.funcs = funcs
        # population size of each age-group is sum of the compartments
        N = self.model.init_model_state.reshape((self.model.K, self.model.n_comp)).sum(1).repeat(self.model.n_comp)
        # contact matrix
        self.C = np.ones((1, model.K, model.K)) if C is None else C
        # factors of contact matrix for symptomatic people, reshape to match C shape
        self.C_sym_factor = np.array([1., 0.09, 0.13, 0.09, 0.06, 0.25])[:, None, None]
        # params for gradual compliance TODO load from config file
        self.beta_0 = -5
        self.beta_1 = 1.404

        # the observation space are compartments x age_groups
        self.observation_space = Box(low=np.zeros(model.n_comp*model.K), high=N, dtype=np.float32)
        # action space is proportional reduction of work, school, leisure
        self.action_space = Box(low=np.zeros(3), high=np.ones(3), dtype=np.float32)
        # reward_space is attack-rate for infections, hospitalizations and reduction in social contact
        self.reward_space = Box(low=np.zeros(3), high=np.array([N.sum(), N.sum(), 1]), dtype=np.float32)

    def init_params(self, p, lp):

        self.lockdown_params = lp
        self.params = p

        return

    def reset(self):
        self.model.current_model_state = self.model.init_model_state
        self.tstep=1
        return self.tstep, self.model.current_model_state

    def step(self, action):
        # action is a 3d continuous vector
        self.lockdown = self.lockdown_params[0]
        if self.tstep < self.lockdown:
            p_w, p_s, p_l = action
        else:
            p_w, p_s, p_l = self.lockdown_params[1], self.lockdown_params[2], self.lockdown_params[3]
        
        # match all C components, reshape to match C shape
        p = np.array([1, p_w, p_w, p_s, p_l, p_l])[:, None, None]
        C_target = self.C*p

        s = self.model.current_model_state

        # simulate for a whole week, sum the daily rewards
        r_ari = r_arh = r_sr = 0.
        # make sure we only take upper diagonal
        for __ in range(1):
            # gradual compliance, C_target is only reached after a number of days
            if self.tstep < self.lockdown:
                w0, w1 = self.params[4], self.params[5]
            else:
                w0, w1 = gradual_compliance_weights(self.tstep-self.lockdown, self.beta_0, self.beta_1)
            
            C = self.C*w0 + C_target*w1
            C_asym = C.sum(axis=0)
            C_sym = (C*self.C_sym_factor).sum(axis=0)
            # TODO the ODE should support C_asym and C_sym
            s_n = self.model.simulate_day(C_asym, C_sym)
            # attack rate infected
            S_s = s[self.funcs.S(np.arange(self.model.K))]
            S_s_n = s_n[self.funcs.S(np.arange(self.model.K))]
            r_ari += -(np.sum(S_s) - np.sum(S_s_n))
            # TODO attack rate hospitalization
            r_arh += 0.
            # reduction in social contact
            R_s_n = s_n[self.funcs.R(np.arange(self.model.K))]
            # all combinations of age groups
            i, j = np.meshgrid(range(self.model.K), range(self.model.K))
            r_sr += (C_asym*S_s_n[i]*S_s_n[j] + C_asym*R_s_n[i]*R_s_n[j]).sum()
            # update state
            s = s_n
            self.tstep+=1

        # next-state, reward, terminal?, info
        return s_n, np.array([r_ari, r_arh, r_sr]), False, {}