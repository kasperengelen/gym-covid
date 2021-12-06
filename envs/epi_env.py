import numpy as np
import gym
from gym.spaces import Box


def reduce(c, p):
    # p contains p_w, p_s, p_l
    # TODO reduce contacts
    return c


class EpiEnv(gym.Env):

    def __init__(self, model):
        super(EpiEnv, self).__init__()
        # under the hood we run this epi model
        self.model = model
        # TODO population size defined in model or in mdp?
        N = 1000
        # contact matrix
        self.C = np.ones((model.K, model.K))

        # the observation space are compartments x age_groups
        self.observation_space = Box(low=np.zeros(model.n_comp*model.K), high=np.full(model.n_comp*model.K, N), dtype=np.float32)
        # action space is proportional reduction of work, school, leisure
        self.action_space = Box(low=np.zeros(3), high=np.ones(3), dtype=np.float32)
        # reward_space is attack-rate for infections, hospitalizations and reduction in social contact
        self.reward_space = Box(low=np.zeros(3), high=np.array([N, N, 1]), dtype=np.float32)

    def reset(self):
        self.model.current_model_state = self.model.init_model_state
        return self.model.current_model_state

    def step(self, action):
        # action is a 3d continuous vector
        # TODO reduction does not happen at once?
        C = reduce(self.C, action)

        s = self.model.current_model_state

        # simulate for a whole week, sum the daily rewards
        r_ari = r_arh = r_sr = 0.
        # make sure we only take upper diagonal
        C_diff = np.triu(C-self.C)
        for _ in range(7):
            s_n = self.model.simulate_day(C)
            # attack rate infected
            S_s = s[self.model.S(np.arange(self.model.K))]
            S_s_n = s_n[self.model.S(np.arange(self.model.K))]
            r_ari += -(np.sum(S_s) - np.sum(S_s_n))
            # TODO attack rate hospitalization
            r_arh += 0.
            # reduction in social contact
            R_s_n = s_n[self.model.R(np.arange(self.model.K))]
            # all combinations of age groups
            i, j = np.meshgrid(range(self.model.K), range(self.model.K))
            r_sr += (C_diff*S_s_n[i]*S_s_n[j] + C_diff*R_s_n[i]*R_s_n[j]).sum()
            # update state
            s = s_n

        # TODO do I need to manually set the state to the new state in the epi model?
        self.C = C
        # next-state, reward, terminal?, info
        return s_n, np.array([r_ari, r_arh, r_sr]), False, {}