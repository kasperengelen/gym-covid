from typing import Any

import numpy as np
import gymnasium
from gymnasium.spaces import Box, Tuple
import datetime

from gym_covid.envs.model import BinomialModel, ODEModel


def gradual_compliance_weights(t, beta_0, beta_1):
    x = beta_0 + beta_1*t
    w1 = np.minimum(1, np.exp(x)/(1+np.exp(x)))
    w0 = 1-w1
    return w0, w1


def school_holidays(C, C_current, C_target):
    # schools are closed, 0% school contacts
    C[3] = C[3]*0.
    C_current[3] = C_current[3]*0.
    C_target[3] = C_target[3]*0.
    return C, C_current, C_target


class EpiEnv(gymnasium.Env):

    def __init__(self, model, C=None,
                 beta_0=None, beta_1=None, datapoints=None):
        super(EpiEnv, self).__init__()
        # under the hood we run this epi model
        self.model = model
        # population size of each age-group is sum of the compartments
        N = self.model.init_state.sum(axis=0)
        self.K = len(N)
        self.N = N.sum()
        # contact matrix
        self.C = np.ones((1, self.K, self.K)) if C is None else C
        # factors of contact matrix for symptomatic people,
        # reshape to match C shape
        self.C_sym_factor = np.array([1., 0.09, 0.13,
                                      0.09, 0.06, 0.25])[:, None, None]
        self.C_full = self.C.copy()
        self.days_per_timestep = 7

        self.beta_0 = beta_0
        self.beta_1 = beta_1
        if isinstance(self.model, ODEModel):
            ptype = np.float64
        elif isinstance(self.model, BinomialModel):
            ptype = np.int64
        else:
            raise ValueError(f"Error: unsupported model type '{type(self.model)}'")

        # action space is proportional reduction of work, school, leisure
        self.action_space = Box(low=np.zeros(3), high=np.ones(3),
                                dtype=np.float64)
        # the observation space is composed of
        # (1) days per timestep x compartments x age_groups
        # (2) a boolean array for holidays per day
        # (3) the previous action
        self.observation_space = Tuple([
            Box(low=np.zeros((self.days_per_timestep, model.n_comp, self.K)),
                high=np.tile(N,  # FIXME: pop. not preserved!
                             (self.days_per_timestep, model.n_comp, 1)),
                dtype=ptype),
            Box(low=np.tile(False, (self.days_per_timestep, 1)),
                high=np.tile(True, (self.days_per_timestep, 1)),
                dtype=np.bool),
            Box(low=np.zeros(3), high=np.ones(3), dtype=np.float64)])

        self.datapoints = datapoints
        self.today = datetime.date(2020, 3, 1)

        self.events = {}
        # include school holiday
        for holiday_start, \
            holiday_end in ((datetime.date(2020, 7, 1),
                             datetime.date(2020, 8, 31)),
                            (datetime.date(2020, 11, 2),
                             datetime.date(2020, 11, 8)),
                            (datetime.date(2020, 12, 21),
                             datetime.date(2021, 1, 3))):
            # enforce holiday-event (0% school contacts)
            # every day of school-holiday
            for i in range((holiday_end-holiday_start).days+1):
                day = holiday_start+datetime.timedelta(days=i)
                self.events[day] = school_holidays

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        super(EpiEnv, self).reset(seed=seed, options=options)
        self.model.current_state = self.model.init_state.copy()
        self.current_C = self.C
        self.today = datetime.date(2020, 3, 1)
        # check for events on this day
        event = self.today in self.events
        # NOTE: intead of returning a boolean, this returns now a dictionary with a boolean inside it.
        res = (np.tile(self.model.current_state, (self.days_per_timestep, 1, 1)),
               np.tile(False, (self.days_per_timestep, 1)),
               np.zeros(3)),  {"has_event": event}
        o, i = res
        # print(f"size of reset obs = {len(o)}")
        return res

    def step(self, action):
        # action is a 3d continuous vector
        p_w, p_s, p_l = action

        # match all C components, reshape to match C shape
        p = np.array([1, p_w, p_w, p_s, p_l, p_l])[:, None, None]
        C_target = self.C*p

        s = self.model.current_state.copy()

        # simulate for a whole week, sum the daily rewards
        r_ari = r_arh = r_sr = 0.

        # the binomial model uses integers, the ODE model uses floats.
        if isinstance(self.model, ODEModel):
            state_n = np.empty((self.days_per_timestep,
                                self.model.n_comp,
                                self.K), dtype=np.float64)
        elif isinstance(self.model, BinomialModel):
            state_n = np.empty((self.days_per_timestep,
                                self.model.n_comp,
                                self.K), dtype=np.int64)
        else:
            raise ValueError(f"Error: unsupported model type '{type(self.model)}'")
        event_n = np.zeros((self.days_per_timestep, 1), dtype=bool)
        for day in range(self.days_per_timestep):
            # every day check if there are events on the calendar
            today = self.today + datetime.timedelta(days=day)
            if today in self.events:
                C_full, C_c, C_t = self.events[today](self.C.copy(), self.current_C, C_target)
                #C_full = C_full.sum(0)
                # today is a school holiday
                event_n[day] = True
            else:
                C_full, C_c, C_t = self.C_full, self.current_C, C_target

            # gradual compliance, C_target is only reached after a number of days
            w0, w1 = gradual_compliance_weights(day, self.beta_0, self.beta_1)
            
            C_asym = C_c*w0 + C_t*w1
            #C_asym = C.sum(axis=0)
            C_sym = (C_asym*self.C_sym_factor)#.sum(axis=0)

            s_n = self.model.simulate_day(C_asym.sum(axis=0), C_sym.sum(axis=0))
            state_n[day] = s_n
            # attack rate infected
            S_s = s[self.model.S]
            S_s_n = s_n[self.model.S]
            r_ari += -(np.sum(S_s) - np.sum(S_s_n))
            # attack rate hospitalization
            I_hosp_new_s_n = s_n[self.model.I_hosp_new] + s_n[self.model.I_icu_new]
            r_arh += -np.sum(I_hosp_new_s_n)
            # reduction in social contact
            R_s_n = s_n[self.model.R]
            # all combinations of age groups
            i, j = np.meshgrid(range(self.K), range(self.K))
            C_diff = C_asym-C_full
            # divide by total population to get lost contacts/person, for each social environment
            r_sr += (C_diff*S_s_n[None,i]*S_s_n[None,j] + C_diff*R_s_n[None,i]*R_s_n[None,j]).sum(axis=(1,2))/self.N
            # update state
            s = s_n

        # update current contact matrix
        self.current_C = C_target
        # update date
        self.today = self.today + datetime.timedelta(days=self.days_per_timestep)
        # social reduction for work, school and leisure
        r_sr_w = r_sr[1]+r_sr[2]
        r_sr_s = r_sr[3]
        r_sr_l = r_sr[4]+r_sr[5]
        # the reward is the attack-rate for infections,
        # hospitalizations and reduction in social contact
        # we aggregate all of this with a magical formula
        rew = (3 * r_ari + 4 * r_arh  # hospitalized > infectious
               + 0.02 * r_sr_w  # schools > work, leisure
               + 0.03 * r_sr_s  # social contact << attack rate
               + 0.02 * r_sr_l)

        # next-state , reward, terminal?, truncated, info
        # provide action as proxy for current SCM, impacts progression of epidemic
        # NOTE: recent versions of gym require a "truncated" return value. I set this to return False.
        return (state_n, event_n, action.copy()),\
                rew, False, False, {"state": state_n}
