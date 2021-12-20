import numpy as np
from scipy.stats import binom

from envs.epi_model import *
from envs.epi_model import __funcs__
import math

class BinomialEpiModel:

    def __init__(self, K, n_comp, init_model_state):
        #age groups
        self.K = K
        self.n_comp = n_comp
        self.init_model_state = init_model_state
        self.current_model_state = init_model_state        

        self.mf = __funcs__()
        self.p, self.ap = self.mf.init_params()

        #TODO:hacky
        self.s = self.init_model_state


    def next(self, n, rate):
        
        return binom.rvs(int(n), 1 - np.exp(rate))

    def age_step(self, k, C_asym, C_sym):
        E_n = self.next(self.s[self.mf.S(k)], -self.h * self.mf.lambda_(k, C_asym, C_sym, self.s))
        I_presym_n = self.next(self.s[self.mf.E(k)], -self.h * self.p.gamma)
        I_asym_n = self.next(self.s[self.mf.I_presym(k)], -self.h * self.ap[k].p * self.p.theta)
        I_mild_n = self.next(self.s[self.mf.I_presym(k)], -self.h * (1-self.ap[k].p) * self.p.theta)
        I_sev_n = self.next(self.s[self.mf.I_mild(k)], -self.h * self.ap[k].psi(self.p))
        I_hosp_n = self.next(self.s[self.mf.I_sev(k)], -self.h * self.ap[k].phi1 * self.ap[k].omega)
        I_icu_n = self.next(self.s[self.mf.I_sev(k)], -self.h * (1-self.ap[k].phi1) * self.ap[k].omega)
        D_hosp_n = self.next(self.s[self.mf.I_hosp(k)], -self.h * self.ap[k].tau1)
        D_icu_n = self.next(self.s[self.mf.I_icu(k)], -self.h * self.ap[k].tau1)
        R_asym_n = self.next(self.s[self.mf.I_asym(k)], -self.h * self.p.delta1)
        R_mild_n = self.next(self.s[self.mf.I_mild(k)], -self.h * self.ap[k].delta2)
        R_hosp_n = self.next(self.s[self.mf.I_hosp(k)], -self.h * self.ap[k].delta3)
        R_icu_n = self.next(self.s[self.mf.I_icu(k)], -self.h * self.ap[k].delta3)

        self.s[self.mf.S(k)] = self.s[self.mf.S(k)] - E_n
        self.s[self.mf.E(k)] = self.s[self.mf.E(k)] + E_n - I_presym_n
        self.s[self.mf.I_presym(k)] = self.s[self.mf.I_presym(k)] + I_presym_n - I_asym_n - I_mild_n
        self.s[self.mf.I_asym(k)] = self.s[self.mf.I_asym(k)] + I_asym_n - R_asym_n
        self.s[self.mf.I_mild(k)] = self.s[self.mf.I_mild(k)] + I_mild_n - I_sev_n - R_mild_n
        self.s[self.mf.I_sev(k)] = self.s[self.mf.I_sev(k)] + I_sev_n - I_hosp_n - I_icu_n
        self.s[self.mf.I_hosp(k)] = self.s[self.mf.I_hosp(k)] + I_hosp_n - D_hosp_n - R_hosp_n
        self.s[self.mf.I_icu(k)] = self.s[self.mf.I_icu(k)] + I_icu_n - D_icu_n - R_icu_n
        self.s[self.mf.D(k)] = self.s[self.mf.D(k)] + D_hosp_n + D_icu_n
        self.s[self.mf.R(k)] = self.s[self.mf.R(k)] + R_asym_n + R_mild_n + R_hosp_n + R_icu_n

        return self.s
        
    def simulate_day(self, C_asym, C_sym):
        
        self.s = self.current_model_state.copy()
        self.h = 1/24

        for step in range(24):
            for k in range(self.K):
                self.s = self.age_step(k, C_asym, C_sym)

        self.current_model_state = self.s.copy()
        return self.current_model_state
        
            
        
        
