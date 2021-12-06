import numpy as np
from scipy.stats import binom

from epi_model import *

class BinomialEpiModel:
    #TODO: let's assume that init_model_state is a numpy array
    def __init__(self, K, init_model_state, parameters, age_parameters):
        #age groups
        self.K = K
        self.init_model_state = init_model_state
        self.current_model_state = init_model_state
        self.parameters = parameters
        self.age_parameters = age_parameters

        #TODO:hacky
        self.new = np.zeros(self.K * epi.n_comp)

    def next(n, rate):
        return binom.rvs(n, 1 - np.exp(rate))[0]

    def age_step(self, k, C_asym, C_sym, a_p, p):
        E_n = next(s[S(k)], 1 - np.exp(-h*lambda_(k,state)))
        I_presym_n = next(s[E(k)], -h*p.gamma)
        I_asym_n = next(s[I_presym(k)], -h * a_p[k].p * p.theta)
        I_mild_n = next(s[I_presym(k)], -h * (1-a_p[k].p) * p.theta)
        I_sev_n = next(s[I_mild(k)], -h * a_p[k].psi(p))
        I_hosp_n = next(s[I_sev(k)], -h * a_p[k].phi1 * a_p[k].omega)
        I_icu_n = next(s[I_sev(k)], -h * (1-a_p[k].phi1) * a_p[k].omega)
        D_hosp_n = next(s[I_hosp(k)], -h * a_p[k].mu1)
        D_icu_n = next(s[I_icu(k)], -h * a_p[k].mu2)
        R_asym_n = next(s[I_asym(k)], -h * a_p[k].delta1)
        R_mild_n = next(s[I_mild(k)], -h * a_p[k].delta2)
        R_hosp_n = next(s[I_hosp(k)], -h * a_p[k].delta3)
        R_icu_n = next(s[I_icu(k)], -h * a_p[k].delta3)

        n = self.n
        n[S(k)] = s[S(k)] - E_n
        n[E(k)] = s[E(k)] + E_n - I_presym_n
        n[I_presym(k)] = s[I_presym(k)] + I_presym_n - I_asym_n - I_mild_n
        n[I_asym(k)] = s[I_asym(k)] + I_asym_n - R_asym_n
        n[I_mild(k)] = s[I_mild(k)] + I_mild_n - I_sev_n - R_mild_n
        n[I_sev(k)] = s[I_sev(k)] + I_sev_n - I_hosp_n - R_sev_n
        n[I_hosp(k)] = s[I_hosp(k)] + I_hosp_n - D_hosp_n - R_hosp_n
        n[I_icu(k)] = s[I_icu(k)] + I_icu_n - D_icu_n - R_icu_n
        n[D(k)] = s[D(k)] + D_hosp_n + D_icu_n
        n[R(k)] = s[R(k)] + R_asym_n + R_mild_n + R_hosp_n + R_icu_n
        
    #C_sym, C_asym -> contact matrices
    def simulate_day(self, C_asym, C_sym):
        p = self.parameters
        a_p = self.age_parameters
        
        s = self.current_model_state
        for h in range(24):
            for k in range(K):
                age_step(k, C_asym, C_sym, a_p, p)
            self.new = s
            
        return self.current_model_state
        
            
        
        
