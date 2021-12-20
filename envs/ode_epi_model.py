#init and simulate_day are the interface of EpiModel, we will have a BiomialEpiModel as well, 
# that implements this same interface

from scipy.integrate import odeint, solve_ivp
from envs.epi_model import Parameters
from envs.epi_model import AgeParameters
from envs.epi_model import __funcs__
import numpy as np
import pandas as pd
import sys

np.set_printoptions(threshold=sys.maxsize)

from envs.epi_model import *

class OdeEpiModel:
    
    def __init__(self, N, K, init_model_state, beta, gamma):
        #age groups
        self.K = K        
        #initial model state
        self.init_model_state = init_model_state
        #current model state
        self.current_model_state = init_model_state
        #parameters
        self.beta = beta
        self.gamma = gamma
        self.N = N
        self.n_comp = 10
        self.mf = __funcs__(self.n_comp, self.K)
        self.p, self.ap = self.mf.init_params()

    def simulate_day(self, C_asym, C_sym):

        def deriv(y, t):
            d_ = np.zeros([(self.n_comp * (self.K))])

            for k in range(self.K):

                _lambda = self.mf.lambda_(k, C_asym, C_sym, y)

                d_[self.mf.S(k)] = -_lambda * y[self.mf.S(k)]

                d_[self.mf.E(k)] = _lambda * y[self.mf.S(k)] - self.p.gamma * y[self.mf.E(k)]

                d_[self.mf.I_presym(k)] = self.p.gamma * y[self.mf.E(k)] - self.p.theta * y[self.mf.I_presym(k)]

                d_[self.mf.I_asym(k)] = self.p.theta * self.ap[k].p * y[self.mf.I_presym(k)] - self.p.delta1 * y[self.mf.I_asym(k)]

                d_[self.mf.I_mild(k)] = self.p.theta * (1 - self.ap[k].p) * y[self.mf.I_presym(k)] - (self.ap[k].psi(self.p) + self.ap[k].delta2) * y[self.mf.I_mild(k)]

                d_[self.mf.I_sev(k)] = self.ap[k].psi(self.p) * y[self.mf.I_mild(k)] - self.ap[k].omega * y[self.mf.I_sev(k)]

                d_[self.mf.I_hosp(k)] = self.ap[k].phi1 * self.ap[k].omega * y[self.mf.I_sev(k)] - (self.ap[k].delta3 + self.ap[k].mu1) * y[self.mf.I_hosp(k)]

                d_[self.mf.I_icu(k)] = (1 - self.ap[k].phi1) * self.ap[k].omega * y[self.mf.I_sev(k)] - (self.ap[k].delta3 + self.ap[k].tau1) * y[self.mf.I_icu(k)]

                d_[self.mf.D(k)] = self.ap[k].tau1 * y[self.mf.I_hosp(k)] + self.ap[k].tau1 * y[self.mf.I_icu(k)] 

                d_[self.mf.R(k)] = self.p.delta1 * y[self.mf.I_asym(k)] + self.ap[k].delta2 * y[self.mf.I_mild(k)
                ] + self.ap[k].delta3 * y[self.mf.I_hosp(k)] + self.ap[k].delta3 * y[self.mf.I_icu(k)]

            return d_

        # Initial conditions vector
        y0 = self.current_model_state   

        # Integrate the SIR equations over the time grid, t.        
        # time for each hour of the day - needs to be defined        
        t = np.linspace(0,1,3)
        ret = odeint(deriv, y0, t)

        # state will be the last time period
        self.current_model_state = ret[-1].T

        return self.current_model_state
        
            
