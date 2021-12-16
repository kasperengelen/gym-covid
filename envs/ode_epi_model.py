#init and simulate_day are the interface of EpiModel, we will have a BiomialEpiModel as well, 
# that implements this same interface

from scipy.integrate import odeint, solve_ivp
from envs.epi_model import Parameters
from envs.epi_model import AgeParameters
from envs.epi_model import Model_Funcs
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
        self.mf = Model_Funcs(self.n_comp, self.K)
        self.p, self.ap = self.mf.read_parameters_from_csv('https://raw.githubusercontent.com/lwillem/stochastic_model_BE/main/data/config/MCMCmulti_20211123_wave1.csv?token=AI4WOQFDQCWTZGFKPRPKQ2DBYJPXW')

    def simulate_day(self, C_asym, C_sym):

        def deriv(y, t):
            d_ = np.empty([(self.n_comp * (self.K))])

            for k in range(self.K):

                _lambda = self.mf.lambda_(k, C_asym, C_sym, y)

                d_[self.mf.S(k)] = -_lambda * y[self.mf.S(k)]

                d_[self.mf.E(k)] = _lambda * y[self.mf.S(k)] - self.p.gamma * y[self.mf.E(k)]

                d_[self.mf.I_presym(k)] = self.p.gamma * y[self.mf.E(k)] - self.p.theta * y[self.mf.I_presym(k)]

                d_[self.mf.I_asym(k)] = self.p.theta * self.ap[k].p * y[self.mf.I_presym(k)] - self.p.delta1 * y[self.mf.I_asym(k)]

                d_[self.mf.I_mild(k)] = self.p.theta * (1 - self.ap[k].p) * y[self.mf.I_presym(k)] - (self.ap[k].psi(self.p) + self.p.delta2) * y[self.mf.I_mild(k)]

                d_[self.mf.I_sev(k)] = self.ap[k].psi(self.p) * y[self.mf.I_mild(k)] - self.ap[k].omega * y[self.mf.I_sev(k)]

                d_[self.mf.I_hosp(k)] = self.ap[k].phi1 * self.ap[k].omega * y[self.mf.I_sev(k)] - (self.p.delta3 + self.ap[k].mu1) * y[self.mf.I_hosp(k)]

                d_[self.mf.I_icu(k)] = (1 - self.ap[k].phi1) * self.ap[k].omega * y[self.mf.I_sev(k)] - (self.p.delta3 + self.ap[k].mu2) * y[self.mf.I_icu(k)]

                d_[self.mf.D(k)] = self.ap[k].mu1 * y[self.mf.I_hosp(k)] + self.ap[k].mu2 * y[self.mf.I_icu(k)] 

                d_[self.mf.R(k)] = self.p.delta1 * y[self.mf.I_asym(k)] + self.p.delta2 * y[self.mf.I_mild(k)
                ] + self.p.delta3 * y[self.mf.I_hosp(k)] + self.p.delta3 * y[self.mf.I_icu(k)]

            #print(d_)
            return d_

        # Initial conditions vector
        y0 = self.current_model_state   

        # Integrate the SIR equations over the time grid, t.        
        # time for each hour of the day - needs to be defined        
        t = np.linspace(0,23,24)
        ret = odeint(deriv, y0, t)
        #ret = solve_ivp(deriv, t, y0)
        #print(ret)
       
        
        # state will be the last time period
        self.current_model_state = ret[-1].T

        return self.current_model_state
        
            
