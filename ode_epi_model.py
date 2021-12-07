#init and simulate_day are the interface of EpiModel, we will have a BiomialEpiModel as well, 
# that implements this same interface

from scipy.integrate import odeint
from wrappers.epi_model import Parameters
from wrappers.epi_model import AgeParameters
import numpy as np
import pandas as pd

from epi_model import *

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

    def simulate_day(self, C_asym, C_sym):
        #TODO: check the size of C

        def deriv(y, t, N, beta, gamma):
            d_ = np.empty([(self.n_comp * self.K)])

            for k in range(self.K):

                _lambda = lambda_(k, C_asym, C_sym, y)
                d_[S(k)] = -_lambda * y[S(k)]
                d_[E(k)] = _lambda * y[S(k)] - p[k].gamma * y[E(k)]
                d_[I_presym(k)] = p[k].gamma * y[E(k)] - p[k].delta1 * y[I_presym(k)]
                d_[I_asym(k)] = p[k].theta * p[k].p * y[I_presym[k]] - p[k].delta1 * y[I_asym(k)]
                d_[I_mild(k)] = p[k].theta * (1 - p[k].p) * y[I_presym[k]] - (ap[k].psi + p[k].delta2) * y[I_mild[k]]
                d_[I_sev(k)] = ap[k].psi * y[I_mild[k]] - ap[k].omega * y[I_sev[k]]
                d_[I_hosp(k)] = ap[k].phi1 * ap[k].omega * y[I_sev[k]] - (p[k].delta3 + ap[k].mu1) * y[I_hosp[k]]
                d_[I_icu(k)] = (1 - ap[k].phi1) * ap[k].omega * y[I_sev[k]] - (p[k].delta3 + ap[k].mu2) * y[I_icu[k]]
                d_[D(k)] = ap[k].mu1 * y[I_hosp[k]] + ap[k].mu_2 * y[I_icu[k]] 
                d_[R(k)] = p[k].delta1 * y[I_asym[k]] +  p[k].delta2 * y[I_mild[k]] + p[k].delta3 * y[I_hosp[k]] + p[k].delta3 * y[I_icu[k]]

            return d_

        # Initial conditions vector
        y0 = self.current_model_state   

        # Integrate the SIR equations over the time grid, t.        
        # time for each our of the day - needs to be defined        
        t = np.linspace(0,23,24)
        ret = odeint(deriv, y0, t, args=(self.N, self.beta, self.gamma))
        
        # state will be the last time period
        self.current_model_state = ret[-1].T

        return self.current_model_state
        
            
"""       
def main():
    # Parameters for SIR
    # K = number of age groups
    k = 10
    # init_model_state
    init_state = [49, 1, 0, 48, 1, 1]
    # beta
    beta = 0.05
    # gamma
    gamma = 1/3

    timesteps = 1
    ode_epi_model = OdeEpiModel(100, k, init_state, beta, gamma)

    for i in range(timesteps):        
        #print("state:", ode_epi_model.simulate_day())
        ode_epi_model.read_parameters_from_csv('./data/MCMCmulti_20211123_wave1.csv')        

if __name__ == "__main__":
    main()
"""

