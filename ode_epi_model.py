#init and simulate_day are the interface of EpiModel, we will have a BiomialEpiModel as well, 
# that implements this same interface

from scipy.integrate import odeint
from wrappers.epi_model import Parameters
from wrappers.epi_model import AgeParameters
import numpy as np
import pandas as pd

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

    def intialise_parameters(self, file, row=-1):
        df = pd.read_csv(file)
        df = df.iloc[row]
        p_vec = [0.94,0.90,0.84,0.61,0.49,0.21,0.02,0.02,0.02,0.02]

        self.ap = []
        self.p = []

        # model parameters
        self.p.append(Parameters([[1, 1], [1, 1]], df['log_gamma'], df['log_theta'], df['log_delta1'], 
        df['log_delta2'], df['log_delta3']))

        # age parameters
        for k in range(self.K):
            if k < 3:
                self.ap.append(AgeParameters(k, p_vec[k], df['log_omega_age' + str(k+1)], df['log_phi0_age' + str(k+1)], 
                df['log_phi1_age' + str(k+1)], 0, 0))

            else:
                self.ap.append(AgeParameters(k, [], df['log_omega_age' + str(k+1)], df['log_phi0_age' + str(k+1)], 
                df['log_phi1_age' + str(k+1)], df['log_mu_sev_age' + str(k+1)], df['log_mu2_sev_age' + str(k+1)]))



    #number of compartments in each age groups

    #functions to compute the index, for the input and output vectors
    #return the idx for compartment S for age group k

    def S(self, k):
        return (self.n_comp*k) + 0

    def E(self, k):
        return (self.n_comp*k) + 1

    def I_presym(self, k):
        return (self.n_comp*k) + 2

    def I_asym(self, k):
        return (self.n_comp*k) + 3

    def I_mild(self, k):
        return (self.n_comp*k) + 4

    def I_sev(self, k):
        return (self.n_comp*k) + 5

    def I_hosp(self, k):
        return (self.n_comp*k) + 6

    def I_icu(self, k):
        return (self.n_comp*k) + 7

    def D(self, k):
        return (self.n_comp*k) + 8

    def R(self, k):
        return (self.n_comp*k) + 9
    

    def lambda_(self, k, C_asym, C_sym, state):
        l_ = 0.0
        for k_prime in range(self.K):
            beta_asym = self.parameters.q_sym * C_sym[k][k_prime]
            beta_sym = self.parameters.q_asym * C_asym[k][k_prime]
            l_ += beta_asym * state[self.I_presym(k_prime)] + state[self.I_asym(k_prime)]
            l_ += beta_sym * state[self.I_mild(k_prime)] + state[self.I_sev(k_prime)]
        return l_


    def simulate_day(self, C_asym, C_sym):
        #TODO: check the size of C

        def deriv(y, t, N, beta, gamma):
            d_ = np.empty([(self.n_comp * self.K)])

            for k in range(self.K):

                _lambda = self.lambda_(k, C_asym, C_sym, y)
                d_[self.S(k)] = -_lambda * y[self.S(k)]
                d_[self.E(k)] = _lambda * y[self.S(k)] - self.p[k].gamma * y[self.E(k)]
                d_[self.I_presym(k)] = self.p[k].gamma * y[self.E(k)] - self.p[k].delta1 * y[self.I_presym(k)]
                d_[self.I_asym(k)] = self.p[k].theta * self.p[k].p * y[self.I_presym[k]] - self.p[k].delta1 * y[self.I_asym(k)]
                d_[self.I_mild(k)] = self.p[k].theta * (1 - self.p[k].p) * y[self.I_presym[k]] - (self.ap[k].psi + self.p[k].delta2) * y[self.I_mild[k]]
                d_[self.I_sev(k)] = self.ap[k].psi * y[self.I_mild[k]] - self.ap[k].omega * y[self.I_sev[k]]
                d_[self.I_hosp(k)] = self.ap[k].phi1 * self.ap[k].omega * y[self.I_sev[k]] - (self.p[k].delta3 + self.ap[k].mu1) * y[self.I_hosp[k]]
                d_[self.I_icu(k)] = (1 - self.ap[k].phi1) * self.ap[k].omega * y[self.I_sev[k]] - (self.p[k].delta3 + self.ap[k].mu2) * y[self.I_icu[k]]
                d_[self.D(k)] = self.ap[k].mu1 * y[self.I_hosp[k]] + self.ap[k].mu_2 * y[self.I_icu[k]] 
                d_[self.R(k)] = self.p[k].delta1 * y[self.I_asym[k]] +  self.p[k].delta2 * y[self.I_mild[k]] + self.p[k].delta3 * y[self.I_hosp[k]] + self.p[k].delta3 * y[self.I_icu[k]]

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
        ode_epi_model.intialise_parameters('./data/MCMCmulti_20211123_wave1.csv')        

if __name__ == "__main__":
    main()
"""

