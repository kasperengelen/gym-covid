import pandas as pd
import numpy as np

class __funcs__():

    def __init__(self, n_comp=0, K=0):
        self.n_comp = n_comp
        self.K = K
        #self.C = C

    def read_parameters_from_csv(self, file, row=-1):
        df = pd.read_csv(file)
        df = df.iloc[row]
        p_vec = [0.94,0.90,0.84,0.61,0.49,0.21,0.02,0.02,0.02,0.02]
        
        self.ap = []
        
        
        # model parameters
        self.p = (Parameters(df['log_q'], df['log_gamma'],
                                 df['log_theta'], df['log_delta1'], 
                                 df['log_delta2'], df['log_delta3']))
        
        # age parameters
        for k in range(self.K):
            if k <= 1:
                self.ap.append(AgeParameters(k, p_vec[k],
                                             df['log_omega_age' + str(k+1)],
                                             df['log_phi0_age' + str(k+1)], 
                                             df['log_phi1_age' + str(k+1)], 
                                             0, 
                                             0))
                
            else:
                self.ap.append(AgeParameters(k, p_vec[k],
                                             df['log_omega_age' + str(k+1)],
                                             df['log_phi0_age' + str(k+1)], 
                                             df['log_phi1_age' + str(k+1)],
                                             df['log_mu_sev_age' + str(k+1)],
                                             df['log_mu2_sev_age' + str(k+1)]))

        return self.p, self.ap

    def lambda_(self, k, C_asym, C_sym, state):
        l_ = 0.0
        for k_prime in range(self.K):
            beta_asym = self.p.q_asym * C_asym[k][k_prime]
            beta_sym = self.p.q_sym * C_sym[k][k_prime]
            l_ += beta_asym * (state[self.I_presym(k_prime)] + state[self.I_asym(k_prime)])
            l_ += beta_sym * (state[self.I_mild(k_prime)] + state[self.I_sev(k_prime)])# + state[self.I_hosp(k_prime)] + state[self.I_icu(k_prime)])
        return l_


    #functions to compute the index of each compartement
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

    def R(self, k):
        return (self.n_comp*k) + 8

    def D(self, k):
        return (self.n_comp*k) + 9

class Parameters:
    def __init__(self, q, gamma, theta,
                 delta1, delta2, delta3):
        self.q_sym = q
        self.q_asym = self.q_sym * 0.51
        self.gamma = gamma
        self.theta = theta
        self.delta1 = delta1
        self.delta2 = delta2
        self.delta3 = delta3

class AgeParameters:
    def __init__(self, k, p, omega, phi0, phi1, mu1, mu2):
        self.k = k
        self.p = p
        self.omega = omega
        self.phi0 = phi0
        self.phi1 = phi1
        self.mu1 = mu1
        self.mu2 = mu2

    def psi(self, p):
        delta_2_star = p.delta2 / self.phi0
        return (1-self.phi0)*delta_2_star
