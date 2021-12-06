def lambda_(k, C_asym, C_sym, state):
    l_ = 0.0
    for k_prime in range(K):
        beta_asym = parameters.q_sym * C_sym[k][k_prime]
        beta_sym = parameters.q_asym * C_asym[k][k_prime]
        l_ += beta_asym * state[I_presym(k_prime)] + state[I_asym(k_prime)]
        l_ += beta_sym * state[I_mild(k_prime)] + state[I_sev(k_prime)]
    return l_

#number of compartments in each age groups
n_comp = 10

#functions to compute the index of each compartement
def S(k):
    return (n_com*k) + 0

def E(k):
    return (n_com*k) + 1

def I_presym(k):
    return (n_com*k) + 2

def I_asym(k):
    return (n_com*k) + 3

def I_mild(k):
    return (n_com*k) + 4

def I_mild(k):
    return (n_com*k) + 5

def I_sev(k):
    return (n_com*k) + 6

def I_hosp(k):
    return (n_com*k) + 7

def I_icu(k):
    return (n_com*k) + 8

def R(k):
    return (n_com*k) + 9

def D(k):
    return (n_com*k) + 10

class Parameters:
    def __init__(self, q_sym, q_asym, gamma, theta,
                 delta1, delta2, delta3, delta4):
        self.q_sym = q_sym
        self.q_asym = q_asym
        self.gamma = gamma
        self.theta = theta
        self.delta1 = delta1
        self.delta2 = delta2
        self.delta3 = delta3
        self.delta4 = delta4

class AgeParameters:
    def __init__(self, k, p, omega, phi, psi):
        self.k = k
        self.p = p
        self.omega = omega
        self.phi = phi
        self.psi = psi
