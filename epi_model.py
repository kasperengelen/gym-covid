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
