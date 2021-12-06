#init and simulate_day are the interface of EpiModel, we will have a BiomialEpiModel as well, that implements this same interface
class OdeEpiModel:
    #having a ctor that allows initation from a particular state,
    #and a function to forward simulate a day, should allow us
    #to do everything we want,
    #standard trajectories via an MDP
    #running parts of a trajectory (for value iteration methods)
    #?
    
    def __init__(self, K, init_model_state, beta, gamma):
        #age groups
        self.K = K
        #initial model state
        self.init_model_state = init_model_state
        #current model state
        self.current_model_state = init_model_state
        #parameters
        self.beta = beta
        self.gamma = gamma

        #number of compartments in each age groups
        self.n_comp = 3

    #functions to compute the index, for the input and output vectors
    #return the idx for compartment S for age group k
    def S(self, k):
        return (self.n_comp*k) + 0

    def I(self, k):
        return (self.n_comp*k) + 1

    def R(self, k):
        return (self.n_comp*k) + 2

    #C is the contact matrix to be used,
    #would be used in beta
    def simulate_day(self, C):
        #TODO: check the size of C
        def deriv(y, t, N, beta, gamma):
            d_ = [] #initialize properly
            for k in range(K):
                d_[S(k)] = -beta * y[S(k)] * y[I(k)] / N
                d_[I(k)] = beta * y[S(k)] * y[I(k)] / N - gamma * y[I(k)]
                d_[R(k)] = gamma * y[I(k)]
            return d_

        #run the ode, something like this:
        # Initial conditions vector
        y0 = self.current_model_state
        # Integrate the SIR equations over the time grid, t.
        #TODO: decoide on t,
        #important, we will simulate one day here,
        #that should be OK for an ODE approximation, I think...
        ret = odeint(deriv, y0, t, args=(N, beta, gamma))
        self.current_model_state = ret.T

        return self.current_model_state
        
            
        
        
