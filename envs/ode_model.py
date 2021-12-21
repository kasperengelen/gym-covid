import numpy as np
from scipy.integrate import odeint


#Summary:
#Alles is uit de paper gehaald
#behalve:
# - contact matrices, gekregen van Lander
# - n0 (uit Lander's code: lijn 88 in lib_model_core.R)
# - eind-datum om de cases te bereken, is 13/03/2020 ipv 12/03/2020
# - beta_0 en beta_1

class ODEModel(object):

    def __init__(self, 
                 init_state,
                 ):

        self.init_state = init_state
        self.current_state = init_state

        compartments = ['S', 'E', 'I_presym', 'I_asym', 'I_mild', 'I_sev', 'I_hosp', 'I_icu', 'R', 'D']
        # provide indexes for compartments: self.S = 0, self.E = 1, etc
        for i, name in enumerate(compartments):
            setattr(self, name, i)
        self.compartments = compartments

	#Table F1 Appendix
        delta2_star = 0.756
        delta3_star = 0.185
        phi0 = np.array([0.972, 0.992, 0.984, 0.987, 0.977, 0.971, 0.958, 0.926, 0.956, 0.926])
        #mu[1] = 0, as children are assumed not to die from COVID
	self.mu = mu = np.array([0, 0.005, 0.005, 0.024, 0.037, 0.068, 0.183, 0.325, 0.446, 0.611])
        q = 0.051

        # == from code
        f = 0.51   # relative infectiousness of asymptomatic vs. symptomatic cases
        self.q_asym = f*q
        self.q_sym = q
        # ==

	#Table F1 Appendix
        self.gamma = 0.729
        self.theta = 0.475

	#Section B.2, Appendix
        self.p = np.array([0.94,0.90,0.84,0.61,0.49,0.21,0.02,0.02,0.02,0.02])

	#Table F1 Appendix
        self.delta1 = 0.24
	
        self.delta2 = phi0*delta2_star #Section 2.1, 1st paragraph
        self.delta3 = (1-mu)*delta3_star #Appendix, F1
        self.delta4 = self.delta3 #Communication with Lander
        self.psi = (1-phi0)*delta2_star #Section 2.1, 1st paragraph

	#Table F1 Appendix
        self.omega = np.array([0.167, 0.095, 0.099, 0.162, 0.338, 0.275, 0.343, 0.378, 0.334, 0.302])
        
	#Table B1 (Appendix)
	#last element was copied
        #TODO: in Lander's code, the first 2 entries are 99
	self.phi1 = np.array([100, 100, 85, 85, 76, 76, 73, 69, 74, 74])/100 # last element was missing, as group was [80,100)
       
	#F.1 Appendix 
	self.tau1 = mu*delta3_star
        self.tau2 = self.tau1

    def deriv(self, y, t, C_asym, C_sym):
        # was flattened for ode
        y = y.reshape(self.init_state.shape)
        d_dt = np.zeros(y.shape)
        # compute lambda
        beta_asym = self.q_asym*C_asym
        beta_sym = self.q_sym*C_sym

        lambda_asym = beta_asym*(y[self.I_presym] + y[self.I_asym])
        lambda_sym = beta_sym*(y[self.I_mild] + y[self.I_sev])
        lambda_ = lambda_asym.sum(1) + lambda_sym.sum(1)

        d_dt[self.S] = -lambda_*y[self.S]
        d_dt[self.E] = lambda_*y[self.S] - self.gamma*y[self.E]
        d_dt[self.I_presym] = self.gamma*y[self.E] - self.theta*y[self.I_presym]
        d_dt[self.I_asym] = self.p*self.theta*y[self.I_presym] - self.delta1*y[self.I_asym]
        d_dt[self.I_mild] = self.theta*(1-self.p)*y[self.I_presym]-(self.psi+self.delta2)*y[self.I_mild]
        d_dt[self.I_sev] = self.psi*y[self.I_mild]-self.omega*y[self.I_sev]
        d_dt[self.I_hosp] = self.phi1*self.omega*y[self.I_sev]-(self.delta3+self.tau1)*y[self.I_hosp]
        d_dt[self.I_icu] = (1-self.phi1)*self.omega*y[self.I_sev]-(self.delta4+self.tau2)*y[self.I_icu]
        d_dt[self.D] = self.tau1*y[self.I_hosp] + self.tau2*y[self.I_icu]
        d_dt[self.R] = self.delta1*y[self.I_asym]+self.delta2*y[self.I_mild]+self.delta3*y[self.I_hosp]+self.delta4*y[self.I_icu]
        # flatten for ode
        return d_dt.flatten()


    def simulate_day(self, C_asym, C_sym):

        y0 = self.current_state
        # scale of t is "day", so one days passes for each increment of t
        t = np.array([0, 1])

        ret = odeint(self.deriv, y0.flatten(), t, args=(C_asym, C_sym))
        # state will be the last time period
        self.current_state = ret[-1].reshape(self.init_state.shape)

        return self.current_state
        

if __name__ == '__main__':
    import pandas as pd
    import matplotlib.pyplot as plt

    def gradual_compliance_weights(t, beta_0, beta_1):
        x = beta_0 + beta_1*t
        w1 = np.minimum(1, np.exp(x)/(1+np.exp(x)))
        w0 = 1-w1
        return w0, w1

    def process_contact_csv(ct):
        c = pd.read_csv(f'data/contact_matrix/original/c_{ct}.csv', header=None).values
        #copy the 90+ age group from the 80-90 age group (col and row)
        c = np.hstack((c, c[:, [-1]]))
        c = np.vstack((c, c[-1]))
        return c
        
    contact_types = ['home', 'work', 'transport', 'school', 'leisure', 'otherplace']
    c = [process_contact_csv(ct) for ct in contact_types]
    c = np.array(c)

    n_comp = 10
    #Confirmed cases by age, sex and province taken from:
    #https://epistat.wiv-isp.be/covid/
    population = pd.read_csv('data/population_2020-01-01.csv')
    # sort by age, keep only age and population count (2 last columns)
    population = population.sort_values(by=['Leeftijd']).iloc[:,-2:].values
    # drop last row as it is global population (sum of all ages)
    population = population[:-1]
    # [ 0., 10., 20., 30., 40., 50., 60., 70., 80., 90., inf]
    age_groups = np.concatenate((np.arange(0, 100, 10), (np.inf,)))
    population_groups = np.empty(len(age_groups)-1)
    for i in range(len(age_groups)-1):
        group_index = np.logical_and(population[:,0] >= age_groups[i], population[:,0] < age_groups[i+1])
        population_groups[i] = np.sum(population[:,1][group_index])

    # load confirmed cases, based on section 2.7.3
    #Population taken from:
    #https://statbel.fgov.be/nl/themas/bevolking/structuur-van-de-bevolking#figures
    #and then clicking on:
    #"Bevolking per geslacht en leeftijdsgroep voor België"

    cases = pd.read_csv('data/cases.csv')
    # keep first two weeks to compute the frequency if confirmed cases for each age group
    cases = cases[cases['DATE'] >= '2020-03-01']
    #TODO: the paper states 12 March 2020? (13 march from Lander's code)
    # TODO: Mathieu compared start state with R, and got the same with 13/03/2020
    cases = cases[cases['DATE'] < '2020-03-14']
    age_cases = cases.groupby('AGEGROUP').agg(np.sum)
    rel_age_cases = age_cases/age_cases.sum()
    # Age-dependent asymptomatic proportions
    ##Section B.2, Appendix (same as self.p)
    p_vec = np.array([0.94,0.90,0.84,0.61,0.49,0.21,0.02,0.02,0.02,0.02])
    #lijn 88 in lib_model_core.R
    n0 = np.exp(7.75220356739557)
    imported_cases = np.round(rel_age_cases.values.flatten()*n0*(1/(1-p_vec)),0)
    S = population_groups-imported_cases
    E = imported_cases

    initial_state = np.zeros((n_comp, len(population_groups)))
    initial_state[0] = S
    initial_state[1] = E

    model = ODEModel(initial_state)

    #These parameters make the compliance converge to 100% after a week,
    #more specifically, 95% after 6 days, 99% after 7 days.
    beta_0 = -5
    beta_1 = 1.404

    states = [initial_state]
    start_lockdown = 14
    for day in range(1, 60):
        # before lockdown
        if day < start_lockdown:
            p_w = p_s = p_l = 1.
            w0, w1 = 1, 0
        else:
            p_w, p_s, p_l = 0.2, 0.0, 0.1
            w0, w1 = gradual_compliance_weights(day-start_lockdown, beta_0, beta_1)
            
        C_sym_factor = np.array([1., 0.09, 0.13, 0.09, 0.06, 0.25])[:, None, None]
        p = np.array([1, p_w, p_w, p_s, p_l, p_l])[:, None, None]
        C_target = c*p
        C = c*w0 + C_target*w1
        c_asym = C.sum(0)
        c_sym = (C*C_sym_factor).sum(0)

        
        next_state = model.simulate_day(c_asym, c_sym)
        states.append(next_state)
    states = np.array(states)

    i_hosp = states[:, model.I_hosp].sum(1)
    i_icu = states[:, model.I_icu].sum(1)
    d = states[:, model.D].sum(1)

    plt.figure()
    plt.plot(i_hosp, label='hosp')
    plt.plot(i_icu, label='icu')
    plt.plot(i_hosp+i_icu, label='hosp+icu')
    plt.plot(d, label='deaths')
    plt.legend()
    plt.show()
