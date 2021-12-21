import sys
import os
sys.path.append(os.getcwd())
import matplotlib.pyplot as plt
import argparse


def plot_simulation(states):

    plt.figure()
    i_hosp = states[:,6::10].sum(axis=1)
    i_icu = states[:,7::10].sum(axis=1)
    d = states[:,9::10].sum(axis=1)
    plt.plot(i_hosp, label='hosp')
    plt.plot(i_icu, label='icu')
    plt.plot(i_hosp+i_icu, label='hosp+icu')
    plt.plot(d, label='deaths')
    plt.legend()
    plt.xlabel('day')
    plt.ylabel('n hosp')
    plt.show()


if __name__ == '__main__':
    import gym
    import envs
    from gym.wrappers import TimeLimit
    import numpy as np
    

    parser = argparse.ArgumentParser(description='')
    #parser.add_argument('--algorithm', default = '', type = str)
    parser.add_argument('--env', default = 'binomial', type = str, help='Environment model to run. Options : binomial or ode. Default : binomial')
    parser.add_argument('--runs', default = 1, type = int, help='Number of experiments to be ran. Default : 1')
    parser.add_argument('--timesteps', default = 60, type=int, help='Number of timesteps to run the model . Default : 60')
    parser.add_argument('--parameters', default = [0, 1, 1, 1, 1, 0], type = list, help='Lockdown parameters. [0, p_w, p_s, p_l, w0, w1] Default : [0, 1, 1, 1, 1, 0]')
    parser.add_argument('--lockdown', default = 14, type = int, help='Timestep when lockdown in enforced. Default : 14')
    parser.add_argument('--lockdown_parameters', default = [0, 0.2, 0.0, 0.1, 1, 0], type = list, help='Lockdown parameters. [lockdown, p_w, p_s, p_l, w0, w1] Default : [0, 0.2, 0.0, 0.1, 1, 0]')
    

    args = parser.parse_args()
    print(args)

    env = args.env
    runs = args.runs
    timesteps = args.timesteps
    lockdown = args.lockdown 
    lockdown_params = args.lockdown_parameters
    lockdown_params[0] = lockdown
    parameters = args.parameters

    if env == 'binomial':
        env = gym.make('EpiBelgiumBinomialContinuous-v0')

    if env == 'ode':
        env = gym.make('EpiBelgiumODEContinuous-v0')

    
    for run in range(runs):
        env = TimeLimit(env, timesteps)
        states = []
        t, s = env.reset()
        env.init_params(parameters, lockdown_params)
        d = False
        states.append(s)
        print("initial state", s)

        while not d:
            # action doesnt matter, hardcoded in MDP
            a = np.ones(3)          
            s, r, d, _ = env.step(a)
            #print(s, r)
            states.append(s)
        
        states = np.array(states)
        # plots assume 3 compartments
        plot_simulation(states)