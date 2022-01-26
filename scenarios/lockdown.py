import sys
import os
sys.path.append(os.getcwd())
import matplotlib.pyplot as plt
import argparse
import pandas as pd


def plot_states(states, alpha):
    i_hosp_new = states[:,-3].sum(axis=1)
    i_icu_new = states[:,-2].sum(axis=1)
    d_new = states[:,-1].sum(axis=1)

    axs = plt.gcf().axes
    # hospitalizations
    ax = axs[0]
    ax.plot(i_hosp_new, alpha=alpha, label='hosp', color='blue')
    ax.plot(i_icu_new,  alpha=alpha, label='icu', color='green')
    ax.plot(i_hosp_new+i_icu_new, label='hosp+icu',  alpha=alpha, color='orange')

    # deaths
    ax = axs[1]
    ax.plot(d_new, alpha=alpha, label='deaths', color='red')


def plot_simulation(states_per_stoch_run, ode_states, datapoints=None):
    _, axs = plt.subplots(2, 1)

    for states in states_per_stoch_run:
        plot_states(states, 0.2)
    plot_states(ode_states, 1.)

    if datapoints is not None:
        h = datapoints['hospitalizations']
        axs[0].scatter(np.arange(len(h)), h, facecolors='none', edgecolors='black')
        d = datapoints['deaths']
        axs[1].scatter(np.arange(len(d)), d, facecolors='none', edgecolors='black')

    axs[0].set_xlabel('days')
    axs[0].set_ylabel('hospitalizations')

    axs[1].set_xlabel('days')
    axs[1].set_ylabel('deaths')
        
    plt.show()


def simulate_lockdown(env):
    states = []
    s = env.reset()
    d = False
    week = 0
    # states.append(s)

    while not d:
        if week < 2:
            action = np.ones(3)
        # it's lockdown time
        else:
            action = np.array([0.2, 0.0, 0.1])
        s, r, d, info = env.step(action)
        states.append(s)
        week += 1
    # array of shape [Week DayOfWeek Compartment AgeGroup]
    states = np.array(states)
    # reshape to [Day Compartment AgeGroup]
    return np.array(states).reshape(states.shape[0]*states.shape[1], *states.shape[2:])


if __name__ == '__main__':
    import gym
    import envs
    from gym.wrappers import TimeLimit
    import numpy as np
    

    parser = argparse.ArgumentParser(description='')
    #parser.add_argument('--algorithm', default = '', type = str)
    parser.add_argument('--env', default = 'binomial', type = str, help='Environment model to run. Options : binomial or ode. Default : binomial')
    parser.add_argument('--runs', default = 1, type = int, help='Number of experiments to be ran. Default : 1')
    parser.add_argument('--seed', default = 22122021, type = int, help='RNG seed. Default : 22122021')
    parser.add_argument('--timesteps', default = 15, type=int, help='Number of timesteps to run the model . Default : 60')
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

    np.random.seed(seed=args.seed)

    if env == 'binomial':
        env = gym.make('EpiBelgiumBinomialContinuous-v0')
        
    ode_env = gym.make('EpiBelgiumODEContinuous-v0')
    if env == 'ode':
        env = ode_env
    env = TimeLimit(env, timesteps)
    ode_env = TimeLimit(ode_env, timesteps)

    states_per_run = []
    for run in range(runs):
        states = simulate_lockdown(env)
        states_per_run.append(states)
        
    # plots assume 3 compartments
    ode_states = simulate_lockdown(ode_env)
    plot_simulation(states_per_run, ode_states, env.datapoints)
