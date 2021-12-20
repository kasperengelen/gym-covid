import sys
import os
sys.path.append(os.getcwd())
import matplotlib.pyplot as plt

def plot_simulation(states):
    #fig, axs = plt.subplots(2, 1)

    i_hosp = states[:, 6::10].sum(axis=1)
    plt.plot(i_hosp)
    plt.xlabel("weeks")
    plt.ylabel("n hosp")
    print(len(states))

    """
    for ag in range(2):
        ax = axs[ag]
        s, i, r = states[:, ag*3+0], states[:, ag*3+1], states[:, ag*3+2]
        # lst[0::10]
        
        s, e = [sum(state[0::10]) for state in states], [sum(state[1::11]) for state in states]
        i_presym, i_asym =  [sum(state[2::12]) for state in states], [sum(state[3::13]) for state in states]
        i_mild, i_sev = [sum(state[4::14]) for state in states], [sum(state[5::15]) for state in states]
        i_hosp, i_icu = [sum(state[6::16]) for state in states], [sum(state[7::17]) for state in states]
        d, r = [sum(state[8::18]) for state in states], [sum(state[9::19]) for state in states]


        print(s)
        ax.plot(s, c='b', label='s')
        ax.plot(i_hosp, c='r', label='i_hosp')
        #ax.plot(r, c='g', label='r')
        ax.set_xlabel('week')
        ax.set_ylabel('pop')
        ax.legend()
    """
    plt.show()

def plot_i_hosp(states):

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
    
    # load ode model
    #env = gym.make('EpiBelgiumODEContinuous-v0')

    # load binomial model
    env = gym.make('EpiBelgiumBinomialContinuous-v0')

    env = TimeLimit(env, 60)
    states = []
    s = env.reset()
    d = False
    states.append(s)
    print("initial state", s)
    while not d:
        # action doesnt matter, hardcoded in MDP
        a = np.ones(3)
        s, r, d, _ = env.step(a)
        print(s, r)
        states.append(s)
    
    states = np.array(states)
    # plots assume 3 compartments
    # plot_simulation(states)
    plot_i_hosp(states)