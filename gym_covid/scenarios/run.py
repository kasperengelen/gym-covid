import sys
import os

from numba import jit

sys.path.append(os.getcwd())
import matplotlib.pyplot as plt
import argparse
import pandas as pd
import datetime
import numpy as np

import gym_covid


def plot_states(trajectories):
    i_hosp = trajectories[:, :, -3].sum(axis=2)
    i_icu = trajectories[:, :, -2].sum(axis=2)
    deaths = trajectories[:, :, -1].sum(axis=2)
    i_hosp_std = i_hosp.std(axis=0)
    i_hosp = i_hosp.mean(axis=0)
    i_icu_std = i_icu.std(axis=0)
    i_icu = i_icu.mean(axis=0)
    deaths_std = deaths.std(axis=0)
    deaths = deaths.mean(axis=0)

    deaths_up = deaths + deaths_std
    print(f"Deaths + sigma <= {deaths_up.max()}")
    print(f"Deaths <= {deaths.max()}")

    axs = plt.gcf().axes
    # hospitalizations
    ax = axs[0]
    ax.plot(i_hosp,  # alpha=alpha,
            label='Other',
            color='green')
    #print(i_hosp.shape)
    ax.fill_between(np.arange(len(i_hosp)),
                    i_hosp - i_hosp_std,
                    i_hosp + i_hosp_std,
                    color='green', alpha=0.2)
    ax.plot(i_icu,  # alpha=alpha,
            label='ICU',
            color='orange')
    ax.fill_between(np.arange(len(i_icu)),
                    i_icu - i_icu_std,
                    i_icu + i_icu_std,
                    color='orange', alpha=0.2)
    #ax.plot(i_hosp_new+i_icu_new,
    #        label='Total',
    #        alpha=alpha, color='blue')
    ax.legend(loc="best")

    # deaths
    ax = axs[1]
    ax.plot(deaths,  # alpha=alpha,
            label='Deaths',
            color='red')
    ax.fill_between(np.arange(len(deaths)),
                    deaths - deaths_std,
                    deaths + deaths_std,
                    color='red', alpha=0.2)


def plot_states_small(trajectories):
    i_hosp = trajectories[:, :, -3].sum(axis=2)
    i_icu = trajectories[:, :, -2].sum(axis=2)
    deaths = trajectories[:, :, -1].sum(axis=2)
    i_hosp_std = i_hosp.std(axis=0)
    i_hosp = i_hosp.mean(axis=0)
    i_icu_std = i_icu.std(axis=0)
    i_icu = i_icu.mean(axis=0)
    deaths_std = deaths.std(axis=0)
    deaths = deaths.mean(axis=0)

    deaths_up = deaths + deaths_std
    print(f"Deaths + sigma <= {deaths_up.max()}")
    print(f"Deaths <= {deaths.max()}")

    axs = plt.gcf().axes
    # hospitalizations
    ax = axs[0]
    ax.plot(i_hosp,  # alpha=alpha,
            label='Other hosp.',
            color='green')
    #print(i_hosp.shape)
    ax.fill_between(np.arange(len(i_hosp)),
                    i_hosp - i_hosp_std,
                    i_hosp + i_hosp_std,
                    color='green', alpha=0.2)
    ax.plot(i_icu,  # alpha=alpha,
            label='ICU',
            color='orange')
    ax.fill_between(np.arange(len(i_icu)),
                    i_icu - i_icu_std,
                    i_icu + i_icu_std,
                    color='orange', alpha=0.2)
    #ax.plot(i_hosp_new+i_icu_new,
    #        label='Total',
    #        alpha=alpha, color='blue')
    ax.legend(loc="best")

    # deaths
    ax = axs[0]
    ax.plot(deaths,  # alpha=alpha,
            label='Deaths',
            color='red')
    ax.fill_between(np.arange(len(deaths)),
                    deaths - deaths_std,
                    deaths + deaths_std,
                    color='red', alpha=0.2)


def plot_simulation(states_per_stoch_run, ode_states=None, datapoints=None,
                    xlim=183, ylim=[1000, 300]):

    _, axs = plt.subplots(2, 1)

    # these are the colored lines that indicate the compartment values
    #if ode_states is not None:
    #    plot_states(ode_states, 1.)
    # we also have multiple lighter lines for the stochastic states
    states_array = np.array(states_per_stoch_run)

    #for i, states in enumerate(states_per_stoch_run):
    #    plot_states(states, 0.2, i)
    plot_states(states_array)

    # these are the dots on the plot
    if datapoints is not None:
        h = datapoints['hospitalizations']
        axs[0].scatter(np.arange(len(h)), h,
                       facecolors='none', edgecolors='black')
        d = datapoints['deaths']
        axs[1].scatter(np.arange(len(d)), d,
                       facecolors='none', edgecolors='black')

    #axs[0].set_xlabel('days')
    axs[0].set_ylabel('hospitalizations')

    axs[1].set_xlabel('days')
    axs[1].set_ylabel('deaths')
    for ax in axs:
        ax.set_xlim([0, xlim])
    axs[0].set_ylim([0, ylim[0]])
    axs[1].set_ylim([0, ylim[1]])

    plt.legend(loc="best")
    plt.show()


def plot_smaller(states_per_stoch_run, ode_states=None, datapoints=None,
                 xlim=183, ylim=[1000, 300]):

    plt.rcParams.update({'font.size': 20})
    _, axs = plt.subplots(1, 1)
    axs.xaxis.label.set_fontsize(20)
    axs.yaxis.label.set_fontsize(20)

    # these are the colored lines that indicate the compartment values
    #if ode_states is not None:
    #    plot_states(ode_states, 1.)
    # we also have multiple lighter lines for the stochastic states
    states_array = np.array(states_per_stoch_run)

    #for i, states in enumerate(states_per_stoch_run):
    #    plot_states(states, 0.2, i)
    plot_states_small(states_array)

    #axs[0].set_xlabel('days')
    #axs[0].set_ylabel('hospitalizations')

    axs.set_xlabel('days')
    axs.set_ylabel('people')
    axs.set_xlim([0, xlim])
    axs.set_ylim([0, ylim[0]])
    #axs[0].set_ylim([0, ylim[1]])

    plt.legend(loc="best")
    plt.show()


def simulate_scenario(env, scenario):
    states = []
    s = env.reset()
    d = False
    trunc = False
    timestep = 0
    ret = 0
    # at start of simulation, no restrictions are applied
    action = np.ones(3)
    actions = []
    rewards = []
    # today = datetime.date(2020, 3, 1)
    days = []

    while not (d or trunc):
        # at every timestep check if there are new restrictions
        s = scenario[scenario['timestep'] == timestep]
        if len(s):
            # print(f'timesteps {timestep}: {s["phase"]}')
            # found new restrictions
            action = np.array([s['work'].iloc[0], s['school'].iloc[0],
                               s['leisure'].iloc[0]])

        s, r, d, trunc, info = env.step(action)
        # state is tuple (compartments, events, prev_action),
        # only keep compartments
        states.append(s[0])
        timestep += 1
        ret += r
        actions.append(action)
        rewards.append(r)
        for i in range(7):
            days.append(datetime.date(2020, 3, 1) +
                        datetime.timedelta(days=(timestep-1)*7+i))
    # array of shape [Week DayOfWeek Compartment AgeGroup]
    states = np.stack(states, 0)
    # print(f"shape of states 1 = {states.shape}")
    # print(ret)
    # reshape to [Day Compartment AgeGroup]
    states = np.array(states).reshape(states.shape[0]*states.shape[1],
                                      *states.shape[2:])
    # print(f"shape of states 2 = {states.shape}")

    #with open('/tmp/run.csv', 'a') as f:
    #    f.write('dates,i_hosp_new,i_icu_new,d_new,p_w,p_s,p_l')
    #    i_hosp_new = states[:, -3].sum(axis=1)
    #    i_icu_new = states[:, -2].sum(axis=1)
    #    d_new = states[:, -1].sum(axis=1)
    #    # actions.append(actions[-1])
    #    actions = np.array(actions)
    #    rewards = np.stack(rewards, 0)
    #    actions = actions.repeat(7, 0)
    #    rewards = rewards.repeat(7, 0)
    #    for i in range(len(i_hosp_new)):
    #        f.write(f'{days[i]},{i_hosp_new[i]},{i_icu_new[i]},'
    #                f'{d_new[i]},{actions[i][0]},{actions[i][1]},'
    #                f'{actions[i][2]}\n')

    return states


@jit(nopython=True)
def set_seed_numba(seed):
    """
        Set the seed used by numba.
    """
    np.random.seed(seed)


if __name__ == '__main__':
    import gymnasium
    # from gym_covid import envs
    from gymnasium.wrappers import TimeLimit

    parser = argparse.ArgumentParser(description='')
    parser.add_argument('scenario', type=str, help='Scenario file to be run.')
    parser.add_argument('--runs', default=1, type=int,
                        help='Number of binomial runs. '
                             'Use 0 for ODE run only. Default : 1')
    parser.add_argument('--seed', default=22122021, type=int,
                        help='RNG seed. Default : 22122021')

    args = parser.parse_args()
    # print(args)
    runs = args.runs

    np.random.seed(seed=args.seed)
    # for numba we need to set the seed in a JIT-compiled context as well.
    set_seed_numba(seed=args.seed)
    # load the environments
    bin_env = gymnasium.make('BECovidBinomialContinuous-v0')
    ode_env = gymnasium.make('BECovidODEContinuous-v0')
    days_per_timestep = 7

    # simulation timesteps in weeks
    start = datetime.date(2020, 3, 1)
    #end = datetime.date(2020, 9, 5)
    #timesteps = round((end-start).days/days_per_timestep)

    # apply timestep limit to environments
    #bin_env = TimeLimit(bin_env, timesteps)
    #ode_env = TimeLimit(ode_env, timesteps)

    # load scenario and convert phase-dates to timesteps
    scenario = pd.read_csv(args.scenario)
    scenario['date'] = scenario['date'].astype(str)

    def to_timestep(d):
        return round((datetime.datetime.strptime(d, '%Y-%m-%d')
                     .date()-start).days/days_per_timestep)

    scenario['timestep'] = [to_timestep(d) for d in scenario['date']]
    # print(scenario)

    states_per_run = []
    for run in range(runs):
        states = simulate_scenario(bin_env, scenario)
        states_per_run.append(states)

    # plots assume 3 compartments
    ode_states = simulate_scenario(ode_env, scenario)
    plot_simulation(states_per_run, ode_states)  # , bin_env.datapoints)
