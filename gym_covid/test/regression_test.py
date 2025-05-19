import datetime
from pathlib import Path

import gym
import numpy as np
import pandas as pd
from gym.wrappers import TimeLimit

import gym_covid


def get_test_cases_root():
    return Path(gym_covid.test.__file__) / "test_cases"


def simulate_scenario(env, scenario):
    states = []
    s = env.reset()
    d = False
    timestep = 0
    ret = 0
    # at start of simulation, no restrictions are applied
    action = np.ones(3)
    actions = []
    rewards = []
    today = datetime.date(2020, 3, 1)
    days = []

    while not d:
        # at every timestep check if there are new restrictions
        s = scenario[scenario['timestep'] == timestep]
        if len(s):
            print(f'timesteps {timestep}: {s["phase"]}')
            # found new restrictions
            action = np.array([s['work'].iloc[0], s['school'].iloc[0], s['leisure'].iloc[0]])

        s, r, d, info = env.step(action)
        # state is tuple (compartments, events, prev_action), only keep compartments
        states.append(s[1])
        timestep += 1
        ret += r
        actions.append(action)
        rewards.append(r)
        for i in range(7):
            days.append(datetime.date(2020, 3, 1)+datetime.timedelta(days=(timestep-1)*7+i))
    # array of shape [Week DayOfWeek Compartment AgeGroup]

    states = np.stack(states, 0)
    print(ret)
    # reshape to [Day Compartment AgeGroup]
    states = np.array(states).reshape(states.shape[0]*states.shape[1], *states.shape[2:])

    return states


def test_regression_bin():
    """
        Small test to verify that the numerical outputs of the model are unchanged w.r.t. an earlier reference point.

        This test s the "BECovidBinomialContinuous-v0" model, with seed "22122021" and a single run. The code
        for this test is taken from the "scenarios/run.py" file.

        The scenario CSV is located in "test/test_cases" and is identical to the "scenarios/baseline.csv" file.
    """
    np.random.seed(seed=22122021)

    # load the environments
    bin_env = gym.make('BECovidBinomialContinuous-v0')
    days_per_timestep = bin_env.days_per_timestep
    runs = 1

    # simulation timesteps in weeks
    start = datetime.date(2020, 3, 1)
    end = datetime.date(2020, 9, 5)
    timesteps = round((end - start).days / days_per_timestep)

    # apply timestep limit to environments
    bin_env = TimeLimit(bin_env, timesteps)

    # load scenario and convert phase-dates to timesteps
    scenario_path = get_test_cases_root() / "baseline.csv"
    scenario = pd.read_csv(scenario_path)
    scenario['date'] = scenario['date'].astype(str)
    to_timestep = lambda d: round((datetime.datetime.strptime(d, '%Y-%m-%d').date() - start).days / days_per_timestep)
    scenario['timestep'] = [to_timestep(d) for d in scenario['date']]
    print(scenario)

    states_per_run = []
    for run in range(runs):
        states = simulate_scenario(bin_env, scenario)
        states_per_run.append(states)

    # TODO: store states_per_run, ode_states, bin_env.datapoints in pickle
    # plot_simulation(states_per_run, ode_states, bin_env.datapoints)


def test_regression_ode():
    """
        Small test to verify that the numerical outputs of the model are unchanged w.r.t. an earlier reference point.

        This test s the "BECovidODEContinuous-v0" model, with seed "22122021" and a single run. The code
        for this test is taken from the "scenarios/run.py" file.

        The scenario CSV is located in "test/test_cases" and is identical to the "scenarios/baseline.csv" file.
    """
    np.random.seed(seed=22122021)

    # load the environments
    ode_env = gym.make('BECovidODEContinuous-v0')
    days_per_timestep = ode_env.days_per_timestep
    runs = 1

    # simulation timesteps in weeks
    start = datetime.date(2020, 3, 1)
    end = datetime.date(2020, 9, 5)
    timesteps = round((end - start).days / days_per_timestep)

    # apply timestep limit to environments
    ode_env = TimeLimit(ode_env, timesteps)

    # load scenario and convert phase-dates to timesteps
    scenario_path = get_test_cases_root() / "baseline.csv"
    scenario = pd.read_csv(scenario_path)
    scenario['date'] = scenario['date'].astype(str)
    to_timestep = lambda d: round((datetime.datetime.strptime(d, '%Y-%m-%d').date() - start).days / days_per_timestep)
    scenario['timestep'] = [to_timestep(d) for d in scenario['date']]
    print(scenario)

    # plots assume 3 compartments
    ode_states = simulate_scenario(ode_env, scenario)

    # TODO: store states_per_run, ode_states, bin_env.datapoints in pickle
    # plot_simulation(states_per_run, ode_states, bin_env.datapoints)
