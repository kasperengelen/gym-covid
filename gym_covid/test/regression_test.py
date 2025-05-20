import datetime
import pickle
from pathlib import Path

import gym
import numpy as np
import pandas as pd
from gym.wrappers import TimeLimit
from numba import jit
import json_numpy

import gym_covid


def get_test_cases_root():
    return Path(gym_covid.test.__file__).parent / "test_cases"


def write_traj_to_json(traj: np.ndarray, file: Path):
    """
        Write trajectory to a JSON file.
    """
    with open(file, "w") as f:
        json_numpy.dump(obj=traj, fp=f)


def read_traj_from_json(file: Path):
    """
        Read trajectory from a JSON file.
    """
    with open(file, "r") as f:
        traj = json_numpy.load(f)
        return traj


@jit(nopython=True)
def set_seed_numba(seed):
    np.random.seed(seed)


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
    # NOTE: when using numba you to set a seed separately in a JIT compiled function!
    set_seed_numba(seed=22122021)

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

    # Shape = [Days, Compartments, AgeGroups]
    states = simulate_scenario(bin_env, scenario)

    # retrieve reference
    json_path = get_test_cases_root() / "baseline_seed=22122021_bin.json"
    # write_traj_to_json(traj=states, file=json_path)
    reference = read_traj_from_json(file=json_path)

    # compare every compartment at every day
    for day in range(states.shape[0]):
        for compartment in range(states.shape[1]):
            # check that all age groups are the same
            assert np.allclose(states[day, compartment], reference[day, compartment]), \
                f"Error: mismatch in compartment '{compartment}' on day '{day}'"


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
    assert scenario_path.is_file(), "Error: missing scenario CSV file"
    scenario = pd.read_csv(scenario_path)
    scenario['date'] = scenario['date'].astype(str)
    to_timestep = lambda d: round((datetime.datetime.strptime(d, '%Y-%m-%d').date() - start).days / days_per_timestep)
    scenario['timestep'] = [to_timestep(d) for d in scenario['date']]

    # plots assume 3 compartments
    # Shape = [Days, Compartments, AgeGroups]
    ode_states = simulate_scenario(ode_env, scenario)

    # get reference data
    json_path = get_test_cases_root() / "baseline_seed=22122021_ode.json"
    # write_traj_to_json(traj=ode_states, file=json_path)
    reference = read_traj_from_json(file=json_path)

    # compare every compartment at every day
    for day in range(ode_states.shape[0]):
        for compartment in range(ode_states.shape[1]):
            # check that all age groups are the same
            assert np.allclose(ode_states[day, compartment], reference[day, compartment]), \
                f"Error: mismatch in compartment '{compartment}' on day '{day}'"
