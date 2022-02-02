from gym.envs.registration import register
from gym.wrappers import TimeLimit
from gym_covid.envs.model import ODEModel, BinomialModel
from gym_covid.envs.epi_env import EpiEnv
from gym_covid.envs.discrete_actions import DiscreteAction
from gym_covid.envs.lockdown import Lockdown
import numpy as np
import pandas as pd
import json
from pathlib import Path
from importlib_resources import files, as_file
import datetime


def be_config():
    resources = files('gym_covid')    
    config_file = 'config/wave1.json'
    with open(resources / config_file, 'r') as f:
        config = json.load(f)

    contact_types = ['home', 'work', 'transport', 'school', 'leisure', 'otherplace']
    csm = [pd.read_csv(resources / config['social_contact_dir'] / f'{ct}.csv', header=None).values for ct in contact_types]
    csm = np.array(csm)

    ## set paths correctly for 'population' and 'cases' items
    config['cases'] = resources / config['cases']
    config['population'] = resources / config['population']

    ## DATAPOINTS
    df = pd.read_csv(resources / config['deaths'])
    df = df.sort_values('DATE')
    df = df[df['PLACE'] != 'Nursing home']
    df = df.groupby('DATE').agg('count')
    deaths = df['ID'].values
    # startdate is 10/03/22
    deaths = np.concatenate((np.zeros(9), deaths))

    df = pd.read_csv(resources / config['hospitalizations'])
    hospitalization = df['NewPatientsNotReferredHospital'].values.flatten()
    #startdate is 08/03/22
    hospitalization = np.concatenate((np.zeros(7), hospitalization))

    datapoints = {
        'hospitalizations': hospitalization,
        'deaths': deaths,
        }

    return config, csm, datapoints


def be_ode():
    config, csm, datapoints = be_config()
    model = ODEModel.from_config(config)
    env = EpiEnv(model, C=csm, beta_0=config['beta_0'], beta_1=config['beta_1'], datapoints=datapoints)
    return env


def be_binomial():
    config, csm, datapoints = be_config()
    model = BinomialModel.from_config(config)
    env = EpiEnv(model, C=csm, beta_0=config['beta_0'], beta_1=config['beta_1'], datapoints=datapoints)
    return env


def until_2020_09_01(env):
    end = datetime.date(2020, 9, 1)
    timesteps = round((end-env.today).days/env.days_per_timestep)
    return TimeLimit(env, timesteps)


def discretize_actions(env, work=None, school=None, leisure=None):
    if work is None:
        work = np.array([0, 30, 60])/100
    if school is None:
        school = np.array([0, 50, 100])/100
    if leisure is None:
        leisure = np.array([30, 60, 90])/100
    # all combinations of work, school, leisure
    actions = np.meshgrid(work, school, leisure)
    actions = np.stack(actions).reshape(3, -1).T
    return DiscreteAction(env, actions)


def create_env(env_type='ODE', discrete_actions=False, simulate_lockdown=True):
    if env_type == 'ODE':
        env = be_ode()
    else:
        env = be_binomial()
    env = until_2020_09_01(env)
    if simulate_lockdown:
        env = Lockdown(env)
    if discrete_actions:
        env = discretize_actions(env)
    return env


for env_type in ('ODE', 'Binomial'):
    for discrete_actions in (False, True):
        for simulate_lockdown in (False, True):
            a = 'Discrete' if discrete_actions else 'Continuous'
            l = 'WithLockdown' if simulate_lockdown else ''
            # envs
            register(
                id=f'BECovid{l}{env_type}{a}-v0',
                entry_point='gym_covid.envs:create_env',
                kwargs={
                    'env_type': env_type,
                    'discrete_actions': discrete_actions,
                    'simulate_lockdown': simulate_lockdown}
                )
