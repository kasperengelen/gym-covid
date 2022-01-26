from gym.envs.registration import register
from envs.model import ODEModel, BinomialModel
from envs.epi_env import EpiEnv
import numpy as np
import pandas as pd
import json
from pathlib import Path


def be_config():
    config_file = 'config/wave1.json'
    with open(config_file, 'r') as f:
        config = json.load(f)

    contact_types = ['home', 'work', 'transport', 'school', 'leisure', 'otherplace']
    csm = [pd.read_csv(Path(config['social_contact_dir']) / f'{ct}.csv', header=None).values for ct in contact_types]
    csm = np.array(csm)

    ## DATAPOINTS
    df = pd.read_csv(config['deaths'])
    df = df.sort_values('DATE')
    df = df[df['PLACE'] != 'Nursing home']
    df = df.groupby('DATE').agg('count')
    deaths = df['ID'].values
    # startdate is 10/03/22
    deaths = np.concatenate((np.zeros(9), deaths))

    df = pd.read_csv(config['hospitalizations'])
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


# envs
register(
    id='EpiBelgiumODEContinuous-v0',
    entry_point='envs:be_ode',
    )

register(
    id='EpiBelgiumBinomialContinuous-v0',
    entry_point='envs:be_binomial',
    )

