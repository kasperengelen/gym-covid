from gym.envs.registration import register
from envs.epi_env import EpiEnv
from envs.ode_sir_model import OdeSIREpiModel
from envs.ode_epi_model import OdeEpiModel
from envs.epi_model import __funcs__
import numpy as np
import pandas as pd


def load_c_be2010():
    def process_contact_csv(ct):
        c = pd.read_csv(f'data/contact_matrix/original/c_{ct}.csv', header=None).values
        #copy the 90+ age group from the 80-90 age group (col and row)
        c = np.hstack((c, np.tile(c[:, [-1]], 1)))
        c = np.vstack((c, np.tile(c[-1], 1)))
        return c
        
    contact_types = ['home', 'work', 'transport', 'school', 'leisure', 'otherplace']
    c = [process_contact_csv(ct) for ct in contact_types]
    c = np.array(c)
    return c


def load_initial_state():
    n_comp = 10
    funcs = __funcs__()
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

    # load confirmed cases
    cases = pd.read_csv('data/cases.csv')
    # keep first two weeks to compute the frequency if confirmed cases for each age group
    cases = cases[cases['DATE'] >= '2020-03-01']
    cases = cases[cases['DATE'] < '2020-03-14']
    age_cases = cases.groupby('AGEGROUP').agg(np.sum)
    rel_age_cases = age_cases/age_cases.sum()
    # Age-dependent asymptomatic proportions
    p_vec = np.array([0.94,0.90,0.84,0.61,0.49,0.21,0.02,0.02,0.02,0.02])
    # param from model, TODO load from config file
    n0 = np.exp(7.75220356739557)
    imported_cases = np.round(rel_age_cases.values.flatten()*n0*(1/(1-p_vec)),0)
    S = population_groups-imported_cases
    E = imported_cases

    initial_state = np.zeros(n_comp * len(population_groups))
    initial_state[funcs.S(np.arange(len(population_groups)))] = S
    initial_state[funcs.E(np.arange(len(population_groups)))] = E

    return initial_state

def epi_sir_belgium_ode():
    C = load_c_be2010()
    initial_state = load_initial_state()
    population = initial_state[:10] + initial_state[10:20]
    # TODO replace with our model
    model = OdeSIREpiModel(population, 10, initial_state, 0.05, 1/3)
    env = EpiEnv(model, __funcs__(), C=C)

    return env


def epi_belgium_ode():
    C = load_c_be2010()
    initial_state = load_initial_state()
    population = initial_state[:10] + initial_state[10:20]
    # TODO replace with our model
    model = OdeEpiModel(population, 10, initial_state, 0.05, 1/3)
    env = EpiEnv(model, __funcs__(), C=C)

    return env


def epi_ode():
    # run for `steps` weeks
    # model with 2 age groups
    model = OdeEpiModel(np.array([2.5e6, 7.5e6]), 2, np.array([2.5e6-1, 1, 0., 7.5e6-1, 1, 0]), 0.05, 1/3)
    # contact matrix
    C = np.array([[18, 6], [3, 12]])
    env = EpiEnv(model, __funcs__(), C=C[None])
    return env


# split_env
register(
    id='EpiODEContinuous-v0',
    entry_point='envs:epi_ode',
    )

register(
    id='EpiBelgiumODESIRContinuous-v0',
    entry_point='envs:epi_sir_belgium_ode',
    )

register(
    id='EpiBelgiumODEContinuous-v0',
    entry_point='envs:epi_belgium_ode',
    )

