# gym_covid
## A gym environment for a compartment model modelling the first COVID wave in Belgium.

A Python implementation of [1]. This is a compartment model used to analyse the first wave of COVID that occured in Belgium starting from 01 March 2020. It explicitly models hospitalizations and ICU-hospitalizations, as well as contact-reduction for symptomatic people (as they have to stay at home).

We implement both the stochastic (binomial) model as well as the ODE model described in the paper as Markov Decision Processes (MDPs) that follow the [Gym](https://github.com/openai/gym) API. More details about the compartment model can be found in the original paper.

 We now describe the different aspects of this MDP:

### States

Each state is a tuple that contains 3 elements:
The first is a 13x10 matrix that represent the 13 different compartments for each of the 10 different age-groups.

 - The different compartments are (in order): `S`, `E`, `I_presym`, `I_asym`, `I_mild`, `I_sev`, `I_hosp`, `I_icu`, `R`, `D`, `I_hosp_new`, `I_icu_new`, `D_new`

   Compared to [1], there are 3 additional compartments: `I_hosp_new`, `I_icu_new`, `D_new` that explicitly model the daily new hospitalizations and deaths. These compartments have no impact on the model dynamics and are only there for information-tracking purposes.

 - The different age-groups are (in order): [0, 10), [10, 20), [20, 30), [30, 40), [40, 50), [50, 60), [60, 70), [70, 80), [80, 90), [90, inf)

 The second is a Boolean flag telling if the current day is a school holiday.

 Finally, the third element is the previous action. This acts as a proxy for providing the full Social Contact Matrix (see below), which is required as the previous matrix impacts the way the epidemic evolves.

### Timesteps

At every timestep, you can choose an action (more on that later) that will lead you from state `S` to `S'`. Concretely, the model will simulate the evolution of the epidemic for **1 week** (and thus return the state of the epidemic one week later) before allowing you to take another action. 

Here are some key dates:
 - **2020-03-01**: start of the epidemic
 - **2020-03-14**: start of first lockdown in Belgium
 - **2020-05-04**: end of first lockdown, start of exit-strategy (reopening of schools, bars, reducing telework, etc)
 - **2020-07-01**: start of summer holidays (schools are closed)
 - **2020-09-01**: end of summer holidays (and end of simulation)

We provide 2 variants of the MDPs:
 - One starts at the start of the epidemic (2020-03-01), allowing you to simulate alternative strategies to the lockdown that occured in Belgium
 - One simulates the lockdown, and starts at the end of the lockdown (2020-05-04), allowing you to simulate alternative exit-strategies

### Actions

A deciding factor impacting the evolution of the epidemic is the physical contacts made by the population. How people interact witch eath other is modelled using a **Social Contact Matrix (SCM)**. It shows, for each age-group, how many persons of each other age-group they come in contact with every day. The SCMs are also environment-specific.

 - We define 6 SCMs (in order): `home`, `work`, `transport`, `school`, `leisure`, `otherplace`

   > The total SCM is the sum of these 6 matrices.

Government policies are implemented by enforcing restrictions on these environments. For example, the government could decide to close schools, resulting in zero physical contacts in the `school` environment.

Concretely, we have 3 *proportional reduction* factors that modulate the SCMs (a factor of 0 means zero-contact, while 1 means business-as-usual).

 - They are: `p_work`, `p_school` and `p_leisure`.

   > The total SCM  taking into account these factors is:  
   `total = home + p_work*work + p_work*transport + p_school*school + p_leisure*leisure + p_leisure*otherwise`

We provide 2 variants of the MDPs:
 - Either a 3-dimensional continuous-action MDP, where you can directly decide `p_work`, `p_school` and `p_leisure`.
 - A discretized variant with a pre-selection of possible reductions:
   * work: 0, 0.3, 0.6
   * school: 0, 0.5, 1
   * leisure: 0.3, 0.6, 0.9

   This results in the following actions:

    |action number|work|school|leisure|
    |:-----------:|:--:|:----:|:-----:|
    |0|0.0|0.0|0.3|
    |1|0.0|0.0|0.6|
    |2|0.0|0.0|0.9|
    |3|0.3|0.0|0.3|
    |4|0.3|0.0|0.6|
    |5|0.3|0.0|0.9|
    |6|0.6|0.0|0.3|
    |7|0.6|0.0|0.6|
    |8|0.6|0.0|0.9|
    |9|0.0|0.5|0.3|
    |10|0.0|0.5|0.6|
    |11|0.0|0.5|0.9|
    |12|0.3|0.5|0.3|
    |13|0.3|0.5|0.6|
    |14|0.3|0.5|0.9|
    |15|0.6|0.5|0.3|
    |16|0.6|0.5|0.6|
    |17|0.6|0.5|0.9|
    |18|0.0|1.0|0.3|
    |19|0.0|1.0|0.6|
    |20|0.0|1.0|0.9|
    |21|0.3|1.0|0.3|
    |22|0.3|1.0|0.6|
    |23|0.3|1.0|0.9|
    |24|0.6|1.0|0.3|
    |25|0.6|1.0|0.6|
    |26|0.6|1.0|0.9|

   > You can easily make your own discretization, using the `discretize_actions` function implemented in `envs/__init__.py`

________________
Finally, the measures take time to be applied. This is modelled by a *gradual compliance* logistic function. Full compliance is estimated to occur after 6 days. Thus, after each timestep (which is one week), population is fully compliant and the *gradual compliance* has no effect on the Markov property of our MDP. You can easily change the number of simulated days per timestep, but be aware this may affect your MDP.

## Rewards

At each timestep, the environment also provides a reward. We propose a **Multi-objective** reward, to analyze the different aspects that are affected by government policies.
We have 3 objectives, that are defined as costs:
 - attack rate of susceptibles: compares the difference of suceptibles between current and next state. `-(Sum(state_susceptible) - Sum(next_state_susceptible))`
 - newly hospitalized: `-Sum(state_hosp_new + state_icu_new)`
 - a simple proxy for social burden. We see this a the loss in contacts compared to business-as-usual

## Installation

You can install this environment by cloning the repository and using `pip`. It requires Python3.7+

```
pip install .
```

The different environments are named as follows:
```
BECovid<WithLockdown|><ODE|Binomial><Discrete|Continuous>-v0
```

To create an environment:
```
# env that first runs lockdown, uses the ODE model and discretizes actions
env = gym.make('BECovidWithLockdownODEDiscrete-v0')
# env the starts at beginning of pandemic, uses Binomial model and continuous actions
env = gym.make('BECovidBinomialContinuous-v0')
```

## Data

All parameters used in our implementation were taken from [1]. More information on how the data (SCMs, population counts, etc) was obtained can be found in the `data/` folder. The different exit-strategies analysed in the paper are present in the `scenarios/` folder and can be executed as follows (`baseline.csv` is lockdown only):
```
cd gym_covid
python scenarios/run.py scenarios/baseline.csv
```

Here is a plot showing the simulated hospitalization and deaths vs the reported ones during the lockdown period:
![wave 1 simulation](/docs/assets/wave1.png "A simulation of the first wave")

The MDPs without lockdown start at timestep 0, while the ones with lockdown start at the last timestep on this plot. 

[1]: Abrams, S., Wambua, J., Santermans, E., Willem, L., Kuylen, E., Coletti, P., ... & Hens, N. (2021). Modelling the early phase of the Belgian COVID-19 epidemic using a stochastic compartmental model and studying its implied future trajectories. Epidemics, 35, 100449.
