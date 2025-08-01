from typing import Any

import gymnasium
import datetime
import numpy as np


class Lockdown(gymnasium.Wrapper):

    def __init__(self, env):
        super(Lockdown, self).__init__(env)
        # start and end of lockdown
        start = datetime.date(2020, 3, 14)
        end = datetime.date(2020, 5, 4)
        # convert in timesteps
        def to_timestep(d):
            return round((d-self.today).days/self.days_per_timestep)
        self.lockdown_start = to_timestep(start)
        self.lockdown_end = to_timestep(end)
        # lockdown policy
        self.lockdown_policy = np.array([0.2, 0.0, 0.1])

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        # reset base-env
        s = super(Lockdown, self).reset(seed=seed, options=options)
        # action under no restrictions
        action = np.ones(3)
        # let first wave pass under lockdown, start-state is start of exit strategy
        for t in range(self.lockdown_end):
            if t == self.lockdown_start:
                action = self.lockdown_policy
            s, _, _, _, _ = self.env.step(action)
        return s


if __name__ == '__main__':
    import gym_covid.envs
    env = gymnasium.make('EpiBelgiumODEContinuous-v0')
    env = gymnasium.wrappers.TimeLimit(env, 27)
    env = Lockdown(env)
    env.reset()
    d = False
    t = 0
    while not d:
        print(t)
        t += 1
        s, _, d, _, _ = env.step(np.array([0.3,0,0.2]))
    print(s[-1, 0])