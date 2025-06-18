import gymnasium


class DiscreteAction(gymnasium.ActionWrapper):

    def __init__(self, env, action_values):
        super(DiscreteAction, self).__init__(env)

        self.action_space = gymnasium.spaces.Discrete(len(action_values))
        self.action_map = action_values

    def action(self, action):
        return self.action_map[action]