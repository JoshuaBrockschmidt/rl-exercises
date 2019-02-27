import gym
import gym_minigrid

from gym_minigrid.minigrid import Floor, Goal, DIR_TO_VEC, MiniGridEnv

class GridEnv:
    ACTIONS = MiniGridEnv.Actions

    _ENV_NAME = "MiniGrid-SimpleCrossingS11N5-v0"

    def __init__(self):
        self.env = gym.make(self._ENV_NAME)
        self.width = self.env.grid.width
        self.height = self.env.grid.height
        self.reset()

    def reset(self):
        self.env.reset()
        self.is_done = False
        if hasattr(self.env, 'mission'):
            print('mission: %s' % self.env.mission)

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        self.is_done = done

    def render(self):
        self.env.render('human')

    def is_walkable(self, x, y):
        obj = self.env.grid.get(x, y)
        if obj is None:
            return True
        else:
            return isinstance(obj, Floor) or isinstance(obj, Goal)

    def is_in_grid(self, x, y):
        return x >= 0 and x < self.width and y >= 0 and y < self.height

    def get_agent_pos(self):
        """
        Get the agent's position.

        Returns:
        	2D Numpy array representing the agent's position.
        """
        return self.env.agent_pos

    def get_agent_dir(self):
        """
        Get agent's direction as a vector.

        Return:
        	2D Numpy array representing the agent's direction.
        """
        return DIR_TO_VEC[self.env.agent_dir]

    def find_goals(self):
        """
        Finds all goals within the grid.

        Returns:
        	Collection of coordinates of all goals.
        """
        goals = []
        for x in range(0, self.width):
            for y in range(0, self.height):
                obj = self.env.grid.get(x, y)
                if isinstance(obj, Goal):
                    goals.append((x, y))
        return goals
