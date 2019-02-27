from gridenv import GridEnv

import numpy as np

class Navigator:
    """
    Executes a navigation plan.
    """

    def __init__(self, env, path):
        """
        Args:
        	env: GridEnv to execute plan within.
        	path: Ordered collection of tile coordinates to visit ending with the destination.
        """
        self.env = env
        self.path = list(path)
        self._trim_path()

    def _trim_path(self):
        # Removes any tile coordinates at the top of the path queue that share the agent's position.
        pos = self.env.get_agent_pos()
        while (len(self.path) > 0 and self.path[0][0] == pos[0] and self.path[0][1] == pos[1]):
            self.path.pop(0)

    def has_next(self):
        """
        Checks whether there are any actions left to perform.

        Returns:
        	True if there are actions to perform and False otherwise.
        """
        return len(self.path) > 0

    def do_next_step(self):
        """
        Performs the next planned action, if any.
        """

        if not self.has_next():
            return

        pos = self.env.get_agent_pos()
        needed_dir = np.subtract(self.path[0], pos)
        actual_dir = self.env.get_agent_dir()
        needed_theta = np.arctan2(needed_dir[1], needed_dir[0])
        actual_theta = np.arctan2(actual_dir[1], actual_dir[0])

        # Different between the direction we need to face and the agent's current direction.
        theta_diff = (needed_theta - actual_theta) % (2 * np.pi)

        # Calculate what multiple of pi/2 theta_diff is nearest to.
        theta_mult = int((theta_diff * (2 / np.pi)) + 0.5)

        if theta_mult == 0:
            # Go forward.
            self.env.step(GridEnv.ACTIONS.forward)
            self._trim_path()
        elif theta_mult == 1:
            # Turn left.
            self.env.step(GridEnv.ACTIONS.right)
        else:
            # Turn right, including for U-turns.
            self.env.step(GridEnv.ACTIONS.left)
