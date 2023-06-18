class ImpEnv(object):
    """ Interface for creating IMP environments.

    Methods:
        reset
        step
    """
    def reset(self) -> dict:
        """
        Returns damage probabilities in a dictionary and resets the environment.
        """
        raise NotImplementedError

    def step(self, action: dict):
        """
        Returns damage probabilities, reward, terminated, inspection_infos and transitions the environment one time step.

        Args:
            action: A dictionary containing the actions assigned by each agent.
        """
        raise NotImplementedError
