# Interface for IMP environment

class ImpEnv(object):
    def reset(self) -> dict:
        """
        Returns damage probabilities in a dictionary.
        """
        raise NotImplementedError

    def step(self, action: dict):
        """
        Returns damage probabilities, reward, terminated, inspection_infos.
        """
        raise NotImplementedError
