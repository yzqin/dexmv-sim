class XMLError(Exception):
    """Exception raised for errors related to xml."""

    pass


class SimulationError(Exception):
    """Exception raised for errors during runtime."""

    pass


class RandomizationError(Exception):
    """Exception raised for really really bad RNG."""

    pass


class ModelError(Exception):
    """Exception raised for not allowed Mujoco Model before compilation."""

    pass
