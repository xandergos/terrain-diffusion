from abc import ABC, abstractmethod

class Sampler(ABC):
    """
    Base class for samplers.
    """

    @abstractmethod
    def get_region(self, top: int, left: int, bottom: int, right: int, generate=True):
        """
        Get a region of the image.

        Args:
            top (int): The top coordinate of the region to retrieve.
            left (int): The left coordinate of the region to retrieve.
            bottom (int): The bottom coordinate of the region to retrieve.
            right (int): The right coordinate of the region to retrieve.
            generate (bool, optional): Whether to generate the region or not. Defaults to True. If false, the region may be incomplete/noisy.
        Returns:
            A tensor containing the requested region of the image.
        """
        pass
