import numpy as np
from abc import ABC, abstractmethod
from typing import Dict, Any, Tuple, List, Union

class BaseArtifact(ABC):
    """
    Abstract base class for all artifacts applied during SEM image generation.
    """
    def __init__(self, parameters: Dict[str, Any]):
        """
        Initialize the artifact with its specific parameters.

        Args:
            parameters (Dict[str, Any]): Dictionary containing parameters specific
                                         to this artifact instance (e.g., amplitude,
                                         sigma, probability), already sampled from
                                         the configured ranges by the raffle mechanism.
        """
        self.params = parameters

    @abstractmethod
    def apply(self, image_data: np.ndarray, **kwargs) -> Union[np.ndarray, Tuple[np.ndarray, List[np.ndarray]]]:
        """
        Apply the artifact effect.

        The exact signature and behavior depend on the artifact type (shape-level vs image-level).
        Subclasses must implement this method according to their needs.

        Args:
            image_data (np.ndarray): The primary image data (can be a mask for shape-level,
                                     or the composed image for image-level).
            **kwargs: Additional arguments needed by specific artifacts. Common examples:
                masks (List[np.ndarray]): List of associated masks to transform (for image-level geom.).
                vertices (np.ndarray): Shape vertices (for some shape-level).
                shape_center (Tuple[float, float]): Center of the shape (for some shape-level).

        Returns:
            Union[np.ndarray, Tuple[np.ndarray, List[np.ndarray]]]:
            - For shape-level artifacts primarily modifying geometry via vertices: returns modified vertices.
            - For shape-level artifacts modifying masks: returns the modified mask.
            - For image-level intensity/blur artifacts: returns the modified image_data.
            - For image-level geometric artifacts: returns a tuple (modified_image_data, modified_masks).
        """
        pass

    def get_param(self, key: str, default: Any = None) -> Any:
        """Helper to safely get a parameter value."""
        return self.params.get(key, default)