from typing import Dict, Any, Type

from .base_artifact import BaseArtifact
# Import specific artifact classes
from .shape_level import EdgeRipple, BreaksHoles # Add others as implemented
from .image_level import AffineTransform, ElasticMeshDeform, ProbePSF, Charging, TopographicShading # Add others
from .image_level.defocus_blur import DefocusBlur # Import new class
from .image_level.gradient_illumination import GradientIllumination # Import new class


# Map artifact type names (lowercase, from config) to their classes
ARTIFACT_CLASS_MAP: Dict[str, Type[BaseArtifact]] = {
    # Shape Level
    "edge_ripple": EdgeRipple,
    "breaks_holes": BreaksHoles,
    # "elastic_deformation_local": ElasticDeformationLocal, # If implemented
    # "segment_displacement": SegmentDisplacement, # If implemented
    # "contour_smoothing": ContourSmoothing, # If implemented
    # "local_brightness_thickness": LocalBrightnessThickness, # If implemented

    # Image Level
    "affine": AffineTransform,
    "elastic_mesh_deform": ElasticMeshDeform,
    "probe_psf": ProbePSF,
    "charging": Charging,
    "topographic_shading": TopographicShading,
    # "perspective": PerspectiveTransform, # If implemented
    # "rolling_banding": RollingBanding, # If implemented
    "gradient_illumination": GradientIllumination,
    # "striping_smearing": StripingSmearing, # Needs careful split for geom/intensity?
    # "scanline_drop": ScanlineDrop, # If implemented
    "defocus_blur": DefocusBlur,
    # "detector_fixed_pattern": FixedPatternNoise, # If implemented
}

def create_artifact(artifact_config: Dict[str, Any]) -> BaseArtifact:
    """
    Factory function to create an artifact instance based on its configuration.

    Args:
        artifact_config (Dict[str, Any]): Dictionary containing the artifact's 'type'
                                          and its specific 'parameters'.

    Returns:
        BaseArtifact: An instance of the requested artifact class.

    Raises:
        ValueError: If the artifact type is unknown or parameters are missing.
    """
    artifact_type = artifact_config.get('type', '').lower()
    parameters = artifact_config.get('parameters', {})

    if not artifact_type:
        raise ValueError("Artifact configuration must include a 'type' key.")

    ArtifactClass = ARTIFACT_CLASS_MAP.get(artifact_type)
    if ArtifactClass is None:
        raise ValueError(f"Unknown or unsupported artifact type: '{artifact_type}'")

    try:
        # Pass only the parameters dict to the artifact constructor
        return ArtifactClass(parameters=parameters)
    except KeyError as e:
        raise ValueError(f"Missing required parameter '{e}' for artifact type '{artifact_type}'") from e
    except Exception as e:
        raise RuntimeError(f"Error instantiating artifact type '{artifact_type}': {e}") from e


__all__ = [
    "BaseArtifact",
    "create_artifact",
    # Expose specific classes if needed directly
    "EdgeRipple",
    "BreaksHoles",
    "AffineTransform",
    "ElasticMeshDeform",
    "ProbePSF",
    "Charging",
    "TopographicShading",
    "DefocusBlur",
    "GradientIllumination",
]