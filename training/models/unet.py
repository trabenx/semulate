import torch
import segmentation_models_pytorch as smp

def build_model(arch: str, encoder_name: str, encoder_weights: str, in_channels: int, classes: int):
    """Builds a segmentation model using segmentation-models-pytorch."""
    print(f"Building model: {arch} with encoder {encoder_name}...")
    if arch.lower() == 'unet':
        model = smp.Unet(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=in_channels,
            classes=classes,
            # activation=None # Output logits, apply sigmoid/softmax in loss or post-processing
        )
    elif arch.lower() == 'deeplabv3+':
         model = smp.DeepLabV3Plus(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=in_channels,
            classes=classes,
         )
    # Add other architectures (FPN, PSPNet, Linknet...)
    else:
        raise ValueError(f"Unsupported architecture: {arch}")

    return model