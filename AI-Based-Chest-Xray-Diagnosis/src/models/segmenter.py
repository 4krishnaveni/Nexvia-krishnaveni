import torch
from monai.networks.nets import UNet


def build_unet() -> UNet:
    """2-D U-Net (single-channel in, single-channel logit out).
    NOTE: the brief says 3D U-Net; RSNA is 2-D X-ray so we use spatial_dims=2.
    For real volumes (MSD Task06) change spatial_dims to 3."""
    return UNet(
        spatial_dims=2, in_channels=1, out_channels=1,
        channels=(16, 32, 64, 128, 256), strides=(2, 2, 2, 2), num_res_units=2,
    )


def load_unet(path: str, device="cpu"):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model = build_unet()
    model.load_state_dict(ckpt["state_dict"])
    return model.to(device).eval(), ckpt
