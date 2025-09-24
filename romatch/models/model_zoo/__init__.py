from typing import Union
import torch
from .roma_models import roma_model, tiny_roma_v1_model


weight_urls = {
    "romatch": {
        "outdoor": "https://github.com/Parskatt/storage/releases/download/roma/roma_outdoor.pth",
        "indoor": "https://github.com/Parskatt/storage/releases/download/roma/roma_indoor.pth",
    },
    "tiny_roma_v1": {
        "outdoor": "https://github.com/Parskatt/storage/releases/download/roma/tiny_roma_v1_outdoor.pth",
    },
    "dinov2": "https://dl.fbaipublicfiles.com/dinov2/dinov2_vitl14/dinov2_vitl14_pretrain.pth",  # hopefully this doesnt change :D
}


def tiny_roma_v1_outdoor(device, weights=None, xfeat=None):
    if weights is None:
        weights = torch.hub.load_state_dict_from_url(
            weight_urls["tiny_roma_v1"]["outdoor"], map_location=device
        )
    if xfeat is None:
        xfeat = torch.hub.load(
            "verlab/accelerated_features", "XFeat", pretrained=True, top_k=4096
        ).net

    return tiny_roma_v1_model(weights=weights, xfeat=xfeat).to(device)


def roma_outdoor(
    device,
    weights=None,
    dinov2_weights=None,
    coarse_res: Union[int, tuple[int, int]] = 560,
    upsample_res: Union[int, tuple[int, int]] = 864,
    amp_dtype: torch.dtype = torch.float16,
    symmetric=True,
    use_custom_corr=True,
    upsample_preds=True,
):
    if weights is None:
        weights = torch.hub.load_state_dict_from_url(
            weight_urls["romatch"]["outdoor"], map_location=device
        )
    if dinov2_weights is None:
        dinov2_weights = torch.hub.load_state_dict_from_url(
            weight_urls["dinov2"], map_location=device
        )
    model = roma_model(
        resolution=coarse_res,
        upsample_preds=upsample_preds,
        weights=weights,
        dinov2_weights=dinov2_weights,
        device=device,
        amp_dtype=amp_dtype,
        symmetric=symmetric,
        use_custom_corr=use_custom_corr,
        upsample_res=upsample_res,
    )
    return model


def roma_indoor(
    device,
    weights=None,
    dinov2_weights=None,
    coarse_res: Union[int, tuple[int, int]] = 560,
    upsample_res: Union[int, tuple[int, int]] = 864,
    amp_dtype: torch.dtype = torch.float16,
    symmetric=True,
    use_custom_corr=True,
    upsample_preds=True,
):
    if weights is None:
        weights = torch.hub.load_state_dict_from_url(
            weight_urls["romatch"]["indoor"], map_location=device
        )
    if dinov2_weights is None:
        dinov2_weights = torch.hub.load_state_dict_from_url(
            weight_urls["dinov2"], map_location=device
        )
    model = roma_model(
        resolution=coarse_res,
        upsample_preds=upsample_preds,
        weights=weights,
        dinov2_weights=dinov2_weights,
        device=device,
        amp_dtype=amp_dtype,
        symmetric=symmetric,
        use_custom_corr=use_custom_corr,
        upsample_res=upsample_res,
    )
    return model