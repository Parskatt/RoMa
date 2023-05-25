import torch
from .roma_models import roma_model

weight_urls = {
    "roma": {
        "outdoor": "https://github.com/Parskatt/storage/releases/download/roma/roma_outdoor.pth",
        "indoor": "https://github.com/Parskatt/storage/releases/download/roma/roma_indoor.pth",
    },
    "dinov2": "https://dl.fbaipublicfiles.com/dinov2/dinov2_vitl14/dinov2_vitl14_pretrain.pth", #hopefully this doesnt change :D
}

def roma_outdoor(device, weights=None, dinov2_weights=None):
    if weights is None:
        weights = torch.hub.load_state_dict_from_url(weight_urls["roma"]["outdoor"],
                                                     map_location=device)
    if dinov2_weights is None:
        dinov2_weights = torch.hub.load_state_dict_from_url(weight_urls["dinov2"],
                                                     map_location=device)
    return roma_model(resolution=(14*8*6,14*8*6), upsample_preds=True,
               weights=weights,dinov2_weights = dinov2_weights,device=device)

def roma_indoor(device, weights=None, dinov2_weights=None):
    if weights is None:
        weights = torch.hub.load_state_dict_from_url(weight_urls["roma"]["indoor"],
                                                     map_location=device)
    if dinov2_weights is None:
        dinov2_weights = torch.hub.load_state_dict_from_url(weight_urls["dinov2"],
                                                     map_location=device)
    return roma_model(resolution=(14*8*5,14*8*5), upsample_preds=False,
               weights=weights,dinov2_weights = dinov2_weights,device=device)
