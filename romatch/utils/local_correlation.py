import torch
import torch.nn.functional as F

def local_correlation(
    feature0,
    feature1,
    local_radius,
    padding_mode="zeros",
    flow = None,
    sample_mode = "bilinear",
):
    r = local_radius
    K = (2*r+1)**2
    B, c, h, w = feature0.size()
    corr = torch.empty((B,K,h,w), device = feature0.device, dtype=feature0.dtype)
    if flow is None:
        # If flow is None, assume feature0 and feature1 are aligned
        coords = torch.meshgrid(
                (
                    torch.linspace(-1 + 1 / h, 1 - 1 / h, h, device=feature0.device),
                    torch.linspace(-1 + 1 / w, 1 - 1 / w, w, device=feature0.device),
                ))
        coords = torch.stack((coords[1], coords[0]), dim=-1)[
            None
        ].expand(B, h, w, 2)
    else:
        coords = flow.permute(0,2,3,1) # If using flow, sample around flow target.
    local_window = torch.meshgrid(
                (
                    torch.linspace(-2*local_radius/h, 2*local_radius/h, 2*r+1, device=feature0.device),
                    torch.linspace(-2*local_radius/w, 2*local_radius/w, 2*r+1, device=feature0.device),
                ))
    local_window = torch.stack((local_window[1], local_window[0]), dim=-1)[
            None
        ].expand(1, 2*r+1, 2*r+1, 2).reshape(1, (2*r+1)**2, 2)
    for _ in range(B):
        with torch.no_grad():
            local_window_coords = (coords[_,:,:,None]+local_window[:,None,None]).reshape(1,h,w*(2*r+1)**2,2)
            window_feature = F.grid_sample(
                feature1[_:_+1], local_window_coords, padding_mode=padding_mode, align_corners=False, mode = sample_mode, #
            )
            window_feature = window_feature.reshape(c,h,w,(2*r+1)**2)
        corr[_] = (feature0[_,...,None]/(c**.5)*window_feature).sum(dim=0).permute(2,0,1)
    return corr
