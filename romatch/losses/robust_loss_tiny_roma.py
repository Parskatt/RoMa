from einops.einops import rearrange
import torch
import torch.nn as nn
import torch.nn.functional as F
from romatch.utils.utils import get_gt_warp
import wandb
import romatch
import math

# This is slightly different than regular romatch due to significantly worse corresps
# The confidence loss is quite tricky here //Johan

class RobustLosses(nn.Module):
    def __init__(
        self,
        robust=False,
        center_coords=False,
        scale_normalize=False,
        ce_weight=0.01,
        local_loss=True,
        local_dist=None,
        smooth_mask = False,
        depth_interpolation_mode = "bilinear",
        mask_depth_loss = False,
        relative_depth_error_threshold = 0.05,
        alpha = 1.,
        c = 1e-3,
        epe_mask_prob_th = None,
        cert_only_on_consistent_depth = False,
    ):
        super().__init__()
        if local_dist is None:
            local_dist = {}
        self.robust = robust  # measured in pixels
        self.center_coords = center_coords
        self.scale_normalize = scale_normalize
        self.ce_weight = ce_weight
        self.local_loss = local_loss
        self.local_dist = local_dist
        self.smooth_mask = smooth_mask
        self.depth_interpolation_mode = depth_interpolation_mode
        self.mask_depth_loss = mask_depth_loss
        self.relative_depth_error_threshold = relative_depth_error_threshold
        self.avg_overlap = dict()
        self.alpha = alpha
        self.c = c
        self.epe_mask_prob_th = epe_mask_prob_th
        self.cert_only_on_consistent_depth = cert_only_on_consistent_depth

    def corr_volume_loss(self, mnn:torch.Tensor, corr_volume:torch.Tensor, scale):
        b, h,w, h,w = corr_volume.shape
        inv_temp = 10
        corr_volume = corr_volume.reshape(-1, h*w, h*w)
        nll = -(inv_temp*corr_volume).log_softmax(dim = 1) - (inv_temp*corr_volume).log_softmax(dim = 2)
        corr_volume_loss = nll[mnn[:,0], mnn[:,1], mnn[:,2]].mean()
        
        losses = {
            f"gm_corr_volume_loss_{scale}": corr_volume_loss.mean(),
        }
        wandb.log(losses, step = romatch.GLOBAL_STEP)
        return losses

    

    def regression_loss(self, x2, prob, flow, certainty, scale, eps=1e-8, mode = "delta"):
        epe = (flow.permute(0,2,3,1) - x2).norm(dim=-1)
        if scale in self.local_dist:
            prob = prob * (epe < (2 / 512) * (self.local_dist[scale] * scale)).float()
        if scale == 1:
            pck_05 = (epe[prob > 0.99] < 0.5 * (2/512)).float().mean()
            wandb.log({"train_pck_05": pck_05}, step = romatch.GLOBAL_STEP)
        if self.epe_mask_prob_th is not None:
            # if too far away from gt, certainty should be 0
            gt_cert = prob * (epe < scale * self.epe_mask_prob_th)
        else:
            gt_cert = prob
        if self.cert_only_on_consistent_depth:
            ce_loss = F.binary_cross_entropy_with_logits(certainty[:, 0][prob > 0], gt_cert[prob > 0])
        else:    
            ce_loss = F.binary_cross_entropy_with_logits(certainty[:, 0], gt_cert)
        a = self.alpha[scale] if isinstance(self.alpha, dict) else self.alpha
        cs = self.c * scale
        x = epe[prob > 0.99]
        reg_loss = cs**a * ((x/(cs))**2 + 1**2)**(a/2)
        if not torch.any(reg_loss):
            reg_loss = (ce_loss * 0.0)  # Prevent issues where prob is 0 everywhere
        losses = {
            f"{mode}_certainty_loss_{scale}": ce_loss.mean(),
            f"{mode}_regression_loss_{scale}": reg_loss.mean(),
        }
        wandb.log(losses, step = romatch.GLOBAL_STEP)
        return losses

    def forward(self, corresps, batch):
        scales = list(corresps.keys())
        tot_loss = 0.0
        # scale_weights due to differences in scale for regression gradients and classification gradients
        for scale in scales:
            scale_corresps = corresps[scale]
            scale_certainty, flow_pre_delta, delta_cls, offset_scale, scale_gm_corr_volume, scale_gm_certainty, flow, scale_gm_flow = (
                scale_corresps["certainty"],
                scale_corresps.get("flow_pre_delta"),
                scale_corresps.get("delta_cls"),
                scale_corresps.get("offset_scale"),
                scale_corresps.get("corr_volume"),
                scale_corresps.get("gm_certainty"),
                scale_corresps["flow"],
                scale_corresps.get("gm_flow"),

            )
            if flow_pre_delta is not None:
                flow_pre_delta = rearrange(flow_pre_delta, "b d h w -> b h w d")
                b, h, w, d = flow_pre_delta.shape
            else:
                # _ = 1
                b, _, h, w = scale_certainty.shape
            gt_warp, gt_prob = get_gt_warp(                
            batch["im_A_depth"],
            batch["im_B_depth"],
            batch["T_1to2"],
            batch["K1"],
            batch["K2"],
            H=h,
            W=w,
            )
            x2 = gt_warp.float()
            prob = gt_prob
                        
            if scale_gm_corr_volume is not None:
                gt_warp_back, _ = get_gt_warp(                
                batch["im_B_depth"],
                batch["im_A_depth"],
                batch["T_1to2"].inverse(),
                batch["K2"],
                batch["K1"],
                H=h,
                W=w,
                )
                grid = torch.stack(torch.meshgrid(torch.linspace(-1+1/w, 1-1/w, w), torch.linspace(-1+1/h, 1-1/h, h), indexing='xy'), dim =-1).to(gt_warp.device)
                #fwd_bck = F.grid_sample(gt_warp_back.permute(0,3,1,2), gt_warp, align_corners=False, mode = 'bilinear').permute(0,2,3,1)
                #diff = (fwd_bck - grid).norm(dim = -1)
                with torch.no_grad():
                    D_B = torch.cdist(gt_warp.float().reshape(-1,h*w,2), grid.reshape(-1,h*w,2))
                    D_A = torch.cdist(grid.reshape(-1,h*w,2), gt_warp_back.float().reshape(-1,h*w,2))
                    inds = torch.nonzero((D_B == D_B.min(dim=-1, keepdim = True).values) 
                                        * (D_A == D_A.min(dim=-2, keepdim = True).values)
                                        * (D_B < 0.01)
                                        * (D_A < 0.01))

                gm_cls_losses = self.corr_volume_loss(inds, scale_gm_corr_volume, scale)
                gm_loss = gm_cls_losses[f"gm_corr_volume_loss_{scale}"]
                tot_loss = tot_loss + gm_loss
            elif scale_gm_flow is not None:
                gm_flow_losses = self.regression_loss(x2, prob, scale_gm_flow, scale_gm_certainty, scale, mode = "gm")
                gm_loss = self.ce_weight * gm_flow_losses[f"gm_certainty_loss_{scale}"] + gm_flow_losses[f"gm_regression_loss_{scale}"]
                tot_loss = tot_loss +  gm_loss
            delta_regression_losses = self.regression_loss(x2, prob, flow, scale_certainty, scale)
            reg_loss = self.ce_weight * delta_regression_losses[f"delta_certainty_loss_{scale}"] + delta_regression_losses[f"delta_regression_loss_{scale}"]
            tot_loss = tot_loss + reg_loss
        return tot_loss
