import os
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
import torch
from PIL import Image
import torch.nn.functional as F
import numpy as np
from romatch.utils.utils import tensor_to_pil

from romatch import tiny_roma_v1_outdoor

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if torch.backends.mps.is_available():
    device = torch.device('mps')

if __name__ == "__main__":
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument("--im_A_path", default="assets/sacre_coeur_A.jpg", type=str)
    parser.add_argument("--im_B_path", default="assets/sacre_coeur_B.jpg", type=str)
    parser.add_argument("--save_A_path", default="demo/tiny_roma_warp_A.jpg", type=str)
    parser.add_argument("--save_B_path", default="demo/tiny_roma_warp_B.jpg", type=str)

    args, _ = parser.parse_known_args()
    im1_path = args.im_A_path
    im2_path = args.im_B_path

    # Create model
    roma_model = tiny_roma_v1_outdoor(device=device)

    # Match
    warp, certainty1 = roma_model.match(im1_path, im2_path)
    
    h1, w1 = warp.shape[:2]
    
    # maybe im1.size != im2.size
    im1 = Image.open(im1_path).resize((w1, h1))
    im2 = Image.open(im2_path)
    x1 = (torch.tensor(np.array(im1)) / 255).to(device).permute(2, 0, 1)
    x2 = (torch.tensor(np.array(im2)) / 255).to(device).permute(2, 0, 1)
    
    h2, w2 = x2.shape[1:]
    g1_p2x = w2 / 2 * (warp[..., 2] + 1)
    g1_p2y = h2 / 2 * (warp[..., 3] + 1)
    g2_p1x = torch.zeros((h2, w2), dtype=torch.float32).to(device) - 2
    g2_p1y = torch.zeros((h2, w2), dtype=torch.float32).to(device) - 2

    x, y = torch.meshgrid(
        torch.arange(w1, device=device),
        torch.arange(h1, device=device),
        indexing="xy",
    )
    g2x = torch.round(g1_p2x[y, x]).long()
    g2y = torch.round(g1_p2y[y, x]).long()
    idx_x = torch.bitwise_and(0 <= g2x, g2x < w2)
    idx_y = torch.bitwise_and(0 <= g2y, g2y < h2)
    idx = torch.bitwise_and(idx_x, idx_y)
    g2_p1x[g2y[idx], g2x[idx]] = x[idx].float() * 2 / w1 - 1
    g2_p1y[g2y[idx], g2x[idx]] = y[idx].float() * 2 / h1 - 1

    certainty2 = F.grid_sample(
        certainty1[None][None],
        torch.stack([g2_p1x, g2_p1y], dim=2)[None],
        mode="bilinear",
        align_corners=False,
    )[0]
    
    white_im1 = torch.ones((h1, w1), device = device)
    white_im2 = torch.ones((h2, w2), device = device)
    
    certainty1 = F.avg_pool2d(certainty1[None], kernel_size=5, stride=1, padding=2)[0]
    certainty2 = F.avg_pool2d(certainty2[None], kernel_size=5, stride=1, padding=2)[0]
    
    vis_im1 = certainty1 * x1 + (1 - certainty1) * white_im1
    vis_im2 = certainty2 * x2 + (1 - certainty2) * white_im2
    
    tensor_to_pil(vis_im1, unnormalize=False).save(args.save_A_path)
    tensor_to_pil(vis_im2, unnormalize=False).save(args.save_B_path)