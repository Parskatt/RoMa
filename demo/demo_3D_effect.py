from PIL import Image
import torch
import torch.nn.functional as F
import numpy as np
from roma.utils.utils import tensor_to_pil

from roma import roma_outdoor

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


if __name__ == "__main__":
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument("--im_A_path", default="assets/toronto_A.jpg", type=str)
    parser.add_argument("--im_B_path", default="assets/toronto_B.jpg", type=str)
    parser.add_argument("--save_path", default="demo/gif/roma_warp_toronto", type=str)

    args, _ = parser.parse_known_args()
    im1_path = args.im_A_path
    im2_path = args.im_B_path
    save_path = args.save_path

    # Create model
    roma_model = roma_outdoor(device=device, coarse_res=560, upsample_res=(864, 1152))
    roma_model.symmetric = False

    H, W = roma_model.get_output_resolution()

    im1 = Image.open(im1_path).resize((W, H))
    im2 = Image.open(im2_path).resize((W, H))

    # Match
    warp, certainty = roma_model.match(im1_path, im2_path, device=device)
    # Sampling not needed, but can be done with model.sample(warp, certainty)
    x1 = (torch.tensor(np.array(im1)) / 255).to(device).permute(2, 0, 1)
    x2 = (torch.tensor(np.array(im2)) / 255).to(device).permute(2, 0, 1)

    coords_A, coords_B = warp[...,:2], warp[...,2:]
    for i, x in enumerate(np.linspace(0,2*np.pi,200)):
        t = (1 + np.cos(x))/2
        interp_warp = (1-t)*coords_A + t*coords_B
        im2_transfer_rgb = F.grid_sample(
        x2[None], interp_warp[None], mode="bilinear", align_corners=False
        )[0]
        tensor_to_pil(im2_transfer_rgb, unnormalize=False).save(f"{save_path}_{i:03d}.jpg")