import os
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
import torch
from PIL import Image
import torch.nn.functional as F
import numpy as np
from romatch.utils.utils import tensor_to_pil

from romatch import roma_indoor

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if torch.backends.mps.is_available():
    device = torch.device('mps')

if __name__ == "__main__":
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument("--im_A_path", default="00886.jpg", type=str)
    parser.add_argument("--im_B_path", default="00888.jpg", type=str)
    parser.add_argument("--save_path", default="1latest.jpg", type=str)

    args, _ = parser.parse_known_args()
    im1_path = args.im_A_path
    im2_path = args.im_B_path
    save_path = args.save_path

    # Create model
    roma_model = roma_indoor(device=device, coarse_res=560, upsample_res=(864, 1152))

    H, W = roma_model.get_output_resolution()

    im1 = Image.open(im1_path).resize((W, H))
    im2 = Image.open(im2_path).resize((W, H))

    # Match
    warp, certainty = roma_model.match(im1_path, im2_path, device=device)
    # Sampling not needed, but can be done with model.sample(warp, certainty)
    x1 = (torch.tensor(np.array(im1)) / 255).to(device).permute(2, 0, 1)
    x2 = (torch.tensor(np.array(im2)) / 255).to(device).permute(2, 0, 1)

    # certainty_1 = certainty[:, :1152]
    # #print(certainty_1.shape)
    # certainty_expanded_1 = certainty_1.unsqueeze(0)
    # #print(certainty_expanded.shape)
    # dissimilarity_image1 = x1 * (1 - certainty_expanded_1)
    # tensor_to_pil(dissimilarity_image1, unnormalize=False).save(save_path)
    confidence_threshold = 0.009 #0.009

    # Create a mask where the confidence is below the threshold (dissimilar regions)
    low_confidence_mask = certainty < confidence_threshold

    # Convert the mask to a float tensor for multiplication
    low_confidence_mask = low_confidence_mask.float()

    # certainty_1 = low_confidence_mask[:, :1152]
    # certainty_expanded_1 = certainty_1.unsqueeze(0)
    # #print(certainty_expanded.shape)
    # dissimilarity_image1 = x1 * certainty_expanded_1#(1 - certainty_expanded_2)
    # tensor_to_pil(dissimilarity_image1, unnormalize=False).save(save_path)
    


    certainty_2 = low_confidence_mask[:,1152:]
    certainty_expanded_2 = certainty_2.unsqueeze(0)
    #print(certainty_expanded.shape)
    dissimilarity_image2 = x2 * certainty_expanded_2#(1 - certainty_expanded_2)
    tensor_to_pil(dissimilarity_image2, unnormalize=False).save(save_path)


    # im2_transfer_rgb = F.grid_sample(
    # x2[None], warp[:,:W, 2:][None], mode="bilinear", align_corners=False
    # )[0]
    # im1_transfer_rgb = F.grid_sample(
    # x1[None], warp[:, W:, :2][None], mode="bilinear", align_corners=False
    # )[0]
    # warp_im = torch.cat((im2_transfer_rgb,im1_transfer_rgb),dim=2)
    # white_im = torch.ones((H,2*W),device=device)
    # vis_im = certainty * warp_im + (1 - certainty) * white_im
    # tensor_to_pil(vis_im, unnormalize=False).save(save_path)
