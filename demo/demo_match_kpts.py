from PIL import Image
import torch
import cv2    
import numpy as np
from romatch import roma_outdoor

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if torch.backends.mps.is_available():
    device = torch.device('mps')

if __name__ == "__main__":
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument("--im_A_path", default="assets/sacre_coeur_A.jpg", type=str)
    parser.add_argument("--im_B_path", default="assets/sacre_coeur_B.jpg", type=str)
    parser.add_argument("--num_matches", default=50, type=int)
    parser.add_argument("--save_path", default="demo/roma_draw_matched_kpts.jpg", type=str)

    args, _ = parser.parse_known_args()
    im1_path = args.im_A_path
    im2_path = args.im_B_path
    num_matches = args.num_matches
    save_path = args.save_path
    # Create model
    roma_model = roma_outdoor(device=device)

    img1 = cv2.imread(im1_path)
    img2 = cv2.imread(im2_path)

    H_A, W_A = img1.shape[:2]
    H_B, W_B = img2.shape[:2]

    # Match
    warp, certainty = roma_model.match(im1_path, im2_path, device=device)
    # Sample matches for estimation
    matches, certainty = roma_model.sample(warp, certainty)
    kpts1, kpts2 = roma_model.to_pixel_coordinates(matches, H_A, W_A, H_B, W_B)    

    # Align the heights of both images
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]
    max_h = max(h1, h2)

    # Add padding to match heights
    if h1 < max_h:
        pad = max_h - h1
        img1 = np.pad(img1, ((0, pad), (0, 0), (0, 0)), mode='constant')
    if h2 < max_h:
        pad = max_h - h2
        img2 = np.pad(img2, ((0, pad), (0, 0), (0, 0)), mode='constant')
    
    # Horizontally concatenate the two images
    combined_img = np.hstack([img1, img2])
    
    # Connect matched keypoints with lines
    kpts1_np = kpts1.cpu().numpy()
    kpts2_np = kpts2.cpu().numpy()
    
    # Visualize matching results
    max_matches = min(num_matches, len(kpts1_np))
    for i in range(max_matches):
        x1, y1 = int(kpts1_np[i][0]), int(kpts1_np[i][1])
        x2, y2 = int(kpts2_np[i][0]) + w1, int(kpts2_np[i][1])
        
        # Draw matching lines (green)
        cv2.line(combined_img, (x1, y1), (x2, y2), (0, 255, 0), 1)
        # Mark keypoints (blue: first image, red: second image)
        cv2.circle(combined_img, (x1, y1), 3, (255, 0, 0), -1)
        cv2.circle(combined_img, (x2, y2), 3, (0, 0, 255), -1)
    
    # Add text
    text = f'Matched Keypoints: {max_matches} matches'
    cv2.putText(combined_img, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    
    cv2.imwrite(save_path, combined_img)
    