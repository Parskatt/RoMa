from PIL import Image
import numpy as np

import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt



if __name__ == "__main__":
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument("--im_A_path", default="assets/toronto_A.jpg", type=str)
    parser.add_argument("--im_B_path", default="assets/toronto_B.jpg", type=str)
    parser.add_argument("--save_path", default="demo/roma_warp_toronto.jpg", type=str)

    args, _ = parser.parse_known_args()
    im1_path = args.im_A_path
    im2_path = args.im_B_path
    save_path = args.save_path

    img1 = cv.imread(im1_path,cv.IMREAD_GRAYSCALE)          # queryImage
    img2 = cv.imread(im2_path,cv.IMREAD_GRAYSCALE) # trainImage
    # Initiate SIFT detector
    sift = cv.SIFT_create()
    # find the keypoints and descriptors with SIFT
    kp1, des1 = sift.detectAndCompute(img1,None)
    kp2, des2 = sift.detectAndCompute(img2,None)
    # BFMatcher with default params
    bf = cv.BFMatcher()
    matches = bf.knnMatch(des1,des2,k=2)
    # Apply ratio test
    good = []
    for m,n in matches:
        if m.distance < 0.75*n.distance:
            good.append([m])
    # cv.drawMatchesKnn expects list of lists as matches.
    draw_params = dict(matchColor = (255,0,0), # draw matches in red color
                   singlePointColor = None,
                   flags = 2)

    img3 = cv.drawMatchesKnn(img1,kp1,img2,kp2,good,None,**draw_params)
    Image.fromarray(img3).save("demo/sift_matches.png")
