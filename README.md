# 
<p align="center">
  <h1 align="center"> <ins>RoMa</ins> 🏛️:<br> Robust Dense Feature Matching <br> ⭐CVPR 2024⭐</h1>
  <p align="center">
    <a href="https://scholar.google.com/citations?user=Ul-vMR0AAAAJ">Johan Edstedt</a>
    ·
    <a href="https://scholar.google.com/citations?user=HS2WuHkAAAAJ">Qiyu Sun</a>
    ·
    <a href="https://scholar.google.com/citations?user=FUE3Wd0AAAAJ">Georg Bökman</a>
    ·
    <a href="https://scholar.google.com/citations?user=6WRQpCQAAAAJ">Mårten Wadenbäck</a>
    ·
    <a href="https://scholar.google.com/citations?user=lkWfR08AAAAJ">Michael Felsberg</a>
  </p>
  <h2 align="center"><p>
    <a href="https://arxiv.org/abs/2305.15404" align="center">Paper</a> | 
    <a href="https://parskatt.github.io/RoMa" align="center">Project Page</a>
  </p></h2>
  <div align="center"></div>
</p>
<br/>
<p align="center">
    <img src="https://github.com/Parskatt/RoMa/assets/22053118/15d8fea7-aa6d-479f-8a93-350d950d006b" alt="example" width=80%>
    <br>
    <em>RoMa is the robust dense feature matcher capable of estimating pixel-dense warps and reliable certainties for almost any image pair.</em>
</p>

## Setup/Install
In your python environment (tested on Linux python 3.10), run:
```bash
pip install -e .
```
## Demo / How to Use
We provide two demos in the [demos folder](demo).
Here's the gist of it:
```python
from roma import roma_outdoor
roma_model = roma_outdoor(device=device)
# Match
warp, certainty = roma_model.match(imA_path, imB_path, device=device)
# Sample matches for estimation
matches, certainty = roma_model.sample(warp, certainty)
# Convert to pixel coordinates (RoMa produces matches in [-1,1]x[-1,1])
kptsA, kptsB = roma_model.to_pixel_coordinates(matches, H_A, W_A, H_B, W_B)
# Find a fundamental matrix (or anything else of interest)
F, mask = cv2.findFundamentalMat(
    kptsA.cpu().numpy(), kptsB.cpu().numpy(), ransacReprojThreshold=0.2, method=cv2.USAC_MAGSAC, confidence=0.999999, maxIters=10000
)
```

**New**: You can also match arbitrary keypoints with RoMa. A demo for this will be added soon.
## Settings

### Resolution
By default RoMa uses an initial resolution of (560,560) which is then upsampled to (864,864). 
You can change this at construction (see roma_outdoor kwargs).
You can also change this later, by changing the roma_model.w_resized, roma_model.h_resized, and roma_model.upsample_res.

### Sampling
roma_model.sample_thresh controls the thresholding used when sampling matches for estimation. In certain cases a lower or higher threshold may improve results.


## Reproducing Results
The experiments in the paper are provided in the [experiments folder](experiments).

### Training
1. First follow the instructions provided here: https://github.com/Parskatt/DKM for downloading and preprocessing datasets.
2. Run the relevant experiment, e.g.,
```bash
torchrun --nproc_per_node=4 --nnodes=1 --rdzv_backend=c10d experiments/roma_outdoor.py
```
### Testing
```bash
python experiments/roma_outdoor.py --only_test --benchmark mega-1500
```
## License
All our code except DINOv2 is MIT license.
DINOv2 has an Apache 2 license [DINOv2](https://github.com/facebookresearch/dinov2/blob/main/LICENSE).

## Acknowledgement
Our codebase builds on the code in [DKM](https://github.com/Parskatt/DKM).

## BibTeX
If you find our models useful, please consider citing our paper!
```
@article{edstedt2024roma,
title={{RoMa: Robust Dense Feature Matching}},
author={Edstedt, Johan and Sun, Qiyu and Bökman, Georg and Wadenbäck, Mårten and Felsberg, Michael},
journal={IEEE Conference on Computer Vision and Pattern Recognition},
year={2024}
}
```
