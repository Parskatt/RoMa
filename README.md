# RoMa: Revisiting Robust Losses for Dense Feature Matching
### [Project Page (TODO)](https://parskatt.github.io/RoMa) | [Paper](https://arxiv.org/abs/2305.15404)
<br/>

> RoMa: Revisiting Robust Lossses for Dense Feature Matching  
> [Johan Edstedt](https://scholar.google.com/citations?user=Ul-vMR0AAAAJ), [Qiyu Sun](https://scholar.google.com/citations?user=HS2WuHkAAAAJ), [Georg Bökman](https://scholar.google.com/citations?user=FUE3Wd0AAAAJ), [Mårten Wadenbäck](https://scholar.google.com/citations?user=6WRQpCQAAAAJ), [Michael Felsberg](https://scholar.google.com/citations?&user=lkWfR08AAAAJ)  
> Arxiv 2023

**NOTE!!! Very early code, there might be bugs**

The codebase is in the [roma folder](roma).

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
Due to our dependency on [DINOv2](https://github.com/facebookresearch/dinov2/blob/main/LICENSE), the license is sadly non-commercial only for the moment.

## Acknowledgement
Our codebase builds on the code in [DKM](https://github.com/Parskatt/DKM).

## BibTeX
If you find our models useful, please consider citing our paper!
```
@article{edstedt2023roma,
title={{RoMa}: Revisiting Robust Lossses for Dense Feature Matching},
author={Edstedt, Johan and Sun, Qiyu and Bökman, Georg and Wadenbäck, Mårten and Felsberg, Michael},
journal={arXiv preprint arXiv:2305.15404},
year={2023}
}
```
