# RoMa: Revisiting Robust Losses for Dense Feature Matching

**NOTE!!! Very early code, there might be bugs**

The experiments in the paper are provided in the [experiments folder](experiments).
The codebase is in the [roma folder](roma).

## Setup/Install
In your python environment (tested on Linux python 3.10), run:
```bash
pip install -e .
```

## Training
1. First follow the instructions provided here: https://github.com/Parskatt/DKM for downloading and preprocessing datasets.
2. Run the relevant experiment, e.g.,
```bash
torchrun --nproc_per_node=4 --nnodes=1 --rdzv_backend=c10d experiments/roma_outdoor.py
```
## Testing
```bash
python experiments/roma_outdoor.py --only_test --benchmark mega-1500
```

## Acknowledgement
Our codebase builds on the code in [DKM](https://github.com/Parskatt/DKM).